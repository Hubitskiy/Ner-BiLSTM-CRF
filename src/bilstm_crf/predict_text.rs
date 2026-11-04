use burn::prelude::*;
use indexmap::IndexMap;
use serde::Serialize;

use neuro::tokenizers::tokenizer::{normalizer, tokenize_with_offset};
use neuro::vocabs::vocab::KeyValeBasedVocabulary;
use crate::bilstm_crf::{
    bilstmcrf::BiLstmCRFModel,
    types::NerEntity,
    vocab::{NerChars, NerSamples, NerVocab},
};


/// Prediction of [`predict_ner_entitties`] for a single text.
#[derive(Debug, Serialize)]
pub struct NerPrediction {
    /// Normalized text.
    text: String,
    /// Tokens of the normalized text.
    tokens: Vec<String>,
    /// Predicted BIO label of each token.
    labels: Vec<String>,
    /// Entity spans assembled from the labels.
    entities: Vec<NerEntity>,
    /// First `send_from` and `send_to` values found among the entities.
    slots: IndexMap<String, String>,
}


impl NerPrediction {

    /// Returns the normalized text.
    #[inline]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the tokens of the normalized text.
    #[inline]
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Returns the predicted BIO label of each token.
    #[inline]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Returns the entity spans assembled from the labels.
    #[inline]
    pub fn entities(&self) -> &[NerEntity] {
        &self.entities
    }

    /// Returns the first `send_from` and `send_to` values found among the entities.
    #[inline]
    pub fn slots(&self) -> &IndexMap<String, String> {
        &self.slots
    }
}


/// Assembles entity spans from per-token BIO labels.
///
/// A `B-` label always starts a new entity. An `I-` label extends the current
/// entity if it has the same type, otherwise it starts a new one. `O` and labels
/// without a BIO prefix close the current entity.
///
/// # Arguments
///
/// * `offsets` - byte offsets `(start, end)` of each token in `text`.
/// * `labels` - BIO label of each token.
/// * `text` - normalized text the offsets point into.
///
/// # Returns
///
/// Entity spans in the order they appear in the text.
pub fn decode_bio(
    offsets: &[(usize, usize)],
    labels: &[String],
    text: &str,
) -> Vec<NerEntity> {
    let mut entities = Vec::new();
    // (entity type, first token, last token) of the entity being assembled
    let mut current: Option<(&str, usize, usize)> = None;

    let mut close = |current: &mut Option<(&str, usize, usize)>| {
        if let Some((entity_type, start_token, end_token)) = current.take() {
            let start = offsets[start_token].0;
            let end = offsets[end_token].1;
            entities.push(NerEntity::define(start, end, text[start..end].to_string(), entity_type.to_string()));
        }
    };

    for (index, label) in labels.iter().enumerate() {
        let Some((prefix, entity_type)) = label.split_once('-') else {
            close(&mut current);
            continue;
        };

        match current {
            Some((current_type, _, ref mut end_token)) if prefix == "I" && current_type == entity_type => {
                *end_token = index;
            }
            _ => {
                close(&mut current);
                current = Some((entity_type, index, index));
            }
        }
    }

    close(&mut current);

    entities
}


/// Picks the first `send_from` and `send_to` values among `entities`.
pub fn extract_slots(entities: &[NerEntity]) -> IndexMap<String, String> {
    let mut slots = IndexMap::new();

    for entity in entities {
        if matches!(entity.entity(), "send_from" | "send_to") && !slots.contains_key(entity.entity()) {
            slots.insert(entity.entity().to_string(), entity.value().to_string());
        }
    }

    slots
}


/// Tags a single text with the model and extracts its entities and slots.
///
/// The text is normalized and tokenized the same way as the training data.
/// Tokens and characters missing from their vocabularies get the
/// [`NerVocab::unk_value`] id.
///
/// # Arguments
///
/// * `model` - trained model loaded on a device without autodiff.
/// * `text` - raw text, not normalized.
/// * `words_vocab` - vocabulary of tokens.
/// * `chars_vocab` - vocabulary of characters.
/// * `inv_labels_vocab` - maps label ids back to BIO labels, see [`NerVocab::invert`].
/// * `device` - device on which the input tensors are allocated.
///
/// # Returns
///
/// Normalized text with its tokens, labels, entities and slots. A text without
/// tokens yields an empty prediction.
///
/// # Panics
///
/// Panics if the model predicts a label id missing from `inv_labels_vocab`.
pub fn predict_ner_entitties(
    model: &BiLstmCRFModel,
    text: &str,
    words_vocab: &NerVocab<NerSamples>,
    chars_vocab: &NerVocab<NerChars>,
    inv_labels_vocab: &IndexMap<i64, String>,
    device: &Device,
) -> NerPrediction {
    let text = normalizer(text);
    let (tokens, offsets) = tokenize_with_offset(&text, normalizer).own_parts();

    if tokens.is_empty() {
        return NerPrediction {
            text,
            tokens,
            labels: Vec::new(),
            entities: Vec::new(),
            slots: IndexMap::new(),
        };
    }

    // Offsets must point into `text`, otherwise `decode_bio` slices the wrong bytes.
    debug_assert!(
        offsets.iter().all(|&(start, end)| {
            start <= end && end <= text.len() && text.is_char_boundary(start) && text.is_char_boundary(end)
        }),
        "Token offsets don't match the normalized text"
    );

    // Words → ids
    let word_unk = NerVocab::<NerSamples>::unk_value();
    let token_ids = tokens
        .iter()
        .map(|token| words_vocab.get(token).copied().unwrap_or(word_unk))
        .collect::<Vec<i64>>();

    // Characters → ids
    let char_unk = NerVocab::<NerChars>::unk_value();
    let char_ids = tokens
        .iter()
        .map(|token| {
            token
                .chars()
                .map(|ch| chars_vocab.get(&ch.to_string()).copied().unwrap_or(char_unk))
                .collect::<Vec<i64>>()
        })
        .collect::<Vec<_>>();

    let seq_len = token_ids.len();
    let max_char_len = char_ids.iter().map(Vec::len).max().unwrap_or(1).max(1);

    // Flat [seq_len * max_char_len] buffer padded with the char padding id
    let mut padded_char_ids = vec![NerVocab::<NerChars>::pad_value(); seq_len * max_char_len];
    for (index, chars) in char_ids.iter().enumerate() {
        let start = index * max_char_len;
        padded_char_ids[start..start + chars.len()].copy_from_slice(chars);
    }

    // Batch of a single sentence
    let token_tensor = Tensor::<2, Int>::from_data(
        TensorData::new(token_ids, [1, seq_len]),
        device,
    );
    let char_tensor = Tensor::<3, Int>::from_data(
        TensorData::new(padded_char_ids, [1, seq_len, max_char_len]),
        device,
    );
    let mask = Tensor::<2, Bool>::from_data(
        TensorData::new(vec![true; seq_len], [1, seq_len]),
        device,
    );

    // Viterbi path: one label id per token
    let mut pred_paths = model.decode(token_tensor, char_tensor, mask);
    let path = pred_paths.swap_remove(0);

    debug_assert_eq!(path.len(), seq_len, "Viterbi path length must match the number of tokens");

    let labels = path
        .into_iter()
        .map(|id| {
            inv_labels_vocab
                .get(&(id as i64))
                .unwrap_or_else(|| panic!("Label id {id} is missing from the labels vocabulary"))
                .clone()
        })
        .collect::<Vec<String>>();

    let entities = decode_bio(&offsets, &labels, &text);
    let slots = extract_slots(&entities);

    NerPrediction {
        text,
        tokens,
        labels,
        entities,
        slots,
    }
}


#[cfg(test)]
pub mod test {
    use super::*;

    fn labels(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    fn spans(entities: &[NerEntity]) -> Vec<(&str, usize, usize, &str)> {
        entities.iter().map(|e| (e.entity(), e.start(), e.end(), e.value())).collect()
    }

    #[test]
    fn decode_bio_joins_multi_token_entities() {
        let text = "from new york to paris";
        let offsets = vec![(0, 4), (5, 8), (9, 13), (14, 16), (17, 22)];

        let entities = decode_bio(&offsets, &labels(&["O", "B-send_from", "I-send_from", "O", "B-send_to"]), text);

        assert_eq!(spans(&entities), vec![("send_from", 5, 13, "new york"), ("send_to", 17, 22, "paris")]);
    }

    #[test]
    fn decode_bio_starts_new_entity_on_mismatched_inside_label() {
        let text = "a b c";
        let offsets = vec![(0, 1), (2, 3), (4, 5)];

        let entities = decode_bio(&offsets, &labels(&["B-send_from", "I-send_to", "I-send_to"]), text);

        assert_eq!(spans(&entities), vec![("send_from", 0, 1, "a"), ("send_to", 2, 5, "b c")]);
    }

    #[test]
    fn extract_slots_keeps_first_value() {
        let entities = vec![
            NerEntity::define(0, 1, "a".to_string(), "send_to".to_string()),
            NerEntity::define(2, 3, "b".to_string(), "other".to_string()),
            NerEntity::define(4, 5, "c".to_string(), "send_to".to_string()),
            NerEntity::define(6, 7, "d".to_string(), "send_from".to_string()),
        ];

        let slots = extract_slots(&entities);

        assert_eq!(slots.get("send_to").map(String::as_str), Some("a"));
        assert_eq!(slots.get("send_from").map(String::as_str), Some("d"));
        assert_eq!(slots.len(), 2);
    }
}
