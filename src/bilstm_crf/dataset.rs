use std::io::Error;
use burn::data::dataset::{Dataset, DatasetError};
use serde::{Deserialize, Serialize};
use neuro::tokenizers::tokenizer::ToTokenizedWithOffset;
use neuro::vocabs::vocab::KeyValeBasedVocabulary;
use crate::bilstm_crf::types::{NerExampleSample};
use crate::bilstm_crf::vocab::{NerChars, NerLabels, NerSamples, NerVocab};


/// Tokenized and id-encoded NER sample returned by [`NerDataset`].
#[derive(Debug, Clone, PartialOrd, PartialEq, Deserialize, Serialize)]
pub struct DatasetNerSample {
    /// Normalized text.
    text: String,
    /// Tokens of the normalized text.
    tokens: Vec<String>,
    /// Byte offsets `(start, end)` of each token in the normalized text.
    offsets: Vec<(usize, usize)>,
    /// Word id of each token.
    token_ids: Vec<i64>,
    /// Character ids of each token.
    char_ids: Vec<Vec<i64>>,
    /// BIO label id of each token.
    label_ids: Vec<i64>
}


impl DatasetNerSample {

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

    /// Returns the byte offsets `(start, end)` of each token in the normalized text.
    #[inline]
    pub fn offsets(&self) -> &[(usize, usize)] {
        &self.offsets
    }

    /// Returns the word id of each token.
    #[inline]
    pub fn token_ids(&self) -> &[i64] {
        &self.token_ids
    }

    /// Returns the character ids of each token.
    #[inline]
    pub fn char_ids(&self) -> &[Vec<i64>] {
        &self.char_ids
    }

    /// Returns the BIO label id of each token.
    #[inline]
    pub fn label_ids(&self) -> &[i64] {
        &self.label_ids
    }
}


impl DatasetNerSample {

    /// Creates a sample from already encoded parts.
    ///
    /// # Arguments
    ///
    /// * `text` - normalized text.
    /// * `tokens` - tokens of the normalized text.
    /// * `offsets` - byte offsets of each token in the normalized text.
    /// * `token_ids` - word id of each token.
    /// * `char_ids` - character ids of each token.
    /// * `label_ids` - BIO label id of each token.
    pub fn define(
        text: String,
        tokens: Vec<String>,
        offsets: Vec<(usize, usize)>,
        token_ids: Vec<i64>,
        char_ids: Vec<Vec<i64>>,
        label_ids: Vec<i64>
    ) -> Self {

        Self {
            text,
            tokens,
            offsets,
            token_ids,
            char_ids,
            label_ids
        }

    }

}

/// Dataset of raw NER samples that are encoded on access.
///
/// Samples are tokenized and converted to ids in [`Dataset::get`], not up front.
pub struct NerDataset{
    /// Raw annotated samples.
    raw_samples: Vec<NerExampleSample>,
    /// Maps tokens to word ids.
    words_vocab: NerVocab<NerSamples>,
    /// Maps characters to character ids.
    chars_vocab: NerVocab<NerChars>,
    /// Maps BIO labels to label ids.
    labels_vocab: NerVocab<NerLabels>,
}


impl NerDataset{

    /// Creates a dataset from raw samples and their vocabularies.
    ///
    /// # Arguments
    ///
    /// * `raw_samples` - annotated samples.
    /// * `words_vocab` - vocabulary of tokens.
    /// * `chars_vocab` - vocabulary of characters.
    /// * `labels_vocab` - vocabulary of BIO labels.
    pub fn create(
        raw_samples: Vec<NerExampleSample>,
        words_vocab: NerVocab<NerSamples>,
        chars_vocab: NerVocab<NerChars>,
        labels_vocab: NerVocab<NerLabels>,
    ) -> Self{
        Self {
            raw_samples,
            words_vocab,
            chars_vocab,
            labels_vocab
        }
    }
}


/// Converts the entity spans of a sample to one BIO label per token.
///
/// A token belongs to an entity if their offset ranges overlap. The first such
/// token gets `B-<entity>`, the following ones get `I-<entity>`, and tokens outside
/// of any entity get `O`. If entities overlap, the one listed later wins.
///
/// # Arguments
///
/// * `tokens` - tokens of the sample.
/// * `offsets` - byte offsets of each token in the normalized text.
/// * `sample` - sample that provides the entity spans.
///
/// # Returns
///
/// BIO labels, one per token.
pub fn spans_to_bio_labels(
    tokens: &[String],
    offsets: &[(usize, usize)],
    sample: &NerExampleSample
) -> Vec<String> {
    let mut labels = vec!["O".to_string(); tokens.len()];

    for entity in sample.entities() {
        let mut is_first_covered_token = true;

        for (index, &(token_start, token_end)) in offsets.iter().enumerate() {
            let overlap = token_start.max(entity.start()) < token_end.min(entity.end());

            if !overlap {
                continue;
            }

            let prefix = if is_first_covered_token { "B" } else { "I" };
            labels[index] = format!("{prefix}-{}", entity.entity());

            is_first_covered_token = false;
        }
    }

    labels
}

impl Dataset<DatasetNerSample> for NerDataset {

    /// Tokenizes the sample at `index` and encodes it to ids.
    ///
    /// Tokens and characters missing from their vocabularies get the
    /// [`NerVocab::unk_value`] id, and so do labels missing from the labels vocabulary.
    ///
    /// # Arguments
    ///
    /// * `index` - position of the sample.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] if `index` is out of bounds.
    fn get(&self, index: usize) -> Result<DatasetNerSample, DatasetError> {
        let sample = self.raw_samples
            .get(index)
            .ok_or_else( ||
                DatasetError::new(
                    Error::other(format!("Sample index out of bounds: {index}"))
                )
            )?;

        let text = sample.to_normalized();
        let tokenized_offset = sample.to_tokenized_with_offset();

        let token_ids  = tokenized_offset.tokens()
            .iter()
            .map(|token|{
                *self.words_vocab.get(token).unwrap_or(&NerVocab::<NerSamples>::unk_value())
            })
            .collect::<Vec<_>>();

        let char_ids = tokenized_offset.tokens()
            .iter()
            .map(|token| token.chars().map(|ch| ch.to_string()).collect::<Vec<String>>())
            .flat_map(|chars| {
                vec![
                    chars.iter()
                        .map(
                            |ch| *self.chars_vocab.get(ch).unwrap_or(&NerVocab::<NerSamples>::unk_value())
                        ).collect::<Vec<_>>()
                ]
            }
            )
            .collect::<Vec<Vec<i64>>>();

        let labels = spans_to_bio_labels(tokenized_offset.tokens(), tokenized_offset.offsets(), &sample);

        let label_ids = labels.iter()
            .map(|label| *self.labels_vocab.get(label).unwrap_or(&NerVocab::<NerSamples>::unk_value()))
            .collect::<Vec<i64>>();

        let (tokens, offsets) = tokenized_offset.own_parts();

        Ok(
            DatasetNerSample::define(
                text,
                tokens,
                offsets,
                token_ids,
                char_ids,
                label_ids
            )
        )

    }

    /// Returns the number of samples.
    fn len(&self) -> usize {
        self.raw_samples.len()
    }
}


pub mod test {

    use super::*;

    /// Checks that the first sample is tokenized and encoded to the expected ids.
    #[test]
    fn create_dataset() {
        let samples_json = r#"
            [
                {
                    "text": "does any airline have an early afternoon flight from boston to pittsburgh",
                    "entities": [
                        {
                            "start": 53,
                            "end": 59,
                            "value": "boston",
                            "entity": "send_from"
                        },
                        {
                            "start": 63,
                            "end": 73,
                            "value": "pittsburgh",
                            "entity": "send_to"
                        }
                    ]
                },
                {
                    "text": "show me the flights on monday",
                    "entities": []
                },
                {
                    "text": "what is the earliest flight that serves a meal",
                    "entities": []
                }
            ]
        "#;

        let samples: Vec<NerExampleSample> = serde_json::from_str(samples_json).expect("Can`t parse samples");

        let words_vocab = NerVocab::<NerSamples>::from(samples.as_slice());
        let chars_vocab = NerVocab::<NerChars>::from(samples.as_slice());
        let labels_vocab = NerVocab::<NerLabels>::from(samples.as_slice());

        let dataset = NerDataset::create(samples, words_vocab, chars_vocab, labels_vocab);

        let example = DatasetNerSample::define(
            "does any airline have an early afternoon flight from boston to pittsburgh".to_string(),
            [
                "does", "any", "airline",
                "have", "an", "early",
                "afternoon", "flight", "from",
                "boston", "to", "pittsburgh"
            ]
                .iter().map(|s| s.to_string())
                .collect(),
            vec![
                (0, 4), (5, 8),
                (9, 16), (17, 21),
                (22, 24), (25, 30),
                (31, 40), (41, 47),
                (48, 52), (53, 59),
                (60, 62), (63, 73)
            ],
            vec![8, 6, 4, 14, 5, 10, 3, 11, 13, 7, 25, 20],
            vec![
                vec![5, 14, 6, 17],
                vec![3, 13, 22],
                vec![3, 10, 16, 11, 10, 13, 6],
                vec![9, 3, 20, 6],
                vec![3, 13],
                vec![6, 3, 16, 11, 22],
                vec![3, 7, 18, 6, 16, 13, 14, 14, 13],
                vec![7, 11, 10, 8, 9, 18],
                vec![7, 16, 14, 12],
                vec![4, 14, 17, 18, 14, 13],
                vec![18, 14],
                vec![15, 10, 18, 18, 17, 4, 19, 16, 8, 9]
            ],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2]
        );

        assert_eq!(example, dataset.get(0).unwrap());
    }
}