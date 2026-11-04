use burn::{
    data::dataloader::batcher::Batcher,
    prelude::{Device, TensorData, Tensor, Bool, Int, s}
};
use serde::{Deserialize, Serialize};
use neuro::vocabs::vocab::KeyValeBasedVocabulary;

use crate::bilstm_crf::dataset::DatasetNerSample;
use crate::bilstm_crf::vocab::{NerSamples, NerVocab};

/// Source data of a [`NerBatch`], kept to map predictions back to the text.
///
/// Every field holds one entry per sample of the batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerBatchMetadata {

    /// Normalized text of each sample.
    texts: Vec<String>,
    /// Tokens of each sample.
    tokens: Vec<Vec<String>>,
    /// Byte offsets `(start, end)` of each token in the normalized text.
    offsets: Vec<Vec<(usize, usize)>>,

}


impl NerBatchMetadata {

    /// Creates metadata from per-sample texts, tokens and offsets.
    ///
    /// # Arguments
    ///
    /// * `texts` - normalized text of each sample.
    /// * `tokens` - tokens of each sample.
    /// * `offsets` - byte offsets of each token of each sample.
    fn define(texts: Vec<String>, tokens: Vec<Vec<String>>, offsets: Vec<Vec<(usize, usize)>>) -> Self {
        Self {
            texts,
            tokens,
            offsets
        }
    }
}


/// Padded tensors of a [`NerBatch`], ready to be passed to the model.
///
/// Sequences are right-padded up to the longest sample of the batch, and tokens
/// up to the longest token of the batch.
#[derive(Debug, Clone)]
pub struct NerTensorizedData {
    /// Word ids, shape `[batch_size, max_seq_len]`.
    pub token_ids: Tensor<2, Int>,
    /// Character ids, shape `[batch_size, max_seq_len, max_char_len]`.
    pub char_ids: Tensor<3, Int>,
    /// Gold label ids, shape `[batch_size, max_seq_len]`.
    pub label_ids: Tensor<2, Int>,
    /// `true` for real tokens, `false` for padding, shape `[batch_size, max_seq_len]`.
    pub mask: Tensor<2, Bool>,
}


impl NerTensorizedData {

    /// Creates tensorized data from already padded tensors.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - word ids, shape `[batch_size, max_seq_len]`.
    /// * `char_ids` - character ids, shape `[batch_size, max_seq_len, max_char_len]`.
    /// * `label_ids` - gold label ids, shape `[batch_size, max_seq_len]`.
    /// * `mask` - `true` for real tokens, `false` for padding, shape `[batch_size, max_seq_len]`.
    fn define(
        token_ids: Tensor<2, Int>,
        char_ids: Tensor<3, Int>,
        label_ids: Tensor<2, Int>,
        mask: Tensor<2, Bool>
    ) -> Self {
        Self{
            token_ids,
            char_ids,
            label_ids,
            mask
        }
    }
}


/// Batch of NER samples produced by [`NerBatcher`].
#[derive(Debug, Clone)]
pub struct NerBatch {
    /// Source texts, tokens and offsets of the samples.
    pub metadata: NerBatchMetadata,
    /// Padded model inputs and gold labels.
    pub tensorized: NerTensorizedData
}

impl NerBatch {

    /// Creates a batch from its metadata and tensor parts.
    ///
    /// # Arguments
    ///
    /// * `texts` - normalized text of each sample.
    /// * `tokens` - tokens of each sample.
    /// * `offsets` - byte offsets of each token of each sample.
    /// * `token_ids` - word ids, shape `[batch_size, max_seq_len]`.
    /// * `char_ids` - character ids, shape `[batch_size, max_seq_len, max_char_len]`.
    /// * `label_ids` - gold label ids, shape `[batch_size, max_seq_len]`.
    /// * `mask` - `true` for real tokens, `false` for padding, shape `[batch_size, max_seq_len]`.
    pub fn define(
        texts: Vec<String>,
        tokens: Vec<Vec<String>>,
        offsets: Vec<Vec<(usize, usize)>>,
        token_ids: Tensor<2, Int>,
        char_ids: Tensor<3, Int>,
        label_ids: Tensor<2, Int>,
        mask: Tensor<2, Bool>
    ) -> Self {

        Self {
            metadata: NerBatchMetadata::define(texts, tokens, offsets),
            tensorized: NerTensorizedData::define(token_ids, char_ids, label_ids, mask)
        }
    }
}



/// Collates [`DatasetNerSample`] items into a padded [`NerBatch`].
#[derive(Clone)]
pub struct NerBatcher;

impl NerBatcher {
    
    pub fn new() -> Self { Self {} }
}


impl Batcher< DatasetNerSample, NerBatch> for NerBatcher {
    /// Pads the samples to a common shape and stacks them into a [`NerBatch`].
    ///
    /// Token, character and label tensors are filled with the padding id
    /// [`NerVocab::pad_value`] wherever a sample is shorter than the batch maximum.
    /// Padded positions are marked `false` in the mask.
    ///
    /// # Arguments
    ///
    /// * `items` - samples to collate.
    /// * `device` - device on which the tensors are allocated.
    ///
    /// # Returns
    ///
    /// Batch whose tensors have the sizes of the longest sample and the longest token.
    ///
    /// # Panics
    ///
    /// Panics if `items` is empty, if a sample has no tokens, if a sample's
    /// `label_ids`, `char_ids`, `tokens` or `offsets` differ in length from its
    /// `token_ids`, or if a token has an empty `char_ids` list.
    fn batch(&self, items: Vec<DatasetNerSample>, device: &Device) -> NerBatch {
        assert!(
            !items.is_empty(),
            "Can`t gather empty NER batch",
        );

        let batch_size = items.len();

        for (i, item) in items.iter().enumerate() {
            let seq_len = item.token_ids().len();

            assert!(
                seq_len > 0,
                "Sample {i}: tokens sequence are empty",
            );

            assert_eq!(
                item.label_ids().len(),
                seq_len,
                "Sample {i}: label_ids count is not equal to token_ids count",
            );

            assert_eq!(
                item.char_ids().len(),
                seq_len,
                "Sample {i}: char_ids list count is not equal to token_ids",
            );

            assert_eq!(
                item.tokens().len(),
                seq_len,
                "Sample {i}: tokens count is not equal to token_ids",
            );

            assert_eq!(
                item.offsets().len(),
                seq_len,
                "Sample {i}: offsets count is not equal to token_ids",
            );

            assert!(
                item.char_ids().iter().all(|chars| !chars.is_empty()),
                "Sample {i}: found token with empty char_ids list",
            );
        }
        let max_seq_len = items
            .iter()
            .map(|item| item.token_ids().len())
            .max()
            .unwrap_or(0);


        let max_char_len = items
            .iter()
            .flat_map(|item| item.char_ids().iter())
            .map(|chars| chars.len())
            .max()
            .unwrap_or(1);

        let mut token_tensor = Tensor::<2, Int>::full(
            [batch_size, max_seq_len],
            NerVocab::<NerSamples>::pad_value(),
            device
        );

        let mut char_tensor = Tensor::<3, Int>::full(
            [batch_size, max_seq_len, max_char_len],
            NerVocab::<NerSamples>::pad_value(),
            device
        );

        let mut label_tensor = Tensor::<2, Int>::full(
            [batch_size, max_seq_len],
            NerVocab::<NerSamples>::pad_value(),
            device
        );

        let mut mask = Tensor::<2, Bool>::zeros(
            [batch_size, max_seq_len],
            device,
        );

        let mut texts = Vec::with_capacity(batch_size);
        let mut tokens = Vec::with_capacity(batch_size);
        let mut offsets = Vec::with_capacity(batch_size);

        for (i, item) in items.iter().enumerate() {

            let seq_len = item.token_ids().len();


            token_tensor = token_tensor.slice_assign(
                s![i..i + 1_usize, 0..seq_len],
                Tensor::<2, Int>::from_data(
                    TensorData::new(item.token_ids().to_vec(), [1, seq_len]),
                    device
                )
            );

            label_tensor = label_tensor.slice_assign(
                s![i..i + 1_usize, 0..seq_len],
                Tensor::<2, Int>::from_data(
                    TensorData::new(item.label_ids().to_vec(), [1, seq_len]),
                    device
                )
            );

            mask = mask.slice_assign(
                s![i..i + 1_usize, 0..seq_len],
                Tensor::<2, Bool>::from_data(
                    TensorData::new(vec![true; seq_len], [1, seq_len]),
                    device,
                )
            );

            for (j, chrs) in item.char_ids().iter().enumerate() {

                let chars_len = chrs.len();

                if chars_len == 0 {
                    continue;
                }

                char_tensor = char_tensor.slice_assign(
                    s![i..i + 1_usize, j..j + 1_usize, 0..chars_len],
                    Tensor::<3, Int>::from_data(
                        TensorData::new(chrs.clone(), [1, 1, chars_len]),
                        device
                    )
                )

            }

            texts.push(item.text().to_string());
            tokens.push(item.tokens().to_vec());
            offsets.push(item.offsets().to_vec());

        }

        NerBatch::define(
            texts,
            tokens,
            offsets,
            token_tensor,
            char_tensor,
            label_tensor,
            mask
        )
    }
}

#[cfg(test)]
pub mod test {

    use super::*;
    use burn::data::dataset::Dataset;
    use crate::bilstm_crf::dataset::NerDataset;
    use crate::bilstm_crf::types::NerExampleSample;
    use crate::bilstm_crf::vocab::{NerChars, NerLabels};

    /// Builds a three-sample dataset with vocabularies created from the same samples.
    fn create_dataset() -> NerDataset {
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

        NerDataset::create(samples, words_vocab, chars_vocab, labels_vocab)
    }

    /// Reads an integer tensor of any rank into a flat `Vec<i64>`.
    fn to_i64_vec<const D: usize>(tensor: Tensor<D, Int>) -> Vec<i64> {
        tensor.into_data().convert::<i64>().try_to_vec::<i64>().expect("Can`t read int tensor")
    }

    /// Checks metadata, shapes and padded values of a batch made from a whole dataset.
    #[test]
    fn batch_dataset_samples() {
        let dataset = create_dataset();
        let items: Vec<DatasetNerSample> = (0..dataset.len())
            .map(|i| dataset.get(i).expect("Can`t get dataset sample"))
            .collect();

        let pad = NerVocab::<NerSamples>::pad_value();

        // sequence lengths: 12, 6, 9 tokens; the longest token is "pittsburgh" (10 chars)
        let batch_size = items.len();
        let max_seq_len = 12;
        let max_char_len = 10;

        let mut expected_tokens = vec![pad; batch_size * max_seq_len];
        let mut expected_labels = vec![pad; batch_size * max_seq_len];
        let mut expected_mask = vec![false; batch_size * max_seq_len];
        let mut expected_chars = vec![pad; batch_size * max_seq_len * max_char_len];

        for (i, item) in items.iter().enumerate() {
            for j in 0..item.token_ids().len() {
                let pos = i * max_seq_len + j;
                expected_tokens[pos] = item.token_ids()[j];
                expected_labels[pos] = item.label_ids()[j];
                expected_mask[pos] = true;

                for (k, ch) in item.char_ids()[j].iter().enumerate() {
                    expected_chars[pos * max_char_len + k] = *ch;
                }
            }
        }

        let expected_texts: Vec<String> = items.iter().map(|item| item.text().to_string()).collect();
        let expected_token_strs: Vec<Vec<String>> = items.iter().map(|item| item.tokens().to_vec()).collect();
        let expected_offsets: Vec<Vec<(usize, usize)>> = items.iter().map(|item| item.offsets().to_vec()).collect();

        let device = Device::default();
        let batch = NerBatcher.batch(items, &device);

        // metadata
        assert_eq!(batch.metadata.texts, expected_texts);
        assert_eq!(batch.metadata.tokens, expected_token_strs);
        assert_eq!(batch.metadata.offsets, expected_offsets);
        assert_eq!(batch.metadata.texts[1], "show me the flights on monday");

        // shapes
        let tensorized = batch.tensorized;
        assert_eq!(tensorized.token_ids.dims(), [batch_size, max_seq_len]);
        assert_eq!(tensorized.label_ids.dims(), [batch_size, max_seq_len]);
        assert_eq!(tensorized.mask.dims(), [batch_size, max_seq_len]);
        assert_eq!(tensorized.char_ids.dims(), [batch_size, max_seq_len, max_char_len]);

        // values
        let token_ids = to_i64_vec(tensorized.token_ids);
        let label_ids = to_i64_vec(tensorized.label_ids);
        let char_ids = to_i64_vec(tensorized.char_ids);
        let mask: Vec<bool> = to_i64_vec(tensorized.mask.int()).into_iter().map(|v| v != 0).collect();

        // first row is not padded and matches the create_dataset example
        assert_eq!(&token_ids[..max_seq_len], &[8, 6, 4, 14, 5, 10, 3, 11, 13, 7, 25, 20]);
        assert_eq!(&label_ids[..max_seq_len], &[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2]);
        assert_eq!(&char_ids[..max_char_len], &[5, 14, 6, 17, pad, pad, pad, pad, pad, pad]);
        assert_eq!(
            &char_ids[11 * max_char_len..12 * max_char_len],
            &[15, 10, 18, 18, 17, 4, 19, 16, 8, 9]
        );

        // second row has 6 tokens followed by padding
        assert_eq!(
            &mask[max_seq_len..2 * max_seq_len],
            &[true, true, true, true, true, true, false, false, false, false, false, false]
        );

        assert_eq!(token_ids, expected_tokens);
        assert_eq!(label_ids, expected_labels);
        assert_eq!(mask, expected_mask);
        assert_eq!(char_ids, expected_chars);
    }

    /// Checks that batching an empty list of items panics.
    #[test]
    #[should_panic(expected = "Can`t gather empty NER batch")]
    fn batch_empty_items_panics() {
        NerBatcher.batch(Vec::new(), &Device::default());
    }
}
