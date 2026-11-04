use burn::{
    nn::{Linear, LinearConfig,BiLstm, BiLstmConfig},
    train::{TrainStep, TrainOutput, InferenceStep},
    prelude::*,
};
use neuro::models::{
    crf::{CRF, CRFConfig},
    embeddings::{PaddingEmbedding, PaddingEmbeddingConfig},
    encoders::{NerCharEncoder, NerCharEncoderConfig},
};
use neuro::vocabs::prelude::*;
use crate::bilstm_crf::{
    batcher::NerBatch,
    types::NerOutput
};
use super::vocab::{NerVocab, NerSamples};


/// BiLSTM-CRF sequence tagger.
///
/// Each token is represented by its word embedding concatenated with a
/// character-level encoding. A [`BiLstm`] runs over these representations, a
/// linear layer turns its output into per-token label scores, and a [`CRF`]
/// scores whole label sequences.
#[derive(Debug, Module)]
pub struct BiLstmCRFModel {
    /// Word embedding table that zeroes padding tokens.
    words_embedding: PaddingEmbedding,
    /// Character-level encoder of each token.
    chars_encoder: NerCharEncoder,
    /// Bidirectional LSTM running over the token representations.
    lstm: BiLstm,
    /// Projects LSTM outputs to per-label emission scores.
    classifier: Linear,
    /// CRF layer that computes the training loss.
    crf: CRF,
}


impl BiLstmCRFModel {

    /// Computes per-token label scores.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - word ids, shape `[batch_size, seq_len]`.
    /// * `char_ids` - character ids, shape `[batch_size, seq_len, char_len]`.
    ///
    /// # Returns
    ///
    /// Emission scores of shape `[batch_size, seq_len, num_labels]`.
    fn emissions(&self, token_ids: Tensor<2, Int>, char_ids: Tensor<3, Int>) -> Tensor<3> {
        let words_embedding = self.words_embedding.forward(token_ids);
        let char_repr = self.chars_encoder.forward(char_ids);

        let word_char_mix = Tensor::cat(vec![words_embedding, char_repr], 2);

        let (output, _) = self.lstm.forward(word_char_mix, None);

        self.classifier.forward(output)
    }

    /// Computes the CRF negative log-likelihood loss of a batch.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - word ids, shape `[batch_size, seq_len]`.
    /// * `char_ids` - character ids, shape `[batch_size, seq_len, char_len]`.
    /// * `label_ids` - gold label ids, shape `[batch_size, seq_len]`.
    /// * `mask` - `true` for real tokens, `false` for padding, shape `[batch_size, seq_len]`.
    ///
    /// # Returns
    ///
    /// Negative log-likelihood averaged over the batch, shape `[1]`.
    pub fn forward(
        &self,
        token_ids: Tensor<2, Int>,
        char_ids: Tensor<3, Int>,
        label_ids: Tensor<2, Int>,
        mask: Tensor<2, Bool>
    ) -> Tensor<1> {

        let emissions_output = self.emissions(token_ids, char_ids);

        self.crf.forward(emissions_output, label_ids, mask)
    }

    /// Predicts the most likely label sequence of every sample in a batch.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - word ids, shape `[batch_size, seq_len]`.
    /// * `char_ids` - character ids, shape `[batch_size, seq_len, char_len]`.
    /// * `mask` - `true` for real tokens, `false` for padding, shape `[batch_size, seq_len]`.
    ///
    /// # Returns
    ///
    /// Label ids of every sample, one per unmasked token.
    pub fn decode(
        &self,
        token_ids: Tensor<2, Int>,
        char_ids: Tensor<3, Int>,
        mask: Tensor<2, Bool>
    ) -> Vec<Vec<usize>> {

        let emissions_output = self.emissions(token_ids, char_ids);

        self.crf.decode(emissions_output, mask)
    }

}


/// Configuration for [`BiLstmCRFModel`].
#[derive(Debug, Config)]
pub struct BiLstmCRFModelConfig {
    /// Number of words in the vocabulary, including padding.
    words_vocab_size: usize,
    /// Number of characters in the vocabulary, including padding.
    char_vocab_size: usize,
    /// Number of distinct labels.
    num_labels: usize,
    /// Dimension of each word embedding.
    word_emb_dim: usize,
    /// Dimension of each character embedding.
    char_emb_dim: usize,
    /// Hidden size of each direction of the character LSTM.
    char_hidden_dim: usize,
    /// Hidden size of each direction of the main LSTM.
    lstm_hidden_dim: usize,
}


impl BiLstmCRFModelConfig {

    /// Creates a [`BiLstmCRFModel`] with freshly initialized parameters.
    ///
    /// The main LSTM takes `word_emb_dim + 2 * char_hidden_dim` features and the
    /// classifier takes `2 * lstm_hidden_dim` features.
    ///
    /// # Arguments
    ///
    /// * `device` - device on which the parameters are allocated.
    ///
    /// # Panics
    ///
    /// Panics if computing the input dimensions overflows `usize`, or if `num_labels` is 0.
    pub fn init(self, device: &Device) -> BiLstmCRFModel {

        let pad_idx = NerVocab::<NerSamples>::pad_value();

        let input_dim = self.word_emb_dim
            .checked_add(
                self.char_hidden_dim
                    .checked_mul(2)
                    .expect("Caught overflow when char_hidden_dim * 2")
            ).expect("Caught overflow when word_emb_dim + char_hidden_dim * 2");

        let classifier_input_dim = self.lstm_hidden_dim.checked_mul(2).expect("Caught overflow when lstm_hidden_dim * 2");

        BiLstmCRFModel {
            words_embedding: PaddingEmbeddingConfig::new(self.words_vocab_size, self.word_emb_dim, pad_idx).init(device),
            chars_encoder: NerCharEncoderConfig::new(self.char_vocab_size, self.char_emb_dim, self.char_hidden_dim, pad_idx).init(device),
            lstm: BiLstmConfig::new(input_dim, self.lstm_hidden_dim, true).with_batch_first(true).init(device),
            classifier: LinearConfig::new(classifier_input_dim, self.num_labels).init(device),
            crf: CRFConfig::new(self.num_labels).init(device)
        }
    }

}

impl TrainStep for BiLstmCRFModel {

    type Input = NerBatch;
    type Output = NerOutput;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {

        let loss = self.forward(
            item.tensorized.token_ids,
            item.tensorized.char_ids,
            item.tensorized.label_ids,
            item.tensorized.mask,
        );

        let grads = loss.backward();

        TrainOutput::new(self, grads, NerOutput::define(loss))
    }

}


impl InferenceStep for BiLstmCRFModel {
    type Input = NerBatch;
    type Output = NerOutput;

    fn step(&self, item: Self::Input) -> Self::Output {
        let loss = self.forward(
            item.tensorized.token_ids,
            item.tensorized.char_ids,
            item.tensorized.label_ids,
            item.tensorized.mask,
        );

        NerOutput::define(loss)
    }
}
