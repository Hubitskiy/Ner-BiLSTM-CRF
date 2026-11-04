use burn::nn::{BiLstm, BiLstmConfig};
use burn::prelude::*;
use crate::models::embeddings::{PaddingEmbedding, PaddingEmbeddingConfig};


/// Character-level word encoder.
///
/// Builds a fixed-size vector for every token from its characters: the
/// characters are embedded with a [`PaddingEmbedding`], run through a
/// [`BiLstm`], and the final hidden states of the forward and backward
/// directions are concatenated. The resulting vectors capture word shape
/// (prefixes, suffixes, capitalization) and are meant to be combined with
/// word-level embeddings before the main sequence encoder.
///
/// Each token is encoded independently: the batch and sequence dimensions
/// are flattened before the LSTM, so a token's vector does not depend on its
/// neighbours.
#[derive(Debug, Module)]
pub struct NerCharEncoder {
    /// Character embedding table that zeroes padding characters.
    embedding: PaddingEmbedding,
    /// Bidirectional LSTM running over the characters of each token.
    lstm: BiLstm,
}


impl NerCharEncoder {

    /// Encodes every token of a batch from its character ids.
    ///
    /// # Arguments
    ///
    /// * `char_ids` - character ids of shape `[batch_size, seq_len, char_len]`,
    ///   where shorter tokens are right-padded with `pad_idx`.
    ///
    /// # Returns
    ///
    /// Token encodings of shape `[batch_size, seq_len, 2 * char_hidden_dim]`:
    /// the last forward hidden state followed by the last backward hidden state.
    ///
    /// # Panics
    ///
    /// Panics if `batch_size * seq_len` overflows `usize`.
    pub fn forward(&self, char_ids: Tensor<3, Int>) -> Tensor<3>{

        let [bsz, seq_len, char_len] = char_ids.dims();

        let num_tokens = bsz.checked_mul(seq_len).expect("NerCharEncoder bsz * seq_len overflow captured");

        let flat = char_ids
            .clone()
            .reshape([num_tokens, char_len]);

        let emb = self.embedding.forward(flat);

        let (_, state) = self.lstm.forward(emb, None);

        let h_n: Tensor<3> = state.hidden;

        let [_, _, hidden_size] = h_n.dims();

        let h_forward: Tensor<2> = h_n
            .clone()
            .slice([0..1, 0..num_tokens, 0..hidden_size])
            .reshape([num_tokens, hidden_size]);

        let h_backward = h_n
            .slice([1..2, 0..num_tokens, 0..hidden_size])
            .reshape([num_tokens, hidden_size]);

        let out: Tensor<2> = Tensor::cat(vec![h_forward, h_backward], 1);

        let [_, feature_dim] = out.dims();

        out.reshape([bsz, seq_len, feature_dim])

    }

}


/// Configuration for [`NerCharEncoder`].
#[derive(Debug, Config)]
pub struct NerCharEncoderConfig {
    /// Number of characters in the vocabulary, including padding.
    char_vocab_size: usize,
    /// Dimension of each character embedding.
    char_embedding_dim: usize,
    /// Hidden size of each LSTM direction; the output dimension is twice this.
    char_hidden_dim: usize,
    /// Character id that marks padding positions.
    pad_idx: i64
}

impl NerCharEncoderConfig {

    /// Creates a [`NerCharEncoder`] with freshly initialized parameters.
    ///
    /// # Arguments
    ///
    /// * `device` - device on which the parameters are allocated.
    pub fn init(self, device: &Device) -> NerCharEncoder {

        NerCharEncoder {
            embedding: PaddingEmbeddingConfig::new(self.char_vocab_size, self.char_embedding_dim, self.pad_idx).init(device),
            lstm: BiLstmConfig::new(self.char_embedding_dim, self.char_hidden_dim, true).with_batch_first(true).init(device)
        }

    }
}

#[cfg(test)]
pub mod test {

    use super::*;

    const VOCAB: usize = 12;
    const EMB: usize = 5;
    const HIDDEN: usize = 3;
    const PAD: i64 = 0;

    fn encoder(device: &Device) -> NerCharEncoder {
        NerCharEncoderConfig::new(VOCAB, EMB, HIDDEN, PAD).init(device)
    }

    fn ids_tensor(ids: Vec<i64>, shape: [usize; 3], device: &Device) -> Tensor<3, Int> {
        Tensor::<3, Int>::from_data(TensorData::new(ids, shape), device)
    }

    fn to_f32_vec(tensor: Tensor<3>) -> Vec<f32> {
        tensor.into_data().convert::<f32>().try_to_vec::<f32>().expect("Can`t read float tensor")
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (l, r) in left.iter().zip(right) {
            assert!((l - r).abs() < 1e-5, "{left:?} != {right:?}");
        }
    }

    #[test]
    fn init_keeps_config_values() {
        let config = NerCharEncoderConfig::new(VOCAB, EMB, HIDDEN, PAD);

        assert_eq!(config.char_vocab_size, VOCAB);
        assert_eq!(config.char_embedding_dim, EMB);
        assert_eq!(config.char_hidden_dim, HIDDEN);
        assert_eq!(config.pad_idx, PAD);
    }

    #[test]
    fn init_creates_expected_number_of_params() {
        let device = Device::default();
        let module = encoder(&device);

        // embedding table + 2 directions * 4 gates * (input and hidden linear layers with bias)
        let embedding = VOCAB * EMB;
        let lstm = 2 * 4 * (EMB * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN);

        assert_eq!(module.num_params(), embedding + lstm);
    }

    #[test]
    fn forward_output_shape() {
        let device = Device::default();
        let module = encoder(&device);

        let ids = ids_tensor((0..24).map(|i| i % VOCAB as i64).collect(), [2, 3, 4], &device);

        assert_eq!(module.forward(ids).dims(), [2, 3, 2 * HIDDEN]);
    }

    #[test]
    fn forward_single_token() {
        let device = Device::default();
        let module = encoder(&device);

        let output = module.forward(ids_tensor(vec![1, 2, 3], [1, 1, 3], &device));

        assert_eq!(output.dims(), [1, 1, 2 * HIDDEN]);
        assert!(to_f32_vec(output).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_is_deterministic() {
        let device = Device::default();
        let module = encoder(&device);

        let ids = vec![1, 2, 3, PAD, 4, 5, PAD, PAD];
        let first = to_f32_vec(module.forward(ids_tensor(ids.clone(), [1, 2, 4], &device)));
        let second = to_f32_vec(module.forward(ids_tensor(ids, [1, 2, 4], &device)));

        assert_close(&first, &second);
    }

    #[test]
    fn forward_encodes_tokens_independently() {
        let device = Device::default();
        let module = encoder(&device);

        // the same word [7, 8, 9] at batch 0 / pos 0 and at batch 1 / pos 1
        let ids = vec![
            7, 8, 9,   1, 2, 3,
            4, 5, 6,   7, 8, 9,
        ];
        let output = to_f32_vec(module.forward(ids_tensor(ids, [2, 2, 3], &device)));

        let feature = 2 * HIDDEN;
        assert_close(&output[..feature], &output[3 * feature..4 * feature]);
    }

    #[test]
    fn forward_matches_single_token_encoding() {
        let device = Device::default();
        let module = encoder(&device);

        let batch = to_f32_vec(module.forward(ids_tensor(vec![1, 2, 3, 4, 5, 6], [1, 2, 3], &device)));
        let alone = to_f32_vec(module.forward(ids_tensor(vec![4, 5, 6], [1, 1, 3], &device)));

        assert_close(&batch[2 * HIDDEN..], &alone);
    }

    #[test]
    fn forward_distinguishes_different_tokens() {
        let device = Device::default();
        let module = encoder(&device);

        // same characters in reverse order must give a different encoding
        let output = to_f32_vec(module.forward(ids_tensor(vec![1, 2, 3, 3, 2, 1], [1, 2, 3], &device)));

        assert_ne!(&output[..2 * HIDDEN], &output[2 * HIDDEN..]);
    }
}
