use burn::nn::Embedding;
use burn::nn::EmbeddingConfig;
use burn::prelude::*;

/// Token embedding layer that zeroes out vectors at padding positions.
///
/// Wraps a regular [`Embedding`] lookup table and, after the lookup, replaces
/// every vector whose input id equals `pad_idx` with an all-zero vector. This
/// keeps padded positions in a batch from leaking information into downstream
/// layers (e.g. the BiLSTM).
///
/// Note that only the *output* is masked: the row of the embedding table at
/// `pad_idx` is still a regular trainable parameter, but since its output is
/// always replaced by zeros it receives no gradient.
#[derive(Module, Debug)]
pub struct PaddingEmbedding {
    /// Underlying lookup table of shape `[size, dim]`.
    embedding: Embedding,
    /// Token id that marks padding positions.
    pad_idx: i64
}


impl PaddingEmbedding {

    /// Looks up embeddings for a batch of token ids and zeroes padding positions.
    ///
    /// # Arguments
    ///
    /// * `ids` - token ids of shape `[batch_size, seq_len]`.
    ///
    /// # Returns
    ///
    /// Embeddings of shape `[batch_size, seq_len, dim]`, where every position
    /// with `ids[b, t] == pad_idx` is a zero vector.
    pub fn forward(&self, ids: Tensor<2, Int>) -> Tensor<3> {

        let pad_mask = ids
            .clone()
            .equal_elem(self.pad_idx)
            .unsqueeze_dim::<3>(2);

        let vector = self.embedding.forward(ids);

        vector.mask_fill(pad_mask, 0.0)

    }

}

/// Configuration for [`PaddingEmbedding`].
#[derive(Debug, Config)]
pub struct PaddingEmbeddingConfig {

    /// Vocabulary size, i.e. the number of rows in the embedding table.
    size: usize,
    /// Dimension of each embedding vector.
    dim: usize,
    /// Token id that marks padding positions.
    pad_idx: i64

}

impl PaddingEmbeddingConfig {

    /// Creates a [`PaddingEmbedding`] with a freshly initialized embedding table.
    ///
    /// # Arguments
    ///
    /// * `device` - device on which the parameters are allocated.
    pub fn init(self, device: &Device) -> PaddingEmbedding {

        PaddingEmbedding{
            embedding: EmbeddingConfig::new(self.size, self.dim).init(device),
            pad_idx: self.pad_idx
        }

    }

}

#[cfg(test)]
pub mod test {

    use super::*;

    const SIZE: usize = 10;
    const DIM: usize = 4;
    const PAD: i64 = 0;

    fn ids_tensor(ids: Vec<i64>, shape: [usize; 2], device: &Device) -> Tensor<2, Int> {
        Tensor::<2, Int>::from_data(TensorData::new(ids, shape), device)
    }

    fn to_f32_vec(tensor: Tensor<3>) -> Vec<f32> {
        tensor.into_data().convert::<f32>().try_to_vec::<f32>().expect("Can`t read float tensor")
    }

    #[test]
    fn forward_output_shape() {
        let device = Device::default();
        let module = PaddingEmbeddingConfig::new(SIZE, DIM, PAD).init(&device);

        let ids = ids_tensor(vec![1, 2, 3, PAD, 4, 5, PAD, PAD], [2, 4], &device);

        assert_eq!(module.forward(ids).dims(), [2, 4, DIM]);
    }

    #[test]
    fn forward_zeroes_padding_positions() {
        let device = Device::default();
        let module = PaddingEmbeddingConfig::new(SIZE, DIM, PAD).init(&device);

        // second row: 2 real tokens followed by padding
        let ids = vec![1, 2, 3, 4, 5, 6, PAD, PAD];
        let output = to_f32_vec(module.forward(ids_tensor(ids.clone(), [2, 4], &device)));

        for (pos, id) in ids.iter().enumerate() {
            let vector = &output[pos * DIM..(pos + 1) * DIM];

            if *id == PAD {
                assert!(vector.iter().all(|v| *v == 0.0), "pad position {pos} is not zeroed: {vector:?}");
            } else {
                assert!(vector.iter().any(|v| *v != 0.0), "token position {pos} is unexpectedly zeroed");
            }
        }
    }

    #[test]
    fn forward_matches_plain_embedding_for_non_pad_tokens() {
        let device = Device::default();
        let module = PaddingEmbeddingConfig::new(SIZE, DIM, PAD).init(&device);

        let ids = vec![1, PAD, 3, 9, PAD, 7];
        let masked = to_f32_vec(module.forward(ids_tensor(ids.clone(), [2, 3], &device)));
        let plain = to_f32_vec(module.embedding.forward(ids_tensor(ids.clone(), [2, 3], &device)));

        for (pos, id) in ids.iter().enumerate() {
            let range = pos * DIM..(pos + 1) * DIM;

            if *id != PAD {
                assert_eq!(&masked[range.clone()], &plain[range]);
            }
        }
    }

    #[test]
    fn forward_respects_non_zero_pad_idx() {
        let device = Device::default();
        let pad = 5;
        let module = PaddingEmbeddingConfig::new(SIZE, DIM, pad).init(&device);

        // id 0 is a regular token here, only id 5 must be masked
        let ids = vec![0, pad, 2];
        let output = to_f32_vec(module.forward(ids_tensor(ids, [1, 3], &device)));

        assert!(output[..DIM].iter().any(|v| *v != 0.0));
        assert!(output[DIM..2 * DIM].iter().all(|v| *v == 0.0));
        assert!(output[2 * DIM..].iter().any(|v| *v != 0.0));
    }

    #[test]
    fn forward_all_padding_returns_zeros() {
        let device = Device::default();
        let module = PaddingEmbeddingConfig::new(SIZE, DIM, PAD).init(&device);

        let output = to_f32_vec(module.forward(ids_tensor(vec![PAD; 6], [2, 3], &device)));

        assert!(output.iter().all(|v| *v == 0.0));
    }
}
