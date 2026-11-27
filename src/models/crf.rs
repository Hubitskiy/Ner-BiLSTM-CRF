use burn::backend::Distribution;
use burn::module::{Param};
use burn::prelude::*;


/// Linear-chain Conditional Random Field.
///
/// Scores whole tag sequences instead of individual tokens: the score of a
/// sequence `y_0..y_n` for per-token emission scores `e` is
///
/// ```text
/// start[y_0] + e[0, y_0] + sum_{t=1..n} (transition[y_{t-1}, y_t] + e[t, y_t]) + end[y_n]
/// ```
///
/// Training minimizes the negative log-likelihood of the gold sequence, i.e.
/// the log-partition over all possible sequences minus the gold score. This
/// lets the model learn which tag transitions are plausible (e.g. `I-PER`
/// must not follow `B-LOC`) on top of the encoder's emission scores.
#[derive(Debug, Module)]
pub struct CRF {
    /// Score of starting a sequence with each tag, shape `[num_tags]`.
    start_transition: Param<Tensor<1>>,
    /// Score of ending a sequence with each tag, shape `[num_tags]`.
    end_transition: Param<Tensor<1>>,
    /// Score of moving from tag `i` (row) to tag `j` (column), shape `[num_tags, num_tags]`.
    transition: Param<Tensor<2>>,
}


impl CRF {

    /// Computes the unnormalized score of the given tag sequences.
    ///
    /// Positions where `mask` is `false` contribute neither emission nor
    /// transition scores, and the end transition is taken from the last
    /// unmasked tag of every sequence.
    ///
    /// # Arguments
    ///
    /// * `emissions` - per-token tag scores, shape `[batch_size, seq_len, num_tags]`.
    /// * `tags` - tag ids, shape `[batch_size, seq_len]`.
    /// * `mask` - `true` for real tokens, `false` for padding, shape `[batch_size, seq_len]`.
    ///   The first position of every sequence is assumed to be unmasked and
    ///   padding is expected only at the end.
    ///
    /// # Returns
    ///
    /// Score of every sequence, shape `[batch_size]`.
    fn score_sentence(
        &self,
        emissions: Tensor<3>,
        tags: Tensor<2, Int>,
        mask: Tensor<2, Bool>,
    ) -> Tensor<1> {
        let [batch_size, seq_len, num_tags] = emissions.dims();

        let transitions = self.transition.val();

        let device = emissions.device();

        let first_tags = tags
            .clone()
            .slice([0..batch_size, 0..1])
            .reshape([batch_size]);

        let mut score = self
            .start_transition
            .val()
            .select(0, first_tags.clone());

        let first_emission = emissions
            .clone()
            .slice([0..batch_size, 0..1, 0..num_tags])
            .reshape([batch_size, num_tags])
            .gather(1, first_tags.reshape([batch_size, 1]))
            .reshape([batch_size]);

        score = score + first_emission;

        for t in 1..seq_len {
            let previous_tags = tags
                .clone()
                .slice([0..batch_size, (t - 1)..t])
                .reshape([batch_size]);

            let current_tags = tags
                .clone()
                .slice([0..batch_size, t..(t + 1)])
                .reshape([batch_size]);

            let transition_score = transitions
                .clone()
                .select(0, previous_tags)
                .gather(
                    1,
                    current_tags.clone().reshape([batch_size, 1]),
                )
                .reshape([batch_size]);

            let emission_score = emissions
                .clone()
                .slice([
                    0..batch_size,
                    t..(t + 1),
                    0..num_tags,
                ])
                .reshape([batch_size, num_tags])
                .gather(
                    1,
                    current_tags.reshape([batch_size, 1]),
                )
                .reshape([batch_size]);

            let active = mask
                .clone()
                .slice([0..batch_size, t..(t + 1)])
                .reshape([batch_size]);

            let increment = Tensor::<1>::zeros(
                [batch_size],
                &device,
            )
                .mask_where(active, transition_score + emission_score);

            score = score + increment;
        }

        let seq_ends = mask.int().sum_dim(1) - 1;

        let last_tags = tags
            .gather(1, seq_ends)
            .reshape([batch_size]);

        score + self.end_transition.val().select(0, last_tags)
    }

    /// Computes the log-partition function with the forward algorithm.
    ///
    /// This is the log-sum-exp of the scores of all possible tag sequences,
    /// computed in `O(seq_len * num_tags^2)` instead of enumerating them.
    /// Masked positions leave the running forward scores unchanged.
    ///
    /// # Arguments
    ///
    /// * `emissions` - per-token tag scores, shape `[batch_size, seq_len, num_tags]`.
    /// * `mask` - `true` for real tokens, `false` for padding, shape `[batch_size, seq_len]`.
    ///
    /// # Returns
    ///
    /// Log-partition of every sequence, shape `[batch_size]`.
    fn compute_log_partition(
        &self,
        emissions: Tensor<3>,
        mask: Tensor<2, Bool>,
    ) -> Tensor<1> {
        let [batch_size, seq_len, num_tags] = emissions.dims();

        let transitions = self.transition.val();

        let first_emissions = emissions
            .clone()
            .slice([0..batch_size, 0..1, 0..num_tags])
            .reshape([batch_size, num_tags]);

        let mut score = self
            .start_transition
            .val()
            .reshape([1, num_tags])
            + first_emissions;

        let broadcast_transitions =
            transitions.reshape([1, num_tags, num_tags]);

        for t in 1..seq_len {
            let broadcast_score = score
                .clone()
                .reshape([batch_size, num_tags, 1]);

            let broadcast_emissions = emissions
                .clone()
                .slice([
                    0..batch_size,
                    t..(t + 1),
                    0..num_tags,
                ])
                .reshape([batch_size, 1, num_tags]);

            let candidates = broadcast_score
                + broadcast_transitions.clone()
                + broadcast_emissions;

            let next_score = Self::logsumexp_keepdim(candidates, 1)
                .reshape([batch_size, num_tags]);

            let active = mask
                .clone()
                .slice([0..batch_size, t..(t + 1)])
                .repeat_dim(1, num_tags);

            score = score.mask_where(active, next_score);
        }

        let final_score = score
            + self
            .end_transition
            .val()
            .reshape([1, num_tags]);

        Self::logsumexp_keepdim(final_score, 1)
            .reshape([batch_size])
    }

    /// Numerically stable `log(sum(exp(values)))` along `dim`.
    ///
    /// The maximum is subtracted before exponentiation to avoid overflow.
    /// The reduced dimension is kept with size 1.
    fn logsumexp_keepdim<const D: usize>(
        values: Tensor<D>,
        dim: usize,
    ) -> Tensor<D> {
        let maximum = values.clone().max_dim(dim);

        let sum_exp = (values - maximum.clone())
            .exp()
            .sum_dim(dim);

        maximum + sum_exp.log()
    }

    /// Computes the CRF negative log-likelihood loss.
    ///
    /// # Arguments
    ///
    /// * `emissions` - per-token tag scores from the encoder, shape `[batch_size, seq_len, num_tags]`.
    /// * `label_ids` - gold tag ids, shape `[batch_size, seq_len]`.
    /// * `mask` - `true` for real tokens, `false` for padding, shape `[batch_size, seq_len]`.
    ///
    /// # Returns
    ///
    /// Negative log-likelihood averaged over the batch, shape `[1]`. The value
    /// is always non-negative.
    pub fn forward(
        &self,
        emissions: Tensor<3>,
        label_ids: Tensor<2, Int>,
        mask: Tensor<2, Bool>
    ) -> Tensor<1> {
        let gold_score = self.score_sentence(emissions.clone(), label_ids.clone(), mask.clone());
        let partition = self.compute_log_partition(emissions.clone(), mask.clone());

        (partition - gold_score).mean()
    }

    /// Finds the best label sequence for every sentence.
    ///
    /// * `emissions` - shape `[batch_size, seq_len, num_labels]`.
    /// * `mask` - `true` for real tokens, shape `[batch_size, seq_len]`.
    ///
    /// Returns label ids per sentence, without padding.
    pub fn decode(&self, emissions: Tensor<3>, mask: Tensor<2, Bool>) -> Vec<Vec<usize>> {
        let [batch_size, seq_len, num_labels] = emissions.dims();

        let em: Vec<f32> = emissions
            .into_data()
            .convert::<f32>()
            .try_to_vec()
            .expect("emissions to vec");

        let mask: Vec<bool> = mask
            .int()
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .expect("mask to vec")
            .into_iter()
            .map(|v| v != 0)
            .collect();

        let start: Vec<f32> = self.start_transition
            .val()
            .into_data()
            .convert::<f32>()
            .try_to_vec()
            .expect("start");

        let end: Vec<f32> = self.end_transition
            .val()
            .into_data()
            .convert::<f32>()
            .try_to_vec()
            .expect("end");

        let trans: Vec<f32> = self.transition
            .val()
            .into_data()
            .convert::<f32>()
            .try_to_vec()
            .expect("trans");

        let e = |b: usize, t: usize, j: usize| em[(b * seq_len + t) * num_labels + j];
        // trans[i, j] — score of moving from label i to label j
        let tr = |i: usize, j: usize| trans[i * num_labels + j];

        let mut result = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let len = (0..seq_len).filter(|&t| mask[b * seq_len + t]).count();
            if len == 0 {
                result.push(Vec::new());
                continue;
            }

            let mut score: Vec<f32> = (0..num_labels).map(|j| start[j] + e(b, 0, j)).collect();
            let mut backptr = vec![vec![0usize; num_labels]; len];

            for t in 1..len {
                let mut next = vec![f32::NEG_INFINITY; num_labels];
                for j in 0..num_labels {
                    for i in 0..num_labels {
                        let s = score[i] + tr(i, j);
                        if s > next[j] {
                            next[j] = s;
                            backptr[t][j] = i;
                        }
                    }
                    next[j] += e(b, t, j);
                }
                score = next;
            }

            let mut best = 0;
            let mut best_score = f32::NEG_INFINITY;
            for j in 0..num_labels {
                let s = score[j] + end[j];
                if s > best_score {
                    best_score = s;
                    best = j;
                }
            }

            let mut path = vec![best; len];
            for t in (1..len).rev() {
                path[t - 1] = backptr[t][path[t]];
            }
            result.push(path);
        }

        result
    }
}


/// Configuration for creating a [`CRF`].
#[derive(Debug, Config)]
pub struct CRFConfig {

    /// Number of distinct tags.
    num_tags: usize,

}


impl CRFConfig {

    /// Creates a [`CRF`] with transition scores sampled from `Uniform(-0.1, 0.1)`.
    ///
    /// # Arguments
    ///
    /// * `device` - device on which the parameters are allocated.
    ///
    /// # Panics
    ///
    /// Panics if `num_tags` is 0.
    pub fn init(self, device: &Device) -> CRF {

        assert!(self.num_tags > 0, "num_tags must be greater than 0");
        
        CRF {
            start_transition: Param::from_tensor(
                Tensor::<1>::random(
                    [self.num_tags], 
                    Distribution::Uniform(-0.1, 0.1), 
                    device
                )
            ),
            end_transition: Param::from_tensor(
                Tensor::<1>::random(
                    [self.num_tags], 
                    Distribution::Uniform(-0.1, 0.1), 
                    device
                )
            ),
            transition: Param::from_tensor(
                Tensor::<2>::random(
                    [self.num_tags, self.num_tags], 
                    Distribution::Uniform(-0.1, 0.1), 
                    device
                )
            ),
        }

    }

}

#[cfg(test)]
pub mod test {

    use super::*;

    const NUM_TAGS: usize = 3;

    fn crf_from_values(start: Vec<f32>, end: Vec<f32>, transition: Vec<f32>, device: &Device) -> CRF {
        let num_tags = start.len();
        CRF {
            start_transition: Param::from_tensor(Tensor::<1>::from_data(TensorData::new(start, [num_tags]), device)),
            end_transition: Param::from_tensor(Tensor::<1>::from_data(TensorData::new(end, [num_tags]), device)),
            transition: Param::from_tensor(Tensor::<2>::from_data(TensorData::new(transition, [num_tags, num_tags]), device)),
        }
    }

    fn emissions_tensor(values: Vec<f32>, shape: [usize; 3], device: &Device) -> Tensor<3> {
        Tensor::<3>::from_data(TensorData::new(values, shape), device)
    }

    fn tags_tensor(ids: Vec<i64>, shape: [usize; 2], device: &Device) -> Tensor<2, Int> {
        Tensor::<2, Int>::from_data(TensorData::new(ids, shape), device)
    }

    fn mask_tensor(values: Vec<bool>, shape: [usize; 2], device: &Device) -> Tensor<2, Bool> {
        Tensor::<2, Bool>::from_data(TensorData::new(values, shape), device)
    }

    fn to_scalar(tensor: Tensor<1>) -> f32 {
        let values = tensor.into_data().convert::<f32>().try_to_vec::<f32>().expect("Can`t read float tensor");
        assert_eq!(values.len(), 1);
        values[0]
    }

    fn assert_close(left: f32, right: f32, tolerance: f32) {
        assert!((left - right).abs() < tolerance, "{left} != {right}");
    }

    /// Negative log-likelihood of one unpadded sequence, computed by enumerating every tag sequence.
    fn brute_force_nll(start: &[f32], end: &[f32], transition: &[f32], emissions: &[f32], tags: &[usize]) -> f32 {
        let num_tags = start.len();
        let seq_len = tags.len();

        let score = |path: &[usize]| {
            let mut total = start[path[0]] + emissions[path[0]];
            for t in 1..seq_len {
                total += transition[path[t - 1] * num_tags + path[t]] + emissions[t * num_tags + path[t]];
            }
            total + end[path[seq_len - 1]]
        };

        let all_scores: Vec<f32> = (0..num_tags.pow(seq_len as u32))
            .map(|mut index| {
                let path: Vec<usize> = (0..seq_len)
                    .map(|_| {
                        let tag = index % num_tags;
                        index /= num_tags;
                        tag
                    })
                    .collect();
                score(&path)
            })
            .collect();

        let maximum = all_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_partition = maximum + all_scores.iter().map(|s| (s - maximum).exp()).sum::<f32>().ln();

        log_partition - score(tags)
    }

    #[test]
    fn forward_returns_scalar() {
        let device = Device::default();
        let crf = CRFConfig::new(NUM_TAGS).init(&device);

        let emissions = Tensor::<3>::random([2, 4, NUM_TAGS], Distribution::Uniform(-1.0, 1.0), &device);
        let tags = tags_tensor(vec![0, 1, 2, 1, 2, 2, 0, 0], [2, 4], &device);
        let mask = mask_tensor(vec![true; 8], [2, 4], &device);

        let loss = crf.forward(emissions, tags, mask);

        assert_eq!(loss.dims(), [1]);
        assert!(to_scalar(loss).is_finite());
    }

    #[test]
    fn forward_is_non_negative() {
        let device = Device::default();
        let crf = CRFConfig::new(NUM_TAGS).init(&device);

        let emissions = Tensor::<3>::random([3, 5, NUM_TAGS], Distribution::Uniform(-3.0, 3.0), &device);
        let tags = tags_tensor((0..15).map(|i| i % NUM_TAGS as i64).collect(), [3, 5], &device);
        let mask = mask_tensor(vec![true; 15], [3, 5], &device);

        assert!(to_scalar(crf.forward(emissions, tags, mask)) >= -1e-5);
    }

    #[test]
    fn forward_with_zero_scores_is_uniform() {
        let device = Device::default();
        let crf = crf_from_values(vec![0.0; NUM_TAGS], vec![0.0; NUM_TAGS], vec![0.0; NUM_TAGS * NUM_TAGS], &device);

        let seq_len = 4;
        let emissions = emissions_tensor(vec![0.0; seq_len * NUM_TAGS], [1, seq_len, NUM_TAGS], &device);
        let tags = tags_tensor(vec![0, 2, 1, 0], [1, seq_len], &device);
        let mask = mask_tensor(vec![true; seq_len], [1, seq_len], &device);

        // every one of NUM_TAGS^seq_len sequences is equally likely
        let expected = seq_len as f32 * (NUM_TAGS as f32).ln();

        assert_close(to_scalar(crf.forward(emissions, tags, mask)), expected, 1e-4);
    }

    #[test]
    fn forward_matches_brute_force() {
        let device = Device::default();

        let start = vec![0.5, -0.2, 0.1];
        let end = vec![-0.3, 0.4, 0.0];
        let transition = vec![
            0.2, -0.5, 0.3,
            0.1, 0.0, -0.4,
            -0.2, 0.6, 0.1,
        ];
        let emissions = vec![
            1.0, 0.2, -0.5,
            -0.3, 0.8, 0.1,
            0.4, -0.6, 0.9,
            0.0, 0.3, -0.2,
        ];
        let tags = vec![0usize, 1, 2, 1];

        let crf = crf_from_values(start.clone(), end.clone(), transition.clone(), &device);
        let loss = crf.forward(
            emissions_tensor(emissions.clone(), [1, 4, NUM_TAGS], &device),
            tags_tensor(tags.iter().map(|&t| t as i64).collect(), [1, 4], &device),
            mask_tensor(vec![true; 4], [1, 4], &device),
        );

        let expected = brute_force_nll(&start, &end, &transition, &emissions, &tags);

        assert_close(to_scalar(loss), expected, 1e-4);
    }

    #[test]
    fn forward_ignores_padded_positions() {
        let device = Device::default();
        let crf = CRFConfig::new(NUM_TAGS).init(&device);

        let real = vec![0.3, -0.1, 0.7, 0.9, 0.2, -0.4];
        let unpadded = crf.forward(
            emissions_tensor(real.clone(), [1, 2, NUM_TAGS], &device),
            tags_tensor(vec![2, 0], [1, 2], &device),
            mask_tensor(vec![true, true], [1, 2], &device),
        );

        // two padded positions with arbitrary emissions and tags
        let mut padded_emissions = real;
        padded_emissions.extend([5.0, -5.0, 2.0, -3.0, 4.0, 1.0]);
        let padded = crf.forward(
            emissions_tensor(padded_emissions, [1, 4, NUM_TAGS], &device),
            tags_tensor(vec![2, 0, 1, 2], [1, 4], &device),
            mask_tensor(vec![true, true, false, false], [1, 4], &device),
        );

        assert_close(to_scalar(unpadded), to_scalar(padded), 1e-5);
    }

    #[test]
    fn forward_averages_over_batch() {
        let device = Device::default();
        let crf = CRFConfig::new(NUM_TAGS).init(&device);

        let first_emissions = vec![0.1, 0.5, -0.2, 0.3, -0.7, 0.4, 0.0, 0.2, 0.6];
        let second_emissions = vec![-0.4, 0.8, 0.1, 0.9, 0.0, -0.3, 0.5, 0.5, -0.1];
        let first_tags = vec![1, 0, 2];
        let second_tags = vec![0, 2, 2];

        let single = |emissions: Vec<f32>, tags: Vec<i64>| {
            to_scalar(crf.forward(
                emissions_tensor(emissions, [1, 3, NUM_TAGS], &device),
                tags_tensor(tags, [1, 3], &device),
                mask_tensor(vec![true; 3], [1, 3], &device),
            ))
        };
        let first = single(first_emissions.clone(), first_tags.clone());
        let second = single(second_emissions.clone(), second_tags.clone());

        let batch = crf.forward(
            emissions_tensor([first_emissions, second_emissions].concat(), [2, 3, NUM_TAGS], &device),
            tags_tensor([first_tags, second_tags].concat(), [2, 3], &device),
            mask_tensor(vec![true; 6], [2, 3], &device),
        );

        assert_close(to_scalar(batch), (first + second) / 2.0, 1e-5);
    }

    #[test]
    fn forward_handles_sequences_of_different_lengths() {
        let device = Device::default();

        let start = vec![0.2, -0.1, 0.4];
        let end = vec![0.0, 0.3, -0.2];
        let transition = vec![
            0.1, 0.2, -0.3,
            -0.1, 0.5, 0.0,
            0.3, -0.4, 0.2,
        ];
        let crf = crf_from_values(start.clone(), end.clone(), transition.clone(), &device);

        let long_emissions = vec![0.5, -0.5, 0.0, 0.1, 0.9, -0.2, -0.6, 0.3, 0.7];
        let short_emissions = vec![0.8, 0.1, -0.4, 0.2, -0.3, 0.6];
        let long_tags = vec![0usize, 1, 2];
        let short_tags = vec![2usize, 0];

        let mut padded_short = short_emissions.clone();
        padded_short.extend([1.0, 1.0, 1.0]);

        let loss = crf.forward(
            emissions_tensor([long_emissions.clone(), padded_short].concat(), [2, 3, NUM_TAGS], &device),
            tags_tensor(vec![0, 1, 2, 2, 0, 0], [2, 3], &device),
            mask_tensor(vec![true, true, true, true, true, false], [2, 3], &device),
        );

        let expected = (brute_force_nll(&start, &end, &transition, &long_emissions, &long_tags)
            + brute_force_nll(&start, &end, &transition, &short_emissions, &short_tags))
            / 2.0;

        assert_close(to_scalar(loss), expected, 1e-4);
    }

    #[test]
    fn forward_single_token_sequence() {
        let device = Device::default();

        let start = vec![0.1, 0.2, 0.3];
        let end = vec![0.3, 0.2, 0.1];
        let emissions = vec![1.0, 0.0, -1.0];
        let crf = crf_from_values(start.clone(), end.clone(), vec![0.0; NUM_TAGS * NUM_TAGS], &device);

        let loss = crf.forward(
            emissions_tensor(emissions.clone(), [1, 1, NUM_TAGS], &device),
            tags_tensor(vec![0], [1, 1], &device),
            mask_tensor(vec![true], [1, 1], &device),
        );

        let expected = brute_force_nll(&start, &end, &[0.0; NUM_TAGS * NUM_TAGS], &emissions, &[0]);

        assert_close(to_scalar(loss), expected, 1e-5);
    }

    /// Best tag sequence of one unpadded sequence, found by enumerating every tag sequence.
    fn brute_force_decode(start: &[f32], end: &[f32], transition: &[f32], emissions: &[f32], seq_len: usize) -> Vec<usize> {
        let num_tags = start.len();

        (0..num_tags.pow(seq_len as u32))
            .map(|mut index| {
                (0..seq_len)
                    .map(|_| {
                        let tag = index % num_tags;
                        index /= num_tags;
                        tag
                    })
                    .collect::<Vec<usize>>()
            })
            .map(|path| {
                let mut total = start[path[0]] + emissions[path[0]];
                for t in 1..seq_len {
                    total += transition[path[t - 1] * num_tags + path[t]] + emissions[t * num_tags + path[t]];
                }
                (total + end[path[seq_len - 1]], path)
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .map(|(_, path)| path)
            .expect("No tag sequences")
    }

    #[test]
    fn decode_matches_brute_force() {
        let device = Device::default();

        let start = vec![0.5, -0.2, 0.1];
        let end = vec![-0.3, 0.4, 0.0];
        let transition = vec![
            0.2, -0.5, 0.3,
            0.1, 0.0, -0.4,
            -0.2, 0.6, 0.1,
        ];
        let emissions = vec![
            1.0, 0.2, -0.5,
            -0.3, 0.8, 0.1,
            0.4, -0.6, 0.9,
            0.0, 0.3, -0.2,
        ];

        let crf = crf_from_values(start.clone(), end.clone(), transition.clone(), &device);
        let paths = crf.decode(
            emissions_tensor(emissions.clone(), [1, 4, NUM_TAGS], &device),
            mask_tensor(vec![true; 4], [1, 4], &device),
        );

        assert_eq!(paths, vec![brute_force_decode(&start, &end, &transition, &emissions, 4)]);
    }

    #[test]
    fn decode_handles_sequences_of_different_lengths() {
        let device = Device::default();

        let start = vec![0.2, -0.1, 0.4];
        let end = vec![0.0, 0.3, -0.2];
        let transition = vec![
            0.1, 0.2, -0.3,
            -0.1, 0.5, 0.0,
            0.3, -0.4, 0.2,
        ];
        let crf = crf_from_values(start.clone(), end.clone(), transition.clone(), &device);

        let long_emissions = vec![0.5, -0.5, 0.0, 0.1, 0.9, -0.2, -0.6, 0.3, 0.7];
        let short_emissions = vec![0.8, 0.1, -0.4, 0.2, -0.3, 0.6];
        let single_emissions = vec![-0.2, 0.0, 0.5];

        let mut padded_short = short_emissions.clone();
        padded_short.extend([9.0, -9.0, 9.0]);
        let mut padded_single = single_emissions.clone();
        padded_single.extend([9.0, 9.0, -9.0, -9.0, 9.0, 9.0]);

        let paths = crf.decode(
            emissions_tensor([long_emissions.clone(), padded_short, padded_single].concat(), [3, 3, NUM_TAGS], &device),
            mask_tensor(vec![true, true, true, true, true, false, true, false, false], [3, 3], &device),
        );

        assert_eq!(
            paths,
            vec![
                brute_force_decode(&start, &end, &transition, &long_emissions, 3),
                brute_force_decode(&start, &end, &transition, &short_emissions, 2),
                brute_force_decode(&start, &end, &transition, &single_emissions, 1),
            ]
        );
    }

    #[test]
    fn decode_single_token_sequence() {
        let device = Device::default();
        let crf = crf_from_values(vec![0.0; NUM_TAGS], vec![0.0; NUM_TAGS], vec![0.0; NUM_TAGS * NUM_TAGS], &device);

        let paths = crf.decode(
            emissions_tensor(vec![-1.0, 2.0, 0.5], [1, 1, NUM_TAGS], &device),
            mask_tensor(vec![true], [1, 1], &device),
        );

        assert_eq!(paths, vec![vec![1]]);
    }

    #[test]
    fn forward_is_near_zero_for_confident_correct_emissions() {
        let device = Device::default();
        let crf = CRFConfig::new(NUM_TAGS).init(&device);

        let tags = vec![2, 0, 1];
        let emissions: Vec<f32> = tags
            .iter()
            .flat_map(|&gold| (0..NUM_TAGS as i64).map(move |tag| if tag == gold { 20.0 } else { -20.0 }))
            .collect();

        let loss = crf.forward(
            emissions_tensor(emissions, [1, 3, NUM_TAGS], &device),
            tags_tensor(tags, [1, 3], &device),
            mask_tensor(vec![true; 3], [1, 3], &device),
        );

        assert_close(to_scalar(loss), 0.0, 1e-3);
    }
}
