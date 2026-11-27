use std::path::Path;
use burn::{
    data::dataloader::DataLoaderBuilder,
    grad_clipping::GradientClippingConfig,
    optim::AdamConfig,
    train::{
        SupervisedTraining,
        Learner,
        metric::LossMetric,
    }
};
use burn::lr_scheduler::constant::ConstantLr;
use burn::module::Module;
use burn::prelude::*;
use indexmap::IndexMap;
use neuro::vocabs::prelude::*;
use crate::bilstm_crf::{
    batcher::NerBatcher,
    types::NerExampleSample,
    bilstmcrf::{
        BiLstmCRFModel,
        BiLstmCRFModelConfig,
    },
    dataset::NerDataset,
    vocab::{BuiltVocab, NerVocab}
};
use crate::bilstm_crf::utils::select_device;

pub mod bilstm_crf;


fn create_train_artifacts_dir(dir: &Path){
    std::fs::remove_dir_all(dir.join("experimental.log")).ok();
    std::fs::create_dir(dir).ok();
}

const NUM_EPOCHS: usize = 5;
const WORDS_VOCAB_EMB_DIM: usize = 128;
const CHARS_VOCAB_EMB_DIM: usize = 32;
const CHAR_HIDDEN_DIM: usize = 32;
const LSTM_HIDDEN_DIM: usize = 128;
const INPUT_DROPOUT: f64 = 0.2;
const OUTPUT_DROPOUT: f64 = 0.2;


fn main() {

    println!("Start training...");
    println!("Step 1: Crate artifacts dir");
    let artifact_dir = "./src/artifacts";

    create_train_artifacts_dir(Path::new(artifact_dir));

    let device = select_device();

    device.seed(42);

    let ad_device = device.autodiff();

    println!("Step 2: Download vocabularies");

    println!("Step 2.1: Create chars Vocab");
    let chars_raw_data = std::fs::read_to_string("./src/data/chars_vocab.json").expect("Can`t read chars data");
    let chars_sample: IndexMap<String, i64> = serde_json::from_str(chars_raw_data.as_str()).expect("Can`t parse chars samples");
    let chars_vocab = NerVocab::<BuiltVocab>::from(chars_sample);

    println!("Step 2.2: Create words Vocab");
    let words_raw_data = std::fs::read_to_string("./src/data/words_vocab.json").expect("Can`t read words data");
    let words_sample: IndexMap<String, i64> = serde_json::from_str(words_raw_data.as_str()).expect("Can`t parse words samples");
    let words_vocab = NerVocab::<BuiltVocab>::from(words_sample);

    println!("Step 2.3: Create labels Vocab");
    let labes_raw_data = std::fs::read_to_string("./src/data/labels_vocab.json").expect("Can`t read labels data");
    let labels_sample: IndexMap<String, i64> = serde_json::from_str(labes_raw_data.as_str()).expect("Can`t parse labels samples");
    let labels_vocab = NerVocab::<BuiltVocab>::from(labels_sample);

    println!("Step 3: Create train and test datasets");

    let train_raw_samples = std::fs::read_to_string("./src/labeled_data/train_augmented.json").expect("Can`t read train data");
    let train_samples: Vec<NerExampleSample> = serde_json::from_str(train_raw_samples.as_str()).expect("Can`t parse train samples");
    let train_dataset = NerDataset::create(train_samples, words_vocab.clone().into(), chars_vocab.clone().into(), labels_vocab.clone().into());

    let test_raw_samples = std::fs::read_to_string("./src/labeled_data/test_augmented.json").expect("Can`t read test data");
    let test_samples: Vec<NerExampleSample> = serde_json::from_str(test_raw_samples.as_str()).expect("Can`t parse test samples");
    let test_dataset = NerDataset::create(test_samples, words_vocab.clone().into(), chars_vocab.clone().into(), labels_vocab.clone().into());

    println!("Step 4: Initialize training");

    let batcher = NerBatcher::new();

    let train_dl = DataLoaderBuilder::new(batcher.clone())
        .batch_size(16)
        .shuffle(42)
        .num_workers(4)
        .build(train_dataset);

    let test_dl = DataLoaderBuilder::new(batcher.clone()).batch_size(16).build(test_dataset);

    let optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(5.0)))
        .init();

    let training = SupervisedTraining::<BiLstmCRFModel>::new(artifact_dir, train_dl, test_dl)
        .metrics((LossMetric::new(),))
        .with_default_checkpointers()
        .num_epochs(NUM_EPOCHS)
        .summary();

    println!("Step 5: Initialize model");

    let model_config = BiLstmCRFModelConfig::new(
        words_vocab.len(),
        chars_vocab.len(),
        labels_vocab.len(),
        WORDS_VOCAB_EMB_DIM,
        CHARS_VOCAB_EMB_DIM,
        CHAR_HIDDEN_DIM,
        LSTM_HIDDEN_DIM,
        INPUT_DROPOUT,
        OUTPUT_DROPOUT,
    );

    println!("Step 6: Save model config");
    model_config.save(format!("{artifact_dir}/config.json")).expect("Can`t save model config");

    let model = model_config.init(&ad_device);

    println!("Step 7: Start training");

    let result = training.launch(Learner::new(model, optimizer, ConstantLr::from(1e-3)));

    println!("Step 8: End training");

    println!("Step 9: Save model weights");
    result.model.into_record().save(format!("{artifact_dir}/model")).expect("Can`t save model weights");
    println!("Step 10: Model weights saved");

}