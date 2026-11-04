use std::fs;
use std::path::Path;

use env_logger::Env;

use neuro::vocabs::vocab::KeyValeBasedVocabulary;
use crate::bilstm_crf::types::NerExampleSample;
use crate::bilstm_crf::vocab::{NerVocab, NerSamples, NerChars, NerLabels};

pub mod config;
pub mod bilstm_crf;


fn main() {

    env_logger::Builder::from_env(Env::default().default_filter_or("info,warning")).init();

    log::info!("Start training...");

    log::info!("Step 1: Create vocabularies");

    let raw_data = fs::read_to_string("./src/labeled_data/train_augmented.json").expect("Failed to read file");
    let samples: Vec<NerExampleSample> = serde_json::from_str(&raw_data).expect("Failed to parse vocab.json");

    let samples_slice = samples.as_slice();

    let words_vocab = NerVocab::<NerSamples>::from(samples_slice);
    let chars_vocab = NerVocab::<NerChars>::from(samples_slice);
    let labels_vocab = NerVocab::<NerLabels>::from(samples_slice);

    words_vocab.save_to_file(Path::new("./src/data/words_vocab.json")).expect("Failed to write words_vocab.json");
    chars_vocab.save_to_file(Path::new("./src/data/chars_vocab.json")).expect("Failed to write chars_vocab.json");
    labels_vocab.save_to_file(Path::new("./src/data/labels_vocab.json")).expect("Failed to write labels_vocab.json");

    log::info!("End Step 1: Vocabulary created");
}
