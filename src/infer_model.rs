use std::path::Path;

use burn::prelude::*;
use burn::store::ModuleRecord;
use indexmap::IndexMap;
use crate::bilstm_crf::bilstmcrf::BiLstmCRFModelConfig;
use crate::bilstm_crf::predict_text::predict_ner_entitties;
use crate::bilstm_crf::vocab::{BuiltVocab, NerChars, NerSamples, NerVocab};
use crate::bilstm_crf::utils::select_device;

mod bilstm_crf;

/// Reads a `key → id` vocabulary from a JSON file.
fn read_vocab_map(path: impl AsRef<Path>) -> IndexMap<String, i64> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("Can`t read vocab {}: {err}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("Can`t parse vocab {}: {err}", path.display()))
}

fn main() {
    let artifact_dir = "./src/artifacts";
    let vocab_dir = "./src/data";

    let device = select_device();

    println!("Step 1: Load model config");
    let model_config = BiLstmCRFModelConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Can`t load model config");

    println!("Step 2: Load model record");
    let model_record = ModuleRecord::load(format!("{artifact_dir}/model"))
        .expect("Can`t load model record");

    println!("Step 3: Initialize model");
    // Plain device without autodiff: inference only
    let model = model_config.init(&device).load_record(model_record);

    println!("Step 4: Load vocabularies");

    println!("Step 4.1: Chars vocab");
    let chars_vocab = NerVocab::<NerChars>::from(NerVocab::<BuiltVocab>::from(
        read_vocab_map(format!("{vocab_dir}/chars_vocab.json")),
    ));

    println!("Step 4.2: Words vocab");
    let words_vocab = NerVocab::<NerSamples>::from(NerVocab::<BuiltVocab>::from(
        read_vocab_map(format!("{vocab_dir}/words_vocab.json")),
    ));

    println!("Step 4.3: Labels vocab");
    let inv_labels_vocab = NerVocab::<BuiltVocab>::from(
        read_vocab_map(format!("{vocab_dir}/labels_vocab.json")),
    )
        .invert();

    println!("Step 5: Predict");
    let infer_examples = [
        "how to fly from macedonia to israel",
        "how to reach england from france",
        "how to travel from usa to denmark",
    ];

    for example in infer_examples {
        let prediction = predict_ner_entitties(
            &model,
            example,
            &words_vocab,
            &chars_vocab,
            &inv_labels_vocab,
            &device,
        );

        println!("\nText:   {}", prediction.text());
        println!("Tokens: {:?}", prediction.tokens());
        println!("Labels: {:?}", prediction.labels());
        println!("Slots:  {:?}", prediction.slots());
    }
}