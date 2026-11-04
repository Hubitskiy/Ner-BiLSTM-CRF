use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::BufWriter;
use std::marker::PhantomData;
use std::path::Path;

use log;
use indexmap::{IndexMap, IndexSet};
use thiserror::Error;
use serde_json;

use neuro::tokenizers::tokenizer::ToTokenizedWithOffset;
use neuro::vocabs::vocab::KeyValeBasedVocabulary;
use crate::bilstm_crf::{types::{NerExampleSample}};


/// Marks a [`NerVocab`] of words.
pub struct NerSamples;

/// Marks a [`NerVocab`] of characters.
pub struct NerChars;

/// Marks a [`NerVocab`] of BIO labels.
pub struct NerLabels;

/// Marks a [`NerVocab`] restored from an existing key-to-id map.
pub struct BuiltVocab;

impl Display for BuiltVocab {
    /// Writes the name of the vocabulary kind.
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Built Vocabulary")
    }
}


/// Vocabulary mapping string keys to integer ids.
///
/// `Kind` is a compile-time marker that selects how the vocabulary is built from
/// samples: [`NerSamples`], [`NerChars`], [`NerLabels`] or [`BuiltVocab`].
#[derive(Debug)]
pub struct NerVocab<Kind> {
    /// Keys and their ids, in insertion order.
    internal: IndexMap<String, i64>,
    /// Binds the vocabulary to its kind.
    _kind: PhantomData<Kind>,
}


impl<Kind> NerVocab<Kind> {

    pub fn invert(&self) -> IndexMap<i64, String> {
        self.internal
            .iter()
            .map(|(key, &id)| (id, key.clone()))
            .collect()
    }
}




impl<Kind> Clone for NerVocab<Kind> {
    fn clone(&self) -> Self {
        Self {
            internal: self.internal.clone(),
            _kind: PhantomData,
        }
    }
}


#[derive(Debug, Error)]
pub enum NerIOVocabError {
    #[error("Got error when try to create file: {0}")]
    CreateFileError(#[from] std::io::Error),

    #[error("Got error when try write to file: {0}")]
    WriteToFileError(#[from] serde_json::Error),
}


const PAD: &str = "<PAD>";

const PAD_IDX: i64 = 0;

const UNK: &str = "<UNK>";

const UNK_IDX: i64 = 1;

impl<Kind> KeyValeBasedVocabulary for NerVocab<Kind> {

    type Key = String;
    type Value = i64;

    type IOErrors = NerIOVocabError;

    /// Creates a vocabulary that holds only the padding and unknown entries.
    fn new() -> Self {

        Self{
            internal:[
                (Self::pad_key(), Self::pad_value()),
                (Self::unk_key(), Self::unk_value())
            ].into_iter().collect::<IndexMap<_, _>>(),
            _kind: PhantomData,
        }

    }

    /// Returns the number of entries.
    #[inline]
    fn len(&self) -> usize { self.internal.len() }

    /// Returns `true` if the vocabulary has no entries.
    #[inline]
    fn is_empty(&self) -> bool {
        self.internal.is_empty()
    }

    /// Returns the id of `key`, if present.
    #[inline]
    fn get(&self, key: &Self::Key) -> Option<&Self::Value> {
        self.internal.get(key)
    }

    /// Inserts `key` with id `value`, replacing the id of an existing key.
    fn insert(&mut self, key: Self::Key, value: Self::Value) {
        self.internal.insert(key, value);
    }

    /// Returns the key of the unknown entry.
    #[inline]
    fn unk_key() -> Self::Key {
        String::from(UNK)
    }

    /// Returns the id of the unknown entry.
    #[inline]
    fn unk_value() -> Self::Value {
        UNK_IDX
    }

    /// Returns the key of the padding entry.
    #[inline]
    fn pad_key() -> Self::Key {
        String::from(PAD)
    }

    /// Returns the id of the padding entry.
    #[inline]
    fn pad_value() -> Self::Value {
        PAD_IDX
    }

    /// Saves the vocabulary to a file as pretty-printed JSON.
    ///
    /// # Arguments
    ///
    /// * `path` - path of the file, which is created or overwritten.
    ///
    /// # Errors
    ///
    /// Returns [`NerIOVocabError`] if the file can't be created or written.
    fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), Self::IOErrors> {
        let to_write = File::create(path)?;

        let writer = BufWriter::new(to_write);

        serde_json::to_writer_pretty(writer, &self.internal)?;

        Ok(())
    }

    /// Returns `true` if `key` is present.
    #[inline]
    fn contains_key(&self, key: &Self::Key) -> bool {
        self.internal.contains_key(key)
    }

    /// Creates a vocabulary with no entries, not even padding and unknown.
    #[inline]
    fn empty() -> Self {
        Self{
            internal: IndexMap::new(),
            _kind: PhantomData,
        }
    }
}


impl From<&[NerExampleSample]> for NerVocab<NerSamples> {

    /// Builds a words vocabulary from the tokens of `samples`.
    ///
    /// Ids follow the padding and unknown entries and are assigned in sorted key order.
    ///
    /// # Arguments
    ///
    /// * `samples` - samples to collect tokens from.
    fn from(samples: &[NerExampleSample]) -> Self {

        log::info!("Start creating Words Vocabulary");

        let samples_count = samples.len();

        log::info!("Try to built Frequency Vocabulary from Samples: {}", samples_count);

        let mut freq_vocab = IndexMap::<String, usize>::new();

        log::info!("Try to create Frequency map");

        samples.iter()
            .for_each(|sample| {

                let tokenized = sample.to_tokenized_with_offset();

                tokenized.tokens().iter()
                    .for_each(|token|
                        *freq_vocab.entry(token.clone()).or_insert(0) +=1
                    );

                }
            );

        log::info!("Frequency map was created");

        log::info!("Try to create Words IDX Vocabulary: {}", freq_vocab.len());
        freq_vocab.sort_keys();

        let mut vocab = Self::new();

        let min_freq = 1_usize;


        freq_vocab.into_iter().for_each(|(k, v)| {
            if v >= min_freq {
                vocab.insert(k, vocab.len() as i64);
            };
        });
        log::info!("Words IDX Vocabulary was created");
        log::info!("End creating Words Vocabulary");

        vocab

    }

}


impl From<NerVocab<BuiltVocab>> for NerVocab<NerSamples> {

    fn from(vocab: NerVocab<BuiltVocab>) -> Self {
        Self {
            internal: vocab.internal,
            _kind: PhantomData,
        }
    }
}


impl From<NerVocab<BuiltVocab>> for NerVocab<NerChars> {

    fn from(vocab: NerVocab<BuiltVocab>) -> Self {
        Self {
            internal: vocab.internal,
            _kind: PhantomData,
        }
    }
}


impl From<NerVocab<BuiltVocab>> for NerVocab<NerLabels> {

    fn from(vocab: NerVocab<BuiltVocab>) -> Self {
        Self {
            internal: vocab.internal,
            _kind: PhantomData,
        }
    }
}


impl From<&[NerExampleSample]> for NerVocab<NerChars> {

    /// Builds a characters vocabulary from the normalized text of `samples`.
    ///
    /// Ids follow the padding and unknown entries and are assigned in sorted character
    /// order. Whitespace left by normalization is included.
    ///
    /// # Arguments
    ///
    /// * `samples` - samples to collect characters from.
    fn from(samples: &[NerExampleSample]) -> Self {
        log::info!("Start creating Chars Vocabulary");

        let mut vocab = Self::new();
        log::info!("Try to build Unique Chars map: {}", samples.len());

        let mut chars = samples.iter()
            .flat_map(|sample| sample.to_normalized().chars().collect::<Vec<char>>())
            .collect::<IndexSet<char>>();

        chars.sort_unstable();
        
        log::info!("Unique Chars map was created");

        log::info!("Try to create Chars IDX Vocabulary");

        chars.into_iter().for_each(|c| { vocab.insert(c.to_string(), vocab.len() as i64); });

        log::info!("Chars IDX Vocabulary was created");

        log::info!("Chars Vocabulary was created");
        vocab
    }
}


impl From<&[NerExampleSample]> for NerVocab<NerLabels> {


    /// Builds a labels vocabulary from the entities of `samples`.
    ///
    /// `O` gets id 0 and the sorted `B-<entity>` and `I-<entity>` tags get ids from 1.
    /// The vocabulary has no padding or unknown entries.
    ///
    /// # Arguments
    ///
    /// * `samples` - samples to collect entity names from.
    fn from(samples: &[NerExampleSample]) -> Self {

        log::info!("Start creating Labels Vocabulary");

        let mut vocab = Self::empty();
        vocab.insert("O".to_owned(), 0);

        log::info!("Try to create Unique Labels Vocabulary");

        let set: HashSet<String> = samples
            .iter()
            .flat_map(|sample| {
                sample.entities().iter().flat_map(|entity| {
                    let name = entity.entity();
                    [format!("B-{name}"), format!("I-{name}")]
                })
            })
            .collect();

        let mut tags: Vec<String> = set.into_iter().collect();
        tags.sort_unstable();

        log::info!("Unique Labels Vocabulary was created");

        log::info!("Try to create Labels IDX Vocabulary");

        for (index, tag) in (1_i64..).zip(tags) {
            vocab.insert(tag, index);
        }

        log::info!("Labels IDX Vocabulary was created");

        log::info!("Labels Vocabulary was created");
        vocab
    }
}


impl From<IndexMap<String, i64>> for NerVocab<BuiltVocab> {

    /// Wraps an existing key-to-id map without adding any entries.
    fn from(vocab: IndexMap<String, i64>) -> Self {
        Self {internal: vocab, _kind: PhantomData }
    }
}


#[cfg(test)]
pub mod test {
    use std::io::BufReader;
    use tempfile::tempdir;
    use std::collections::HashSet;

    use super::*;
    use crate::bilstm_crf::{types::{NerExampleSample}};

    /// Checks that a saved vocabulary keeps all of its entries.
    #[test]
    fn save_vocab_to_file() {
        let temp_dir = tempdir().expect("Can`t create tempdir");

        let output_path = temp_dir.path().join("vocab.json");

        let mut vocab = NerVocab::<NerSamples>::new();

        vocab.insert(String::from("A"), 1);
        vocab.insert(String::from("B"), 2);
        vocab.save_to_file(&output_path).expect("Can`t save vocab to file");

        assert!(output_path.is_file());

        let vocab_file = File::open(&output_path).expect("Can`t open vocab file");
        let reader = BufReader::new(vocab_file);

        let saved_vocab: IndexMap<String, i64> = serde_json::from_reader(reader).expect("Can`t read vocab from file");

        assert_eq!(vocab.len(), saved_vocab.len());

        assert_eq!(vocab.get(&"A".to_string()), saved_vocab.get(&"A".to_string()));
        assert_eq!(vocab.get(&"B".to_string()), saved_vocab.get(&"B".to_string()));

        assert_eq!(vocab.get(&NerVocab::<NerSamples>::pad_key()), saved_vocab.get(&NerVocab::<NerSamples>::pad_key()));
        assert_eq!(vocab.get(&NerVocab::<NerSamples>::unk_key()), saved_vocab.get(&NerVocab::<NerSamples>::unk_key()));

    }


    /// Checks the special and word ids of a vocabulary built from samples.
    #[test]
    fn vocab_from_words_samples() {

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

        let vocab = NerVocab::<NerSamples>::from(samples.as_slice());
        assert!(vocab.len() > 0);
        assert_eq!(vocab.get(&NerVocab::<NerSamples>::pad_key()), Some(&0));
        assert_eq!(vocab.get(&NerVocab::<NerSamples>::unk_key()), Some(&1));
        assert_eq!(vocab.get(&"boston".to_string()), Some(&7));
        assert_eq!(vocab.get(&"flight".to_string()), Some(&11));
        assert_eq!(vocab.len(), vocab.internal.values().copied().collect::<HashSet<i64>>().len());

    }

    /// Checks that a vocabulary restored from a saved map keeps its ids.
    #[test]
    fn vocab_from_char_saved_char_vocab() {
        let chars_vocab = r#"
            {
              "<PAD>": 0,
              "<UNK>": 1,
              "d": 2,
              "o": 3,
              "e": 4,
              "s": 5
            }
        "#;

        let samples: IndexMap<String, i64> = serde_json::from_str(chars_vocab).expect("Can`t parse samples");
        let vocab = NerVocab::<BuiltVocab>::from(samples);
        assert_eq!(vocab.len(), 6);
        assert_eq!(vocab.get(&NerVocab::<BuiltVocab>::pad_key()), Some(&0));
        assert_eq!(vocab.get(&"d".to_string()), Some(&2));


    }

    /// Checks the special and character ids of a vocabulary built from samples.
    #[test]
    fn build_char_vocab() {
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

        let vocab = NerVocab::<NerChars>::from(samples.as_slice());

        assert_eq!(vocab.get(&String::from('a')), Some(&3));
        assert_eq!(vocab.get(&NerVocab::<NerChars>::pad_key()), Some(&0));
        assert_eq!(vocab.get(&NerVocab::<NerChars>::unk_key()), Some(&1));
        assert_eq!(vocab.get(&String::from('i')), Some(&10));

    }

    #[test]
    /// Checks that a labels vocabulary holds `O` and both BIO tags of every entity.
    fn build_labels_vocab() {
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

        let vocab = NerVocab::<NerLabels>::from(samples.as_slice());

        assert_eq!(vocab.len(), 5);
        assert!(vocab.contains_key(&String::from("B-send_from")));
        assert!(vocab.contains_key(&String::from("I-send_from")));
        assert!(vocab.contains_key(&String::from("B-send_to")));
        assert!(vocab.contains_key(&String::from("I-send_to")));
        assert!(vocab.contains_key(&String::from('O')));


    }

}