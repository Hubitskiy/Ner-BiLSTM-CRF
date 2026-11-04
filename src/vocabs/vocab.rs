use std::error::Error;
use std::path::Path;

/// By implementing Vocab to the type you define as type which can be used as vocabulary.
/// Defining fn unk_key(&self), fn pad_key(&self), fn unk_value(&self) and fn pad_key(&self) determine unknown and padding tokens.
///
/// Examples
/// ```
/// use indexmap::IndexMap;
/// use thiserror::Error;
/// use neuro::vocabs::vocab::KeyValeBasedVocabulary;
/// use std::fs::File;
/// use std::io::BufWriter;
/// use std::path::Path;
///
/// const PAD: &str = "<PAD>";
///
/// const PAD_IDX: usize = 0;
///
/// const UNK: &str = "<UNK>";
///
/// const UNK_IDX: usize = 1;
///
/// #[derive(Debug, Error)]
/// pub enum IOVocabError {
///     #[error("Got error when try to create file: {0}")]
///     CreateFileError(#[from] std::io::Error),
///
///     #[error("Got error when try write to file: {0}")]
///     WriteToFileError(#[from] serde_json::Error),
/// }
///
/// pub struct WordsVocab {
///     internal: IndexMap<String, usize>,
/// }
///
/// impl KeyValeBasedVocabulary for WordsVocab {
///     type Key = String;
///     type Value = usize;
///     type IOErrors = IOVocabError;
///
///     fn new() -> Self {
///
///         Self{
///             internal:[
///                 (Self::pad_key(), Self::pad_value()),
///                 (Self::unk_key(), Self::unk_value())
///             ].into_iter().collect::<IndexMap<_, _>>()
///         }
///
///     }
///
///     #[inline]
///     fn len(&self) -> usize { self.internal.len() }
///
///     #[inline]
///     fn is_empty(&self) -> bool {
///         self.internal.is_empty()
///     }
///
///     #[inline]
///     fn get(&self, key: &Self::Key) -> Option<&Self::Value> {
///         self.internal.get(key)
///     }
///
///     fn insert(&mut self, key: Self::Key, value: Self::Value) {
///         self.internal.insert(key, value);
///     }
///
///     #[inline]
///     fn unk_key() -> Self::Key {
///         String::from(UNK)
///     }
///
///     #[inline]
///     fn unk_value() -> Self::Value {
///         UNK_IDX
///     }
///
///     #[inline]
///     fn pad_key() -> Self::Key {
///         String::from(PAD)
///     }
///
///     #[inline]
///     fn pad_value() -> Self::Value {
///         PAD_IDX
///     }
///
///     fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), Self::IOErrors> {
///         let to_write = File::create(path)?;
///
///         let writer = BufWriter::new(to_write);
///
///         serde_json::to_writer_pretty(writer, &self.internal)?;
///
///         Ok(())
///     }
///
///     fn contains_key(&self, key: &Self::Key) -> bool {
///         self.internal.contains_key(key)
///     }
///
///     #[inline]
///     fn empty() -> Self {
///         Self{
///             internal: IndexMap::new(),
///         }
///     }
///
/// }
///
///
/// let mut words_vocab = WordsVocab::new();
///
/// words_vocab.insert("Hello".to_string(), 2);
/// words_vocab.insert("World".to_string(), 3);
///
/// assert_eq!(words_vocab.len(), 4);
/// assert_eq!(words_vocab.get(&"Hello".to_string()), Some(&2));
/// assert_eq!(words_vocab.get(&WordsVocab::pad_key()), Some(&WordsVocab::pad_value()));
/// assert_eq!(words_vocab.get(&WordsVocab::unk_key()), Some(&WordsVocab::unk_value()));
/// assert_eq!(words_vocab.is_empty(), false);
/// ```
pub trait KeyValeBasedVocabulary {

    type Key;
    type Value;

    type IOErrors: Error;

    fn new() -> Self;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool;

    fn get(&self, key: &Self::Key) -> Option<&Self::Value>;

    fn insert(&mut self, key: Self::Key, value: Self::Value);

    fn unk_key() -> Self::Key;

    fn unk_value() -> Self::Value;

    fn pad_key() -> Self::Key;

    fn pad_value() -> Self::Value;

    fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), Self::IOErrors>;

    fn contains_key(&self, key: &Self::Key) -> bool;

    fn empty() -> Self;

}
