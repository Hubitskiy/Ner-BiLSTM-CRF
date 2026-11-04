use regex::Regex;

type Start = usize;
type End = usize;


// TokenizedString contains words with offsets
#[derive(Debug)]
pub struct TokenizedString{
    tokens: Vec<String>,
    offsets: Vec<(Start, End)>
}


impl TokenizedString{

    pub fn new(tokens: Vec<String>, offsets: Vec<(Start, End)>) -> Self{
        Self{tokens, offsets}
    }

    #[inline]
    pub fn tokens(&self) -> &Vec<String> {
        &self.tokens
    }

    #[inline]
    pub fn offsets(&self) -> &Vec<(Start, End)> {
        &self.offsets
    }

    #[inline]
    pub fn own_parts(self) -> (Vec<String>, Vec<(Start, End)>) {
        (self.tokens, self.offsets)
    }
}

/// Tokenize words into a TokenizedString structure with word and position vectors.
///
/// # Examples
/// ```
///     use neuro::tokenizers::tokenizer::{tokenize_with_offset, normalizer};
///     let to_tokenize = String::from("Brave new world");
///
///     let result = tokenize_with_offset(&to_tokenize, normalizer);
///
///     assert_eq!(result.tokens(), &vec!["brave".to_string(), "new".to_string(), "world".to_string()]);
///     assert_eq!(result.offsets(), &vec![(0, 5), (6, 9), (10, 15)]);
/// ```
pub fn  tokenize_with_offset<N>(text: &str, normalizer: N) -> TokenizedString
where N: FnOnce(&str) -> String
{
    let normalized_string = normalizer(text);

    let re = Regex::new(r"\w+|[^\w\s]").expect("Failed to compile tokenize words regex");

    let matches = re.find_iter(&normalized_string).collect::<Vec<_>>();

    let capacity = matches.len();

    let (mut tokens, mut offsets) = (
        Vec::with_capacity(capacity),
        Vec::with_capacity(capacity)
    );

    matches
        .into_iter()
        .for_each(
            |m| {
                let (start, end, token)= (m.start(), m.end(), m.as_str().to_string());
                tokens.push(token);
                offsets.push((start, end));
            }
        );

    TokenizedString::new(tokens, offsets)
}


/// Normalize string by escaping whitespaces and make it lowercase
///
/// # Examples
/// ```
/// use neuro::tokenizers::tokenizer::normalizer;
/// let result = normalizer(&String::from(" Hello  world" ));
///
///assert_eq!(result, String::from("hello world"));
/// ```
pub fn normalizer(text: &str) -> String {
    let re = Regex::new(r"\s+").expect("Failed to compile normalize words regex");

    re.replace_all(text, " ").trim().to_string().to_lowercase()
}


/// Trait implements tokenization with offsets to struct
///
/// # Examples
/// ```
/// use neuro::tokenizers::tokenizer::{
///     tokenize_with_offset,
///     normalizer,
///     TokenizedString,
///     ToTokenizedWithOffset,
/// };
///
/// struct SomeText {
///     text: String
/// }
///
/// impl ToTokenizedWithOffset for SomeText {
///     fn to_tokenized_with_offset(&self) -> TokenizedString {
///         tokenize_with_offset(&self.text, normalizer)
///     }
///
///     fn to_normalized(&self) -> String {
///         normalizer(&self.text)
///     }
/// }
/// let some_text = SomeText { text: String::from(" Brave  new  world" )};
/// let tokenized_result = some_text.to_tokenized_with_offset();
/// let normalized_result = some_text.to_normalized();
///
/// assert_eq!(tokenized_result.tokens(), &vec!["brave".to_string(), "new".to_string(), "world".to_string()]);
/// assert_eq!(tokenized_result.offsets(), &vec![(0, 5), (6, 9), (10, 15)]);
/// assert_eq!(normalized_result, String::from("brave new world"));
/// ```
///
pub trait ToTokenizedWithOffset {
    fn to_tokenized_with_offset(&self) -> TokenizedString;

    fn to_normalized(&self) -> String;
}