#[test]
fn test_tokenizer() {
    use crate::tokenizers::tokenizer::{tokenize_with_offset, normalizer};

    let to_tokenize = String::from("Brave new world");

    let result = tokenize_with_offset(&to_tokenize, normalizer);

    assert_eq!(result.tokens(), &vec!["brave".to_string(), "new".to_string(), "world".to_string()]);
    assert_eq!(result.offsets(), &vec![(0, 5), (6, 9), (10, 15)]);

}

#[test]
fn test_tokenize_empty_string() {
    use crate::tokenizers::tokenizer::{tokenize_with_offset, normalizer};

    let to_tokenize = String::from("");

    let result = tokenize_with_offset(&to_tokenize, normalizer);
    let tokens_example: Vec<String> = vec![];
    let offsets_example: Vec<(usize, usize)> = vec![];
    assert_eq!(result.tokens(), &tokens_example);
    assert_eq!(result.offsets(), &offsets_example);
}

#[test]
fn test_tokenize_string_with_punctuation_marks() {
    use crate::tokenizers::tokenizer::{tokenize_with_offset, normalizer};

    let to_tokenize = String::from("Brave, new world?");

    let result = tokenize_with_offset(&to_tokenize, normalizer);

    assert_eq!(result.tokens(), &vec!["brave".to_string(), ",".to_string(), "new".to_string(), "world".to_string(), "?".to_string()]);
    assert_eq!(result.offsets(), &vec![(0, 5), (5, 6), (7, 10), (11, 16), (16, 17)]);

}


#[test]
fn test_tokenize_string_with_emoji() {
    use crate::tokenizers::tokenizer::{tokenize_with_offset, normalizer};

    let to_tokenize = String::from("Brave, new world 😊");

    let result = tokenize_with_offset(&to_tokenize, normalizer);
    println!("{:?}", result);
    assert_eq!(result.tokens(), &vec![
        "brave".to_string(),
        ",".to_string(),
        "new".to_string(),
        "world".to_string(),
        "😊".to_string()
    ]);
    assert_eq!(result.offsets(), &vec![(0, 5), (5, 6), (7, 10), (11, 16), (17, 21)]);

}


#[test]
fn test_already_normalized_string() {
    use crate::tokenizers::tokenizer::normalizer;

    let to_normalize = String::from("Brave new world");

    let result = normalizer(&to_normalize);

    assert_eq!(result, String::from("brave new world"));
}
