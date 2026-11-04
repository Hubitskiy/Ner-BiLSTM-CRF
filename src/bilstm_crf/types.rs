use burn::{
    backend::{ExecutionError},
    train::{metric::{Adaptor, LossInput, ItemLazy}},
    Tensor
};
use serde::{Serialize, Deserialize};
use neuro::tokenizers::tokenizer::{
    ToTokenizedWithOffset, 
    TokenizedString,
    tokenize_with_offset,
    normalizer
};


/// Annotated entity span of a [`NerExampleSample`].
#[derive(Debug, Serialize, Deserialize)]
pub struct NerEntity {
    /// Byte offset where the entity starts in the normalized text.
    start: usize,
    /// Byte offset right after the entity's last byte in the normalized text.
    end: usize,
    /// Text of the entity.
    value: String,
    /// Name of the entity type, without a BIO prefix.
    entity: String,
}


impl NerEntity {
    /// Creates an entity span.
    ///
    /// # Arguments
    ///
    /// * `start` - byte offset where the entity starts in the normalized text.
    /// * `end` - byte offset right after the entity's last byte in the normalized text.
    /// * `value` - text of the entity.
    /// * `entity` - name of the entity type, without a BIO prefix.
    pub fn define(start: usize, end: usize, value: String, entity: String) -> Self {
        Self { start, end, value, entity }
    }

    /// Returns the byte offset where the entity starts in the normalized text.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Returns the byte offset right after the entity's last byte in the normalized text.
    pub fn end(&self) -> usize {
        self.end
    }

    /// Returns the text of the entity.
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Returns the name of the entity type, without a BIO prefix.
    pub fn entity(&self) -> &str {
        &self.entity
    }
}


/// Raw NER sample: a text with its annotated entities.
#[derive(Debug, Serialize, Deserialize)]
pub struct NerExampleSample {
    /// Source text, not normalized.
    text: String,
    /// Entities annotated in the text.
    entities: Vec<NerEntity>
}


impl NerExampleSample {
    /// Returns the entities annotated in the text.
    pub fn entities(&self) -> &Vec<NerEntity> {
        &self.entities
    }
}


impl ToTokenizedWithOffset for NerExampleSample {
    
    /// Normalizes the text and splits it into tokens with byte offsets.
    #[inline]
    fn to_tokenized_with_offset(&self) -> TokenizedString {
        tokenize_with_offset(&self.text, normalizer)
    }
    
    /// Returns the normalized text.
    #[inline]
    fn to_normalized(&self) -> String {
        normalizer(&self.text)
    }
}


pub struct NerOutput {
    pub loss: Tensor<1>
}


impl NerOutput {
    
    pub fn define(loss: Tensor<1>) -> Self {
        Self { loss }
    }
    
}


impl ItemLazy for NerOutput {

    fn sync(self) -> Result<Self, ExecutionError> {
        self.loss.device().flush()?;

        Ok(NerOutput { loss: self.loss.without_autodiff() })
    }

}


impl Adaptor<LossInput> for NerOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.loss.clone())
    }
}