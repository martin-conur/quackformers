use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{
    BertModel, Config, HiddenAct, PositionEmbeddingType, DTYPE,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokenizers::{PaddingParams, Tokenizer};
mod jina_implementation;
use jina_implementation::{Config as JinaConfig, JinaModel};

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("IO error {0}")]
    Io(#[from] std::io::Error),

    #[error("Serde JSON error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("HF Hub error: {0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),

    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Model type error: {0}")]
    ModelTypeError(String),
}

pub struct TextEmbedder {
    model: Box<dyn EmbedModel>,
    tokenizer: Tokenizer,
}

#[derive(Clone, Debug)]
pub enum ModelType {
    Bert(Device),
    Jina(Device),
}

impl ModelType {
    fn get_model_id(&self) -> String {
        match &self {
            Self::Bert(_) => "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            Self::Jina(_) => "jinaai/jina-embeddings-v2-base-en".to_string(),
        }
    }

    fn get_jina_model(&self, vb: VarBuilder) -> Result<JinaModel, EmbeddingError> {
        match &self {
            Self::Jina(_) => {
                let config = JinaConfig::v2_base();
                Ok(JinaModel::load(vb, &config)?)
            }
            _ => Err(EmbeddingError::ModelTypeError(
                "Incorrect Model Type".into(),
            )),
        }
    }

    fn get_bert_model(&self, vb: VarBuilder) -> Result<BertModel, EmbeddingError> {
        match &self {
            Self::Bert(_) => {
                let config = Config {
                    vocab_size: 30522,
                    hidden_size: 384,
                    num_hidden_layers: 6,
                    num_attention_heads: 12,
                    intermediate_size: 1536,
                    hidden_act: HiddenAct::Gelu,
                    hidden_dropout_prob: 0.1,
                    max_position_embeddings: 512,
                    type_vocab_size: 2,
                    initializer_range: 0.02,
                    layer_norm_eps: 1e-12,
                    pad_token_id: 0,
                    position_embedding_type: PositionEmbeddingType::Absolute,
                    use_cache: true,
                    classifier_dropout: None,
                    model_type: Some("bert".to_string()),
                };
                Ok(BertModel::load(vb, &config)?)
            }
            _ => Err(EmbeddingError::ModelTypeError(
                "Incorrect Model Type".into(),
            )),
        }
    }

    fn get_local_model_path(&self) -> Option<PathBuf> {
        match &self {
            Self::Bert(_) => std::env::var("BERT_MODEL_FOLDER").ok().map(PathBuf::from),
            Self::Jina(_) => std::env::var("JINA_MODEL_FOLDER").ok().map(PathBuf::from),
        }
    }

    fn load_from_local(&self, local_path: &Path) -> Result<(PathBuf, PathBuf), EmbeddingError> {
        let tokenizer_path = local_path.join("tokenizer.json");
        let weights_path = local_path.join("model.safetensors");

        if !tokenizer_path.exists() {
            return Err(EmbeddingError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Tokenizer file not found at {:?}", tokenizer_path),
            )));
        }

        if !weights_path.exists() {
            return Err(EmbeddingError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Model weights file not found at {:?}", weights_path),
            )));
        }

        Ok((tokenizer_path, weights_path))
    }

    fn load_from_hub(&self) -> Result<(PathBuf, PathBuf), EmbeddingError> {
        let model_id = self.get_model_id();
        let repo = Repo::new(model_id, RepoType::Model);
        let api = Api::new()?;
        let api = api.repo(repo);
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;
        Ok((tokenizer, weights))
    }

    pub fn build_text_embedder(&self) -> Result<TextEmbedder, EmbeddingError> {
        let device = match &self {
            Self::Bert(device) => device,
            Self::Jina(device) => device,
        };

        // Try to load from local path first, fall back to HuggingFace Hub
        let (tokenizer_filename, weights_filename) =
            if let Some(local_path) = self.get_local_model_path() {
                self.load_from_local(&local_path)?
            } else {
                self.load_from_hub()?
            };

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)?;
        configure_batch_padding(&mut tokenizer);

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, device)? };

        let model: Box<dyn EmbedModel> = match &self {
            Self::Bert(_) => Box::new(self.get_bert_model(vb)?),
            Self::Jina(_) => Box::new(self.get_jina_model(vb)?),
        };

        Ok(TextEmbedder { model, tokenizer })
    }
}

pub trait Embed {
    fn embed(
        &self,
        column: Vec<String>,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}

pub trait EmbedModel: Send + Sync {
    fn device(&self) -> &Device;
    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, EmbeddingError>;
}

impl EmbedModel for BertModel {
    fn device(&self) -> &Device {
        &self.device
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, EmbeddingError> {
        Ok(self.forward(input_ids, token_type_ids, attention_mask)?)
    }
}

impl EmbedModel for JinaModel {
    fn device(&self) -> &Device {
        &self.device
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, EmbeddingError> {
        Ok(self.forward(input_ids, token_type_ids, attention_mask)?)
    }
}

impl Embed for TextEmbedder {
    fn embed(
        &self,
        column: Vec<String>,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let device = self.model.device();

        // chunk based approach
        let mut all_embeddings = Vec::with_capacity(column.len());

        for chunk in column.chunks(batch_size) {
            let tokens = self.tokenizer.encode_batch(chunk.to_vec(), true)?;

            let token_ids = tokens
                .iter()
                .map(|tokens| {
                    let tokens = tokens.get_ids().to_vec();
                    Ok(Tensor::new(tokens.as_slice(), device)?)
                })
                .collect::<Result<Vec<_>, EmbeddingError>>()?;

            let attention_mask = tokens
                .iter()
                .map(|tokens| {
                    let tokens = tokens.get_attention_mask().to_vec();
                    Ok(Tensor::new(tokens.as_slice(), device)?)
                })
                .collect::<Result<Vec<_>, EmbeddingError>>()?;

            let token_ids = Tensor::stack(&token_ids, 0)?;
            let attention_mask = Tensor::stack(&attention_mask, 0)?;
            let token_type_ids = token_ids.zeros_like()?;

            let embeddings =
                self.model
                    .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

            let attention_mask = attention_mask.to_dtype(candle_core::DType::F32)?;
            let masked_embeddings = embeddings.broadcast_mul(&attention_mask.unsqueeze(2)?)?;
            let sum_embeddings = masked_embeddings.sum(1)?;
            let real_token_counts = attention_mask.sum(1)?.maximum(1e-8)?;
            let mean_embeddings = sum_embeddings.broadcast_div(&real_token_counts.unsqueeze(1)?)?;
            let normalized_embeddings = normalize_l2(&mean_embeddings)?;

            let chunk_embeddings = normalized_embeddings.to_vec2()?;
            all_embeddings.extend(chunk_embeddings);
        }
        Ok(all_embeddings)
    }
}

fn configure_batch_padding(tokenizer: &mut Tokenizer) {
    if let Some(padding) = tokenizer.get_padding_mut() {
        padding.strategy = tokenizers::PaddingStrategy::BatchLongest;
    } else {
        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
    }
}

fn normalize_l2(v: &Tensor) -> Result<Tensor, EmbeddingError> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::thread;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    struct TestModel {
        device: Device,
    }

    impl EmbedModel for TestModel {
        fn device(&self) -> &Device {
            &self.device
        }

        fn forward(
            &self,
            input_ids: &Tensor,
            _token_type_ids: &Tensor,
            _attention_mask: Option<&Tensor>,
        ) -> Result<Tensor, EmbeddingError> {
            let input_ids = input_ids.to_dtype(DType::F32)?.unsqueeze(2)?;
            let squared_input_ids = input_ids.sqr()?;
            Ok(Tensor::cat(&[&input_ids, &squared_input_ids], 2)?)
        }
    }

    fn test_embedder() -> TextEmbedder {
        let vocab = HashMap::from([
            ("[UNK]".to_string(), 0),
            ("hello".to_string(), 1),
            ("world".to_string(), 2),
        ]);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Whitespace);
        configure_batch_padding(&mut tokenizer);

        TextEmbedder {
            model: Box::new(TestModel {
                device: Device::Cpu,
            }),
            tokenizer,
        }
    }

    #[test]
    fn shared_embedder_supports_concurrent_calls() {
        let embedder = Arc::new(test_embedder());
        let handles = [
            ("hello", [0.70710677, 0.70710677]),
            ("world", [0.4472136, 0.8944272]),
        ]
        .into_iter()
        .map(|(text, expected)| {
            let embedder = Arc::clone(&embedder);
            thread::spawn(move || {
                let embeddings = embedder.embed(vec![text.into()], 1).unwrap();
                (embeddings, expected)
            })
        })
        .collect::<Vec<_>>();

        for handle in handles {
            let (embeddings, expected) = handle.join().unwrap();
            assert_eq!(embeddings.len(), 1);
            assert_eq!(embeddings[0].len(), expected.len());
            assert!(embeddings[0].iter().all(|value| value.is_finite()));
            for (actual, expected) in embeddings[0].iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }
}
