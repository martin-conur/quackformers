use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{
    BertModel, Config, HiddenAct, PositionEmbeddingType, DTYPE,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use thiserror::Error;
use tokenizers::{PaddingParams, Tokenizer};
mod jina_implementation;
use jina_implementation::{Config as JinaConfig, JinaModel};

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("IO error {0}")]
    Io(#[from] std::io::Error),

    #[error("Serde JSON error: {0}")]
    Sede(#[from] serde_json::Error),

    #[error("HF Hub error: {0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),

    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

pub struct TextEmbedder<M> {
    model: M,
    tokenizer: Tokenizer,
}

// first I'm gonna do separate implementations, just to have a fast iteration
// but later I'm gonna implment generic Model abstraction so we could have
// different models and not repeat myselft too much (DRY)
pub fn build_model_and_tokenizer_jina() -> Result<TextEmbedder<JinaModel>, EmbeddingError> {
    let device = Device::Cpu;
    let model_id = "jinaai/jina-embeddings-v2-base-en".to_string();
    let repo = Repo::new(model_id, RepoType::Model);
    let (tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;
        (tokenizer, weights)
    };
    let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

    let config = JinaConfig::v2_base();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
    let model = JinaModel::load(vb, &config)?;
    Ok(TextEmbedder { model, tokenizer })
}

pub fn build_model_and_tokenizer(
    approximate_gelu: bool,
) -> Result<TextEmbedder<BertModel>, EmbeddingError> {
    let device = Device::Cpu;
    let model_id = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    let detault_revision = "refs/pr/21".to_string();

    let repo = Repo::with_revision(model_id, RepoType::Model, detault_revision);
    let (tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;
        (tokenizer, weights)
    };
    let mut config = Config {
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
    let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
    if approximate_gelu {
        config.hidden_act = HiddenAct::GeluApproximate;
    }
    let model = BertModel::load(vb, &config)?;
    Ok(TextEmbedder { model, tokenizer })
}

pub trait Embed {
    fn embed(
        &mut self,
        column: Vec<String>,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}

pub trait EmbedModel {
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

impl<M: EmbedModel> Embed for TextEmbedder<M> {
    fn embed(
        &mut self,
        column: Vec<String>,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let device = self.model.device();

        // padding
        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }

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

fn normalize_l2(v: &Tensor) -> Result<Tensor, EmbeddingError> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
