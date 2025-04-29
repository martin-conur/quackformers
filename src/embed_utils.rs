use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use thiserror::Error; // replace this to thiserror or custom error
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

#[derive(Error, Debug)]
pub enum EmbedError {
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

fn build_model_and_tokenizer(model_id: Option<String>, approximate_gelu:bool) ->Result<(BertModel, Tokenizer), EmbedError> {
    let device = Device::Cpu;
    let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    let detault_revision = "refs/pr/21".to_string();
    
    let model_id = model_id.unwrap_or(default_model);

    let repo = Repo::with_revision(model_id, RepoType::Model, detault_revision);
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;
        (config, tokenizer, weights)
    };
    let config = std::fs::read_to_string(config_filename)?;
    let mut config: Config = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
    if approximate_gelu {
        config.hidden_act = HiddenAct::GeluApproximate;
    }
    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer))
}

pub fn embed(column: Vec<String>, batch_size: usize) -> Result<Vec<Vec<f32>>, EmbedError> {

    let (model, mut tokenizer) = build_model_and_tokenizer(Some("sentence-transformers/all-MiniLM-L6-v2".to_string()),false)?;
    let device = &model.device;

    // padding 
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams{
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    // chunk based approach
    let mut all_embeddings = Vec::with_capacity(column.len());

    for chunk in column.chunks(batch_size) {
        let tokens = tokenizer.encode_batch(chunk.to_vec(), true)?;

        let token_ids = tokens
            .iter()
            .map(|tokens|{
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), device)?)
            })
            .collect::<Result<Vec<_>, EmbedError>>()?;

        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Ok(Tensor::new(tokens.as_slice(), device)?)
            })
            .collect::<Result<Vec<_>, EmbedError>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

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

fn normalize_l2(v: &Tensor) -> Result<Tensor, EmbedError> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}