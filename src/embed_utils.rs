use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use anyhow::{Error as E, Result}; // replace this to thiserror or custom error
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

fn build_model_and_tokenizer(model_id: Option<String>, approximate_gelu:bool) ->Result<(BertModel, Tokenizer)> {
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
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
    if approximate_gelu {
        config.hidden_act = HiddenAct::GeluApproximate;
    }
    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer))
}

pub fn embed(prompt: String) -> Result<Vec<f32>> {

    let (model, mut tokenizer) = build_model_and_tokenizer(Some("sentence-transformers/all-MiniLM-L6-v2".to_string()),false)?;
    let device = &model.device;

    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;

    let ys = &model.forward(&token_ids, &token_type_ids, None)?;

    let mean = ys.mean(1)?;

    let mean_normalized = normalize_l2(&mean)?;

    let output_vec = mean_normalized.to_vec2::<f32>()?[0].clone();

    Ok(output_vec)
}

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}