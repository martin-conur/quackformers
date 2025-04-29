#![allow(dead_code, unused_imports)]
use candle_core::{DType, Device, IndexOp, Result, Tensor, Var, D};
use candle_nn::{layer_norm, LayerNorm, Module, VarBuilder, linear, Linear, Embedding, linear_no_bias, embedding};
use candle_nn::ops::softmax_last_dim;
use candle_transformers::models::stable_diffusion::attention;
use serde::de::value;
use serde::Deserialize;

use crate::embed_utils::EmbedError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Alibi;

// Create a config struct with its impl based on https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/main/config.json
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    pub position_embedding_type: Alibi,
}

impl Config {
    pub fn v2_base() -> Self {
        Self {
            vocab_size: 30528,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: candle_nn::Activation::Gelu,
            max_position_embeddings: 8192,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Alibi,
        }
    }

    // If we want a different config, not sure if we're gonna use this 
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        hidden_act: candle_nn::Activation,
        max_position_embeddings: usize,
        type_vocab_size: usize,
        initializer_range: f64,
        layer_norm_eps: f64,
        pad_token_id: usize,
        position_embedding_type: Alibi,
    ) -> Self {
        Self {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            max_position_embeddings,
            type_vocab_size,
            initializer_range,
            layer_norm_eps,
            pad_token_id,
            position_embedding_type
        }
    }
}


// Based on candle implementation of BertEmbedding but without 
// position embeddings, one of the main differences with Jina 
#[derive(Clone, Debug)]
struct BertEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(config.vocab_size, config.vocab_size, vb.pp("word_embeddings"))?;
        let token_type_embeddings = embedding(config.type_vocab_size, config.hidden_size, vb.pp("token_type_embeddings"))?;
        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self{
            word_embeddings,
            token_type_embeddings,
            layer_norm
        })
    }
}



impl Module for BertEmbeddings {
    fn forward(&self, input_ids: &Tensor) ->Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = Tensor::zeros(seq_len, DType::U32, input_ids.device())?
            .broadcast_left(b_size)?;
        let embeddings = (&input_embeddings + token_type_embeddings)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        Ok(embeddings)
    }
}


// Architecture
#[derive(Clone, Debug)]
struct BertSelfAttention {
    query: Linear, 
    key: Linear,
    value: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl BertSelfAttention {
    fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let hidden_size = config.hidden_size;

        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;

        Ok(Self { 
            query, 
            key, 
            value, 
            num_attention_heads: config.num_attention_heads, 
            attention_head_size 
        })
    }

    fn transpose_for_score(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x_shape = xs.dims().to_vec();
        x_shape.pop();
        x_shape.push(self.num_attention_heads);
        x_shape.push(self.attention_head_size);
        xs.reshape(x_shape)?.transpose(1, 2)?.contiguous()
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let query_layer = self.query.forward(xs)?;
        let value_layer = self.value.forward(xs)?;
        let key_layer = self.key.forward(xs)?;

        let query_layer = self.transpose_for_score(&query_layer)?;
        let value_layer = self.transpose_for_score(&value_layer)?;
        let key_layer = self.transpose_for_score(&key_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores = attention_scores.broadcast_add(bias)?;
        let attention_probs = softmax_last_dim(&attention_scores)?;
        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(D::Minus2)?;
        
        Ok(context_layer)
    }
}

#[derive(Clone, Debug)]
struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertSelfOutput {
    fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self {
            dense,
            layer_norm
        })
    }

    fn forward(&self, xs: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let xs = self.dense.forward(xs)?;
        self.layer_norm.forward(&(xs + input_tensor)?)
    }
}

#[derive(Clone, Debug)]
struct BertAttention{
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
}

impl BertAttention {
    fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let self_attention = BertSelfAttention::new(vb.pp("self"), config)?;
        let self_output = BertSelfOutput::new(vb.pp("output"), config)?;
        Ok(Self {
            self_attention, 
            self_output
        })
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward(xs, bias)?;
        let attention_output = self.self_output.forward(&self_outputs, xs)?;
        Ok(attention_output)
    }
}

// GLUMP, this is particular to Jina architectures
#[derive(Clone, Debug)]
struct BertGLUMLP {
    gated_layers: Linear,
    act: candle_nn::Activation,
    wo: Linear,
    layernorm: LayerNorm,
    intermediate_size: usize,
}

impl BertGLUMLP {
    fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let gated_layers = linear_no_bias(config.hidden_size, config.intermediate_size * 2, vb.pp("gated_layers"))?;
        let act = candle_nn::Activation::Gelu;
        let wo = linear(config.intermediate_size, config.hidden_size, vb.pp("wo"))?;
        let layernorm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("layernorm"))?;
        Ok(Self {
            gated_layers,
            act,
            wo,
            layernorm,
            intermediate_size: config.intermediate_size
        })
    }
}

impl Module for BertGLUMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.gated_layers)?;
        let gated = xs.narrow(D::Minus1, 0, self.intermediate_size)?;
        let non_gated = xs.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let xs_res = (gated.apply(&self.act) * non_gated)?.apply(&self.wo);
        (xs_res + residual)?.apply(&self.layernorm)
    }
}

#[derive(Clone, Debug)]
struct BertLayer {
    attention: BertAttention,
    mlp: BertGLUMLP,
}

impl BertLayer {
    fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = BertAttention::new(vb.pp("attention"), config)?;
        let mlp = BertGLUMLP::new(vb.pp("mlp"), config)?;
        Ok(Self {
            attention,
            mlp
        })
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor) ->Result<Tensor> {
        self.attention.forward(xs, bias)?.apply(&self.mlp)
    }
}

fn build_alibi_bias(config: &Config) -> Result<Tensor> {
    let n_heads = config.num_attention_heads;
    let seq_len = config.max_position_embeddings;
    let alibi_bias = Tensor::arange(0, seq_len as i64, &Device::Cpu)?.to_dtype(DType::F32)?;
    let alibi_bias = {
        let a1 = alibi_bias.reshape((1, seq_len))?;
        let a2 = alibi_bias.reshape((seq_len, 1))?;
        a1.broadcast_sub(&a2)?.abs()?.broadcast_left(n_heads)?
    };

    let mut n_heads2 = 1;
    while n_heads2 < n_heads {
        n_heads2 *= 2
    }
    let slopes = (1..=n_heads2)
        .map(|v| -1f32 / 2f32.powf((v * 8) as f32 / n_heads2 as f32))
        .collect::<Vec<_>>();
    let slopes = if n_heads2 == n_heads {
        slopes
    } else {
        slopes
            .iter()
            .skip(1)
            .step_by(2)
            .chain(slopes.iter().step_by(2))
            .take(n_heads)
            .cloned()
            .collect::<Vec<f32>>()
    };
    let slopes = Tensor::new(slopes, &Device::Cpu)?.reshape((1, (), 1, 1))?;
    alibi_bias.to_dtype(DType::F32)?.broadcast_mul(&slopes)
}