use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::{
    Client,
    operation::converse::{ConverseError, ConverseOutput},
    primitives::Blob,
    types::{ContentBlock, ConversationRole, Message},
};
use serde_json::json;
// TODO: add lazy call
// use once_cell::sync::Lazy;
// use tokio::runtime::Runtime;

pub enum FundationModel {
    AmazonTitanTextExpressV1,
}

pub enum EmbeddingModel {
    AmazonTitanEmbedTextV1,
}

const EMBEDDING_MODEL_ID: &str = "amazon.titan-embed-text-v1";
const REGION: &str = "us-east-1";

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TitanResponse {
    pub embedding: Vec<f32>,
    input_text_token_count: i128,
}

// TODO: I would change this to thiserror Error
#[derive(Debug)]
pub struct BedrockConverseError(String);
impl std::fmt::Display for BedrockConverseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Can't invoke model. Reason: {}", self.0)
    }
}
impl std::error::Error for BedrockConverseError {}
impl From<&str> for BedrockConverseError {
    fn from(value: &str) -> Self {
        BedrockConverseError(value.to_string())
    }
}
impl From<&ConverseError> for BedrockConverseError {
    fn from(value: &ConverseError) -> Self {
        BedrockConverseError::from(match value {
            ConverseError::ModelTimeoutException(_) => "Model took too long",
            ConverseError::ModelNotReadyException(_) => "Model is not ready",
            _ => "Unknown",
        })
    }
}

fn get_converse_output_text(output: ConverseOutput) -> Result<String, BedrockConverseError> {
    let text = output
        .output()
        .ok_or("no output")?
        .as_message()
        .map_err(|_| "output not a message")?
        .content()
        .first()
        .ok_or("no content in message")?
        .as_text()
        .map_err(|_| "content is not text")?
        .to_string();
    Ok(text)
}

pub async fn embedrock(message: &str) -> Result<TitanResponse, BedrockConverseError> {
    let sdk_config = aws_config::defaults(BehaviorVersion::latest())
        .region(REGION)
        .load()
        .await;
    let client = Client::new(&sdk_config);
    let embeddings_prompt = json!({
        "inputText": message
    }).to_string();
    let embedding_response = client
        .invoke_model()
        .model_id(EMBEDDING_MODEL_ID)
        .body(Blob::new(embeddings_prompt.as_bytes().to_vec()))
        .send()
        .await
        .unwrap();

    let titan_response =
        serde_json::from_slice::<TitanResponse>(&embedding_response.body().clone().into_inner())
            .unwrap();
    Ok(titan_response)
}

pub async fn bedrock_invoke(
    prompt: &str,
    model: FundationModel,
) -> Result<String, BedrockConverseError> {

    let model_id = match model {
        FundationModel::AmazonTitanTextExpressV1 => "amazon.titan-text-express-v1",
    };

    let sdk_config = aws_config::defaults(BehaviorVersion::latest())
        .region(REGION)
        .load()
        .await;
    let client = Client::new(&sdk_config);
    let response = client
        .converse()
        .model_id(model_id)
        .messages(
            Message::builder()
                .role(ConversationRole::User)
                .content(ContentBlock::Text(prompt.to_string()))
                .build()
                .map_err(|_| "failed to build message")?,
        )
        .send()
        .await;
    match response {
        Ok(output) => {
            let text = get_converse_output_text(output)?;
            Ok(text)
        }
        Err(e) => Err(e
            .as_service_error()
            .map(BedrockConverseError::from)
            .unwrap_or_else(|| BedrockConverseError("Unknown service error".into()))),
    }
}