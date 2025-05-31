#![allow(dead_code)]
#![allow(unused_imports)]
extern crate duckdb;
extern crate duckdb_loadable_macros;
extern crate libduckdb_sys;

use candle_core::Device;
use duckdb::ffi;
use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vscalar::{ScalarFunctionSignature, VScalar},
    vtab::arrow::WritableVector,
    Connection, Result,
};
use duckdb_loadable_macros::duckdb_entrypoint_c_api;
use libduckdb_sys::{duckdb_string_t, duckdb_string_t_data, duckdb_string_t_length};
use std::error::Error;
use std::slice;
use once_cell::sync::{Lazy, OnceCell};
use futures::future::join_all;
use std::sync::Mutex;
mod embed_utils;
mod bedrock_utils;
use embed_utils::{Embed, EmbeddingError, ModelType, TextEmbedder};
use bedrock_utils::{embedrock, bedrock_invoke, BedrockConverseError, TitanResponse};
use tokio::runtime::Runtime;
use tracing_subscriber::FmtSubscriber;

static TRACING: OnceCell<()> = OnceCell::new();

static TOKIO: Lazy<Runtime> = Lazy::new(|| Runtime::new().unwrap());

const DEVICE: Device = Device::Cpu;
const BERT_FUNCTION_NAME: &str = "embed";
const JINA_FUNCTION_NAME: &str = "embed_jina";
const BEDROCK_EMBEDDING_NAME: &str = "embedrock";
const BEDROCK_INVOKE_NAME: &str = "bedrock_invoke";

fn init_tracing() {
    TRACING.get_or_init(|| {
        let subscriber = FmtSubscriber::builder().with_max_level(tracing::Level::INFO).finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

fn duckdb_string_to_owned_string(word: &duckdb_string_t) -> String {
    unsafe {
        let len = duckdb_string_t_length(*word);
        let c_ptr = duckdb_string_t_data(word as *const _ as *mut _);
        let bytes = slice::from_raw_parts(c_ptr as *const u8, len as usize);
        String::from_utf8_lossy(bytes).into_owned()
    }
}

fn process_strings(input_slice: &[duckdb_string_t]) -> Result<Vec<String>, EmbeddingError> {
    input_slice
        .iter()
        .map(|word| Ok(duckdb_string_to_owned_string(word)))
        .collect::<Result<Vec<String>, EmbeddingError>>()
}

/// Load & JIT once on first use:
static BERT_EMBEDDER: Lazy<Mutex<TextEmbedder>> = Lazy::new(|| {
    let mut embedder = ModelType::Bert(DEVICE)
        .build_text_embedder()
        .expect("failed to load BERT embedder");
    // Warm up: do one dummy forward to JIT kernels
    let dummy = ["hello world".to_string()].to_vec();
    let _ = embedder.embed(dummy, /*batch_size=*/1);
    Mutex::new(embedder)
});

static JINA_EMBEDDER: Lazy<Mutex<TextEmbedder>> = Lazy::new(|| {
    let mut embedder = ModelType::Jina(DEVICE)
        .build_text_embedder()
        .expect("failed to load Jina embedder");
    let dummy = ["hello world".to_string()].to_vec();
    let _ = embedder.embed(dummy, /*batch_size=*/1);
    Mutex::new(embedder)
});

// Local embedding function (uses vanilla Bert or JINA)
unsafe fn generic_embed_invoke(
    input: &mut DataChunkHandle,
    output: &mut dyn WritableVector,
    use_jina: bool,
) -> Result<(), Box<dyn Error>> {
    let input_vec = input.flat_vector(0);
    // slice of strings
    let input_slice = input_vec.as_slice_with_len::<duckdb_string_t>(input.len());

    let mut output_list_vector = output.list_vector();

    // Bert embed
    let vect_phrases = process_strings(input_slice)?;
    // choose the already-loaded embedder
    let mut guard = if use_jina {
        JINA_EMBEDDER.lock().unwrap()
    } else {
        BERT_EMBEDDER.lock().unwrap()
    };

    let embedded_phrases = guard.embed(vect_phrases, /*batch_size=*/32)?;
    let total_len: usize = embedded_phrases.iter().map(|v| v.len()).sum();
    let mut child_vector = output_list_vector.child(total_len);

    let mut offset = 0;
    for (i, embedded_phrase) in embedded_phrases.iter().enumerate() {
        child_vector.as_mut_slice_with_len(offset + embedded_phrase.len())
            [offset..offset + embedded_phrase.len()]
            .copy_from_slice(embedded_phrase);

        output_list_vector.set_entry(i, offset, embedded_phrase.len());

        offset += embedded_phrase.len();
    }
    output_list_vector.set_len(embedded_phrases.len());

    Ok(())
}

struct EmbedFunc;

impl VScalar for EmbedFunc {
    type State = ();

    /// # Safety
    /// This function is called by DuckDB when executing the UDF (user-defined function).
    /// - `input` must be a valid and initialized DataChunkHandle.
    /// - `output` must be a valid and writable WritableVector.
    /// - Caller (DuckDB) must guarantee input and output are valid for the duration of the call.
    unsafe fn invoke(
        _state: &(),
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        generic_embed_invoke(input, output, /*use_jina=*/ false)
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeId::Varchar.into()],
            LogicalTypeHandle::list(&LogicalTypeId::Float.into()),
        )]
    }
}

struct EmbedJinaFunc;

impl VScalar for EmbedJinaFunc {
    type State = ();

    /// # Safety
    /// This function is called by DuckDB when executing the UDF (user-defined function).
    /// - `input` must be a valid and initialized DataChunkHandle.
    /// - `output` must be a valid and writable WritableVector.
    /// - Caller (DuckDB) must guarantee input and output are valid for the duration of the call.
    unsafe fn invoke(
        _state: &(),
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        generic_embed_invoke(input, output, /*use_jina=*/ true)
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeId::Varchar.into()],
            LogicalTypeHandle::list(&LogicalTypeId::Float.into()),
        )]
    }
}


// Bedrock functionalities
async unsafe fn embedrock_invoke(
    input: &mut DataChunkHandle,
    output: &mut dyn WritableVector,
) -> Result<(), Box<dyn Error>> {
    let input_vec = input.flat_vector(0);
    // formatting and processing input
    // slice of strings
    let input_slice = input_vec.as_slice_with_len::<duckdb_string_t>(input.len());
    let mut output_list_vector = output.list_vector();
    // duckdb type to rust type
    let vect_phrases = process_strings(input_slice)?;
   // embedd fn calling, it should return a Vec<Vec<f32>>. TODO: implement a faster/lighter dtype
   // this is the only line you should change for different implementations. 
   // TODO: create a generic func that accepts the emebedder as a parameter.
    let embedding_futures = vect_phrases
        .iter()
        .map(|phrase| async {
            let bedrock_response = embedrock(&phrase.as_str());
            let embedding: TitanResponse = bedrock_response.await.unwrap();
            embedding.embedding
        });

    let embedded_phrases: Vec<Vec<f32>> = join_all(embedding_futures).await;

    // from rust type back to duckdb type
    let total_len: usize = embedded_phrases.iter().map(|v| v.len()).sum();
    let mut child_vector = output_list_vector.child(total_len);
    let mut offset = 0;
    for (i, embedded_phrase) in embedded_phrases.iter().enumerate() {
        child_vector.as_mut_slice_with_len(offset + embedded_phrase.len())
            [offset..offset + embedded_phrase.len()]
            .copy_from_slice(embedded_phrase);

        output_list_vector.set_entry(i, offset, embedded_phrase.len());

        offset += embedded_phrase.len();
    }
    output_list_vector.set_len(embedded_phrases.len());

    Ok(())
}

struct EmbedRockFunc;

impl VScalar for EmbedRockFunc {
    type State = ();

    /// # Safety
    /// This function is called by DuckDB when executing the UDF (user-defined function).
    /// - `input` must be a valid and initialized DataChunkHandle.
    /// - `output` must be a valid and writable WritableVector.
    /// - Caller (DuckDB) must guarantee input and output are valid for the duration of the call.
    unsafe fn invoke(
        _state: &(),
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        TOKIO.block_on(embedrock_invoke(input, output))
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeId::Varchar.into()],
            LogicalTypeHandle::list(&LogicalTypeId::Float.into()),
        )]
    }
}

#[duckdb_entrypoint_c_api]
/// # Safety
/// This function must only be called by DuckDB's extension loader system.
/// The `Connection` must be a valid and open DuckDB connection provided by DuckDB.
/// Caller must guarantee that DuckDB is properly initialized and not in an error state.
pub unsafe fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    init_tracing();
    // Force the model + tokenizer to load & JIT right now,
    // so the *very first* SQL call is fast.
    Lazy::force(&BERT_EMBEDDER);
    Lazy::force(&JINA_EMBEDDER);
    con.register_scalar_function::<EmbedFunc>(BERT_FUNCTION_NAME)
        .expect("Failed to register embed() function");
    con.register_scalar_function::<EmbedJinaFunc>(JINA_FUNCTION_NAME)
        .expect("Failed to register embed_jina() function");
    con.register_scalar_function::<EmbedRockFunc>(BEDROCK_EMBEDDING_NAME)
        .expect("Failed to register embed_jina() function");
    Ok(())
}
