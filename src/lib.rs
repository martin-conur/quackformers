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

mod embed_utils;
use embed_utils::{Embed, EmbeddingError, ModelType, TextEmbedder};

const DEVICE: Device = Device::Cpu;

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

unsafe fn generic_embed_invoke<F>(
    input: &mut DataChunkHandle,
    output: &mut dyn WritableVector,
    build_model_fn: F,
) -> std::result::Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce() -> Result<TextEmbedder, EmbeddingError>,
{
    // Extract the input word
    let input_vec = input.flat_vector(0);
    // slice of strings
    let input_slice = input_vec.as_slice_with_len::<duckdb_string_t>(input.len());

    // let output_flat_vector = output.flat_vector();
    let mut output_list_vector = output.list_vector();

    // Bert embed
    let vect_phrases = process_strings(input_slice)?;
    let mut model = build_model_fn()?;
    let embedded_phrases = model.embed(vect_phrases, 32)?;

    let total_len: usize = embedded_phrases.iter().map(|v| v.len()).sum();
    let mut child_vector = output_list_vector.child(total_len);

    let mut offset = 0;
    for (i, embedded_phrase) in embedded_phrases.iter().enumerate() {
        child_vector.as_mut_slice_with_len(offset + embedded_phrase.len())
            [offset..offset + embedded_phrase.len()]
            .copy_from_slice(embedded_phrase);

        output_list_vector.set_entry(i, offset, embedded_phrase.len());

        offset += embedded_phrase.len();
        // let embedded_phrase_string = CString::new(format!("{:?}", embedded_phrase))?;
        // output_flat_vector.insert(i, embedded_phrase_string);
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
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Extract the input word
        generic_embed_invoke(input, output, || {
            ModelType::Bert(DEVICE).build_text_embedder()
        })
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
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Extract the input word
        generic_embed_invoke(input, output, || {
            ModelType::Jina(DEVICE).build_text_embedder()
        })
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeId::Varchar.into()],
            LogicalTypeHandle::list(&LogicalTypeId::Float.into()),
        )]
    }
}

const BERT_FUNCTION_NAME: &str = "embed";
const JINA_FUNCTION_NAME: &str = "embed_jina";

#[duckdb_entrypoint_c_api]
/// # Safety
/// This function must only be called by DuckDB's extension loader system.
/// The `Connection` must be a valid and open DuckDB connection provided by DuckDB.
/// Caller must guarantee that DuckDB is properly initialized and not in an error state.
pub unsafe fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    con.register_scalar_function::<EmbedFunc>(BERT_FUNCTION_NAME)
        .expect("Failed to register embed() function");
    con.register_scalar_function::<EmbedJinaFunc>(JINA_FUNCTION_NAME)
        .expect("Failed to register embed_jina() function");
    Ok(())
}
