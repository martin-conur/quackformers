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
use once_cell::sync::Lazy;
use std::error::Error;
use std::slice;
use std::sync::{Condvar, Mutex};
mod embed_utils;
use embed_utils::{Embed, ModelType, TextEmbedder};

const DEVICE: Device = Device::Cpu;
const EMBEDDING_BATCH_SIZE: usize = 32;
// Keep forward-pass memory bounded even on hosts with many logical CPUs.
const MAX_EMBEDDING_CONCURRENCY: usize = 4;

struct Semaphore {
    permits: Mutex<usize>,
    available: Condvar,
}

struct SemaphorePermit<'a> {
    semaphore: &'a Semaphore,
}

impl Semaphore {
    fn new(permits: usize) -> Self {
        assert!(permits > 0, "a semaphore needs at least one permit");
        Self {
            permits: Mutex::new(permits),
            available: Condvar::new(),
        }
    }

    fn acquire(&self) -> SemaphorePermit<'_> {
        let mut permits = self
            .permits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while *permits == 0 {
            permits = self
                .available
                .wait(permits)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        *permits -= 1;
        SemaphorePermit { semaphore: self }
    }
}

impl Drop for SemaphorePermit<'_> {
    fn drop(&mut self) {
        let mut permits = self
            .semaphore
            .permits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *permits += 1;
        self.semaphore.available.notify_one();
    }
}

fn embedding_concurrency_limit() -> usize {
    std::thread::available_parallelism()
        .map(|cores| (cores.get() / 2).max(1).min(MAX_EMBEDDING_CONCURRENCY))
        .unwrap_or(1)
}

static EMBEDDING_SEMAPHORE: Lazy<Semaphore> =
    Lazy::new(|| Semaphore::new(embedding_concurrency_limit()));

fn duckdb_string_to_owned_string(word: &duckdb_string_t) -> String {
    unsafe {
        let len = duckdb_string_t_length(*word);
        let c_ptr = duckdb_string_t_data(word as *const _ as *mut _);
        let bytes = slice::from_raw_parts(c_ptr as *const u8, len as usize);
        String::from_utf8_lossy(bytes).into_owned()
    }
}

/// Load & JIT once on first use:
static BERT_EMBEDDER: Lazy<TextEmbedder> = Lazy::new(|| {
    let embedder = ModelType::Bert(DEVICE)
        .build_text_embedder()
        .expect("failed to load BERT embedder");
    // Warm up: do one dummy forward to JIT kernels
    let dummy = ["hello world".to_string()].to_vec();
    let _ = embedder.embed(dummy, /*batch_size=*/ 1);
    embedder
});

static JINA_EMBEDDER: Lazy<TextEmbedder> = Lazy::new(|| {
    let embedder = ModelType::Jina(DEVICE)
        .build_text_embedder()
        .expect("failed to load Jina embedder");
    let dummy = ["hello world".to_string()].to_vec();
    let _ = embedder.embed(dummy, /*batch_size=*/ 1);
    embedder
});

unsafe fn generic_embed_invoke(
    input: &mut DataChunkHandle,
    output: &mut dyn WritableVector,
    use_jina: bool,
) -> Result<(), Box<dyn Error>> {
    let input_vec = input.flat_vector(0);
    // slice of strings
    let input_slice = input_vec.as_slice_with_len::<duckdb_string_t>(input.len());

    // let output_flat_vector = output.flat_vector();
    let mut output_list_vector = output.list_vector();

    // here we'll track the not-null rows
    let mut texts: Vec<String> = Vec::with_capacity(input.len());
    let mut rows: Vec<usize> = Vec::with_capacity(input.len());

    for row in 0..input.len() {
        if input_vec.try_row_is_null(row as u64)? {
            continue;
        }
        texts.push(duckdb_string_to_owned_string(&input_slice[row]));
        rows.push(row);
    }
    // choose the already-loaded embedder
    let embedder = if use_jina {
        &*JINA_EMBEDDER
    } else {
        &*BERT_EMBEDDER
    };
    let _permit = EMBEDDING_SEMAPHORE.acquire();
    let embedded_phrases = embedder.embed(texts, EMBEDDING_BATCH_SIZE)?;
    let total_len: usize = embedded_phrases.iter().map(|v| v.len()).sum();
    let mut child_vector = output_list_vector.child(total_len);

    // put the not null rows in the output vector with the right entry index (what we tracked in texts and rows)
    let mut offset = 0;
    for (i, embedded_phrase) in embedded_phrases.iter().enumerate() {
        let row = rows[i];
        child_vector.as_mut_slice_with_len(offset + embedded_phrase.len())
            [offset..offset + embedded_phrase.len()]
            .copy_from_slice(embedded_phrase);

        output_list_vector.set_entry(row, offset, embedded_phrase.len());

        offset += embedded_phrase.len();
    }

    // a seconds for loop that will set to null entries with null rows
    for row in 0..input.len() {
        if input_vec.try_row_is_null(row as u64)? {
            output_list_vector.set_entry(row, 0, 0);
            output_list_vector.set_null(row);
        }
    }

    output_list_vector.set_len(input.len());

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
    fn invoke(
        _state: &(),
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            generic_embed_invoke(input, output, /*use_jina=*/ false)
        }
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
    fn invoke(
        _state: &(),
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            generic_embed_invoke(input, output, /*use_jina=*/ true)
        }
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
    // Force the model + tokenizer to load & JIT right now,
    // so the *very first* SQL call is fast.
    Lazy::force(&BERT_EMBEDDER);
    Lazy::force(&JINA_EMBEDDER);
    con.register_scalar_function::<EmbedFunc>(BERT_FUNCTION_NAME)
        .expect("Failed to register embed() function");
    con.register_scalar_function::<EmbedJinaFunc>(JINA_FUNCTION_NAME)
        .expect("Failed to register embed_jina() function");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    };
    use std::thread;
    use std::time::Duration;

    #[test]
    fn embedding_concurrency_limit_is_bounded() {
        let limit = embedding_concurrency_limit();

        assert!((1..=MAX_EMBEDDING_CONCURRENCY).contains(&limit));
    }

    #[test]
    fn semaphore_limits_active_permits() {
        let semaphore = Arc::new(Semaphore::new(2));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(5));

        let handles = (0..4)
            .map(|_| {
                let semaphore = Arc::clone(&semaphore);
                let active = Arc::clone(&active);
                let peak = Arc::clone(&peak);
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    start.wait();
                    let _permit = semaphore.acquire();
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(current, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(10));
                    active.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect::<Vec<_>>();

        start.wait();
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(peak.load(Ordering::SeqCst), 2);
    }
}
