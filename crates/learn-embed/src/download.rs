//! Model download helpers.
//!
//! `ensure_default_model` locates or downloads the BGE-large-en-v1.5 files
//! into `~/.cache/learn-rs/models/bge-large-en-v15/`.
//! Files are written to a temp path first, then atomically renamed.

use camino::Utf8PathBuf;
use learn_core::{LearnError, Result};
use std::io::Write as _;
use std::path::PathBuf;
use tracing::info;

const HF_BASE: &str = "https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main";
const MODEL_FILE: &str = "onnx/model.onnx";
const TOKENIZER_FILE: &str = "tokenizer.json";
const CACHE_SUBDIR: &str = "learn-rs/models/bge-large-en-v15";

/// Return the directory containing `model.onnx` and `tokenizer.json`,
/// downloading them first if absent.
pub fn ensure_default_model() -> Result<Utf8PathBuf> {
    let cache_dir = resolve_cache_dir()?;
    std::fs::create_dir_all(&cache_dir)
        .map_err(|e| LearnError::Embed(format!("create cache dir: {e}")))?;

    let dir = Utf8PathBuf::try_from(cache_dir)
        .map_err(|e| LearnError::Embed(format!("non-UTF-8 cache path: {e}")))?;

    maybe_download(&dir, "model.onnx", &format!("{HF_BASE}/{MODEL_FILE}"))?;
    maybe_download(
        &dir,
        "tokenizer.json",
        &format!("{HF_BASE}/{TOKENIZER_FILE}"),
    )?;

    Ok(dir)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn resolve_cache_dir() -> Result<PathBuf> {
    let base = dirs::cache_dir()
        .ok_or_else(|| LearnError::Embed("cannot determine cache dir".to_owned()))?;
    Ok(base.join(CACHE_SUBDIR))
}

/// Download `url` into `dir/filename` unless it already exists.
fn maybe_download(dir: &Utf8PathBuf, filename: &str, url: &str) -> Result<()> {
    let dest = dir.join(filename);
    if dest.exists() {
        info!(?dest, "model file already present, skipping download");
        return Ok(());
    }
    info!(%url, ?dest, "downloading model file");
    let response = ureq::get(url)
        .call()
        .map_err(|e| LearnError::Embed(format!("download {url}: {e}")))?;

    // Write to a temp file in the same directory, then atomic rename.
    let tmp_path = dir.join(format!("{filename}.tmp"));
    {
        let mut f = std::fs::File::create(tmp_path.as_path())
            .map_err(|e| LearnError::Embed(format!("create tmp file: {e}")))?;

        let mut reader = response.into_body().into_reader();
        let mut buf = [0u8; 65536];
        let mut total = 0u64;
        loop {
            let n = std::io::Read::read(&mut reader, &mut buf)
                .map_err(|e| LearnError::Embed(format!("read download: {e}")))?;
            if n == 0 {
                break;
            }
            f.write_all(&buf[..n])
                .map_err(|e| LearnError::Embed(format!("write tmp: {e}")))?;
            total += n as u64;
            eprint!("\r  {filename}: {:.1} MB", total as f64 / 1_048_576.0);
        }
        eprintln!();
    }

    std::fs::rename(tmp_path.as_path(), dest.as_path())
        .map_err(|e| LearnError::Embed(format!("rename tmp: {e}")))?;

    info!(?dest, "download complete");
    Ok(())
}
