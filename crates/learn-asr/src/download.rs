//! Model download helper: ensure the default ggml-base.en model is present.

use camino::Utf8PathBuf;
use learn_core::{LearnError, Result};
use std::io::Write;
use tracing::info;

const MODEL_FILENAME: &str = "ggml-base.en.bin";
const MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin";

/// Return the canonical default model path: `~/.cache/learn-rs/models/ggml-base.en.bin`.
pub fn default_model_path() -> Result<Utf8PathBuf> {
    let home = std::env::var("HOME").map_err(|_| LearnError::Transcribe("$HOME not set".into()))?;
    let mut p = Utf8PathBuf::from(home);
    p.push(".cache/learn-rs/models");
    p.push(MODEL_FILENAME);
    Ok(p)
}

/// Ensure the default model file exists, downloading it if needed.
///
/// Uses a tempfile + atomic rename so partial downloads are never visible.
pub fn ensure_default_model() -> Result<Utf8PathBuf> {
    let model_path = default_model_path()?;

    if model_path.exists() {
        info!(path = %model_path, "model already present");
        return Ok(model_path);
    }

    let dir = model_path
        .parent()
        .ok_or_else(|| LearnError::Transcribe("model path has no parent dir".into()))?;

    std::fs::create_dir_all(dir)
        .map_err(|e| LearnError::Transcribe(format!("create model dir: {e}")))?;

    info!(%model_path, source = %MODEL_URL, "downloading whisper model");

    let response = ureq::get(MODEL_URL)
        .call()
        .map_err(|e| LearnError::Transcribe(format!("HTTP GET failed: {e}")))?;

    // Write to a temp file in the same directory, then rename atomically.
    let tmp_path = format!("{}.tmp", model_path);
    let mut tmp_file = std::fs::File::create(&tmp_path)
        .map_err(|e| LearnError::Transcribe(format!("create temp file: {e}")))?;

    let mut reader = response.into_reader();
    let mut buf = [0u8; 65536];
    let mut total: u64 = 0;

    loop {
        let n = std::io::Read::read(&mut reader, &mut buf)
            .map_err(|e| LearnError::Transcribe(format!("read download stream: {e}")))?;
        if n == 0 {
            break;
        }
        tmp_file
            .write_all(&buf[..n])
            .map_err(|e| LearnError::Transcribe(format!("write temp file: {e}")))?;
        total += n as u64;
        if total % (10 * 1024 * 1024) < n as u64 {
            info!(mb = total as f64 / 1_048_576.0, "model download progress");
        }
    }

    tmp_file
        .flush()
        .map_err(|e| LearnError::Transcribe(format!("flush temp file: {e}")))?;
    drop(tmp_file);

    std::fs::rename(&tmp_path, model_path.as_std_path())
        .map_err(|e| LearnError::Transcribe(format!("rename temp file: {e}")))?;

    info!(
        path = %model_path,
        bytes = total,
        mb = total as f64 / 1_048_576.0,
        "model download complete"
    );

    Ok(model_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_path_ends_with_filename() {
        let p = default_model_path().expect("HOME must be set in test env");
        assert!(p.as_str().ends_with(MODEL_FILENAME));
        assert!(p.as_str().contains(".cache/learn-rs/models"));
    }

    #[test]
    #[ignore = "performs network I/O — run with --ignored to actually download"]
    fn ensure_default_model_downloads_if_absent() {
        // This will download ~150 MB from HuggingFace.
        let path = ensure_default_model().expect("download should succeed");
        assert!(path.exists(), "model file should exist after download");
    }
}
