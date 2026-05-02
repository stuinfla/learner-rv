//! `learn-asr` — local Whisper transcription via whisper-rs (Metal on Apple Silicon).
//!
//! Uses whisper-rs 0.16 which vendors whisper.cpp. The `metal` feature enables
//! the Metal GPU backend on aarch64-apple-darwin; CPU fallback applies elsewhere.

#![deny(unsafe_code)]

mod audio;
mod download;
mod engine;

pub use engine::{AsrConfig, WhisperEngine};

use camino::Utf8Path;
use learn_core::{Result, Transcript};

/// One-shot helper: load model, transcribe, drop.
pub fn transcribe_file(
    audio_path: &Utf8Path,
    video_id: &str,
    cfg: &AsrConfig,
) -> Result<Transcript> {
    let mut engine = WhisperEngine::load(cfg)?;
    engine.transcribe(audio_path, video_id)
}

/// Locate or download the default model. Returns the path.
///
/// Default location: `~/.cache/learn-rs/models/ggml-base.en.bin`
///
/// If absent, downloads from the official ggerganov/whisper.cpp HuggingFace mirror
/// using a tempfile + atomic rename. Progress is printed to stderr.
pub fn ensure_default_model() -> Result<camino::Utf8PathBuf> {
    download::ensure_default_model()
}
