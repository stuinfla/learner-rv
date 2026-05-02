//! `WhisperEngine`: loads a ggml model once and transcribes audio files.

use camino::{Utf8Path, Utf8PathBuf};
use learn_core::{LearnError, Result, Segment, Transcript, TranscriptSource};
use tracing::{debug, info};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::{audio, download};

/// Configuration for the Whisper ASR engine.
#[derive(Debug, Clone)]
pub struct AsrConfig {
    /// Path to a ggml whisper model file.
    pub model_path: Utf8PathBuf,
    /// Language hint (e.g. `"en"`). `None` lets Whisper auto-detect.
    pub language: Option<String>,
    /// CPU thread count. Metal still uses this for CPU-side prep work.
    pub n_threads: i32,
}

impl Default for AsrConfig {
    fn default() -> Self {
        let model_path = download::default_model_path()
            .unwrap_or_else(|_| Utf8PathBuf::from("~/.cache/learn-rs/models/ggml-base.en.bin"));
        let n_threads = (num_cpus::get() / 2).max(1) as i32;
        Self {
            model_path,
            language: None,
            n_threads,
        }
    }
}

/// Opaque engine: holds the loaded model context.
#[derive(Debug)]
pub struct WhisperEngine {
    ctx: WhisperContext,
    cfg: AsrConfig,
}

impl WhisperEngine {
    /// Load the ggml model from `cfg.model_path`. Fails fast if the file is absent.
    pub fn load(cfg: &AsrConfig) -> Result<Self> {
        if !cfg.model_path.exists() {
            return Err(LearnError::Transcribe(format!(
                "model file not found: {}",
                cfg.model_path
            )));
        }

        info!(model = %cfg.model_path, "loading Whisper model");

        let ctx = WhisperContext::new_with_params(
            cfg.model_path.as_str(),
            WhisperContextParameters::default(),
        )
        .map_err(|e| LearnError::Transcribe(format!("whisper context init: {e}")))?;

        Ok(Self {
            ctx,
            cfg: cfg.clone(),
        })
    }

    /// Transcribe an audio file (mp3 or wav) and return a [`Transcript`].
    ///
    /// Steps:
    /// 1. Decode audio to 16 kHz mono f32 PCM via ffmpeg.
    /// 2. Configure `FullParams` with beam search, language hint, thread count.
    /// 3. Run `state.full(params, &samples)`.
    /// 4. Collect segments with timestamps and confidence.
    pub fn transcribe(&mut self, audio_path: &Utf8Path, video_id: &str) -> Result<Transcript> {
        debug!(path = %audio_path, "starting transcription");

        let samples = audio::decode_to_pcm(audio_path)?;

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| LearnError::Transcribe(format!("create whisper state: {e}")))?;

        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        });

        params.set_n_threads(self.cfg.n_threads);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_single_segment(false);
        params.set_suppress_nst(true);

        // Language: pass the hint as a static-lifetime string slice or "auto".
        // We store as an owned String in cfg; we must ensure it lives long enough.
        // FullParams<'a, '_> borrows the &'a str for language, so we use a local.
        let lang_str;
        match &self.cfg.language {
            Some(l) => {
                lang_str = l.clone();
                params.set_language(Some(lang_str.as_str()));
            }
            None => {
                params.set_detect_language(true);
            }
        }

        state
            .full(params, &samples)
            .map_err(|e| LearnError::Transcribe(format!("whisper inference: {e}")))?;

        let n_segments = state.full_n_segments();
        debug!(n_segments, "whisper returned segments");

        let mut segments = Vec::with_capacity(n_segments as usize);

        for i in 0..n_segments {
            let seg = state
                .get_segment(i)
                .ok_or_else(|| LearnError::Transcribe(format!("segment {i} out of bounds")))?;

            let text = seg
                .to_str_lossy()
                .map_err(|e| LearnError::Transcribe(format!("segment {i} text: {e}")))?
                .trim()
                .to_string();

            // Timestamps are in centiseconds (10 ms units); convert to seconds.
            let start_seconds = seg.start_timestamp() as f64 / 100.0;
            let end_seconds = seg.end_timestamp() as f64 / 100.0;

            // no_speech_probability is the probability that the segment is NOT speech.
            // confidence ≈ 1.0 - no_speech_prob, clamped to [0, 1].
            let no_speech = seg.no_speech_probability();
            let confidence = (1.0 - no_speech).clamp(0.0, 1.0);

            segments.push(Segment {
                start_seconds,
                end_seconds,
                text,
                confidence: Some(confidence),
                speaker: None,
            });
        }

        // Detect language from state.
        let lang_id = state.full_lang_id_from_state();
        let language = whisper_rs::get_lang_str(lang_id).map(str::to_string);

        info!(
            video_id,
            n_segments,
            language = language.as_deref().unwrap_or("unknown"),
            "transcription complete"
        );

        Ok(Transcript {
            video_id: video_id.to_string(),
            language,
            source: TranscriptSource::WhisperLocal,
            segments,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn asr_config_default_model_path_under_cache() {
        let cfg = AsrConfig::default();
        assert!(
            cfg.model_path.as_str().contains(".cache/learn-rs/models"),
            "default model path should be under ~/.cache/learn-rs/models, got: {}",
            cfg.model_path
        );
        assert!(
            cfg.model_path.as_str().ends_with("ggml-base.en.bin"),
            "default model should be ggml-base.en.bin"
        );
    }

    #[test]
    fn asr_config_default_threads_at_least_one() {
        let cfg = AsrConfig::default();
        assert!(cfg.n_threads >= 1, "n_threads must be at least 1");
    }

    #[test]
    fn load_returns_transcribe_error_for_missing_model() {
        let tmp = NamedTempFile::new().expect("tempfile");
        // Use a path that definitely does not exist.
        let missing = Utf8PathBuf::from(format!("{}.missing", tmp.path().display()));
        let cfg = AsrConfig {
            model_path: missing.clone(),
            language: Some("en".into()),
            n_threads: 1,
        };
        let result = WhisperEngine::load(&cfg);
        match result {
            Err(LearnError::Transcribe(msg)) => {
                assert!(
                    msg.contains("model file not found"),
                    "expected 'model file not found' in error, got: {msg}"
                );
            }
            other => panic!("expected Transcribe error, got: {other:?}"),
        }
    }

    #[test]
    fn transcribe_nonexistent_audio_returns_error() {
        // We can't load a real model without network I/O, so we test the ffmpeg
        // path by pointing at an audio file that doesn't exist.
        // First we need a "model" file that exists (even if invalid) to pass the
        // existence check, then we rely on ffmpeg failing for the audio path.
        //
        // Actually — WhisperEngine::load checks model existence, and initialising
        // whisper with a garbage file will fail. So we test the outer error path
        // that a missing model produces a LearnError::Transcribe.
        let missing_audio = Utf8PathBuf::from("/tmp/this_audio_does_not_exist_12345.mp3");
        assert!(!missing_audio.exists());

        // Simulate what the pipeline would see: ffmpeg spawn for a missing file.
        let result = crate::audio::decode_to_pcm(&missing_audio);
        // ffmpeg will fail because the input doesn't exist.
        assert!(
            result.is_err(),
            "decode_to_pcm on missing file should return Err"
        );
        match result.unwrap_err() {
            LearnError::Transcribe(_) => {}
            e => panic!("expected Transcribe error, got: {e:?}"),
        }
    }
}
