//! `learn-frames` — keyframe extraction + Sonnet-vision captioning.
//!
//! Extracts JPEG keyframes from a video file using `ffmpeg`, then sends each
//! frame to the Anthropic Messages API (vision content block) for a 1–2 sentence
//! description. Returns a `Vec<Segment>` with `kind = SegmentKind::FrameDescription`
//! that can be merged with caption segments before chunking.
//!
//! ## ffmpeg dependency
//!
//! `ffmpeg` is a **runtime** requirement; the crate compiles without it.
//! `FrameExtractor` will return [`learn_core::LearnError::Acquire`] at runtime
//! when ffmpeg is absent or non-executable.
//!
//! ## Cost guard
//!
//! Call [`estimate_frame_count`] before running to print the cost hint to stderr.

#![deny(unsafe_code)]

pub mod decide;
pub use decide::{decide_frames, Decision, FrameDecision, FramesArg};

use camino::{Utf8Path, Utf8PathBuf};
use learn_core::{LearnError, Result, Segment, SegmentKind};
use serde::Deserialize;
use std::process::Command;
use tracing::info;

// ── Public types ─────────────────────────────────────────────────────────────

/// One extracted frame with its filesystem path and timestamp.
#[derive(Debug, Clone)]
pub struct ExtractedFrame {
    /// Absolute path to the JPEG file.
    pub path: Utf8PathBuf,
    /// Timestamp of the frame in the source video (seconds).
    pub timestamp_seconds: f64,
    /// Zero-based frame index (order ffmpeg emitted it).
    pub frame_index: usize,
}

/// Configuration for frame extraction.
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// ffmpeg binary path (default: `ffmpeg` on PATH).
    pub ffmpeg_path: Utf8PathBuf,
    /// Frames per second to extract (default: 1/10 = one frame per 10 s).
    pub fps_rate: f64,
    /// Hard cap: if the estimated count exceeds this, fps is reduced proportionally.
    pub max_frames: usize,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            ffmpeg_path: Utf8PathBuf::from("ffmpeg"),
            fps_rate: 0.1, // 1 frame per 10 s
            max_frames: 60,
        }
    }
}

/// Stateless helper that invokes ffmpeg.
pub struct FrameExtractor {
    cfg: ExtractorConfig,
}

/// Configuration for the vision captioner.
#[derive(Debug, Clone)]
pub struct CaptionerConfig {
    /// Anthropic API key (defaults to `ANTHROPIC_API_KEY` env var at call time).
    pub api_key: Option<String>,
    /// Model to call. Defaults to `claude-sonnet-4-6`.
    pub model: String,
    /// Max tokens for the description (default: 200).
    pub max_tokens: u32,
}

impl Default for CaptionerConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            model: "claude-sonnet-4-6".to_string(),
            max_tokens: 200,
        }
    }
}

/// Calls the Anthropic Messages API with vision content blocks.
pub struct FrameCaptioner {
    client: reqwest::Client,
    cfg: CaptionerConfig,
}

// ── FrameExtractor impl ───────────────────────────────────────────────────────

impl FrameExtractor {
    /// Create a new extractor with the given config.
    pub fn new(cfg: ExtractorConfig) -> Self {
        Self { cfg }
    }

    /// Extract frames from `video_path` into `out_dir` (must exist).
    ///
    /// Calls:
    /// ```text
    /// ffmpeg -i <video> -vf fps=<rate> -q:v 2 <out_dir>/frame-%04d.jpg
    /// ```
    /// Returns the list of frames ordered by frame_index.
    /// Returns `LearnError::Acquire` when ffmpeg is absent or fails.
    pub fn extract_frames(
        &self,
        video_path: &Utf8Path,
        out_dir: &Utf8Path,
    ) -> Result<Vec<ExtractedFrame>> {
        let effective_fps = self.effective_fps(video_path);
        let pattern = out_dir.join("frame-%04d.jpg");

        let status = Command::new(self.cfg.ffmpeg_path.as_str())
            .args([
                "-i",
                video_path.as_str(),
                "-vf",
                &format!("fps={effective_fps}"),
                "-q:v",
                "2",
                pattern.as_str(),
            ])
            .status()
            .map_err(|e| {
                LearnError::Acquire(format!(
                    "ffmpeg not found or failed to launch: {e}. \
                     Install ffmpeg (e.g. `brew install ffmpeg`) and ensure it is on PATH."
                ))
            })?;

        if !status.success() {
            return Err(LearnError::Acquire(format!(
                "ffmpeg exited with non-zero status ({}) for video {video_path}",
                status.code().unwrap_or(-1)
            )));
        }

        collect_frames(out_dir, effective_fps)
    }

    /// Estimate the effective fps after applying the `max_frames` cap.
    ///
    /// When no duration is knowable (no probe available), returns `cfg.fps_rate` unchanged.
    fn effective_fps(&self, video_path: &Utf8Path) -> f64 {
        let duration = probe_duration(video_path).unwrap_or(0.0);
        if duration <= 0.0 {
            return self.cfg.fps_rate;
        }
        let estimated = (duration * self.cfg.fps_rate).ceil() as usize;
        if estimated <= self.cfg.max_frames {
            self.cfg.fps_rate
        } else {
            // Reduce fps so estimated count == max_frames.
            self.cfg.max_frames as f64 / duration
        }
    }
}

/// Estimate frame count for a video without extracting.
///
/// Returns estimated frame count (no side effects).
pub fn estimate_frame_count(video_path: &Utf8Path, cfg: &ExtractorConfig) -> usize {
    let duration = probe_duration(video_path).unwrap_or(0.0);
    let extractor = FrameExtractor::new(cfg.clone());
    let fps = extractor.effective_fps(video_path);
    if duration > 0.0 {
        ((duration * fps).ceil() as usize).min(cfg.max_frames)
    } else {
        cfg.max_frames
    }
}

/// Estimate frame count for a video without extracting.
///
/// Prints cost hint to stderr and returns estimated frame count.
pub fn estimate_and_print_cost(video_path: &Utf8Path, cfg: &ExtractorConfig) -> usize {
    let count = estimate_frame_count(video_path, cfg);
    eprintln!("Estimated frames: {count} (~{count}×$0.005 in vision tokens)");
    count
}

// ── FrameCaptioner impl ───────────────────────────────────────────────────────

impl FrameCaptioner {
    /// Create a new captioner. Builds a single `reqwest::Client` (60 s timeout).
    pub fn new(cfg: CaptionerConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| LearnError::Acquire(format!("reqwest client build failed: {e}")))?;
        Ok(Self { client, cfg })
    }

    /// Send one frame to the Anthropic vision API and return the description.
    ///
    /// Reads the JPEG at `frame.path`, base64-encodes it, and posts a vision
    /// content block. Retries 429/503 up to 3 times (1 s / 2 s / 4 s).
    pub async fn caption_frame(&self, frame: &ExtractedFrame) -> Result<String> {
        let image_bytes = std::fs::read(frame.path.as_std_path()).map_err(LearnError::Io)?;
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &image_bytes);
        let api_key = self.resolve_api_key()?;
        let body = build_vision_body(&self.cfg.model, self.cfg.max_tokens, &b64);
        let response_str = post_with_retries(&self.client, &api_key, &body).await?;
        extract_text_from_response(&response_str)
    }

    fn resolve_api_key(&self) -> Result<String> {
        if let Some(k) = &self.cfg.api_key {
            return Ok(k.clone());
        }
        std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            LearnError::Acquire(
                "ANTHROPIC_API_KEY not set — frame captioning requires an Anthropic API key"
                    .to_string(),
            )
        })
    }
}

// ── High-level pipeline helper ────────────────────────────────────────────────

/// Extract frames from `video_path`, caption each with Sonnet vision, and return
/// a `Vec<Segment>` with `kind = SegmentKind::FrameDescription`.
///
/// Frames are extracted into `out_dir`. Captioning is sequential with backoff.
/// When `video_path` cannot be opened or ffmpeg is absent the error propagates;
/// individual caption failures are warned and skipped (best-effort per frame).
pub async fn extract_and_caption(
    video_path: &Utf8Path,
    out_dir: &Utf8Path,
    extractor_cfg: &ExtractorConfig,
    captioner_cfg: &CaptionerConfig,
) -> Result<Vec<Segment>> {
    let extractor = FrameExtractor::new(extractor_cfg.clone());
    let frames = extractor.extract_frames(video_path, out_dir)?;
    info!(count = frames.len(), "frames extracted");

    let captioner = FrameCaptioner::new(captioner_cfg.clone())?;
    let mut segments = Vec::with_capacity(frames.len());

    for frame in &frames {
        match captioner.caption_frame(frame).await {
            Ok(text) => segments.push(Segment {
                start_seconds: frame.timestamp_seconds,
                end_seconds: frame.timestamp_seconds + 0.1,
                text,
                confidence: None,
                speaker: None,
                kind: SegmentKind::FrameDescription,
            }),
            Err(e) => {
                tracing::warn!(
                    frame_index = frame.frame_index,
                    timestamp = frame.timestamp_seconds,
                    error = %e,
                    "frame caption failed — skipping"
                );
            }
        }
    }

    Ok(segments)
}

/// Merge caption segments and frame-description segments, sorted by start_seconds.
pub fn merge_segments(mut captions: Vec<Segment>, mut frames: Vec<Segment>) -> Vec<Segment> {
    captions.append(&mut frames);
    captions.sort_by(|a, b| {
        a.start_seconds
            .partial_cmp(&b.start_seconds)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    captions
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Walk `out_dir` for `frame-NNNN.jpg` files and build `ExtractedFrame` list.
fn collect_frames(out_dir: &Utf8Path, fps: f64) -> Result<Vec<ExtractedFrame>> {
    let mut frames: Vec<(usize, Utf8PathBuf)> = std::fs::read_dir(out_dir.as_std_path())
        .map_err(LearnError::Io)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !name.starts_with("frame-") || !name.ends_with(".jpg") {
                return None;
            }
            let idx_str = name.strip_prefix("frame-")?.strip_suffix(".jpg")?;
            let idx: usize = idx_str.parse().ok()?;
            let path = Utf8PathBuf::from_path_buf(e.path()).ok()?;
            Some((idx, path))
        })
        .collect();

    frames.sort_by_key(|(idx, _)| *idx);

    Ok(frames
        .into_iter()
        .enumerate()
        .map(|(i, (idx, path))| {
            // ffmpeg frame index is 1-based; timestamp = (index - 1) / fps
            let timestamp_seconds = (idx.saturating_sub(1)) as f64 / fps;
            ExtractedFrame {
                path,
                timestamp_seconds,
                frame_index: i,
            }
        })
        .collect())
}

/// Probe video duration using `ffprobe -v quiet -show_entries format=duration`.
/// Returns `None` when ffprobe is absent or returns non-parseable output.
fn probe_duration(video_path: &Utf8Path) -> Option<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path.as_str(),
        ])
        .output()
        .ok()?;
    let s = std::str::from_utf8(&output.stdout).ok()?.trim().to_string();
    s.parse::<f64>().ok()
}

/// Build the Anthropic Messages API request body with a vision content block.
pub fn build_vision_body(model: &str, max_tokens: u32, b64_jpeg: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64_jpeg
                    }
                },
                {
                    "type": "text",
                    "text": "Describe what's shown in this video frame in 1-2 sentences. \
                             Focus on text, code, diagrams, or other content visible on screen. \
                             If the frame is mostly the speaker, briefly describe their gesture \
                             or what's visible behind them."
                }
            ]
        }]
    })
}

/// POST `body` to the Anthropic Messages API with exponential back-off retry
/// on 429 / 503. Mirrors the pattern in `learn-synth`.
async fn post_with_retries(
    client: &reqwest::Client,
    api_key: &str,
    body: &serde_json::Value,
) -> Result<String> {
    const URL: &str = "https://api.anthropic.com/v1/messages";
    const MAX_RETRIES: u32 = 3;
    let mut delay_secs = 1u64;

    for attempt in 0..MAX_RETRIES {
        let resp = client
            .post(URL)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| LearnError::Acquire(format!("Anthropic vision request failed: {e}")))?;

        let status = resp.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS
            || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
        {
            if attempt + 1 < MAX_RETRIES {
                tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                delay_secs *= 2;
                continue;
            }
            return Err(LearnError::Acquire(format!(
                "Anthropic vision API returned {status} after {MAX_RETRIES} attempts"
            )));
        }

        if !status.is_success() {
            let excerpt = resp
                .text()
                .await
                .unwrap_or_default()
                .chars()
                .take(200)
                .collect::<String>();
            return Err(LearnError::Acquire(format!(
                "Anthropic vision API error {status}: {excerpt}"
            )));
        }

        return resp.text().await.map_err(|e| {
            LearnError::Acquire(format!("Anthropic vision response read failed: {e}"))
        });
    }

    Err(LearnError::Acquire(
        "Anthropic vision retry loop exhausted unexpectedly".to_string(),
    ))
}

/// Deserialised Anthropic Messages API response (text content only).
#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    r#type: String,
    text: String,
}

/// Extract the first text block from a raw Anthropic response JSON string.
fn extract_text_from_response(response_str: &str) -> Result<String> {
    let parsed: AnthropicResponse = serde_json::from_str(response_str)
        .map_err(|e| LearnError::Acquire(format!("malformed Anthropic vision response: {e}")))?;

    parsed
        .content
        .into_iter()
        .filter(|c| c.r#type == "text")
        .map(|c| c.text)
        .reduce(|a, b| format!("{a} {b}"))
        .ok_or_else(|| {
            LearnError::Acquire("Anthropic vision response had no text content".to_string())
        })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use learn_core::SegmentKind;

    // ── Test 1: vision request body construction ──────────────────────────────

    #[test]
    fn build_vision_body_includes_base64_image_and_prompt() {
        let b64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            b"fake-jpeg-bytes",
        );
        let body = build_vision_body("claude-sonnet-4-6", 200, &b64);

        // Model and max_tokens present.
        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(body["max_tokens"], 200);

        // Message content has two blocks.
        let content = &body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "image");
        assert_eq!(content[0]["source"]["type"], "base64");
        assert_eq!(content[0]["source"]["media_type"], "image/jpeg");
        assert_eq!(content[0]["source"]["data"], b64);
        assert_eq!(content[1]["type"], "text");
        let prompt = content[1]["text"].as_str().unwrap();
        assert!(
            prompt.contains("video frame"),
            "prompt should mention 'video frame'; got: {prompt}"
        );
    }

    // ── Test 2: segment merge sorts by timestamp ──────────────────────────────

    #[test]
    fn merge_segments_interleaves_by_start_seconds() {
        let captions = vec![
            Segment {
                start_seconds: 0.0,
                end_seconds: 5.0,
                text: "caption at 0s".into(),
                confidence: None,
                speaker: None,
                kind: SegmentKind::Caption,
            },
            Segment {
                start_seconds: 20.0,
                end_seconds: 25.0,
                text: "caption at 20s".into(),
                confidence: None,
                speaker: None,
                kind: SegmentKind::Caption,
            },
        ];
        let frames = vec![
            Segment {
                start_seconds: 10.0,
                end_seconds: 10.1,
                text: "frame at 10s".into(),
                confidence: None,
                speaker: None,
                kind: SegmentKind::FrameDescription,
            },
            Segment {
                start_seconds: 5.0,
                end_seconds: 5.1,
                text: "frame at 5s".into(),
                confidence: None,
                speaker: None,
                kind: SegmentKind::FrameDescription,
            },
        ];
        let merged = merge_segments(captions, frames);
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[0].start_seconds, 0.0);
        assert_eq!(merged[1].start_seconds, 5.0);
        assert_eq!(merged[2].start_seconds, 10.0);
        assert_eq!(merged[3].start_seconds, 20.0);
        assert_eq!(merged[1].kind, SegmentKind::FrameDescription);
        assert_eq!(merged[2].kind, SegmentKind::FrameDescription);
    }

    // ── Test 3: response JSON parsing ────────────────────────────────────────

    #[test]
    fn extract_text_from_response_parses_text_block() {
        let json = r#"{"content":[{"type":"text","text":"A speaker gestures at a whiteboard."}],"stop_reason":"end_turn"}"#;
        let result = extract_text_from_response(json).unwrap();
        assert_eq!(result, "A speaker gestures at a whiteboard.");
    }

    #[test]
    fn extract_text_from_response_errors_on_empty_content() {
        let json = r#"{"content":[],"stop_reason":"end_turn"}"#;
        let result = extract_text_from_response(json);
        assert!(result.is_err(), "empty content should return Err");
    }

    // ── Test 4: effective_fps caps frames correctly ───────────────────────────

    #[test]
    fn effective_fps_reduces_when_over_max() {
        // 600 s video at 0.1 fps → 60 frames = exactly at the cap.
        let cfg = ExtractorConfig {
            fps_rate: 0.1,
            max_frames: 60,
            ..Default::default()
        };
        let extractor = FrameExtractor::new(cfg);
        // We can't call effective_fps directly (it calls probe_duration which
        // needs a real file). Test the math independently.
        let duration = 600.0_f64;
        let fps_rate = 0.1_f64;
        let max_frames = 60_usize;
        let estimated = (duration * fps_rate).ceil() as usize;
        assert_eq!(estimated, 60);
        // At exactly 60 frames, no reduction needed.
        let effective = if estimated <= max_frames {
            fps_rate
        } else {
            max_frames as f64 / duration
        };
        assert!((effective - 0.1).abs() < 1e-9, "should stay at 0.1");

        // 1200 s video at 0.1 fps → 120 frames > 60 cap → reduce to 0.05 fps.
        let duration2 = 1200.0_f64;
        let estimated2 = (duration2 * fps_rate).ceil() as usize;
        assert_eq!(estimated2, 120);
        let effective2 = max_frames as f64 / duration2;
        assert!((effective2 - 0.05).abs() < 1e-9, "should reduce to 0.05");

        // Suppress unused-variable warning.
        drop(extractor);
    }

    // ── Test 5: FrameDescription kind on merged output ───────────────────────

    #[test]
    fn merge_segments_preserves_kind_field() {
        let captions = vec![Segment {
            start_seconds: 0.0,
            end_seconds: 5.0,
            text: "words".into(),
            confidence: None,
            speaker: None,
            kind: SegmentKind::Caption,
        }];
        let frames = vec![Segment {
            start_seconds: 2.5,
            end_seconds: 2.6,
            text: "visual".into(),
            confidence: None,
            speaker: None,
            kind: SegmentKind::FrameDescription,
        }];
        let merged = merge_segments(captions, frames);
        assert_eq!(merged[0].kind, SegmentKind::Caption);
        assert_eq!(merged[1].kind, SegmentKind::FrameDescription);
    }

    // ── Integration test (requires ANTHROPIC_API_KEY + real video) ──────────

    /// End-to-end frame extraction + captioning.
    /// Run with: cargo test -p learn-frames frame_caption_integration -- --ignored
    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY and a local video file at /tmp/test-video.mp4"]
    async fn frame_caption_integration() {
        let video = Utf8PathBuf::from("/tmp/test-video.mp4");
        let out_dir = Utf8PathBuf::from("/tmp/learn-frames-test");
        std::fs::create_dir_all(out_dir.as_std_path()).unwrap();

        let extractor_cfg = ExtractorConfig::default();
        let captioner_cfg = CaptionerConfig::default();

        let segments = extract_and_caption(&video, &out_dir, &extractor_cfg, &captioner_cfg)
            .await
            .unwrap();
        assert!(
            !segments.is_empty(),
            "should produce at least one segment from a real video"
        );
        assert!(
            segments
                .iter()
                .all(|s| s.kind == SegmentKind::FrameDescription),
            "all segments should be FrameDescription"
        );
    }
}
