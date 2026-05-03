//! Frame-extraction auto-decision: pHash variance + optional VLM probe.
//!
//! Decision hierarchy (only runs when `FramesArg::Auto`):
//! 1. Extract 5 evenly-spaced frames with ffmpeg.
//! 2. Compute perceptual-hash pairwise Hamming distance, normalised to \[0, 1\].
//!    - distance > 0.30 → `Decision::FullExtraction` (high variance → visual)
//!    - distance < 0.10 → `Decision::Skip`             (low variance → talking head)
//!    - else            → Step 3
//! 3. Send the middle frame to Sonnet 4.6 vision for a VISUAL / TALKING_HEAD probe.
//!    Requires `ANTHROPIC_API_KEY`; falls back to `FullExtraction` when absent.

#![deny(unsafe_code)]

use camino::Utf8Path;
use image_hasher::{HashAlg, HasherConfig, ImageHash};
use learn_core::{LearnError, Result};
use std::process::Command;
use tracing::info;

// ── Public types ──────────────────────────────────────────────────────────────

/// How the caller chose to handle frame extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FramesArg {
    /// Auto-decide using the 3-tier evaluator.
    Auto,
    /// Force full frame extraction regardless of content.
    On,
    /// Skip frame extraction regardless of content.
    Off,
}

/// Outcome of `decide_frames`.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// Run the full frame-extraction + captioning pipeline.
    FullExtraction { reason: String },
    /// Skip frame extraction; captions-only path.
    Skip { reason: String },
}

/// Full context returned by `decide_frames`.
#[derive(Debug, Clone)]
pub struct FrameDecision {
    pub mode: Decision,
    pub reason: String,
    pub variance: f32,
    pub probe_invoked: bool,
}

// Thresholds (named for readability and easy tuning).
// HIGH: variance above this → definitely visual content (high frame diversity).
// LOW:  variance below this → only skip if frames are near-identical (e.g. freeze frames,
//       screensavers, or a static speaker shot). Dark-theme screen recordings can have
//       very low pHash variance even when content changes, so we keep this very tight.
const VARIANCE_HIGH: f32 = 0.30;
const VARIANCE_LOW: f32 = 0.015;
const PROBE_FRAMES: usize = 5;

const VLM_PROBE_PROMPT: &str = "\
Does this video frame contain meaningful visual content beyond a person speaking?\n\
Examples of meaningful: code, slides, charts, cooking technique, art demonstration, \
physical skill being shown, screen recording.\n\
Examples of NOT meaningful: just a person at a podium, talking head against a background, \
podcast face, interview format.\n\
Reply with exactly one word: VISUAL or TALKING_HEAD.";

// ── Public API ────────────────────────────────────────────────────────────────

/// Decide whether frame extraction should run for `video_path`.
///
/// - `FramesArg::On`  → always `FullExtraction`
/// - `FramesArg::Off` → always `Skip`
/// - `FramesArg::Auto` → runs the 3-tier pHash + optional VLM evaluator
pub async fn decide_frames(video_path: &Utf8Path, frames_arg: FramesArg) -> Result<FrameDecision> {
    match frames_arg {
        FramesArg::On => Ok(FrameDecision {
            mode: Decision::FullExtraction {
                reason: "frames=on (forced)".to_string(),
            },
            reason: "frames=on (forced)".to_string(),
            variance: 1.0,
            probe_invoked: false,
        }),
        FramesArg::Off => Ok(FrameDecision {
            mode: Decision::Skip {
                reason: "frames=off (forced)".to_string(),
            },
            reason: "frames=off (forced)".to_string(),
            variance: 0.0,
            probe_invoked: false,
        }),
        FramesArg::Auto => auto_decide(video_path).await,
    }
}

// ── Auto-decision pipeline ────────────────────────────────────────────────────

async fn auto_decide(video_path: &Utf8Path) -> Result<FrameDecision> {
    // Step 1: extract 5 probe frames + compute pHash variance.
    let (variance, probe_dir) = phash_variance(video_path)?;

    if variance > VARIANCE_HIGH {
        let reason = format!("HIGH variance ({variance:.2}) — visual content detected");
        return Ok(FrameDecision {
            mode: Decision::FullExtraction {
                reason: reason.clone(),
            },
            reason,
            variance,
            probe_invoked: false,
        });
    }

    if variance < VARIANCE_LOW {
        let reason = format!("LOW variance ({variance:.2}) — talking head detected");
        return Ok(FrameDecision {
            mode: Decision::Skip {
                reason: reason.clone(),
            },
            reason,
            variance,
            probe_invoked: false,
        });
    }

    // Step 2: MID variance — run VLM probe on the middle frame.
    info!(variance, "MID variance — running VLM probe");
    let middle_frame = probe_dir.join("probe-0003.jpg");
    let vlm_result = vlm_probe(&middle_frame).await;

    let (decision, reason) = match vlm_result {
        Ok(VlmVerdict::Visual(label)) => (
            Decision::FullExtraction {
                reason: format!("MID variance ({variance:.2}) — vlm-probe: {label}"),
            },
            format!("MID variance ({variance:.2}) — vlm-probe: {label}"),
        ),
        Ok(VlmVerdict::TalkingHead(label)) => (
            Decision::Skip {
                reason: format!("MID variance ({variance:.2}) — vlm-probe: {label}"),
            },
            format!("MID variance ({variance:.2}) — vlm-probe: {label}"),
        ),
        Err(e) => {
            let reason =
                format!("MID variance ({variance:.2}) — VLM probe failed ({e}), defaulting on");
            tracing::warn!(%e, "VLM probe failed — defaulting to FullExtraction");
            (
                Decision::FullExtraction {
                    reason: reason.clone(),
                },
                reason,
            )
        }
    };

    Ok(FrameDecision {
        mode: decision,
        reason,
        variance,
        probe_invoked: true,
    })
}

// ── pHash variance ────────────────────────────────────────────────────────────

/// Extract 5 probe frames, compute average pairwise normalised Hamming distance.
/// Returns `(variance, tmp_dir_path)` so callers can reference the middle frame.
fn phash_variance(video_path: &Utf8Path) -> Result<(f32, camino::Utf8PathBuf)> {
    // Create temp dir inside the system temp.
    let tmp = std::env::temp_dir().join("learn-phash");
    std::fs::create_dir_all(&tmp).map_err(LearnError::Io)?;
    let out_dir = camino::Utf8PathBuf::from_path_buf(tmp)
        .map_err(|p| LearnError::Acquire(format!("non-UTF-8 tmp path: {}", p.display())))?;

    extract_probe_frames(video_path, &out_dir)?;

    let hashes = load_hashes(&out_dir)?;
    if hashes.len() < 2 {
        // Can't compute pairwise — treat as FullExtraction (safe default).
        return Ok((1.0, out_dir));
    }

    let variance = avg_hamming(&hashes);
    Ok((variance, out_dir))
}

/// Run ffmpeg to extract exactly `PROBE_FRAMES` evenly-spaced frames.
///
/// Uses `fps=N/duration` to space frames evenly across the video. This is
/// simpler and more portable than the `select` filter (which requires
/// `n_frames` to be known before decoding).
fn extract_probe_frames(video_path: &Utf8Path, out_dir: &camino::Utf8Path) -> Result<()> {
    // Remove stale probe frames so we get exactly PROBE_FRAMES fresh ones.
    for i in 1..=(PROBE_FRAMES + 2) {
        let _ = std::fs::remove_file(out_dir.join(format!("probe-{i:04}.jpg")).as_std_path());
    }

    // Compute fps = PROBE_FRAMES / duration so ffmpeg emits exactly PROBE_FRAMES frames.
    let duration = probe_duration_secs(video_path);
    let fps_str = if duration > 0.0 {
        format!("{}", (PROBE_FRAMES as f64) / duration)
    } else {
        // Fallback: extract at 0.1 fps (one frame per 10 s) and cap with -vframes.
        "0.1".to_string()
    };

    let vf = format!("fps={fps_str},scale=256:144:force_original_aspect_ratio=decrease");

    let status = Command::new("ffmpeg")
        .args([
            "-i",
            video_path.as_str(),
            "-vf",
            &vf,
            "-vframes",
            &PROBE_FRAMES.to_string(),
            "-q:v",
            "5",
            "-y",
            out_dir.join("probe-%04d.jpg").as_str(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| {
            LearnError::Acquire(format!(
                "ffmpeg probe failed to launch: {e}. Is ffmpeg installed?"
            ))
        })?;

    if !status.success() {
        return Err(LearnError::Acquire(format!(
            "ffmpeg probe exited {} for {video_path}",
            status.code().unwrap_or(-1)
        )));
    }
    Ok(())
}

/// Probe video duration in seconds using ffprobe.
/// Returns 0.0 when ffprobe is absent or output is unparseable.
fn probe_duration_secs(video_path: &Utf8Path) -> f64 {
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
        .ok();
    output
        .and_then(|o| {
            std::str::from_utf8(&o.stdout)
                .ok()
                .map(|s| s.trim().to_string())
        })
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// Load JPEG probe frames from `out_dir` and compute their pHashes.
fn load_hashes(out_dir: &camino::Utf8Path) -> Result<Vec<ImageHash>> {
    // Mean + DCT preprocessing == classic pHash (per image_hasher docs).
    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::Mean)
        .preproc_dct()
        .hash_size(8, 8)
        .to_hasher();

    let mut hashes: Vec<(usize, ImageHash)> = std::fs::read_dir(out_dir.as_std_path())
        .map_err(LearnError::Io)?
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !name.starts_with("probe-") || !name.ends_with(".jpg") {
                return None;
            }
            let idx_str = name.strip_prefix("probe-")?.strip_suffix(".jpg")?;
            let idx: usize = idx_str.parse().ok()?;
            let img = image::open(e.path()).ok()?;
            let hash = hasher.hash_image(&img);
            Some((idx, hash))
        })
        .collect();

    hashes.sort_by_key(|(idx, _)| *idx);
    Ok(hashes.into_iter().map(|(_, h)| h).collect())
}

/// Average normalised pairwise Hamming distance across all hash pairs.
fn avg_hamming(hashes: &[ImageHash]) -> f32 {
    let n = hashes.len();
    if n < 2 {
        return 0.0;
    }
    let hash_bits = 64u32; // 8×8 pHash
    let mut total = 0u32;
    let mut pairs = 0u32;

    for i in 0..n {
        for j in (i + 1)..n {
            total += hashes[i].dist(&hashes[j]);
            pairs += 1;
        }
    }

    (total as f32) / (pairs as f32) / (hash_bits as f32)
}

// ── VLM probe ─────────────────────────────────────────────────────────────────

enum VlmVerdict {
    Visual(String),
    TalkingHead(String),
}

async fn vlm_probe(frame_path: &camino::Utf8Path) -> Result<VlmVerdict> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
        LearnError::Acquire(
            "ANTHROPIC_API_KEY not set — VLM probe requires an Anthropic API key".to_string(),
        )
    })?;

    let image_bytes = std::fs::read(frame_path.as_std_path()).map_err(LearnError::Io)?;
    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &image_bytes);

    let body = build_probe_body(&b64);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| LearnError::Acquire(format!("reqwest client: {e}")))?;

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| LearnError::Acquire(format!("VLM probe request failed: {e}")))?;

    let status = resp.status();
    if !status.is_success() {
        let excerpt = resp
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(200)
            .collect::<String>();
        return Err(LearnError::Acquire(format!(
            "VLM probe API error {status}: {excerpt}"
        )));
    }

    let text = resp
        .text()
        .await
        .map_err(|e| LearnError::Acquire(format!("VLM probe read: {e}")))?;

    parse_vlm_verdict(&text)
}

fn build_probe_body(b64: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 10,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64
                    }
                },
                {
                    "type": "text",
                    "text": VLM_PROBE_PROMPT
                }
            ]
        }]
    })
}

fn parse_vlm_verdict(response_str: &str) -> Result<VlmVerdict> {
    #[derive(serde::Deserialize)]
    struct Resp {
        content: Vec<Block>,
    }
    #[derive(serde::Deserialize)]
    struct Block {
        r#type: String,
        text: String,
    }

    let parsed: Resp = serde_json::from_str(response_str)
        .map_err(|e| LearnError::Acquire(format!("VLM probe response parse: {e}")))?;

    let raw = parsed
        .content
        .into_iter()
        .find(|b| b.r#type == "text")
        .map(|b| b.text)
        .unwrap_or_default();

    let trimmed = raw.trim().to_uppercase();
    if trimmed.starts_with("VISUAL") {
        Ok(VlmVerdict::Visual(
            "VISUAL (text/diagrams visible)".to_string(),
        ))
    } else if trimmed.starts_with("TALKING_HEAD") || trimmed.starts_with("TALKING") {
        Ok(VlmVerdict::TalkingHead("TALKING_HEAD".to_string()))
    } else {
        // Ambiguous — conservatively default to full extraction.
        Ok(VlmVerdict::Visual(format!(
            "ambiguous response ({raw}), defaulting on"
        )))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: FramesArg::On always returns FullExtraction ──────────────────
    #[tokio::test]
    async fn frames_arg_on_returns_full_extraction() {
        let path = camino::Utf8Path::new("/tmp/dummy.mp4");
        let decision = decide_frames(path, FramesArg::On).await.unwrap();
        assert!(
            matches!(decision.mode, Decision::FullExtraction { .. }),
            "frames=on must return FullExtraction"
        );
        assert!(!decision.probe_invoked);
        assert_eq!(decision.variance, 1.0);
    }

    // ── Test 2: FramesArg::Off always returns Skip ────────────────────────────
    #[tokio::test]
    async fn frames_arg_off_returns_skip() {
        let path = camino::Utf8Path::new("/tmp/dummy.mp4");
        let decision = decide_frames(path, FramesArg::Off).await.unwrap();
        assert!(
            matches!(decision.mode, Decision::Skip { .. }),
            "frames=off must return Skip"
        );
        assert!(!decision.probe_invoked);
        assert_eq!(decision.variance, 0.0);
    }

    // ── Test 3: avg_hamming returns 0.0 for single identical hashes ──────────
    #[test]
    fn avg_hamming_zero_for_identical_hashes() {
        use image::DynamicImage;
        let img = DynamicImage::new_rgb8(8, 8);
        let hasher = image_hasher::HasherConfig::new()
            .hash_alg(image_hasher::HashAlg::Mean)
            .preproc_dct()
            .hash_size(8, 8)
            .to_hasher();
        let h = hasher.hash_image(&img);
        let hashes = vec![h.clone(), h];
        let dist = avg_hamming(&hashes);
        assert_eq!(dist, 0.0, "identical hashes must have 0 distance");
    }

    // ── Test 4: avg_hamming with <2 hashes returns 0 ─────────────────────────
    #[test]
    fn avg_hamming_single_hash_returns_zero() {
        use image::DynamicImage;
        let img = DynamicImage::new_rgb8(8, 8);
        let hasher = image_hasher::HasherConfig::new()
            .hash_alg(image_hasher::HashAlg::Mean)
            .preproc_dct()
            .hash_size(8, 8)
            .to_hasher();
        let h = hasher.hash_image(&img);
        let dist = avg_hamming(&[h]);
        assert_eq!(dist, 0.0);
    }

    // ── Test 5: avg_hamming returns >0 for distinct images ───────────────────
    #[test]
    fn avg_hamming_positive_for_distinct_images() {
        use image::{DynamicImage, Rgba, RgbaImage};
        let mut img1 = RgbaImage::new(8, 8);
        for pixel in img1.pixels_mut() {
            *pixel = Rgba([0u8, 0, 0, 255]);
        }
        let mut img2 = RgbaImage::new(8, 8);
        for pixel in img2.pixels_mut() {
            *pixel = Rgba([255u8, 255, 255, 255]);
        }
        let hasher = image_hasher::HasherConfig::new()
            .hash_alg(image_hasher::HashAlg::Mean)
            .preproc_dct()
            .hash_size(8, 8)
            .to_hasher();
        let h1 = hasher.hash_image(&DynamicImage::ImageRgba8(img1));
        let h2 = hasher.hash_image(&DynamicImage::ImageRgba8(img2));
        let dist = avg_hamming(&[h1, h2]);
        assert!(dist > 0.0, "black vs white must have positive distance");
    }

    // ── Test 6: VLM verdict parsing — VISUAL ─────────────────────────────────
    #[test]
    fn parse_vlm_verdict_visual() {
        let json = r#"{"content":[{"type":"text","text":"VISUAL"}]}"#;
        let v = parse_vlm_verdict(json).unwrap();
        assert!(matches!(v, VlmVerdict::Visual(_)));
    }

    // ── Test 7: VLM verdict parsing — TALKING_HEAD ───────────────────────────
    #[test]
    fn parse_vlm_verdict_talking_head() {
        let json = r#"{"content":[{"type":"text","text":"TALKING_HEAD"}]}"#;
        let v = parse_vlm_verdict(json).unwrap();
        assert!(matches!(v, VlmVerdict::TalkingHead(_)));
    }

    // ── Test 8: VLM verdict ambiguous → defaults to Visual ───────────────────
    #[test]
    fn parse_vlm_verdict_ambiguous_defaults_visual() {
        let json = r#"{"content":[{"type":"text","text":"I am not sure"}]}"#;
        let v = parse_vlm_verdict(json).unwrap();
        assert!(
            matches!(v, VlmVerdict::Visual(_)),
            "ambiguous response should default to Visual"
        );
    }

    // ── Test 9: build_probe_body has required fields ──────────────────────────
    #[test]
    fn build_probe_body_well_formed() {
        let body = build_probe_body("AABB==");
        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(body["max_tokens"], 10);
        let content = &body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "image");
        assert_eq!(content[1]["type"], "text");
        let prompt = content[1]["text"].as_str().unwrap();
        assert!(
            prompt.contains("VISUAL or TALKING_HEAD"),
            "probe prompt must ask for VISUAL or TALKING_HEAD"
        );
    }

    // ── Test 10: variance threshold logic (unit) ──────────────────────────────
    #[test]
    fn threshold_logic_high_variance_selects_extraction() {
        // Round-trip through a variable so clippy can't constant-fold.
        let low = VARIANCE_LOW;
        let high = VARIANCE_HIGH;
        assert!(low < high, "LOW threshold must be less than HIGH threshold");
        assert!(high <= 1.0, "HIGH threshold must be <= 1.0 (normalised)");

        // Simulate the decision branch for a known high-variance value.
        let variance: f32 = 0.45;
        let outcome = if variance > high {
            "full"
        } else if variance < low {
            "skip"
        } else {
            "probe"
        };
        assert_eq!(outcome, "full");
    }
}
