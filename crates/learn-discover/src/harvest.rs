//! Candidate harvest via `yt-dlp --flat-playlist`.
//!
//! Results are cached in `$TMPDIR/learn-discover-cache/<query_hash>/flat.ndjson`
//! keyed by (query, depth pool_n) so repeated calls to `discover` in the same
//! session skip the yt-dlp invocation.

use crate::Candidate;
use learn_core::{LearnError, Result};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Run `yt-dlp ytsearch{pool_n}:<topic>` and return parsed candidates.
///
/// Results are cached in a temp-dir keyed by the SHA256 of `(topic, pool_n)`.
/// The cache has no TTL (it is scoped to the OS-managed temp dir).
pub async fn harvest(topic: &str, pool_n: usize) -> Result<Vec<Candidate>> {
    let cache_key = format!("{topic}\x00{pool_n}");
    let digest = hex::encode(Sha256::digest(cache_key.as_bytes()));
    let cache_dir = std::env::temp_dir()
        .join("learn-discover-cache")
        .join(&digest);
    let cache_file = cache_dir.join("flat.ndjson");

    if cache_file.exists() {
        tracing::debug!("harvest: cache hit for pool_n={pool_n}");
        let raw = std::fs::read_to_string(&cache_file).map_err(LearnError::Io)?;
        return parse_ndjson(&raw);
    }

    let query = format!("ytsearch{pool_n}:{topic}");
    tracing::info!("harvest: running yt-dlp for {query:?}");

    let output = tokio::process::Command::new("yt-dlp")
        .args([
            &query,
            "--dump-json",
            "--flat-playlist",
            "--skip-download",
            "--no-warnings",
        ])
        .output()
        .await
        .map_err(|e| LearnError::Acquire(format!("yt-dlp spawn failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(LearnError::Acquire(format!(
            "yt-dlp exited {:?}: {stderr}",
            output.status.code()
        )));
    }

    let raw = String::from_utf8_lossy(&output.stdout).into_owned();
    let candidates = parse_ndjson(&raw)?;

    // Write cache
    std::fs::create_dir_all(&cache_dir).map_err(LearnError::Io)?;
    std::fs::write(&cache_file, &raw).map_err(LearnError::Io)?;

    Ok(candidates)
}

fn parse_ndjson(raw: &str) -> Result<Vec<Candidate>> {
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_flat_entry(line) {
            Ok(c) => out.push(c),
            Err(e) => {
                tracing::warn!("harvest: skipping malformed entry: {e}");
            }
        }
    }
    Ok(out)
}

fn parse_flat_entry(line: &str) -> Result<Candidate> {
    let v: serde_json::Value = serde_json::from_str(line)?;

    let video_id = v["id"]
        .as_str()
        .ok_or_else(|| LearnError::Acquire("missing id".into()))?
        .to_owned();

    let title = v["title"].as_str().unwrap_or(&video_id).to_owned();

    let channel = v["channel"].as_str().map(str::to_owned);
    let channel_id = v["channel_id"].as_str().map(str::to_owned);
    let view_count = v["view_count"].as_u64();
    let duration_seconds = v["duration"].as_f64();
    let upload_date = v["upload_date"].as_str().map(str::to_owned);

    Ok(Candidate {
        video_id,
        title,
        channel,
        channel_id,
        view_count,
        duration_seconds,
        upload_date,
        has_captions: None, // populated by caption_gate
        score: 0.0,
    })
}

/// Fetch full info for a single video to check caption availability.
/// Returns (video_id, has_en_captions).
pub async fn fetch_caption_info(video_id: &str) -> std::result::Result<bool, String> {
    let cache_file = caption_cache_path(video_id);
    let raw = if cache_file.exists() {
        std::fs::read_to_string(&cache_file).map_err(|e| e.to_string())?
    } else {
        let url = format!("https://www.youtube.com/watch?v={video_id}");
        let output = tokio::process::Command::new("yt-dlp")
            .args([&url, "--dump-json", "--skip-download", "--no-warnings"])
            .output()
            .await
            .map_err(|e| format!("yt-dlp spawn failed: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("yt-dlp exit {:?}: {stderr}", output.status.code()));
        }
        let s = String::from_utf8_lossy(&output.stdout).into_owned();
        if let Some(parent) = cache_file.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&cache_file, &s);
        s
    };

    let v: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("parse error: {e}"))?;

    Ok(has_en_captions(&v))
}

fn has_en_captions(v: &serde_json::Value) -> bool {
    for key in ["subtitles", "automatic_captions"] {
        if let Some(obj) = v[key].as_object() {
            if obj.keys().any(|k| k.starts_with("en")) {
                return true;
            }
        }
    }
    false
}

fn caption_cache_path(video_id: &str) -> PathBuf {
    std::env::temp_dir()
        .join("learn-discover-cache")
        .join("captions")
        .join(format!("{video_id}.json"))
}
