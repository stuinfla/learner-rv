//! `learn-acquire` — resolve a video URL to a local `Acquired` with captions.
//!
//! Phase 1: shells out to `yt-dlp --skip-download` to pull `.info.json` and
//! WebVTT captions, then builds a `VideoRef` + `Acquired` from the results.

#![deny(unsafe_code)]

pub mod vtt;

use camino::{Utf8Path, Utf8PathBuf};
use learn_core::{Acquired, LearnError, Result, VideoRef};
use serde::Deserialize;
use std::fs;
use tokio::process::Command;
use tracing::{info, warn};

// ── source classification ─────────────────────────────────────────────────────

/// Coarse kind of a source string as understood by yt-dlp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceKind {
    /// A single watchable video URL (youtube.com/watch, youtu.be, etc.).
    SingleVideo,
    /// A playlist URL (`/playlist?list=…`).
    Playlist,
    /// A YouTube channel handle (`@channelname`) or channel URL.
    Channel,
    /// A yt-dlp `ytsearch[N]:` or `ytsearchdate[N]:` pseudo-scheme.
    Search,
    /// A local directory path.
    LocalDirectory,
    /// A local file path.
    LocalFile,
}

/// Classify a source string into a [`SourceKind`] without hitting the network.
pub fn classify_source(source: &str) -> SourceKind {
    // Local paths take priority — check disk first.
    let p = std::path::Path::new(source);
    if p.exists() {
        return if p.is_dir() {
            SourceKind::LocalDirectory
        } else {
            SourceKind::LocalFile
        };
    }

    // yt-dlp pseudo-schemes.
    if source.starts_with("ytsearch") {
        return SourceKind::Search;
    }

    // Channel handles.
    if source.starts_with('@') {
        return SourceKind::Channel;
    }

    // URL-based classification.
    if let Ok(u) = url::Url::parse(source) {
        let host = u.host_str().unwrap_or("");
        let path = u.path();

        // Channel URLs: /c/, /channel/, /@handle
        if path.starts_with("/c/")
            || path.starts_with("/channel/")
            || path.starts_with("/@")
            || path.contains("/videos")
        {
            return SourceKind::Channel;
        }

        // Playlist URLs.
        let is_playlist_query = u.query_pairs().any(|(k, _)| k == "list");
        let has_no_v = !u.query_pairs().any(|(k, _)| k == "v");
        if (host.contains("youtube.com") && is_playlist_query && has_no_v)
            || path.contains("/playlist")
        {
            return SourceKind::Playlist;
        }
    }

    // Anything else (http single-video URL, unknown URL scheme).
    SourceKind::SingleVideo
}

// ── flat-playlist resolver ────────────────────────────────────────────────────

/// Minimal subset of a yt-dlp flat-playlist JSON entry.
#[derive(Debug, Deserialize)]
struct FlatEntry {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    webpage_url: Option<String>,
    #[serde(default)]
    id: Option<String>,
}

/// Resolve `source` to a list of single-video URLs.
///
/// - [`SourceKind::SingleVideo`] / [`SourceKind::LocalFile`] / [`SourceKind::LocalDirectory`]:
///   returns a one-element `Vec` containing `source` unchanged.
/// - [`SourceKind::Playlist`] / [`SourceKind::Channel`] / [`SourceKind::Search`]:
///   calls `yt-dlp --flat-playlist --dump-json` and parses the NDJSON output
///   into individual watch URLs. If `limit` is `Some(n)`, passes
///   `--playlist-end n` to yt-dlp.
pub async fn resolve_to_videos(source: &str, limit: Option<usize>) -> Result<Vec<String>> {
    validate_source(source)?;

    match classify_source(source) {
        SourceKind::SingleVideo | SourceKind::LocalFile | SourceKind::LocalDirectory => {
            Ok(vec![source.to_owned()])
        }
        SourceKind::Playlist | SourceKind::Channel | SourceKind::Search => {
            resolve_flat_playlist(source, limit).await
        }
    }
}

/// Call `yt-dlp --flat-playlist --dump-json [--playlist-end N] <source>` and
/// parse each NDJSON line into a watch URL.
async fn resolve_flat_playlist(source: &str, limit: Option<usize>) -> Result<Vec<String>> {
    let mut args: Vec<String> = vec!["--flat-playlist".into(), "--dump-json".into()];
    if let Some(n) = limit {
        args.push("--playlist-end".into());
        args.push(n.to_string());
    }
    args.push(source.to_owned());

    let output = Command::new("yt-dlp")
        .args(&args)
        .output()
        .await
        .map_err(|e| LearnError::Acquire(format!("yt-dlp not found or failed to spawn: {e}")))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !stderr.is_empty() {
        warn!(ytdlp.stderr = %stderr.trim());
    }

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        warn!(exit_code = code, "yt-dlp flat-playlist exited non-zero");
    }

    let mut urls: Vec<String> = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<FlatEntry>(trimmed) {
            if let Some(u) = entry.url.or(entry.webpage_url) {
                urls.push(normalise_video_url(u, &entry.id));
                continue;
            }
            if let Some(id) = entry.id {
                urls.push(format!("https://www.youtube.com/watch?v={id}"));
            }
        }
    }

    if urls.is_empty() {
        return Err(LearnError::Acquire(format!(
            "yt-dlp returned no video entries for source {source:?}"
        )));
    }

    Ok(urls)
}

/// Ensure a URL from a flat-playlist entry is a fully-qualified watch URL.
/// If `url` is already absolute, return it. If it looks like a bare video id,
/// build the canonical watch URL.
fn normalise_video_url(url: String, id: &Option<String>) -> String {
    if url.starts_with("http") {
        return url;
    }
    // Bare 11-char YouTube id or relative path.
    if let Some(id) = id {
        return format!("https://www.youtube.com/watch?v={id}");
    }
    url
}

// ── yt-dlp info.json subset ───────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct InfoJson {
    id: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    uploader: Option<String>,
    #[serde(default)]
    channel_id: Option<String>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    upload_date: Option<String>, // YYYYMMDD
}

// ── public API ────────────────────────────────────────────────────────────────

/// Validate that `source` is a safe, recognisable input for yt-dlp.
///
/// Rejects anything starting with `-` (would be interpreted as a yt-dlp flag).
/// Accepts: http(s) URLs, yt-dlp pseudo-schemes (`ytsearch*:`), local file
/// paths that exist on disk, and YouTube channel handles (`@…`).
fn validate_source(source: &str) -> Result<()> {
    if source.starts_with('-') {
        return Err(LearnError::Acquire(format!(
            "source cannot start with '-' (would be interpreted as a yt-dlp flag): {source:?}"
        )));
    }
    let is_url = url::Url::parse(source).is_ok();
    let is_search = source.starts_with("ytsearch") || source.starts_with("ytsearchdate");
    let is_local = std::path::Path::new(source).exists();
    let is_handle = source.starts_with('@');
    if !(is_url || is_search || is_local || is_handle) {
        return Err(LearnError::Acquire(format!(
            "source does not match any known shape (URL, ytsearch:, local path, @handle): {source:?}"
        )));
    }
    Ok(())
}

/// Download captions (no audio/video) for `url` into `raw_dir`.
///
/// Shells out to `yt-dlp`. Success is defined by the presence of
/// `video.info.json` — yt-dlp may exit non-zero even when info was written.
///
/// `raw_dir` must be under `kb_root`; returns `Err(LearnError::Acquire)` if not.
pub async fn acquire_url(url: &str, kb_root: &Utf8Path, raw_dir: &Utf8Path) -> Result<Acquired> {
    validate_source(url)?;
    validate_raw_dir_under_kb_root(kb_root, raw_dir)?;
    fs::create_dir_all(raw_dir)?;

    run_ytdlp(url, raw_dir).await?;

    let info_path = raw_dir.join("video.info.json");
    let info = read_info_json(&info_path)?;

    // Single-video sources that aren't parseable http URLs (e.g. a local file
    // passed directly to acquire_url) get a synthetic file:// URL so VideoRef
    // retains a non-optional url field.
    let video_url = url::Url::parse(url)
        .or_else(|_| url::Url::from_file_path(url).map_err(|_| ()))
        .map_err(|_| LearnError::Acquire(format!("cannot construct URL for source {url:?}")))?;

    let captions_vtt = find_vtt(raw_dir);

    let video = VideoRef {
        video_id: info.id,
        url: video_url,
        title: info.title,
        channel: info.uploader,
        channel_id: info.channel_id,
        duration_seconds: info.duration,
        published_at: info.upload_date,
    };

    info!(video_id = %video.video_id, ?captions_vtt, "acquired");

    Ok(Acquired {
        video,
        captions_vtt,
        audio_mp3: None,
        raw_dir: raw_dir.to_owned(),
    })
}

// ── internals ─────────────────────────────────────────────────────────────────

/// Ensure `raw_dir` is underneath `kb_root`. Validates against the canonical
/// (resolved) form of `kb_root` when it exists; falls back to lexical
/// `starts_with` when it does not (e.g. in tests with non-existent dirs).
fn validate_raw_dir_under_kb_root(kb_root: &Utf8Path, raw_dir: &Utf8Path) -> Result<()> {
    // Use the existing raw_dir or its first existing ancestor for canonicalisation.
    let anchor = if raw_dir.exists() {
        raw_dir.to_owned()
    } else {
        // Walk up until we find a part that exists.
        let mut p: Utf8PathBuf = raw_dir.to_owned();
        loop {
            match p.parent() {
                Some(parent) if !parent.as_str().is_empty() => p = parent.to_owned(),
                _ => break,
            }
            if p.exists() {
                break;
            }
        }
        p
    };

    // Attempt canonical resolution; fall back to raw path.
    let root_canonical = kb_root
        .canonicalize_utf8()
        .unwrap_or_else(|_| kb_root.to_owned());
    let anchor_canonical = anchor
        .canonicalize_utf8()
        .unwrap_or_else(|_| anchor.to_owned());

    if !anchor_canonical.starts_with(&root_canonical) {
        return Err(LearnError::Acquire("raw_dir must be under kb_root".into()));
    }
    Ok(())
}

async fn run_ytdlp(url: &str, raw_dir: &Utf8Path) -> Result<()> {
    let output = Command::new("yt-dlp")
        .args([
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--write-info-json",
            "--sub-lang",
            "en,en-US,en-GB,en-orig",
            "--sub-format",
            "vtt",
            "-o",
            &format!("{}/video.%(ext)s", raw_dir),
            url,
        ])
        .output()
        .await
        .map_err(|e| LearnError::Acquire(format!("yt-dlp not found or failed to spawn: {e}")))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !stdout.is_empty() {
        info!(ytdlp.stdout = %stdout.trim());
    }
    if !stderr.is_empty() {
        warn!(ytdlp.stderr = %stderr.trim());
    }

    // yt-dlp may exit non-zero even when info.json was successfully written.
    // We accept that and check for the info file in the caller.
    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        warn!(
            exit_code = code,
            "yt-dlp exited non-zero; checking for info.json"
        );
    }

    Ok(())
}

fn read_info_json(path: &Utf8Path) -> Result<InfoJson> {
    let raw = fs::read_to_string(path).map_err(|e| {
        LearnError::Acquire(format!(
            "info.json not found at {path} — yt-dlp may have failed: {e}"
        ))
    })?;
    let info: InfoJson = serde_json::from_str(&raw)?;
    Ok(info)
}

/// Find the best VTT file in `dir`. Prefers `*.en.vtt`, falls back to any `.vtt`.
fn find_vtt(dir: &Utf8Path) -> Option<Utf8PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut best: Option<Utf8PathBuf> = None;
    for entry in entries.flatten() {
        let path = Utf8PathBuf::from_path_buf(entry.path()).ok()?;
        let name = path.file_name().unwrap_or("");
        if name.ends_with(".vtt") {
            if name.ends_with(".en.vtt") {
                return Some(path); // exact match; take it immediately
            }
            best = Some(path);
        }
    }
    best
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_vtt_prefers_en() {
        // Uses a real temp dir with fake files to test the selection logic.
        let dir = tempfile::tempdir().unwrap();
        let base = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        fs::write(base.join("video.fr.vtt"), "").unwrap();
        fs::write(base.join("video.en.vtt"), "").unwrap();
        let found = find_vtt(&base).unwrap();
        assert!(
            found.as_str().ends_with(".en.vtt"),
            "expected .en.vtt but got {found}"
        );
    }

    #[test]
    fn find_vtt_fallback_any() {
        let dir = tempfile::tempdir().unwrap();
        let base = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        fs::write(base.join("video.fr.vtt"), "").unwrap();
        let found = find_vtt(&base).unwrap();
        assert!(found.as_str().ends_with(".vtt"));
    }

    #[test]
    fn find_vtt_returns_none_when_empty() {
        let dir = tempfile::tempdir().unwrap();
        let base = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        assert!(find_vtt(&base).is_none());
    }

    #[tokio::test]
    async fn acquire_url_rejects_raw_dir_outside_kb_root() {
        let dir = tempfile::tempdir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        // "/tmp/outside" is not under the temp kb_root.
        let outside = Utf8PathBuf::from("/tmp/outside_kb_root_test");
        let result = acquire_url("https://example.com/video", &kb_root, &outside).await;
        assert!(
            matches!(result, Err(LearnError::Acquire(_))),
            "expected Err(LearnError::Acquire) but got: {result:?}"
        );
    }

    #[tokio::test]
    async fn acquire_url_accepts_raw_dir_inside_kb_root() {
        let dir = tempfile::tempdir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let inside = kb_root.join("_raw").join("test-topic");
        // This should pass validation (yt-dlp will fail but that's ok — we only
        // test the path guard here).
        let result = acquire_url("https://example.com/video", &kb_root, &inside).await;
        // Validation passes; yt-dlp is expected to fail in CI — not Acquire path error.
        if let Err(LearnError::Acquire(msg)) = &result {
            assert!(
                !msg.contains("raw_dir must be under kb_root"),
                "path guard incorrectly fired: {msg}"
            );
        }
    }

    // ── validate_source unit tests ─────────────────────────────────────────

    #[test]
    fn validate_source_accepts_http_url() {
        assert!(validate_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ").is_ok());
        assert!(validate_source("http://example.com/video").is_ok());
    }

    #[test]
    fn validate_source_accepts_ytsearch_pseudo_scheme() {
        assert!(validate_source("ytsearch20:rust async programming").is_ok());
        assert!(validate_source("ytsearchdate5:news today").is_ok());
    }

    #[test]
    fn validate_source_accepts_local_path() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("sample.mp4");
        std::fs::write(&file, b"").unwrap();
        let path_str = file.to_str().unwrap();
        assert!(
            validate_source(path_str).is_ok(),
            "existing local path should be accepted"
        );
    }

    #[test]
    fn validate_source_rejects_dash_prefix() {
        let err = validate_source("--malicious-flag").unwrap_err();
        assert!(
            matches!(err, LearnError::Acquire(_)),
            "expected LearnError::Acquire, got: {err:?}"
        );
        if let LearnError::Acquire(msg) = err {
            assert!(
                msg.contains("cannot start with"),
                "message should explain rejection: {msg}"
            );
        }
    }

    #[tokio::test]
    async fn acquire_url_rejects_dash_prefixed_source() {
        let dir = tempfile::tempdir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let raw_dir = kb_root.join("_raw").join("test");
        let result = acquire_url("--malicious-flag", &kb_root, &raw_dir).await;
        assert!(
            matches!(result, Err(LearnError::Acquire(_))),
            "expected Err(LearnError::Acquire) synchronously, got: {result:?}"
        );
        // Verify yt-dlp was NOT invoked: raw_dir should not have been created.
        assert!(
            !raw_dir.exists(),
            "raw_dir should not have been created — yt-dlp must not have been called"
        );
    }

    // ── classify_source tests ─────────────────────────────────────────────────

    #[test]
    fn classify_source_routes_correctly() {
        use SourceKind::*;
        let cases: &[(&str, SourceKind)] = &[
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", SingleVideo),
            ("https://youtu.be/dQw4w9WgXcQ", SingleVideo),
            ("http://example.com/video.mp4", SingleVideo),
            ("@mkbhd", Channel),
            ("https://www.youtube.com/@LinusTechTips/videos", Channel),
            (
                "https://www.youtube.com/channel/UCXzySgo3V9KysSfELFLMAeA",
                Channel,
            ),
            (
                "https://www.youtube.com/playlist?list=PLbpi6ZahtOH6Ar_3GPy3workX7N7a7hO4",
                Playlist,
            ),
            ("ytsearch5:rust async programming", Search),
            ("ytsearch20:news today", Search),
            ("ytsearchdate10:breaking news", Search),
        ];
        for (source, expected) in cases {
            let got = classify_source(source);
            assert_eq!(
                got, *expected,
                "classify_source({source:?}) should be {expected:?} but got {got:?}"
            );
        }
    }

    #[test]
    fn classify_source_local_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("clip.mp4");
        fs::write(&file, b"").unwrap();
        let path = file.to_str().unwrap();
        assert_eq!(classify_source(path), SourceKind::LocalFile);
    }

    #[test]
    fn classify_source_local_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        assert_eq!(classify_source(path), SourceKind::LocalDirectory);
    }

    // ── resolve_to_videos tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn resolve_to_videos_single_returns_one_url() {
        let url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";
        let result = resolve_to_videos(url, None).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], url);
    }

    #[tokio::test]
    async fn resolve_to_videos_local_file_returns_path() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("sample.mp4");
        fs::write(&file, b"").unwrap();
        let path = file.to_str().unwrap();
        let result = resolve_to_videos(path, None).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], path);
    }

    /// Requires real network and yt-dlp on PATH.
    /// Verifies that a ytsearch source calls yt-dlp with --flat-playlist and
    /// returns multiple video URLs (count matches the N in the prefix).
    #[tokio::test]
    #[ignore = "requires network and yt-dlp"]
    async fn resolve_to_videos_search_calls_yt_dlp_with_flat_playlist() {
        let source = "ytsearch3:rust programming language";
        let result = resolve_to_videos(source, Some(3)).await;
        assert!(result.is_ok(), "expected Ok but got: {result:?}");
        let urls = result.unwrap();
        assert!(!urls.is_empty(), "expected at least one URL from ytsearch");
        assert!(
            urls.len() <= 3,
            "expected at most 3 URLs (limit=3) but got {}",
            urls.len()
        );
        for url in &urls {
            assert!(
                url.starts_with("https://"),
                "each resolved URL should be absolute: {url}"
            );
        }
    }

    /// Network test — requires `yt-dlp` on PATH and internet access.
    #[tokio::test]
    #[ignore]
    async fn acquire_real_video() {
        let dir = tempfile::tempdir().unwrap();
        let base = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let raw_dir = base.join("_raw").join("test");
        let result = acquire_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            &base,
            &raw_dir,
        )
        .await;
        assert!(result.is_ok(), "{result:?}");
        let acq = result.unwrap();
        assert!(!acq.video.video_id.is_empty());
    }
}
