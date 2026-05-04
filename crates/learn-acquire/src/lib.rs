//! `learn-acquire` — resolve a video URL to a local `Acquired` with captions.
//!
//! Phase 1: shells out to `yt-dlp --skip-download` to pull `.info.json` and
//! WebVTT captions, then builds a `VideoRef` + `Acquired` from the results.
//!
//! Non-video sources (PDF, podcast RSS, web page) are also handled here.

#![deny(unsafe_code)]

pub mod vtt;

use camino::{Utf8Path, Utf8PathBuf};
use learn_core::{Acquired, LearnError, Result, VideoRef};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use tokio::process::Command;
use tracing::{info, warn};

// ── source classification ─────────────────────────────────────────────────────

/// Coarse kind of a source string as understood by yt-dlp or native handlers.
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
    /// A URL whose path ends with `.pdf`, or a local file with `.pdf` extension.
    Pdf,
    /// A podcast RSS feed URL (known feed hosts or `.rss`/`.xml` path).
    PodcastRss,
    /// Any http/https URL not matched by earlier rules.
    WebPage,
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

        // Known video-hosting domains are always routed to yt-dlp as SingleVideo.
        // This check must come before the PDF/RSS/WebPage rules so YouTube URLs
        // (e.g. /watch?v=…, /shorts/…, /live/…) are never misclassified.
        let is_video_host = host.contains("youtube.com")
            || host == "youtu.be"
            || host.contains("vimeo.com")
            || host.contains("dailymotion.com")
            || host.contains("twitch.tv");
        if is_video_host {
            return SourceKind::SingleVideo;
        }

        // Only apply PDF/RSS/WebPage classification to http(s) URLs.
        if u.scheme() == "http" || u.scheme() == "https" {
            let path_lower = path.to_lowercase();

            // Direct video/audio file links are routed to yt-dlp as SingleVideo.
            let is_media_file = path_lower.ends_with(".mp4")
                || path_lower.ends_with(".mkv")
                || path_lower.ends_with(".webm")
                || path_lower.ends_with(".avi")
                || path_lower.ends_with(".mov")
                || path_lower.ends_with(".mp3")
                || path_lower.ends_with(".m4a")
                || path_lower.ends_with(".ogg")
                || path_lower.ends_with(".flac");
            if is_media_file {
                return SourceKind::SingleVideo;
            }

            // PDF: URL path ends with .pdf (case-insensitive).
            if path_lower.ends_with(".pdf") {
                return SourceKind::Pdf;
            }

            // Podcast RSS: known feed hosts or path ends with .rss / .xml.
            let is_feed_host = host.starts_with("feeds.")
                || host == "anchor.fm"
                || host.ends_with(".anchor.fm")
                || host == "buzzsprout.com"
                || host.ends_with(".buzzsprout.com")
                || host == "libsyn.com"
                || host.ends_with(".libsyn.com")
                || host == "podbean.com"
                || host.ends_with(".podbean.com")
                || host == "simplecast.com"
                || host.ends_with(".simplecast.com")
                || host == "transistor.fm"
                || host.ends_with(".transistor.fm")
                || host == "spreaker.com"
                || host.ends_with(".spreaker.com");
            let is_feed_path = path_lower.ends_with(".rss") || path_lower.ends_with(".xml");
            if is_feed_host || is_feed_path {
                return SourceKind::PodcastRss;
            }

            // Any other http(s) URL falls through to WebPage.
            return SourceKind::WebPage;
        }
    }

    // Anything else (unknown URL scheme — treat as single-video for yt-dlp).
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
        SourceKind::SingleVideo
        | SourceKind::LocalFile
        | SourceKind::LocalDirectory
        | SourceKind::Pdf
        | SourceKind::PodcastRss
        | SourceKind::WebPage => Ok(vec![source.to_owned()]),
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

/// Perform a HEAD probe on an `http(s)` URL with a 5-second timeout.
///
/// Returns `Ok(())` if the server responds with a 2xx or 3xx status code.
/// Returns `Err(LearnError::Acquire)` for 4xx/5xx or connection errors.
///
/// Non-http(s) sources (local paths, `ytsearch:`, `@handles`) are skipped — the
/// function returns `Ok(())` immediately without touching the network.
pub async fn probe_http_url(source: &str) -> Result<()> {
    // Only probe plain http/https URLs.
    let parsed = match url::Url::parse(source) {
        Ok(u) if u.scheme() == "http" || u.scheme() == "https" => u,
        _ => return Ok(()), // local path, ytsearch:, @handle — skip
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| LearnError::Acquire(format!("could not build HTTP client: {e}")))?;

    let response = client.head(parsed.as_str()).send().await.map_err(|e| {
        LearnError::Acquire(format!(
            "could not fetch URL: connection error — {e}. \
                 Check the URL for typos."
        ))
    })?;

    let status = response.status();
    if status.is_success() || status.is_redirection() {
        Ok(())
    } else {
        Err(LearnError::Acquire(format!(
            "could not fetch URL: {} {}. Did you mean a different URL?",
            status.as_u16(),
            status.canonical_reason().unwrap_or("Unknown")
        )))
    }
}

/// Check whether `raw_dir` already contains data from a video that is NOT in
/// `known_video_ids` for this topic.
///
/// Reads `<raw_dir>/video.info.json` (if it exists) and checks its `id` field
/// against `known_video_ids`. If the cached video_id is unknown to the topic's
/// manifest, the directory is being squatted by an unrelated video and we
/// refuse with a descriptive error.
///
/// Returns `Ok(())` when:
/// - `video.info.json` does not exist yet (clean directory)
/// - `video.info.json` exists and the stored `id` is in `known_video_ids`
///   (legitimate multi-video topic or resume of the same video)
pub fn check_slug_collision(
    raw_dir: &Utf8Path,
    slug: &str,
    known_video_ids: &std::collections::BTreeSet<String>,
) -> Result<()> {
    let info_path = raw_dir.join("video.info.json");
    if !info_path.exists() {
        return Ok(());
    }

    let raw = fs::read_to_string(&info_path).map_err(|e| {
        LearnError::Acquire(format!(
            "could not read existing info.json at {info_path}: {e}"
        ))
    })?;

    #[derive(serde::Deserialize)]
    struct IdOnly {
        id: String,
    }

    let existing: IdOnly = serde_json::from_str(&raw).map_err(|e| {
        LearnError::Acquire(format!(
            "could not parse existing info.json at {info_path}: {e}"
        ))
    })?;

    if !known_video_ids.contains(&existing.id) {
        return Err(LearnError::Acquire(format!(
            "topic '{slug}' already has cached data from video '{}'. \
             To replace, run `learn forget {slug}` first. \
             To add to it, the slug must match the existing topic's purpose.",
            existing.id
        )));
    }

    Ok(())
}

/// Download captions (and optionally the video file) for `url` into `raw_dir`.
///
/// Shells out to `yt-dlp`. Success is defined by the presence of
/// `video.info.json` — yt-dlp may exit non-zero even when info was written.
///
/// When `download_video` is `true` the video is downloaded at low resolution
/// (≤480p) so that frame extraction can run. When `false` only captions and
/// metadata are fetched (`--skip-download`).
///
/// `raw_dir` must be under `kb_root`; returns `Err(LearnError::Acquire)` if not.
pub async fn acquire_url(
    url: &str,
    kb_root: &Utf8Path,
    raw_dir: &Utf8Path,
    download_video: bool,
) -> Result<Acquired> {
    validate_source(url)?;
    // HEAD probe: fail fast for unreachable http(s) URLs before touching disk.
    probe_http_url(url).await?;
    validate_raw_dir_under_kb_root(kb_root, raw_dir)?;
    fs::create_dir_all(raw_dir)?;

    run_ytdlp(url, raw_dir, download_video).await?;

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

/// Build the yt-dlp argument list without spawning a process.
///
/// Extracted into a pure function so it can be unit-tested without network access.
pub fn build_ytdlp_args(url: &str, raw_dir: &Utf8Path, download_video: bool) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();

    if !download_video {
        args.push("--skip-download".into());
    } else {
        // Low-res download to keep bandwidth manageable; prefer mp4.
        args.push("-f".into());
        args.push("best[height<=480][ext=mp4]/best[height<=480]/best".into());
    }

    args.extend([
        "--write-subs".into(),
        "--write-auto-subs".into(),
        "--write-info-json".into(),
        "--sub-lang".into(),
        "en,en-US,en-GB,en-orig".into(),
        "--sub-format".into(),
        "vtt".into(),
        "-o".into(),
        format!("{}/video.%(ext)s", raw_dir),
        url.to_owned(),
    ]);

    args
}

async fn run_ytdlp(url: &str, raw_dir: &Utf8Path, download_video: bool) -> Result<()> {
    let args = build_ytdlp_args(url, raw_dir, download_video);

    let output = Command::new("yt-dlp")
        .args(&args)
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

// ── non-video acquisition ─────────────────────────────────────────────────────

/// Compute a 12-character hex ID from the SHA-256 of an arbitrary string.
fn short_id(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let digest = hasher.finalize();
    hex::encode(&digest[..6]) // 6 bytes → 12 hex chars
}

/// Write a pseudo-VTT file where each paragraph is assigned a 30-second slot.
///
/// The timestamps are synthetic — the content is what matters for chunking/embedding.
fn write_pseudo_vtt(paragraphs: &[String], path: &Utf8Path) -> Result<()> {
    let mut lines: Vec<String> = vec!["WEBVTT".into(), String::new()];
    for (i, para) in paragraphs.iter().enumerate() {
        let start = i as u64 * 30;
        let end = start + 30;
        lines.push(format!(
            "00:{:02}:{:02}.000 --> 00:{:02}:{:02}.000",
            start / 60,
            start % 60,
            end / 60,
            end % 60
        ));
        lines.push(para.clone());
        lines.push(String::new());
    }
    fs::write(path, lines.join("\n"))?;
    Ok(())
}

/// Split text into non-empty paragraphs of at least `min_chars` characters.
fn split_paragraphs(text: &str, min_chars: usize) -> Vec<String> {
    text.split("\n\n")
        .map(|p| p.trim().to_owned())
        .filter(|p| p.len() >= min_chars)
        .collect()
}

/// Acquire a PDF source: extract text via `pdftotext` subprocess and write a
/// pseudo-VTT file.
pub async fn acquire_pdf(url_or_path: &str, raw_dir: &Utf8Path) -> Result<Acquired> {
    fs::create_dir_all(raw_dir)?;

    let id = short_id(url_or_path);

    // Fetch to a local file if the source is a URL.
    let local_path: Utf8PathBuf;
    let needs_download = url_or_path.starts_with("http://") || url_or_path.starts_with("https://");

    if needs_download {
        let dest = raw_dir.join(format!("{id}.pdf"));
        download_binary(url_or_path, &dest).await?;
        local_path = dest;
    } else {
        local_path = Utf8PathBuf::from(url_or_path);
    }

    let text = extract_pdf_text(&local_path)?;
    let paragraphs = split_paragraphs(&text, 50);

    let vtt_path = raw_dir.join(format!("{id}.vtt"));
    write_pseudo_vtt(&paragraphs, &vtt_path)?;

    let source_url = if needs_download {
        url::Url::parse(url_or_path)
            .map_err(|e| LearnError::Acquire(format!("invalid PDF URL: {e}")))?
    } else {
        url::Url::from_file_path(url_or_path).map_err(|_| {
            LearnError::Acquire(format!("cannot build file URL for {url_or_path:?}"))
        })?
    };

    let title = source_url
        .path_segments()
        .and_then(|mut s| s.next_back())
        .map(str::to_owned);

    let video = VideoRef {
        video_id: id,
        url: source_url,
        title,
        channel: None,
        channel_id: None,
        duration_seconds: None,
        published_at: None,
    };

    info!(video_id = %video.video_id, vtt = %vtt_path, "acquired PDF");

    Ok(Acquired {
        video,
        captions_vtt: Some(vtt_path),
        audio_mp3: None,
        raw_dir: raw_dir.to_owned(),
    })
}

/// Extract text from a PDF file. Tries `pdftotext` subprocess first; logs a
/// warning and returns an error if not found (no pure-Rust fallback to keep
/// the dependency surface small — add `pdf-extract` if needed later).
fn extract_pdf_text(path: &Utf8Path) -> Result<String> {
    let output = std::process::Command::new("pdftotext")
        .args(["-layout", path.as_str(), "-"])
        .output();

    match output {
        Ok(o) if o.status.success() => String::from_utf8(o.stdout)
            .map_err(|e| LearnError::Acquire(format!("pdftotext output is not valid UTF-8: {e}"))),
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            Err(LearnError::Acquire(format!(
                "pdftotext exited non-zero for {path}: {stderr}"
            )))
        }
        Err(e) => {
            warn!("pdftotext not found or failed to spawn ({e}); PDF acquisition unavailable");
            Err(LearnError::Acquire(format!(
                "pdftotext not available — install poppler-utils to ingest PDFs: {e}"
            )))
        }
    }
}

/// Download a binary URL to a local file using `reqwest`.
async fn download_binary(url: &str, dest: &Utf8Path) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| LearnError::Acquire(format!("could not build HTTP client: {e}")))?;

    let bytes = client
        .get(url)
        .send()
        .await
        .map_err(|e| LearnError::Acquire(format!("could not download {url}: {e}")))?
        .bytes()
        .await
        .map_err(|e| {
            LearnError::Acquire(format!("could not read response body from {url}: {e}"))
        })?;

    fs::write(dest, &bytes)?;
    Ok(())
}

/// Acquire a podcast RSS feed: fetch the XML, find the first audio enclosure,
/// download the MP3 via yt-dlp, and return an `Acquired` with `audio_mp3` set.
pub async fn acquire_podcast(rss_url: &str, raw_dir: &Utf8Path) -> Result<Acquired> {
    fs::create_dir_all(raw_dir)?;

    let id = short_id(rss_url);

    // Fetch the RSS XML.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| LearnError::Acquire(format!("could not build HTTP client: {e}")))?;

    let xml_text = client
        .get(rss_url)
        .send()
        .await
        .map_err(|e| LearnError::Acquire(format!("could not fetch RSS feed {rss_url}: {e}")))?
        .text()
        .await
        .map_err(|e| LearnError::Acquire(format!("could not read RSS body: {e}")))?;

    // Parse with roxmltree.
    let doc = roxmltree::Document::parse(&xml_text)
        .map_err(|e| LearnError::Acquire(format!("could not parse RSS XML: {e}")))?;

    // Extract channel title (first <title> child of <channel>).
    let channel_title = doc
        .root_element()
        .children()
        .find(|n| n.has_tag_name("channel"))
        .and_then(|ch| {
            ch.children()
                .find(|n| n.has_tag_name("title"))
                .and_then(|t| t.text())
                .map(str::to_owned)
        });

    // Find first <enclosure> with an audio type attribute.
    let audio_url = doc
        .descendants()
        .find(|n| {
            n.has_tag_name("enclosure")
                && n.attribute("type")
                    .map(|t| t.contains("audio"))
                    .unwrap_or(false)
        })
        .and_then(|enc| enc.attribute("url"))
        .map(str::to_owned)
        .ok_or_else(|| {
            LearnError::Acquire(format!("no audio enclosure found in RSS feed: {rss_url}"))
        })?;

    info!(rss_url, audio_url = %audio_url, "found podcast audio enclosure");

    // Download via yt-dlp.
    let mp3_path = raw_dir.join(format!("{id}.mp3"));
    let output = Command::new("yt-dlp")
        .args([
            "-x",
            "--audio-format",
            "mp3",
            "-o",
            mp3_path.as_str(),
            audio_url.as_str(),
        ])
        .output()
        .await
        .map_err(|e| LearnError::Acquire(format!("yt-dlp not found or failed to spawn: {e}")))?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        warn!(ytdlp.stderr = %stderr.trim());
    }
    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        warn!(exit_code = code, "yt-dlp exited non-zero for podcast audio");
    }

    let rss_parsed = url::Url::parse(rss_url)
        .map_err(|e| LearnError::Acquire(format!("invalid RSS URL: {e}")))?;

    let video = VideoRef {
        video_id: id,
        url: rss_parsed,
        title: channel_title,
        channel: None,
        channel_id: None,
        duration_seconds: None,
        published_at: None,
    };

    info!(video_id = %video.video_id, ?mp3_path, "acquired podcast");

    Ok(Acquired {
        video,
        captions_vtt: None,
        audio_mp3: Some(mp3_path),
        raw_dir: raw_dir.to_owned(),
    })
}

/// Acquire a web page: fetch HTML, extract text from article/main/p tags,
/// write a pseudo-VTT file, and return an `Acquired` with `captions_vtt` set.
pub async fn acquire_webpage(url: &str, raw_dir: &Utf8Path) -> Result<Acquired> {
    fs::create_dir_all(raw_dir)?;

    let id = short_id(url);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| LearnError::Acquire(format!("could not build HTTP client: {e}")))?;

    let html = client
        .get(url)
        .header("User-Agent", "learn-acquire/0.1")
        .send()
        .await
        .map_err(|e| LearnError::Acquire(format!("could not fetch {url}: {e}")))?
        .text()
        .await
        .map_err(|e| {
            LearnError::Acquire(format!("could not read response body from {url}: {e}"))
        })?;

    let text = extract_webpage_text(&html);
    let paragraphs = split_paragraphs(&text, 50);

    let vtt_path = raw_dir.join(format!("{id}.vtt"));
    write_pseudo_vtt(&paragraphs, &vtt_path)?;

    let parsed =
        url::Url::parse(url).map_err(|e| LearnError::Acquire(format!("invalid URL: {e}")))?;

    let title = parsed
        .path_segments()
        .and_then(|mut s| s.next_back())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .or_else(|| parsed.host_str().map(str::to_owned));

    let video = VideoRef {
        video_id: id,
        url: parsed,
        title,
        channel: None,
        channel_id: None,
        duration_seconds: None,
        published_at: None,
    };

    info!(video_id = %video.video_id, vtt = %vtt_path, "acquired web page");

    Ok(Acquired {
        video,
        captions_vtt: Some(vtt_path),
        audio_mp3: None,
        raw_dir: raw_dir.to_owned(),
    })
}

/// Extract visible text from HTML using the `scraper` crate.
///
/// Targets `<article>`, `<main>`, and `<p>` elements; skips `<script>` and
/// `<style>` content. Falls back to all `<p>` text if no `<article>`/`<main>`
/// content is found.
fn extract_webpage_text(html: &str) -> String {
    use scraper::{Html, Selector};

    let document = Html::parse_document(html);

    // Try semantic containers first.
    let container_sel = Selector::parse("article, main").expect("valid selector");
    let mut texts: Vec<String> = document
        .select(&container_sel)
        .flat_map(|el| el.text())
        .map(|t| t.trim().to_owned())
        .filter(|t| !t.is_empty())
        .collect();

    // Fall back to all <p> tags if no article/main found.
    if texts.is_empty() {
        let p_sel = Selector::parse("p").expect("valid selector");
        texts = document
            .select(&p_sel)
            .flat_map(|el| el.text())
            .map(|t| t.trim().to_owned())
            .filter(|t| !t.is_empty())
            .collect();
    }

    texts.join("\n\n")
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
        let result = acquire_url("https://example.com/video", &kb_root, &outside, false).await;
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
        let result = acquire_url("https://example.com/video", &kb_root, &inside, false).await;
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
        let result = acquire_url("--malicious-flag", &kb_root, &raw_dir, false).await;
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

    // ── new source kind tests ─────────────────────────────────────────────────

    #[test]
    fn classify_source_pdf_url() {
        assert_eq!(
            classify_source("https://example.com/paper.pdf"),
            SourceKind::Pdf
        );
    }

    #[test]
    fn classify_source_podcast_rss_buzzsprout() {
        assert_eq!(
            classify_source("https://feeds.buzzsprout.com/123.rss"),
            SourceKind::PodcastRss
        );
    }

    #[test]
    fn classify_source_webpage() {
        assert_eq!(
            classify_source("https://example.com/article"),
            SourceKind::WebPage
        );
    }

    /// YouTube detection must still win over the new WebPage fallback.
    #[test]
    fn classify_source_youtube_still_wins() {
        assert_eq!(
            classify_source("https://www.youtube.com/watch?v=abc"),
            SourceKind::SingleVideo
        );
    }

    #[test]
    fn classify_source_http_plain_url_is_webpage() {
        assert_eq!(
            classify_source("http://example.com/article"),
            SourceKind::WebPage
        );
    }

    #[test]
    fn classify_source_feed_host_is_podcast() {
        assert_eq!(
            classify_source("https://feeds.simplecast.com/abc123"),
            SourceKind::PodcastRss
        );
    }

    #[test]
    fn classify_source_rss_extension_is_podcast() {
        assert_eq!(
            classify_source("https://example.com/mypodcast.rss"),
            SourceKind::PodcastRss
        );
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

    // ── build_ytdlp_args unit tests ────────────────────────────────────────────

    #[test]
    fn ytdlp_args_skip_download_when_frames_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let raw_dir = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let args = build_ytdlp_args("https://example.com/v", &raw_dir, false);
        assert!(
            args.contains(&"--skip-download".to_string()),
            "should include --skip-download when download_video=false; args={args:?}"
        );
        assert!(
            !args.iter().any(|a| a == "-f"),
            "should not include -f format selector when download_video=false; args={args:?}"
        );
    }

    #[test]
    fn ytdlp_args_downloads_video_when_frames_enabled() {
        let dir = tempfile::tempdir().unwrap();
        let raw_dir = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let args = build_ytdlp_args("https://example.com/v", &raw_dir, true);
        assert!(
            !args.contains(&"--skip-download".to_string()),
            "should NOT include --skip-download when download_video=true; args={args:?}"
        );
        let f_pos = args.iter().position(|a| a == "-f");
        assert!(
            f_pos.is_some(),
            "should include -f format selector when download_video=true; args={args:?}"
        );
        let format_val = &args[f_pos.unwrap() + 1];
        assert!(
            format_val.contains("height<=480"),
            "format selector should cap resolution at 480p; got={format_val}"
        );
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
            false,
        )
        .await;
        assert!(result.is_ok(), "{result:?}");
        let acq = result.unwrap();
        assert!(!acq.video.video_id.is_empty());
    }

    // ── probe_http_url tests ──────────────────────────────────────────────────

    /// Non-http sources (local path, ytsearch:, @handle) skip the probe and
    /// return Ok(()) immediately.
    #[tokio::test]
    async fn probe_skips_non_http_sources() {
        // ytsearch: pseudo-scheme — not http, must be skipped
        assert!(
            probe_http_url("ytsearch5:rust programming").await.is_ok(),
            "ytsearch: should be skipped (no network probe)"
        );
        // @handle — not a URL at all
        assert!(
            probe_http_url("@mkbhd").await.is_ok(),
            "@handle should be skipped (no network probe)"
        );
    }

    /// A known-good URL (example.com returns 200) must return Ok(()).
    /// Requires network. Marked ignore so CI doesn't depend on it.
    #[tokio::test]
    #[ignore = "requires network"]
    async fn probe_succeeds_for_reachable_url() {
        assert!(
            probe_http_url("https://www.example.com/").await.is_ok(),
            "probe should return Ok for a reachable URL"
        );
    }

    /// A URL on a guaranteed-unreachable host (localhost:19999) must return Err.
    #[tokio::test]
    async fn probe_fails_for_unreachable_host() {
        let result = probe_http_url("http://127.0.0.1:19999/nonexistent").await;
        assert!(
            result.is_err(),
            "expected Err for unreachable host, got: {result:?}"
        );
        if let Err(LearnError::Acquire(msg)) = result {
            assert!(
                msg.contains("could not fetch URL"),
                "error message should say 'could not fetch URL'; got: {msg}"
            );
        }
    }

    // ── check_slug_collision tests ────────────────────────────────────────────

    /// Clean directory (no video.info.json) → no collision.
    #[test]
    fn slug_collision_clean_dir_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let base = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let known = std::collections::BTreeSet::new();
        assert!(
            check_slug_collision(&base, "my-topic", &known).is_ok(),
            "clean directory should not trigger collision"
        );
    }

    /// video.info.json exists and its id IS in known_video_ids → no collision.
    #[test]
    fn slug_collision_known_video_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let base = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        fs::write(
            base.join("video.info.json"),
            r#"{"id":"abc123","title":"Test"}"#,
        )
        .unwrap();
        let mut known = std::collections::BTreeSet::new();
        known.insert("abc123".to_string());
        assert!(
            check_slug_collision(&base, "my-topic", &known).is_ok(),
            "known video id should not trigger collision"
        );
    }

    /// video.info.json exists but its id is NOT in known_video_ids → collision error.
    #[test]
    fn slug_collision_unknown_video_is_err() {
        let dir = tempfile::tempdir().unwrap();
        let base = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        fs::write(
            base.join("video.info.json"),
            r#"{"id":"squatter_video","title":"Other Video"}"#,
        )
        .unwrap();
        let known: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let result = check_slug_collision(&base, "my-topic", &known);
        assert!(
            result.is_err(),
            "unknown cached video id should trigger collision"
        );
        if let Err(LearnError::Acquire(msg)) = result {
            assert!(
                msg.contains("squatter_video"),
                "error should name the conflicting video id; got: {msg}"
            );
            assert!(
                msg.contains("learn forget"),
                "error should suggest 'learn forget'; got: {msg}"
            );
        }
    }

    /// Malformed video.info.json → returns Acquire error (not a panic).
    #[test]
    fn slug_collision_malformed_info_json_returns_err() {
        let dir = tempfile::tempdir().unwrap();
        let base = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        fs::write(base.join("video.info.json"), b"not valid json").unwrap();
        let known: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let result = check_slug_collision(&base, "my-topic", &known);
        assert!(
            matches!(result, Err(LearnError::Acquire(_))),
            "malformed info.json should return LearnError::Acquire, got: {result:?}"
        );
    }

    // ── helper unit tests ─────────────────────────────────────────────────────

    #[test]
    fn short_id_is_12_hex_chars() {
        let id = short_id("https://example.com/test");
        assert_eq!(id.len(), 12, "short_id must be 12 hex chars; got {id:?}");
        assert!(
            id.chars().all(|c| c.is_ascii_hexdigit()),
            "short_id must be hex; got {id:?}"
        );
    }

    #[test]
    fn short_id_is_deterministic() {
        let a = short_id("hello");
        let b = short_id("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn split_paragraphs_filters_short() {
        let text =
            "Short.\n\nThis is a long enough paragraph that exceeds fifty characters easily.";
        let paras = split_paragraphs(text, 50);
        assert_eq!(
            paras.len(),
            1,
            "short paragraph must be filtered; got {paras:?}"
        );
        assert!(paras[0].contains("long enough"));
    }

    #[test]
    fn write_pseudo_vtt_creates_webvtt_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join("out.vtt")).unwrap();
        let paras = vec![
            "First paragraph.".to_owned(),
            "Second paragraph.".to_owned(),
        ];
        write_pseudo_vtt(&paras, &path).unwrap();
        let content = fs::read_to_string(&path).unwrap();
        assert!(
            content.starts_with("WEBVTT"),
            "VTT must start with WEBVTT header"
        );
        assert!(
            content.contains("First paragraph."),
            "VTT must contain first para"
        );
        assert!(
            content.contains("Second paragraph."),
            "VTT must contain second para"
        );
        assert!(content.contains("-->"), "VTT must contain timestamp arrows");
    }

    #[test]
    fn extract_webpage_text_extracts_paragraph_text() {
        let html = r#"<html><body><p>Hello world.</p><p>Second paragraph.</p></body></html>"#;
        let text = extract_webpage_text(html);
        assert!(
            text.contains("Hello world."),
            "must contain p text; got: {text:?}"
        );
        assert!(
            text.contains("Second paragraph."),
            "must contain second p; got: {text:?}"
        );
    }

    #[test]
    fn extract_webpage_text_prefers_article() {
        let html = r#"<html><body><article><p>Article content here.</p></article><p>Sidebar</p></body></html>"#;
        let text = extract_webpage_text(html);
        assert!(
            text.contains("Article content here."),
            "must extract article content; got: {text:?}"
        );
    }
}
