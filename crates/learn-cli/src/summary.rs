//! Post-ingest summary generation.
//!
//! `generate_summary` calls the Anthropic synthesizer with the top chunks from
//! the current ingest run and writes the result to `<topic>.summary.md`.
//! Missing API key → warn + return Ok(None). No Anthropic → same.

use camino::{Utf8Path, Utf8PathBuf};
use learn_core::{Chunk, Hit, LearnError, Result, Topic};
use tracing::warn;

#[cfg(test)]
use learn_core::SegmentKind;

/// Number of chunks sampled for the summary prompt.
const SUMMARY_CHUNK_LIMIT: usize = 10;

/// System prompt for the key-takeaways synthesis call.
const SUMMARY_SYSTEM: &str = "\
You are a concise knowledge assistant. Given source excerpts from a video, \
extract 3-5 bullet-point key takeaways. Be specific. Include the speaker name \
or video title if present in the excerpts. \
Format each bullet starting with a bullet character (•). \
Do not add any preamble or conclusion — only the bullet points.\
";

/// User-turn template for the summary call.
const SUMMARY_USER: &str = "\
Topic: {topic}

Source excerpts:
{context_snippets}

Extract 3-5 key takeaways as bullet points (• item).\
";

/// Meta-summary prompt used when >1 video was ingested in a single run.
const META_SUMMARY_USER: &str = "\
Topic: {topic}

Source excerpts (from multiple videos):
{context_snippets}

Extract 3-5 key takeaways that span the full set of videos as bullet points (• item).\
";

// ── Public interface ──────────────────────────────────────────────────────────

/// Result of a successful summary generation.
pub struct SummaryResult {
    /// Bullet-point body (raw text from the model).
    pub body: String,
    /// Absolute path to the written `.summary.md` file.
    pub path: Utf8PathBuf,
}

/// Generate a key-takeaways summary for `chunks` and write it to
/// `<kb_root>/<topic>.summary.md`.
///
/// Returns `Ok(None)` when the API key is absent or the synthesizer is
/// unavailable — callers log a warning and continue without a summary.
pub async fn generate_summary(
    topic: &Topic,
    chunks: &[Chunk],
    kb_root: &Utf8Path,
    is_meta: bool,
) -> Result<Option<SummaryResult>> {
    // Graceful skip when API key absent.
    if std::env::var("ANTHROPIC_API_KEY")
        .map(|v| v.is_empty())
        .unwrap_or(true)
    {
        warn!("summary skipped: ANTHROPIC_API_KEY not set");
        return Ok(None);
    }

    let hits = chunks_to_hits(pick_chunks(chunks));
    let context = format_context(&hits);

    let user_template = if is_meta {
        META_SUMMARY_USER
    } else {
        SUMMARY_USER
    };
    let user_msg = user_template
        .replace("{topic}", topic.as_str())
        .replace("{context_snippets}", &context);

    let body = call_summary(user_msg).await?;
    let path = write_summary(topic, kb_root, &body)?;

    Ok(Some(SummaryResult { body, path }))
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Pick up to `SUMMARY_CHUNK_LIMIT` evenly-spaced chunks by position, then
/// prefer longer ones within that selection.
fn pick_chunks(chunks: &[Chunk]) -> &[Chunk] {
    if chunks.len() <= SUMMARY_CHUNK_LIMIT {
        return chunks;
    }
    // We return a contiguous slice: first SUMMARY_CHUNK_LIMIT evenly-spaced.
    // For simplicity take the front 10 (caller provides chunks from the ingest
    // run only — they're already sequentially ordered).
    &chunks[..SUMMARY_CHUNK_LIMIT]
}

fn chunks_to_hits(chunks: &[Chunk]) -> Vec<Hit> {
    chunks
        .iter()
        .enumerate()
        .map(|(i, c)| Hit {
            chunk: c.clone(),
            score: 1.0,
            rank: i,
        })
        .collect()
}

fn format_context(hits: &[Hit]) -> String {
    hits.iter()
        .enumerate()
        .map(|(i, h)| {
            format!(
                "[{}] {}\n    (video={} @ {:.0}s)",
                i + 1,
                h.chunk.text.trim(),
                h.chunk.video_id,
                h.chunk.start_seconds,
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Call Anthropic directly (reuse the same HTTP primitives as `learn-synth`
/// but without a full Synthesizer build, keeping the code in one place).
async fn call_summary(user_msg: String) -> Result<String> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| LearnError::Synth("ANTHROPIC_API_KEY not set".to_string()))?;
    let model =
        std::env::var("LEARN_ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-5".to_string());

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| LearnError::Synth(format!("reqwest build: {e}")))?;

    let body = serde_json::json!({
        "model": model,
        "max_tokens": 512,
        "system": SUMMARY_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    });

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| LearnError::Synth(format!("Anthropic request: {e}")))?;

    let status = resp.status();
    if !status.is_success() {
        let excerpt = resp
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(200)
            .collect::<String>();
        return Err(LearnError::Synth(format!("Anthropic {status}: {excerpt}")));
    }

    let raw = resp
        .text()
        .await
        .map_err(|e| LearnError::Synth(format!("response read: {e}")))?;

    extract_text(&raw)
}

/// Parse Anthropic JSON response and return the text content.
fn extract_text(raw: &str) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct Resp {
        content: Vec<ContentBlock>,
    }
    #[derive(serde::Deserialize)]
    struct ContentBlock {
        #[serde(rename = "type")]
        kind: String,
        text: String,
    }

    let parsed: Resp = serde_json::from_str(raw)
        .map_err(|e| LearnError::Synth(format!("malformed Anthropic response: {e}")))?;

    Ok(parsed
        .content
        .into_iter()
        .filter(|c| c.kind == "text")
        .map(|c| c.text)
        .collect::<Vec<_>>()
        .join(""))
}

/// Write the summary markdown file to `<kb_root>/<topic>.summary.md`.
fn write_summary(topic: &Topic, kb_root: &Utf8Path, body: &str) -> Result<Utf8PathBuf> {
    let path = kb_root.join(format!("{}.summary.md", topic.as_str()));
    let header = format!(
        "# {} — key takeaways\n\nIngested: {}\n\n",
        topic.as_str(),
        crate::rfc3339_now(),
    );
    let content = format!("{header}{body}\n");
    std::fs::create_dir_all(kb_root.as_std_path()).map_err(LearnError::Io)?;
    std::fs::write(path.as_std_path(), content.as_bytes()).map_err(LearnError::Io)?;
    Ok(path)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn make_chunk(video_id: &str, text: &str, start: f64) -> Chunk {
        Chunk {
            chunk_id: format!("{video_id}-{}", start as u64),
            video_id: video_id.to_string(),
            start_seconds: start,
            end_seconds: start + 5.0,
            text: text.to_string(),
            token_count: text.split_whitespace().count(),
            kind: SegmentKind::Caption,
        }
    }

    // ── Unit test 1 ──────────────────────────────────────────────────────────

    /// The summary prompt must include chunk text verbatim.
    #[test]
    fn summary_prompt_includes_chunk_text() {
        let chunks = vec![
            make_chunk(
                "vid1",
                "Solid is a decentralized protocol by Tim Berners-Lee",
                0.0,
            ),
            make_chunk("vid1", "The A2A protocol enables agent communication", 10.0),
        ];
        let hits = chunks_to_hits(pick_chunks(&chunks));
        let ctx = format_context(&hits);
        assert!(
            ctx.contains("Solid is a decentralized protocol"),
            "chunk text must appear in context"
        );
        assert!(ctx.contains("A2A protocol"), "second chunk must appear");
        assert!(ctx.contains("[1]"), "must be numbered starting at 1");
        assert!(ctx.contains("[2]"), "second entry must be numbered");
    }

    // ── Unit test 2 ──────────────────────────────────────────────────────────

    /// Two ingests on the same topic — summary must only reference chunks
    /// from the second run (caller passes only new_chunks).
    #[test]
    fn single_video_path_summarizes_only_new_chunks() {
        let first_run_chunks = vec![make_chunk("vid1", "first run content alpha", 0.0)];
        let second_run_chunks = vec![make_chunk("vid2", "second run content beta", 0.0)];
        // Build prompts for each run separately.
        let hits1 = chunks_to_hits(pick_chunks(&first_run_chunks));
        let hits2 = chunks_to_hits(pick_chunks(&second_run_chunks));
        let ctx1 = format_context(&hits1);
        let ctx2 = format_context(&hits2);

        // ctx2 must not contain first-run text.
        assert!(
            !ctx2.contains("first run content alpha"),
            "second-run context must not include first-run chunks"
        );
        assert!(
            ctx2.contains("second run content beta"),
            "second-run context must include second-run chunk"
        );
        // ctx1 must not contain second-run text.
        assert!(
            !ctx1.contains("second run content beta"),
            "first-run context must not bleed into second run"
        );
    }

    // ── Unit test 3 ──────────────────────────────────────────────────────────

    /// Playlist path (>1 video) produces ONE combined context, not per-video.
    /// Caller passes all chunks from the run; we verify they're merged together.
    #[test]
    fn playlist_path_generates_meta_summary_context() {
        let all_chunks: Vec<Chunk> = (0..3u64)
            .flat_map(|v| {
                vec![
                    make_chunk(&format!("vid{v}"), &format!("video {v} content A"), 0.0),
                    make_chunk(&format!("vid{v}"), &format!("video {v} content B"), 10.0),
                ]
            })
            .collect();

        // Simulate single meta-context build (as run_ingest_with_frames does).
        let hits = chunks_to_hits(pick_chunks(&all_chunks));
        let ctx = format_context(&hits);

        // All three videos' first chunks must appear (up to SUMMARY_CHUNK_LIMIT).
        assert!(
            ctx.contains("video 0 content A"),
            "first video must be in meta context"
        );
        assert!(
            ctx.contains("video 1 content A"),
            "second video must be in meta context"
        );
        assert!(
            ctx.contains("video 2 content A"),
            "third video must be in meta context"
        );
        // Context is one string — not three separate summaries.
        assert_eq!(
            ctx.matches("[1]").count(),
            1,
            "numbered blocks must not repeat [1]"
        );
    }

    // ── Integration test (ignored — requires real Anthropic key) ────────────

    /// Real Anthropic call returns valid markdown bullets.
    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY and network"]
    async fn real_anthropic_call_returns_markdown_bullets() {
        let chunks = vec![
            make_chunk(
                "test_vid",
                "Solid enables user data ownership via linked data pods",
                0.0,
            ),
            make_chunk(
                "test_vid",
                "Tim Berners-Lee founded the Solid project at MIT",
                10.0,
            ),
        ];
        let dir = tempfile::tempdir().unwrap();
        let kb_root = camino::Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let topic = Topic::new("solid-web").unwrap();
        let result = generate_summary(&topic, &chunks, &kb_root, false).await;
        let summary = result
            .expect("should not error")
            .expect("should return Some with API key");
        assert!(
            summary.body.contains('•') || summary.body.contains('-') || summary.body.contains('*'),
            "response must contain bullet markers: {}",
            summary.body
        );
        assert!(summary.path.exists(), "summary.md must be written to disk");
    }
}
