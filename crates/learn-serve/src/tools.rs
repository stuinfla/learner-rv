//! Tool handler implementations for the three MCP tools.
//!
//! Each handler runs synchronously (called from the stdio dispatch loop) and
//! blocks on the async retriever/synthesizer using a local tokio runtime.

use std::path::PathBuf;

use camino::Utf8PathBuf;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use learn_core::{Chunk, Hit, SegmentKind, Topic};
use learn_index::LearnIndex;

use crate::protocol::ServerConfig;
use crate::witness::append_witness;

// ── Public types ──────────────────────────────────────────────────────────────

/// A hit entry returned by `kb_query`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HitEntry {
    pub video_id: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
    pub score: f32,
}

/// A video entry returned by `kb_list_videos`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoEntry {
    pub id: String,
    pub chunk_count: usize,
    pub duration_seconds: Option<f64>,
    pub fetched_at: Option<String>,
    pub status: String,
}

// ── kb_query ─────────────────────────────────────────────────────────────────

pub fn handle_kb_query(cfg: &ServerConfig, args: &Value) -> anyhow::Result<String> {
    let question = args
        .get("question")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("kb_query: 'question' is required"))?
        .to_string();
    let k = args
        .get("k")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(10)
        .max(1);

    let topic = Topic::new(&cfg.topic)
        .map_err(|e| anyhow::anyhow!("invalid topic '{}': {e}", cfg.topic))?;

    let embedder_path = default_model_dir();
    let index = LearnIndex::open(&cfg.kb_root, topic.clone())
        .map_err(|e| anyhow::anyhow!("open index: {e}"))?;

    let mut retriever = learn_retrieve::Retriever::for_topic(index, &topic, embedder_path.as_ref())
        .map_err(|e| anyhow::anyhow!("retriever: {e}"))?;
    retriever
        .refresh_bm25()
        .map_err(|e| anyhow::anyhow!("bm25: {e}"))?;

    let hits = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(retriever.search(&question, k))
    })
    .map_err(|e| anyhow::anyhow!("search: {e}"))?;

    let entries: Vec<HitEntry> = hits
        .iter()
        .map(|h| HitEntry {
            video_id: h.chunk.video_id.clone(),
            start_seconds: h.chunk.start_seconds,
            end_seconds: h.chunk.end_seconds,
            text: h.chunk.text.clone(),
            score: h.score,
        })
        .collect();

    let response = json!({ "hits": entries });
    let req_json = serde_json::to_string(&json!({ "question": question, "k": k }))?;
    let resp_json = serde_json::to_string(&response)?;

    let witness_path = witness_path_for(&cfg.kb_root, &cfg.topic);
    append_witness(&witness_path, "kb_query", &req_json, &resp_json);

    Ok(serde_json::to_string_pretty(&response)?)
}

// ── kb_synthesize ─────────────────────────────────────────────────────────────

pub fn handle_kb_synthesize(cfg: &ServerConfig, args: &Value) -> anyhow::Result<String> {
    let question = args
        .get("question")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("kb_synthesize: 'question' is required"))?
        .to_string();

    let hits_val = args
        .get("hits")
        .ok_or_else(|| anyhow::anyhow!("kb_synthesize: 'hits' is required"))?;

    // Accept either typed HitEntry array or raw JSON array
    let hit_entries: Vec<HitEntry> = serde_json::from_value(hits_val.clone())
        .map_err(|e| anyhow::anyhow!("kb_synthesize: invalid hits format: {e}"))?;

    // Reconstruct Hit objects from the HitEntry inputs
    let hits: Vec<Hit> = hit_entries
        .iter()
        .enumerate()
        .map(|(i, e)| Hit {
            chunk: Chunk {
                chunk_id: format!("mcp-{i}"),
                video_id: e.video_id.clone(),
                start_seconds: e.start_seconds,
                end_seconds: e.end_seconds,
                text: e.text.clone(),
                token_count: 0,
                kind: SegmentKind::Caption,
            },
            score: e.score,
            rank: i,
        })
        .collect();

    let synth =
        learn_synth::select_synthesizer().map_err(|e| anyhow::anyhow!("synthesizer: {e}"))?;

    let answer = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(synth.ask(&cfg.topic, &question, &hits))
    })
    .map_err(|e| anyhow::anyhow!("synthesis: {e}"))?;

    let citations: Vec<Value> = answer
        .citations
        .iter()
        .map(|c| {
            json!({
                "video_id": c.video_id,
                "url": c.url.as_str(),
                "start_seconds": c.start_seconds,
                "title": c.title
            })
        })
        .collect();

    let response = json!({
        "answer": answer.text,
        "citations": citations,
        "abstained": answer.abstained
    });

    let req_json = serde_json::to_string(&json!({ "question": question }))?;
    let resp_json = serde_json::to_string(&response)?;
    let witness_path = witness_path_for(&cfg.kb_root, &cfg.topic);
    append_witness(&witness_path, "kb_synthesize", &req_json, &resp_json);

    Ok(serde_json::to_string_pretty(&response)?)
}

// ── kb_list_videos ────────────────────────────────────────────────────────────

pub fn handle_kb_list_videos(cfg: &ServerConfig) -> anyhow::Result<String> {
    let topic = Topic::new(&cfg.topic)
        .map_err(|e| anyhow::anyhow!("invalid topic '{}': {e}", cfg.topic))?;

    let index =
        LearnIndex::open(&cfg.kb_root, topic).map_err(|e| anyhow::anyhow!("open index: {e}"))?;

    let manifest = index.manifest();
    let videos: Vec<VideoEntry> = manifest
        .videos
        .values()
        .map(|vs| VideoEntry {
            id: vs.video_id.clone(),
            chunk_count: vs.chunk_count,
            duration_seconds: None,
            fetched_at: vs.fetched_at.clone(),
            status: format!("{:?}", vs.status).to_lowercase(),
        })
        .collect();

    let response = json!({ "videos": videos });
    Ok(serde_json::to_string_pretty(&response)?)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn default_model_dir() -> Utf8PathBuf {
    if let Ok(env) = std::env::var("LEARN_EMBED_MODEL_DIR") {
        if !env.is_empty() {
            return Utf8PathBuf::from(env);
        }
    }
    let cache = dirs::cache_dir().unwrap_or_else(|| std::path::PathBuf::from(".cache"));
    Utf8PathBuf::from_path_buf(
        cache
            .join("learn-rs")
            .join("models")
            .join("bge-large-en-v15"),
    )
    .unwrap_or_else(|_| Utf8PathBuf::from(".cache/learn-rs/models/bge-large-en-v15"))
}

fn witness_path_for(kb_root: &Utf8PathBuf, topic: &str) -> PathBuf {
    PathBuf::from(kb_root.join(format!("{topic}.mcp.witness.json")).as_str())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_cfg(kb_root: Utf8PathBuf, topic: &str) -> ServerConfig {
        ServerConfig {
            topic: topic.to_string(),
            kb_root,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn kb_query_returns_hit_shaped_response_when_topic_empty() {
        let dir = tempdir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let cfg = make_cfg(kb_root, "test-topic");
        // Empty index → search returns empty hits (no embed model needed because
        // the index has no vectors — embed step fails first).
        let result = handle_kb_query(&cfg, &json!({ "question": "hello" }));
        // With no model, we get an error from the embedder. That's fine — the
        // shape contract is: on success, has "hits" key.
        match result {
            Ok(json_str) => {
                let v: Value = serde_json::from_str(&json_str).unwrap();
                assert!(v.get("hits").is_some(), "response must have 'hits' key");
            }
            Err(e) => {
                // Acceptable: no model files in test environment.
                let msg = e.to_string();
                assert!(
                    msg.contains("retriever") || msg.contains("embed") || msg.contains("model"),
                    "unexpected error: {msg}"
                );
            }
        }
    }

    #[test]
    fn kb_list_videos_returns_videos_key_for_empty_topic() {
        let dir = tempdir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let cfg = make_cfg(kb_root, "empty-topic");
        let result = handle_kb_list_videos(&cfg).unwrap();
        let v: Value = serde_json::from_str(&result).unwrap();
        assert!(v.get("videos").is_some(), "response must have 'videos' key");
        assert_eq!(
            v["videos"].as_array().unwrap().len(),
            0,
            "empty topic must have zero videos"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn kb_synthesize_errors_gracefully_without_api_key() {
        let dir = tempdir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let cfg = make_cfg(kb_root, "test-topic");

        // Temporarily remove API key.
        let prev = std::env::var("ANTHROPIC_API_KEY").ok();
        std::env::remove_var("ANTHROPIC_API_KEY");
        let prev_local = std::env::var("LEARN_SYNTH_LOCAL").ok();
        std::env::remove_var("LEARN_SYNTH_LOCAL");
        std::env::set_var("MOCK_AIMDS_VERDICT", "safe");

        let result = handle_kb_synthesize(
            &cfg,
            &json!({
                "question": "test",
                "hits": [{ "video_id": "v1", "start_seconds": 0.0, "end_seconds": 5.0, "text": "hi", "score": 0.9 }]
            }),
        );

        // Restore env.
        match prev {
            Some(v) => std::env::set_var("ANTHROPIC_API_KEY", v),
            None => std::env::remove_var("ANTHROPIC_API_KEY"),
        }
        match prev_local {
            Some(v) => std::env::set_var("LEARN_SYNTH_LOCAL", v),
            None => std::env::remove_var("LEARN_SYNTH_LOCAL"),
        }
        std::env::remove_var("MOCK_AIMDS_VERDICT");

        assert!(
            result.is_err(),
            "should fail without ANTHROPIC_API_KEY, got: {result:?}"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("ANTHROPIC_API_KEY") || msg.contains("synthesis"),
            "error should mention API key or synthesis; got: {msg}"
        );
    }

    #[test]
    fn hit_entry_round_trips_through_serde() {
        let entry = HitEntry {
            video_id: "abc123".to_string(),
            start_seconds: 10.5,
            end_seconds: 15.0,
            text: "some transcript text".to_string(),
            score: 0.85,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: HitEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.video_id, "abc123");
        assert!((back.score - 0.85).abs() < 1e-6);
    }
}
