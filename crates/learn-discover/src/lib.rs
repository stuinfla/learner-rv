//! `learn-discover` — autonomous curriculum discovery.
//!
//! Given a natural-language topic description, find the top-N YouTube videos
//! that together form a coherent learning curriculum. Output is a ranked list
//! of `CurriculumPick`s with rationale, ready to be piped into `learn-acquire`.
//!
//! Phase 2.5 pipeline:
//!
//! 1. **Harvest** — `yt-dlp ytsearch{N}:<topic> --dump-json --flat-playlist`
//! 2. **Score** — 5-factor heuristic rubric (pure Rust, no network)
//! 3. **Caption gate** — full info fetch on top 2×surface_count at concurrency 4
//! 4. **Curate** — Claude Opus API call for sub-topic clustering and ordering
//! 5. **Return** — `Curriculum { topic_description, depth, picks }`

pub mod curate;
pub mod harvest;
pub mod score;

use learn_core::{LearnError, Result, VideoRef};
use score::EmbedFn;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Semaphore;

// ── Public types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudyDepth {
    pub max_videos: usize,
    pub allow_long_form: bool,
    /// 0.0 = ignore date, 1.0 = strongly prefer newer content.
    pub recency_bias: f32,
}

impl StudyDepth {
    pub fn quick() -> Self {
        Self {
            max_videos: 5,
            allow_long_form: false,
            recency_bias: 0.5,
        }
    }
    pub fn medium() -> Self {
        Self {
            max_videos: 10,
            allow_long_form: true,
            recency_bias: 0.4,
        }
    }
    pub fn deep() -> Self {
        Self {
            max_videos: 25,
            allow_long_form: true,
            recency_bias: 0.3,
        }
    }

    /// Search pool size per depth (N in `ytsearch{N}:`).
    pub fn pool_size(&self) -> usize {
        match self.max_videos {
            5 => 30,
            10 => 60,
            _ => 150,
        }
    }

    /// Claude input cap per depth (candidates sent to LLM).
    pub fn claude_input_cap(&self) -> usize {
        match self.max_videos {
            5 => 10,
            10 => 20,
            _ => 50,
        }
    }
}

/// One heuristic candidate before curation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub video_id: String,
    pub title: String,
    pub channel: Option<String>,
    pub channel_id: Option<String>,
    pub view_count: Option<u64>,
    pub duration_seconds: Option<f64>,
    pub upload_date: Option<String>,
    /// `None` before caption gate; `Some(true/false)` after.
    pub has_captions: Option<bool>,
    /// Heuristic score computed by `score::score_candidates`.
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumPick {
    pub video: VideoRef,
    pub rationale: String,
    pub sub_topic: String,
    pub rank: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Curriculum {
    pub topic_description: String,
    pub depth: StudyDepth,
    pub picks: Vec<CurriculumPick>,
}

// ── Duration sanity filter ────────────────────────────────────────────────────

/// Drop candidates outside a reasonable duration window before scoring.
/// Very short (<30 s) or very long (>4 h) are structural outliers, not content.
fn duration_filter(candidates: Vec<Candidate>) -> Vec<Candidate> {
    candidates
        .into_iter()
        .filter(|c| match c.duration_seconds {
            Some(d) => (30.0..=14_400.0).contains(&d),
            None => true, // keep unknowns; score will apply 0.5 neutral
        })
        .collect()
}

// ── Caption gate ──────────────────────────────────────────────────────────────

/// Run caption availability check on `candidates` (mutated in place) with
/// a bounded concurrency of `max_concurrent` yt-dlp processes.
async fn run_caption_gate(candidates: &mut [Candidate], max_concurrent: usize) {
    let sem = Arc::new(Semaphore::new(max_concurrent));
    let ids: Vec<String> = candidates.iter().map(|c| c.video_id.clone()).collect();

    let mut handles = Vec::with_capacity(ids.len());
    for id in ids {
        let permit = sem.clone().acquire_owned().await.unwrap();
        handles.push(tokio::spawn(async move {
            let result = harvest::fetch_caption_info(&id).await;
            drop(permit);
            (id, result)
        }));
    }

    let mut results = std::collections::HashMap::new();
    for h in handles {
        if let Ok((id, res)) = h.await {
            match res {
                Ok(has) => {
                    results.insert(id, Some(has));
                }
                Err(e) => {
                    tracing::warn!("caption gate failed for {id}: {e}");
                    results.insert(id, None);
                }
            }
        }
    }

    for c in candidates.iter_mut() {
        if let Some(val) = results.get(&c.video_id) {
            c.has_captions = *val;
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Discover a curriculum for `topic_description` at the given `depth`.
///
/// `title_embedding_fn`: optional closure that produces an embedding for a
/// string; used for `title_alignment` scoring. Pass `None` to use the 0.5
/// placeholder (scores are then based on the other 4 factors).
pub async fn discover(topic_description: &str, depth: StudyDepth) -> Result<Curriculum> {
    discover_impl(topic_description, depth, None, None).await
}

/// Like `discover` but accepts an optional embedding function for title alignment.
pub async fn discover_with_embedder(
    topic_description: &str,
    depth: StudyDepth,
    title_embedding_fn: Option<EmbedFn<'_>>,
) -> Result<Curriculum> {
    let topic_embed: Option<Vec<f32>> = title_embedding_fn.map(|f| f(topic_description));
    discover_impl(
        topic_description,
        depth,
        topic_embed.as_deref(),
        title_embedding_fn,
    )
    .await
}

async fn discover_impl(
    topic_description: &str,
    depth: StudyDepth,
    topic_embed: Option<&[f32]>,
    title_embedding_fn: Option<EmbedFn<'_>>,
) -> Result<Curriculum> {
    let pool_n = depth.pool_size();
    let surface_count = depth.max_videos;
    let caption_gate_n = surface_count * 2;
    let claude_cap = depth.claude_input_cap();

    // 1. Harvest
    let mut candidates = harvest::harvest(topic_description, pool_n).await?;
    if candidates.is_empty() {
        return Err(LearnError::Apply("yt-dlp returned no candidates".into()));
    }

    // 2. Duration sanity filter (pre-score)
    candidates = duration_filter(candidates);

    // 3. Heuristic score — sort descending
    score::score_candidates(&mut candidates, topic_embed, title_embedding_fn, &depth);

    // 4. Caption gate on top 2 × surface_count
    let gate_slice = caption_gate_n.min(candidates.len());
    let (mut to_gate, rest) = {
        let rest = candidates.split_off(gate_slice);
        (candidates, rest)
    };
    run_caption_gate(&mut to_gate, 4).await;
    candidates = to_gate;
    candidates.extend(rest); // rest keeps has_captions=None

    // Re-score with caption info now populated, re-sort
    score::score_candidates(&mut candidates, topic_embed, title_embedding_fn, &depth);

    // Prefer caption-confirmed from the gated slice
    let (captioned, mut not_captioned): (Vec<_>, Vec<_>) = candidates
        .into_iter()
        .partition(|c| c.has_captions == Some(true));

    let mut shortlist = captioned;
    if shortlist.len() < surface_count {
        // relax to include un-gated candidates per memo
        shortlist.append(&mut not_captioned);
    }
    shortlist.truncate(claude_cap);

    if shortlist.is_empty() {
        return Err(LearnError::Apply("no candidates survived scoring".into()));
    }

    // 5. Curation (Anthropic or heuristic fallback)
    let picks = curate::curate(&shortlist, topic_description, surface_count).await?;

    Ok(Curriculum {
        topic_description: topic_description.to_owned(),
        depth,
        picks,
    })
}

// ── Test seam ────────────────────────────────────────────────────────────────

/// Hermetic entry point for unit tests — bypasses harvest and caption gate.
///
/// Accepts pre-built candidates (with `has_captions` already set if desired),
/// runs scoring, then curation (which falls back to heuristic if no API key).
pub async fn discover_with_inputs(
    harvested_candidates: Vec<Candidate>,
    topic_description: &str,
    depth: StudyDepth,
    title_embedding_fn: Option<EmbedFn<'_>>,
) -> Result<Curriculum> {
    let surface_count = depth.max_videos;
    let claude_cap = depth.claude_input_cap();
    let topic_embed: Option<Vec<f32>> = title_embedding_fn.map(|f| f(topic_description));

    let mut candidates = duration_filter(harvested_candidates);
    score::score_candidates(
        &mut candidates,
        topic_embed.as_deref(),
        title_embedding_fn,
        &depth,
    );
    candidates.truncate(claude_cap);

    if candidates.is_empty() {
        return Err(LearnError::Apply("no candidates after filtering".into()));
    }

    let picks = curate::curate(&candidates, topic_description, surface_count).await?;

    Ok(Curriculum {
        topic_description: topic_description.to_owned(),
        depth,
        picks,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(id: &str, score: f32) -> Candidate {
        Candidate {
            video_id: id.to_owned(),
            title: format!("Video {id}"),
            channel: Some("TestChannel".to_owned()),
            channel_id: None,
            view_count: Some(10_000),
            duration_seconds: Some(1200.0), // 20 min — ideal band
            upload_date: Some("20250101".to_owned()),
            has_captions: Some(true),
            score,
        }
    }

    fn make_candidates(n: usize) -> Vec<Candidate> {
        (0..n)
            .map(|i| make_candidate(&format!("vid{i:03}"), (n - i) as f32 / n as f32))
            .collect()
    }

    #[test]
    fn depth_presets_are_distinct() {
        assert!(StudyDepth::quick().max_videos < StudyDepth::medium().max_videos);
        assert!(StudyDepth::medium().max_videos < StudyDepth::deep().max_videos);
    }

    #[test]
    fn depth_serde_round_trip() {
        let d = StudyDepth::medium();
        let s = serde_json::to_string(&d).unwrap();
        let back: StudyDepth = serde_json::from_str(&s).unwrap();
        assert_eq!(d.max_videos, back.max_videos);
    }

    #[test]
    fn depth_pool_sizes_match_memo() {
        assert_eq!(StudyDepth::quick().pool_size(), 30);
        assert_eq!(StudyDepth::medium().pool_size(), 60);
        assert_eq!(StudyDepth::deep().pool_size(), 150);
    }

    #[tokio::test]
    async fn discover_with_inputs_returns_top_n_by_depth_quick() {
        // Ensure no real API key is set for this test
        std::env::remove_var("ANTHROPIC_API_KEY");

        let candidates = make_candidates(50);
        let curriculum = discover_with_inputs(candidates, "test topic", StudyDepth::quick(), None)
            .await
            .unwrap();

        assert_eq!(
            curriculum.picks.len(),
            StudyDepth::quick().max_videos, // 5
            "quick depth must return exactly 5 picks"
        );
    }

    #[tokio::test]
    async fn discover_with_inputs_returns_top_n_by_depth_medium() {
        std::env::remove_var("ANTHROPIC_API_KEY");

        let candidates = make_candidates(50);
        let curriculum = discover_with_inputs(candidates, "test topic", StudyDepth::medium(), None)
            .await
            .unwrap();

        assert_eq!(
            curriculum.picks.len(),
            StudyDepth::medium().max_videos, // 10
            "medium depth must return exactly 10 picks"
        );
    }

    #[tokio::test]
    async fn discover_with_inputs_returns_top_n_by_depth_deep() {
        std::env::remove_var("ANTHROPIC_API_KEY");

        let candidates = make_candidates(50);
        let curriculum = discover_with_inputs(candidates, "test topic", StudyDepth::deep(), None)
            .await
            .unwrap();

        assert_eq!(
            curriculum.picks.len(),
            StudyDepth::deep().max_videos, // 25
            "deep depth must return exactly 25 picks"
        );
    }

    #[tokio::test]
    async fn discover_falls_back_to_heuristic_when_anthropic_key_missing() {
        // Explicitly clear the key so curate falls back
        std::env::set_var("ANTHROPIC_API_KEY", "");

        let candidates = make_candidates(30);
        let curriculum = discover_with_inputs(
            candidates,
            "Rust async programming",
            StudyDepth::quick(),
            None,
        )
        .await
        .unwrap();

        // In heuristic mode every pick gets "(heuristic only)" rationale
        assert!(
            curriculum
                .picks
                .iter()
                .all(|p| p.rationale == "(heuristic only)"),
            "expected all rationales to be '(heuristic only)', got: {:?}",
            curriculum
                .picks
                .iter()
                .map(|p| &p.rationale)
                .collect::<Vec<_>>()
        );

        // sub_topic should be empty strings in heuristic mode
        assert!(
            curriculum.picks.iter().all(|p| p.sub_topic.is_empty()),
            "expected empty sub_topics in heuristic mode"
        );
    }

    /// Live test — requires real yt-dlp and Anthropic key; disabled by default.
    #[tokio::test]
    #[ignore]
    async fn live_discover_rust_async() {
        let curriculum = discover("Rust async programming tokio", StudyDepth::quick())
            .await
            .unwrap();

        assert!(!curriculum.picks.is_empty());
        println!("Picks:");
        for p in &curriculum.picks {
            println!(
                "  [{:>2}] {} — {}",
                p.rank,
                p.video.title.as_deref().unwrap_or("?"),
                p.sub_topic
            );
        }
    }
}
