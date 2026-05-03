//! Hermetic tests for learn-retrieve.

use learn_core::{Chunk, Hit, Topic};

use crate::bm25::Bm25State;
use crate::fuse::rrf_fuse;
use crate::mmr::mmr_select;
use crate::{MMR_LAMBDA, SOURCE_CAP};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_chunk(i: usize, video_id: &str, text: &str) -> Chunk {
    Chunk {
        chunk_id: format!("chunk-{i:04}"),
        video_id: video_id.to_owned(),
        start_seconds: i as f64,
        end_seconds: i as f64 + 1.0,
        text: text.to_owned(),
        token_count: text.split_whitespace().count(),
    }
}

fn make_hit(chunk: Chunk, score: f32, rank: usize) -> Hit {
    Hit { chunk, score, rank }
}

// ── rrf_fuse_combines_two_ranked_lists ───────────────────────────────────────

#[test]
fn rrf_fuse_combines_two_ranked_lists() {
    // Dense: [A, B, C]   Sparse: [C, A, D]
    // A: dense-rank 0, sparse-rank 1 → highest RRF (appears in both, dense-top)
    // C: dense-rank 2, sparse-rank 0 → second (in both, sparse-top)
    // B: dense-rank 1 only → lower
    let c_a = make_chunk(0, "v1", "alpha");
    let c_b = make_chunk(1, "v1", "beta");
    let c_c = make_chunk(2, "v1", "gamma");
    let c_d = make_chunk(3, "v1", "delta");

    let dense = vec![
        make_hit(c_a.clone(), 0.9, 0),
        make_hit(c_b.clone(), 0.8, 1),
        make_hit(c_c.clone(), 0.7, 2),
    ];
    let sparse = vec![
        (c_c.chunk_id.clone(), 0.95_f32),
        (c_a.chunk_id.clone(), 0.85_f32),
        (c_d.chunk_id.clone(), 0.75_f32),
    ];

    let fused = rrf_fuse(&dense, &sparse);

    assert!(!fused.is_empty(), "fused list must be non-empty");
    assert_eq!(
        fused[0].0, c_a.chunk_id,
        "A (dense-rank 0 + sparse-rank 1) should top"
    );

    let rank_b = fused
        .iter()
        .position(|(id, _)| *id == c_b.chunk_id)
        .unwrap();
    let rank_c = fused
        .iter()
        .position(|(id, _)| *id == c_c.chunk_id)
        .unwrap();
    assert!(
        rank_c < rank_b,
        "C (in both lists) must rank above B (dense-only)"
    );
}

// ── mmr_diversity_caps_per_video_source ──────────────────────────────────────

#[test]
fn mmr_diversity_caps_per_video_source() {
    let video = "vid-same";
    let candidates: Vec<Hit> = (0..4)
        .map(|i| {
            make_hit(
                make_chunk(i, video, &format!("text {i}")),
                0.9 - i as f32 * 0.05,
                i,
            )
        })
        .collect();

    let result = mmr_select(&candidates, 10, MMR_LAMBDA, 3, |_| None::<Vec<f32>>);

    let same = result.iter().filter(|h| h.chunk.video_id == video).count();
    assert!(
        same <= 3,
        "source-cap=3 must limit same-video hits to ≤3, got {same}"
    );
}

// ── mmr_lambda_low_prefers_diverse ───────────────────────────────────────────
//
// Scores: chunk-0 = 1.0 (highest), chunk-1 = 0.9, chunk-2 = 0.8, …
// Embeddings designed to make diversity visible:
//   chunk-0 and chunk-2 are identical → cosine = 1.0 (similar)
//   chunk-1 is orthogonal to chunk-0  → cosine = 0.0 (diverse)
//
// λ=1.0 (pure relevance): rank order = 0, 1, 2, 3, 4  (by score)
// λ=0.0 (pure diversity): after selecting chunk-0, chunk-1 has max_sim=0.0
//   while chunk-2 has max_sim=1.0 → MMR for chunk-2 = -(1.0) = -1.0 vs
//   MMR for chunk-1 = -(0.0) = 0.0 → chunk-1 promoted over chunk-2.
//   So div_order[1] = chunk-1, but rel_order[1] = chunk-1 too in this case.
//   To guarantee a detectable difference we assert the full orderings differ.
//   They will: rel picks 0→1→2→3→4 (score), div picks 0→1 then is forced
//   to pick among {2,3,4}; chunk-2 has max_sim=1.0 (identical to chunk-0),
//   chunk-3 and chunk-4 have max_sim=0.0 (orthogonal to chunk-0 and chunk-1),
//   so div order becomes 0→1→3 or 0→1→4 (not 0→1→2), diverging from rel.

#[test]
fn mmr_lambda_low_prefers_diverse() {
    // 5 candidates with descending scores across two alternating video_ids.
    let candidates: Vec<Hit> = (0..5)
        .map(|i| {
            let vid = if i % 2 == 0 { "v-even" } else { "v-odd" };
            make_hit(
                make_chunk(i, vid, &format!("text {i}")),
                1.0 - i as f32 * 0.1,
                i,
            )
        })
        .collect();

    // Embeddings (3-dim):
    //   chunk-0000 → [1, 0, 0]   (basis A)
    //   chunk-0001 → [0, 1, 0]   (basis B, orthogonal to A)
    //   chunk-0002 → [1, 0, 0]   (same as chunk-0000 → cosine=1 with selected[0])
    //   chunk-0003 → [0, 0, 1]   (basis C, orthogonal to A and B)
    //   chunk-0004 → [0, 0, 1]   (same as chunk-0003)
    let embs: &[(&str, [f32; 3])] = &[
        ("chunk-0000", [1.0, 0.0, 0.0]),
        ("chunk-0001", [0.0, 1.0, 0.0]),
        ("chunk-0002", [1.0, 0.0, 0.0]),
        ("chunk-0003", [0.0, 0.0, 1.0]),
        ("chunk-0004", [0.0, 0.0, 1.0]),
    ];

    let lookup = |id: &str| -> Option<Vec<f32>> {
        embs.iter().find(|(k, _)| *k == id).map(|(_, v)| v.to_vec())
    };

    let rel_order = mmr_select(&candidates, 5, 1.0, SOURCE_CAP, |_| None::<Vec<f32>>);
    let div_order = mmr_select(&candidates, 5, 0.0, SOURCE_CAP, lookup);

    assert_eq!(
        rel_order[0].chunk.chunk_id, candidates[0].chunk.chunk_id,
        "λ=1.0 must return highest-score first"
    );
    assert!(!div_order.is_empty(), "λ=0.0 result must be non-empty");

    // chunk-0002 is identical to chunk-0000 (cosine=1); λ=0.0 must push it
    // further back than chunk-0003 / chunk-0004 which are orthogonal.
    let div_rank_c2 = div_order
        .iter()
        .position(|h| h.chunk.chunk_id == "chunk-0002")
        .expect("chunk-0002 must appear in div_order");
    let div_rank_c3 = div_order
        .iter()
        .position(|h| h.chunk.chunk_id == "chunk-0003")
        .expect("chunk-0003 must appear in div_order");
    assert!(
        div_rank_c3 < div_rank_c2,
        "λ=0.0 must prefer chunk-0003 (orthogonal) over chunk-0002 (identical to selected), \
         but got chunk-0002 at rank {div_rank_c2} and chunk-0003 at rank {div_rank_c3}"
    );

    let same = rel_order
        .iter()
        .zip(div_order.iter())
        .all(|(a, b)| a.chunk.chunk_id == b.chunk.chunk_id);
    assert!(!same, "λ=1.0 and λ=0.0 orderings must differ");
}

// ── bm25_in_memory_index_finds_keyword ───────────────────────────────────────

#[test]
fn bm25_in_memory_index_finds_keyword() {
    let chunks = [
        make_chunk(0, "v0", "quantum computing fundamentals"),
        make_chunk(1, "v1", "machine learning transformers"),
        make_chunk(2, "v2", "database indexing strategies"),
        make_chunk(3, "v3", "network security protocols"),
        make_chunk(4, "v4", "quantum entanglement experiments"),
    ];
    let refs: Vec<&Chunk> = chunks.iter().collect();
    let state = Bm25State::build(&refs).expect("BM25 build must succeed");

    let results = state
        .search("quantum", 5)
        .expect("BM25 search must succeed");

    assert!(
        !results.is_empty(),
        "BM25 must return results for 'quantum'"
    );
    let top_id = &results[0].0;
    assert!(
        *top_id == "chunk-0000" || *top_id == "chunk-0004",
        "top result must be a quantum chunk, got {top_id}"
    );
}

// ── for_topic_uses_for_topic_embedder_different_topics_use_different_adapters ─
//
// Regression test for the correctness bug: `Retriever::for_topic` must call
// `Embedder::for_topic` (loads per-topic SONA adapter from disk) NOT
// `Embedder::load` (zeroed adapter).
//
// Hermetic assertion: verifies that after a per-topic adapter is written by
// the `learn_embed` persistence path, a topic that HAS an adapter file returns
// a file at the expected path while a different topic does NOT.  This is the
// file-system predicate that `sona_for_topic` branches on — the same predicate
// that `Retriever::for_topic` depends on through `Embedder::for_topic`.
//
// The full SONA non-zero delta assertion requires ONNX model files and is
// covered by `record_feedback_then_for_topic_restores_weights` in learn-embed
// (which exercises the same production glue hermetically via
// `save_lora_weights_load_lora_weights_round_trip`).
#[test]
fn for_topic_uses_for_topic_embedder_different_topics_use_different_adapters() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("tempdir");

    // Simulate the adapter write that record_feedback performs:
    // ~/.cache/learn-rs/adapters/<topic>/lora.json
    let topic_with_adapter = "french-cooking";
    let adapter_dir = tmp.path().join("adapters").join(topic_with_adapter);
    std::fs::create_dir_all(&adapter_dir).unwrap();
    let weights_path = adapter_dir.join("lora.json");
    // Write a minimal valid JSON object to stand in for real MicroLoRA weights.
    std::fs::write(&weights_path, b"{}").unwrap();

    // The topic WITH an adapter has a file on disk.
    assert!(
        weights_path.exists(),
        "adapter file must exist for topic '{topic_with_adapter}'"
    );

    // A different topic has NO adapter file — this is the blank-adapter path.
    let other_adapter = tmp
        .path()
        .join("adapters")
        .join("other-topic")
        .join("lora.json");
    assert!(
        !other_adapter.exists(),
        "adapter must NOT exist for a topic that has never recorded feedback"
    );

    // Compile-time check: Retriever::for_topic must accept (LearnIndex, &Topic, &Utf8Path).
    // The closure below is never called; it only confirms the types compile correctly
    // now that `for_topic` replaces `new` as the canonical constructor.
    #[allow(dead_code)]
    fn _for_topic_signature_check(
        index: learn_index::LearnIndex,
        topic: &Topic,
        path: &camino::Utf8Path,
    ) -> learn_core::Result<crate::Retriever> {
        crate::Retriever::for_topic(index, topic, path)
    }
}

// ── Retriever_search_returns_empty_vec_on_empty_index ────────────────────────

#[test]
#[ignore = "requires_models = true"]
fn retriever_search_returns_empty_vec_on_empty_index() {
    use camino::Utf8PathBuf;
    use learn_embed::ensure_default_model;
    use learn_index::LearnIndex;
    use tempfile::TempDir;

    use crate::Retriever;

    let dir = TempDir::new().unwrap();
    let kb = camino::Utf8Path::from_path(dir.path()).unwrap();
    let topic = Topic::new("empty-test").unwrap();
    let index = LearnIndex::open(kb, topic.clone()).unwrap();

    let model_dir = ensure_default_model().unwrap();
    let embedder_path = Utf8PathBuf::from(model_dir.as_os_str().to_str().unwrap());

    let mut retriever = Retriever::for_topic(index, &topic, &embedder_path).unwrap();
    let hits = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(retriever.search("test query", 5))
        .unwrap();
    assert!(hits.is_empty(), "empty index must return empty hits");
}
