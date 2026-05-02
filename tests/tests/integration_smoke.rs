// integration_smoke.rs — Phase 1 end-to-end smoke test.
//
// This test is `#[ignore]`-tagged because it requires model files (whisper,
// ort/BGE) that are not present in CI until Phase 1 finishes.
//
// Run with:
//   cargo test --workspace -- --include-ignored integration_smoke
//
// Planned flow (pseudo-code; the test currently panics with TODO to pin the
// API shape before the implementation is wired up):
//
//   1. acquire  — parse fixtures/short.vtt into Vec<Segment>
//   2. chunk    — split segments into Chunk list (target ~200 tokens each)
//   3. embed    — [mocked] return a fixed 1024-dim zero vector per chunk
//   4. index    — write chunks into a temp .rvf file via learn-index
//   5. retrieve — query the .rvf with a test query string
//   6. assert   — hits list must be non-empty
//
// When each crate API stabilises, replace the TODO panic with real calls.

use camino::Utf8Path;
use learn_acquire::vtt;

#[test]
#[ignore = "requires whisper + ort + bge model files; run with --include-ignored after Phase 1 finish"]
fn e2e_ingest_and_retrieve_short_fixture() {
    // ── Step 1: acquire — parse the fixture VTT ──────────────────────────────
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let vtt_path_str = format!("{}/../../fixtures/short.vtt", manifest_dir);
    let vtt_path = Utf8Path::new(&vtt_path_str);

    let segments = vtt::parse_vtt(vtt_path).expect("short.vtt must parse without error");

    assert!(
        !segments.is_empty(),
        "short.vtt should produce at least one segment"
    );

    // ── Step 2: chunk ─────────────────────────────────────────────────────────
    // TODO(Phase 1): call learn_chunk::chunk_segments(&segments, &ChunkConfig::default())
    // Expected: Vec<Chunk>, each ≤ 300 tokens, contiguous timestamps.
    let _chunks: Vec<()> =
        todo_placeholder("learn_chunk::chunk_segments — wire up after learn-chunk API is stable");

    // ── Step 3: embed (mocked) ────────────────────────────────────────────────
    // TODO(Phase 1): call learn_embed::embed_chunks(&chunks, &model_handle).await
    // For the smoke test, inject a mock embedder that returns zeros so we
    // can test the index + retrieve path without downloading model weights.
    // Expected: Vec<Embedded> with embedding.len() == 1024.
    let _embedded: Vec<()> =
        todo_placeholder("learn_embed::embed_chunks — use mock embedder returning f32 zeros");

    // ── Step 4: index ─────────────────────────────────────────────────────────
    // TODO(Phase 1): call learn_index::write_rvf(&embedded, &temp_rvf_path)
    // Expected: Ok(()), temp .rvf file exists on disk.
    let _index_result: () = todo_placeholder(
        "learn_index::write_rvf — wire up after learn-index RVF adapter is stable",
    );

    // ── Step 5: retrieve ──────────────────────────────────────────────────────
    // TODO(Phase 1): let hits = learn_retrieve::query(&temp_rvf_path, "test query", 5).await?
    // Expected: Vec<Hit> with hit.score > 0.0.
    let _hits: Vec<()> = todo_placeholder(
        "learn_retrieve::query — wire up after learn-retrieve hybrid search is stable",
    );

    // ── Step 6: assert ────────────────────────────────────────────────────────
    // TODO(Phase 1): assert!(!hits.is_empty(), "query must return at least one hit");
    todo_placeholder::<()>("final assertion: assert!(!hits.is_empty())");
}

/// Stand-in for unimplemented pipeline stages.
///
/// Panics with a clear TODO message so the test surface is visible without
/// executing unreachable code paths.
#[allow(dead_code)]
fn todo_placeholder<T>(msg: &str) -> T {
    panic!("TODO(Phase 1): {msg}");
}
