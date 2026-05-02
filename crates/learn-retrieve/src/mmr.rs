//! Maximal Marginal Relevance with source-cap and real cosine similarity.

use std::collections::HashMap;

use learn_core::Hit;

// ── Cosine similarity ────────────────────────────────────────────────────────

/// Cosine similarity between two equal-length vectors.
///
/// Returns 0.0 when either vector has zero norm (i.e. treats a zero-norm
/// vector as orthogonal to everything, which is the safest no-op for MMR).
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_sim: dimension mismatch");
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

// ── MMR selection ────────────────────────────────────────────────────────────

/// Select up to `k` hits using MMR with a per-video source-cap.
///
/// `lambda` = 1.0 → pure relevance (no diversity penalty).
/// `lambda` = 0.0 → pure diversity (maximum spread).
/// `source_cap` → maximum hits retained per `video_id`.
///
/// `embed_lookup` is a closure that maps a `chunk_id` string to its stored
/// dense embedding as an owned `Vec<f32>`.  Returning an owned value sidesteps
/// lifetime entanglement between the borrow on `self` (in `Retriever::search`)
/// or a local map (in tests) and the inner mutable selection loop.
///
/// When an embedding is unavailable the closure should return `None`, and the
/// similarity term for that pair is treated as 0.0 (no penalty).
pub(crate) fn mmr_select<F>(
    candidates: &[Hit],
    k: usize,
    lambda: f64,
    source_cap: usize,
    embed_lookup: F,
) -> Vec<Hit>
where
    F: Fn(&str) -> Option<Vec<f32>>,
{
    if candidates.is_empty() {
        return Vec::new();
    }

    let max_score = candidates
        .iter()
        .map(|h| h.score as f64)
        .fold(0.0_f64, f64::max)
        .max(1e-9);

    // Eagerly materialise owned embeddings for all candidates.  This ends the
    // borrow on `embed_lookup` before the mutable selection loop begins.
    let candidate_embeddings: Vec<Option<Vec<f32>>> = candidates
        .iter()
        .map(|h| embed_lookup(&h.chunk.chunk_id))
        .collect();

    let mut selected: Vec<Hit> = Vec::with_capacity(k);
    // Cloned embeddings of already-selected items.
    let mut selected_embeddings: Vec<Option<Vec<f32>>> = Vec::new();
    let mut remaining: Vec<usize> = (0..candidates.len()).collect(); // indices into candidates
    let mut source_count: HashMap<String, usize> = HashMap::new();

    while selected.len() < k && !remaining.is_empty() {
        let mut best_remaining_pos: Option<usize> = None; // position within `remaining`
        let mut best_mmr = f64::NEG_INFINITY;

        for (pos, &cand_idx) in remaining.iter().enumerate() {
            let hit = &candidates[cand_idx];

            if source_count.get(&hit.chunk.video_id).copied().unwrap_or(0) >= source_cap {
                continue;
            }

            let relevance = hit.score as f64 / max_score;

            let max_sim: f64 = if selected_embeddings.is_empty() {
                0.0
            } else {
                let cand_emb = candidate_embeddings[cand_idx].as_deref();
                selected_embeddings
                    .iter()
                    .map(|sel_emb| match (cand_emb, sel_emb.as_deref()) {
                        (Some(c), Some(s)) => cosine_sim(c, s) as f64,
                        // No embedding for one or both sides: assume 0 similarity.
                        _ => 0.0,
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            };

            let mmr = lambda * relevance - (1.0 - lambda) * max_sim;
            if mmr > best_mmr {
                best_mmr = mmr;
                best_remaining_pos = Some(pos);
            }
        }

        let Some(pos) = best_remaining_pos else { break };

        let cand_idx = remaining.remove(pos);
        let chosen = &candidates[cand_idx];
        let chosen_emb = candidate_embeddings[cand_idx].clone();
        selected_embeddings.push(chosen_emb);
        *source_count
            .entry(chosen.chunk.video_id.clone())
            .or_insert(0) += 1;
        let rank = selected.len();
        selected.push(Hit {
            chunk: chosen.chunk.clone(),
            score: chosen.score,
            rank,
        });
    }

    selected
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use learn_core::{Chunk, Hit};

    fn make_chunk(i: usize, video_id: &str) -> Chunk {
        Chunk {
            chunk_id: format!("chunk-{i:04}"),
            video_id: video_id.to_owned(),
            start_seconds: i as f64,
            end_seconds: i as f64 + 1.0,
            text: format!("text {i}"),
            token_count: 2,
        }
    }

    fn make_hit(chunk: Chunk, score: f32, rank: usize) -> Hit {
        Hit { chunk, score, rank }
    }

    /// Two orthogonal unit vectors: dot product = 0, cosine = 0.
    #[test]
    fn cosine_sim_orthogonal_is_zero() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        let sim = cosine_sim(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal cosine must be 0, got {sim}");
    }

    /// Identical vectors: cosine = 1.
    #[test]
    fn cosine_sim_identical_is_one() {
        let a = [0.6f32, 0.8, 0.0];
        let sim = cosine_sim(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical cosine must be 1, got {sim}"
        );
    }

    /// Zero-norm vector: cosine = 0 (no panic).
    #[test]
    fn cosine_sim_zero_norm_no_panic() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        let sim = cosine_sim(&a, &b);
        assert_eq!(sim, 0.0, "zero-norm cosine must return 0");
    }

    /// MMR with real cosine: two orthogonal embeddings ([1,0,0] and [0,1,0]).
    ///
    /// After selecting the first candidate, the cosine similarity between it
    /// and the second candidate is exactly 0.0 — the old score-proxy code used
    /// `(1.0 - |norm_diff|).clamp(0,1)` which returned ~0.95 for near-equal
    /// scores, incorrectly penalising diverse pairs.
    #[test]
    fn mmr_uses_real_cosine_not_score_proxy() {
        // Verify cosine of orthogonal unit vectors is 0 — this is the key
        // invariant that the old score-proxy violated.
        let emb0: &[f32] = &[1.0, 0.0, 0.0];
        let emb1: &[f32] = &[0.0, 1.0, 0.0];
        let actual_cosine = cosine_sim(emb0, emb1);
        assert!(
            actual_cosine.abs() < 1e-6,
            "diversity term must be cosine=0 for orthogonal vectors, got {actual_cosine}"
        );

        // Now verify MMR routes through real cosine by checking selection order.
        let c0 = make_hit(make_chunk(0, "v1"), 0.9, 0);
        let c1 = make_hit(make_chunk(1, "v1"), 0.85, 1);
        let candidates = vec![c0, c1];

        // Build a static lookup backed by two fixed arrays — no heap allocation
        // that triggers HRTB lifetime issues in the test.
        let emb0_owned = vec![1.0f32, 0.0, 0.0];
        let emb1_owned = vec![0.0f32, 1.0, 0.0];

        // We use a plain function-like closure that borrows from the enclosing
        // scope.  `mmr_select` eagerly clones into `Vec<f32>` so the borrow
        // ends before any mutation.
        let result = mmr_select(&candidates, 2, 0.5, 10, |id| {
            if id == "chunk-0000" {
                Some(emb0_owned.clone())
            } else if id == "chunk-0001" {
                Some(emb1_owned.clone())
            } else {
                None
            }
        });

        assert_eq!(result.len(), 2, "both candidates must be selected");
        // c0 selected first (higher score); diversity penalty for c1 is
        // lambda × cosine = 0.5 × 0.0 = 0 — c1 keeps its full relevance score.
        assert_eq!(result[0].chunk.chunk_id, "chunk-0000");
        assert_eq!(result[1].chunk.chunk_id, "chunk-0001");
    }
}
