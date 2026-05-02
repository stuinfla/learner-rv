//! RRF (Reciprocal Rank Fusion) and hit reconstruction.

use std::collections::HashMap;

use learn_core::Hit;
use learn_index::LearnIndex;

/// RRF smoothing constant (Cormack et al. 2009).
const RRF_K: usize = 60;

/// Reciprocal Rank Fusion over a dense and a sparse ranked list.
///
/// Returns `(chunk_id, fused_score)` pairs sorted by descending score.
/// Score: Σ_list 1 / (RRF_K + rank)  where rank is 1-indexed.
pub(crate) fn rrf_fuse(dense: &[Hit], sparse: &[(String, f32)]) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    for (rank, hit) in dense.iter().enumerate() {
        *scores.entry(hit.chunk.chunk_id.clone()).or_insert(0.0) +=
            1.0 / (RRF_K as f64 + (rank + 1) as f64);
    }
    for (rank, (chunk_id, _)) in sparse.iter().enumerate() {
        *scores.entry(chunk_id.clone()).or_insert(0.0) += 1.0 / (RRF_K as f64 + (rank + 1) as f64);
    }

    let mut sorted: Vec<(String, f64)> = scores.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

/// Reconstruct a `Hit` list in RRF order, resolving chunk payloads from the
/// dense list (preferred) or the index sidecar.
pub(crate) fn fused_to_hits(
    fused: &[(String, f64)],
    dense: &[Hit],
    sparse: &[(String, f32)],
    index: &LearnIndex,
) -> Vec<Hit> {
    let dense_map: HashMap<&str, &Hit> = dense
        .iter()
        .map(|h| (h.chunk.chunk_id.as_str(), h))
        .collect();
    // sparse_score_map is consulted only as a fallback signal; suppress
    // unused warning with an underscore prefix on the variable.
    let _sparse_map: HashMap<&str, f32> = sparse.iter().map(|(id, s)| (id.as_str(), *s)).collect();

    fused
        .iter()
        .enumerate()
        .filter_map(|(rank, (chunk_id, rrf_score))| {
            if let Some(hit) = dense_map.get(chunk_id.as_str()) {
                Some(Hit {
                    chunk: hit.chunk.clone(),
                    score: *rrf_score as f32,
                    rank,
                })
            } else {
                index.chunk_by_id(chunk_id).map(|chunk| Hit {
                    chunk,
                    score: *rrf_score as f32,
                    rank,
                })
            }
        })
        .collect()
}
