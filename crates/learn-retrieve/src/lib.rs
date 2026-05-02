//! `learn-retrieve` — dense + BM25 hybrid retrieval, RRF fusion, MMR diversity,
//! source-cap.
//!
//! # Pipeline
//!
//! 1. EMBED query with `Embedder::embed_text`
//! 2. DENSE retrieve top 50 from `LearnIndex::search`
//! 3. SPARSE retrieve top 50 from in-memory tantivy BM25 over chunk text
//! 4. FUSE — straight RRF k=60
//!    // TODO: replace with differentiableSearch when upstream `ruvector-gnn`
//!    //        is added as a workspace dep. Source: ~/RuVector_Clean/crates/
//!    //        ruvector-gnn/src/search.rs `pub fn differentiable_search(query:
//!    //        &[f32], candidate_embeddings: &[Vec<f32>], k: usize, temperature:
//!    //        f32) -> (Vec<usize>, Vec<f32>)`. Shape mismatch vs. ranked-list
//!    //        fusion means a wrapper adapter is needed before it replaces RRF.
//! 5. RERANK — `Reranker::score_pairs` if a reranker model was loaded
//! 6. MMR — λ=0.7, source-cap = max 3 chunks per video_id
//! 7. Return top K

#![deny(unsafe_code)]

mod bm25;
mod fuse;
mod mmr;

#[cfg(test)]
mod tests;

use learn_core::{Chunk, Hit, LearnError, Result};
use learn_embed::{EmbedConfig, Embedder, Reranker};
use learn_index::LearnIndex;
use tracing::{debug, warn};

// ── Constants ────────────────────────────────────────────────────────────────

/// Top-K retrieved from each individual source before fusion.
const RETRIEVE_N: usize = 50;

/// MMR relevance weight. λ=0.7 means 70% relevance, 30% diversity.
const MMR_LAMBDA: f64 = 0.7;

/// Maximum chunks per video_id kept in the final result.
const SOURCE_CAP: usize = 3;

// ── Retriever ────────────────────────────────────────────────────────────────

/// End-to-end hybrid retriever.
///
/// Construct with [`Retriever::new`], then call [`Retriever::search`].
/// Call [`Retriever::refresh_bm25`] after ingesting new chunks to rebuild the
/// sparse index from the updated sidecar.
pub struct Retriever {
    index: LearnIndex,
    embedder: Embedder,
    reranker: Option<Reranker>,
    bm25: Option<bm25::Bm25State>,
}

impl Retriever {
    /// Create a `Retriever` from an open [`LearnIndex`] and an embedder config.
    ///
    /// `embedder_path` must point to a directory containing `model.onnx` and
    /// `tokenizer.json` for the BGE-large model.
    ///
    /// The BM25 index is not built here; call [`Retriever::refresh_bm25`] once
    /// after construction (or after ingesting chunks).
    pub fn new(index: LearnIndex, embedder_path: &camino::Utf8Path) -> Result<Self> {
        let cfg = EmbedConfig {
            model_dir: embedder_path.to_owned(),
            ..Default::default()
        };
        let embedder = Embedder::load(&cfg)?;
        Ok(Self {
            index,
            embedder,
            reranker: None,
            bm25: None,
        })
    }

    /// Attach an optional cross-encoder reranker.
    ///
    /// When set, step 5 of the pipeline calls `Reranker::score_pairs` over the
    /// top-50 RRF candidates before MMR.
    pub fn with_reranker(mut self, reranker: Reranker) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Build (or rebuild) the in-memory BM25 index from the sidecar chunk store.
    ///
    /// Cheap to call repeatedly — the entire tantivy index lives in RAM and is
    /// dropped when a new one is created.  Call after any chunk ingest.
    pub fn refresh_bm25(&mut self) -> Result<()> {
        let chunks = self.index.chunks_snapshot();
        let refs: Vec<&Chunk> = chunks.iter().collect();
        match bm25::Bm25State::build(&refs) {
            Ok(state) => {
                self.bm25 = Some(state);
                Ok(())
            }
            Err(e) => Err(LearnError::Retrieve(format!("BM25 build: {e}"))),
        }
    }

    /// End-to-end retrieval: query string → top-K hits.
    ///
    /// Steps: embed → dense-50 → sparse-50 → RRF-fuse → rerank? → MMR+cap → top-K.
    pub async fn search(&mut self, query: &str, k: usize) -> Result<Vec<Hit>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        // Step 1 — embed query.
        let query_vec = self.embedder.embed_text(query)?;

        // Step 2 — dense top-50.
        let dense_hits = self.index.search(&query_vec, RETRIEVE_N)?;
        debug!(n = dense_hits.len(), "dense hits");

        // Step 3 — sparse top-50.
        let sparse_hits = if let Some(bm25) = &self.bm25 {
            match bm25.search(query, RETRIEVE_N) {
                Ok(h) => h,
                Err(e) => {
                    warn!("BM25 search failed: {e}; using empty sparse list");
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        debug!(n = sparse_hits.len(), "sparse hits");

        // Step 4 — RRF fusion.
        let fused = fuse::rrf_fuse(&dense_hits, &sparse_hits);
        debug!(n = fused.len(), "fused candidates");
        if fused.is_empty() {
            return Ok(Vec::new());
        }

        let mut candidates = fuse::fused_to_hits(&fused, &dense_hits, &sparse_hits, &self.index);
        candidates.truncate(RETRIEVE_N);

        // Step 5 — optional cross-encoder rerank.
        if let Some(reranker) = &mut self.reranker {
            let texts: Vec<&str> = candidates.iter().map(|h| h.chunk.text.as_str()).collect();
            match reranker.score_pairs(query, &texts) {
                Ok(scores) => {
                    for (hit, score) in candidates.iter_mut().zip(scores.iter()) {
                        hit.score = *score;
                    }
                    candidates.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                Err(e) => {
                    warn!("reranker failed: {e}; skipping rerank pass");
                }
            }
        }

        // Step 6 — MMR with source-cap.
        // Pass a closure that resolves chunk_id → stored embedding (owned) so
        // that MMR uses real cosine similarity instead of the score-proxy.
        // The borrow on `self.index` is released before the selection loop.
        Ok(mmr::mmr_select(
            &candidates,
            k,
            MMR_LAMBDA,
            SOURCE_CAP,
            |id| self.index.embedding_for_chunk_id(id).map(|s| s.to_vec()),
        ))
    }
}
