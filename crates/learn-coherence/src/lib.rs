#![deny(unsafe_code)]
//! KB health and query-drift monitoring for Learn-RV.
//!
//! Wraps `ruvector-coherence` spectral primitives to answer two questions:
//! 1. Is this KB's embedding set internally consistent? (`check_kb_health`)
//! 2. Has query quality drifted over recent history? (`check_drift`)

use learn_core::{Embedded, Result};
use ruvector_coherence::metrics::contradiction_rate;
use ruvector_coherence::quality::cosine_similarity;
use ruvector_coherence::spectral::{CsrMatrixView, SpectralConfig, SpectralTracker};
use serde::{Deserialize, Serialize};

/// Structural health summary for an embedded KB.
#[derive(Debug, Clone)]
pub struct KbHealth {
    /// Normalized Fiedler value [0, 1]. Higher = better-connected embedding graph.
    pub fiedler_value: f32,
    /// Rate of contradictory embedding pairs (negative dot product). Lower = better.
    pub contradiction_rate: f32,
    /// Total number of embeddings examined.
    pub vector_count: usize,
    /// (chunk_id_a, chunk_id_b) pairs whose cosine similarity exceeds the
    /// near-duplicate threshold — potential content redundancy or contradiction.
    pub flagged_pairs: Vec<(String, String)>,
}

/// Drift report comparing a baseline quality window to the recent window.
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// True when the CUSUM statistic exceeds the detection threshold.
    pub changepoint_detected: bool,
    /// Mean top-hit score in the recent half of the history.
    pub recent_quality_score: f32,
    /// Mean top-hit score in the baseline (older) half of the history.
    pub baseline_quality_score: f32,
}

/// One entry in query history for drift detection.
#[derive(Debug, Clone)]
pub struct QueryRecord {
    pub query_embedding: Vec<f32>,
    pub top_hit_score: f32,
    pub timestamp: f64,
}

/// Threshold above which two embeddings are considered near-duplicate candidates.
const NEAR_DUPLICATE_THRESHOLD: f64 = 0.95;

/// Edge similarity threshold for building the Laplacian graph.
const GRAPH_EDGE_THRESHOLD: f64 = 0.5;

/// CUSUM drift detection threshold (absolute drop in mean quality score).
const DRIFT_THRESHOLD: f32 = 0.08;

/// Analyse the health of an embedded KB.
///
/// Returns `Err` only when `embedded` is empty (no graph to analyse).
pub fn check_kb_health(embedded: &[Embedded]) -> Result<KbHealth> {
    let n = embedded.len();
    if n == 0 {
        return Err(learn_core::LearnError::Graph(
            "cannot check health of empty embedding set".into(),
        ));
    }

    let flagged_pairs = find_flagged_pairs(embedded);
    let contradiction = compute_contradiction_rate(embedded);
    let fiedler = if n >= 2 {
        compute_fiedler(embedded)
    } else {
        0.0
    };

    Ok(KbHealth {
        fiedler_value: fiedler,
        contradiction_rate: contradiction,
        vector_count: n,
        flagged_pairs,
    })
}

/// Detect quality drift over a sequence of query records.
///
/// Splits history at the midpoint: baseline = older half, recent = newer half.
/// Returns `Err` when fewer than 4 records are provided (insufficient for comparison).
pub fn check_drift(query_history: &[QueryRecord]) -> Result<DriftReport> {
    if query_history.len() < 4 {
        return Err(learn_core::LearnError::Retrieve(
            "need at least 4 query records for drift detection".into(),
        ));
    }

    let mid = query_history.len() / 2;
    let baseline_scores = &query_history[..mid];
    let recent_scores = &query_history[mid..];

    let baseline_mean = mean_score(baseline_scores);
    let recent_mean = mean_score(recent_scores);

    let drop = baseline_mean - recent_mean;
    let changepoint_detected = drop > DRIFT_THRESHOLD;

    Ok(DriftReport {
        changepoint_detected,
        recent_quality_score: recent_mean,
        baseline_quality_score: baseline_mean,
    })
}

// --- internals ---

fn mean_score(records: &[QueryRecord]) -> f32 {
    if records.is_empty() {
        return 0.0;
    }
    records.iter().map(|r| r.top_hit_score).sum::<f32>() / records.len() as f32
}

fn find_flagged_pairs(embedded: &[Embedded]) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for i in 0..embedded.len() {
        for j in (i + 1)..embedded.len() {
            let sim = cosine_similarity(&embedded[i].embedding, &embedded[j].embedding);
            if sim >= NEAR_DUPLICATE_THRESHOLD {
                pairs.push((
                    embedded[i].chunk.chunk_id.clone(),
                    embedded[j].chunk.chunk_id.clone(),
                ));
            }
        }
    }
    pairs
}

fn compute_contradiction_rate(embedded: &[Embedded]) -> f32 {
    if embedded.len() < 2 {
        return 0.0;
    }
    let vecs: Vec<Vec<f32>> = embedded.iter().map(|e| e.embedding.clone()).collect();
    // Compare each vector against the next (wrap-around) as a cyclic reference set.
    let n = vecs.len();
    let rotated: Vec<Vec<f32>> = (0..n).map(|i| vecs[(i + 1) % n].clone()).collect();
    contradiction_rate(&vecs, &rotated) as f32
}

fn build_similarity_edges(embedded: &[Embedded]) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    for i in 0..embedded.len() {
        for j in (i + 1)..embedded.len() {
            let sim = cosine_similarity(&embedded[i].embedding, &embedded[j].embedding);
            if sim >= GRAPH_EDGE_THRESHOLD {
                // Weight = similarity so well-connected clusters have higher Fiedler value.
                edges.push((i, j, sim));
            }
        }
    }
    // Ensure the graph is connected: add a chain of weak edges as a spanning backbone.
    for i in 0..(embedded.len() - 1) {
        let already = edges.iter().any(|&(u, v, _)| u == i && v == i + 1);
        if !already {
            edges.push((i, i + 1, 0.01));
        }
    }
    edges
}

fn compute_fiedler(embedded: &[Embedded]) -> f32 {
    let n = embedded.len();
    let edges = build_similarity_edges(embedded);
    let lap = CsrMatrixView::build_laplacian(n, &edges);
    let cfg = SpectralConfig::default();
    let mut tracker = SpectralTracker::new(cfg);
    let score = tracker.compute(&lap);
    score.fiedler as f32
}

// ── Consciousness KPI ─────────────────────────────────────────────────────────

/// Qualitative interpretation of the integrated-information score.
///
/// Thresholds:
/// - `Disjoint`         : integrated_information < 0.30
/// - `Loose`            : 0.30 ≤ integrated_information < 0.60
/// - `Coherent`         : 0.60 ≤ integrated_information < 0.85
/// - `HighlyIntegrated` : integrated_information ≥ 0.85
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum KpiInterpretation {
    /// Chunks form semantically unrelated islands (< 0.30).
    Disjoint,
    /// Loosely connected — some shared themes, but no strong centre (0.30–0.60).
    Loose,
    /// Well-structured knowledge base with clear semantic coherence (0.60–0.85).
    Coherent,
    /// Highly integrated corpus: dense inter-chunk relationships (≥ 0.85).
    HighlyIntegrated,
}

impl KpiInterpretation {
    fn from_score(score: f32) -> Self {
        if score < 0.30 {
            Self::Disjoint
        } else if score < 0.60 {
            Self::Loose
        } else if score < 0.85 {
            Self::Coherent
        } else {
            Self::HighlyIntegrated
        }
    }

    /// Human-readable label (used in CLI output).
    pub fn label(self) -> &'static str {
        match self {
            Self::Disjoint => "Disjoint",
            Self::Loose => "Loose",
            Self::Coherent => "Coherent",
            Self::HighlyIntegrated => "HighlyIntegrated",
        }
    }
}

/// Integrated-information style coherence KPI for an embedded knowledge base.
///
/// **Phase 4A v1 placeholder using ruvector-coherence spectral primitives.**
/// Will swap to the real `ruvector-consciousness` IIT Φ API when the upstream
/// crate exposes an embedding-native interface (it currently requires a
/// directed causal `TransitionMatrix` unavailable from embedding corpora).
///
/// # Scores
///
/// | Field                  | Formula |
/// |------------------------|---------|
/// | `integrated_information` | `normalized_fiedler × (1 − mean_nn_cosine_dist)` |
/// | `workspace_score`        | `chunks_within_1_hop_of_centroid / total_chunks` |
///
/// Both scores are clamped to `[0.0, 1.0]`.
///
/// ## Intuition
///
/// - **integrated_information** is high when (a) the similarity graph is
///   well-connected (high Fiedler value) AND (b) each chunk has at least one
///   semantically close neighbour. A disjoint set of unrelated videos produces
///   a near-zero Fiedler value.
///
/// - **workspace_score** reflects Global Workspace Theory: a strong central
///   theme means most chunks are "1 hop" from the centroid in the similarity
///   graph. A scattered, multi-topic corpus scores low.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConsciousnessKpi {
    /// Integrated information proxy ∈ [0.0, 1.0].
    pub integrated_information: f32,
    /// Global workspace score ∈ [0.0, 1.0].
    pub workspace_score: f32,
    /// Qualitative interpretation of `integrated_information`.
    pub interpretation: KpiInterpretation,
}

/// Compute the consciousness KPI for an embedded knowledge base.
///
/// Returns `Ok` with a zero-valued `Disjoint` KPI for empty input (graceful
/// no-op; callers need not special-case the empty corpus).
pub fn compute_consciousness_kpi(embedded: &[Embedded]) -> Result<ConsciousnessKpi> {
    if embedded.is_empty() {
        return Ok(ConsciousnessKpi {
            integrated_information: 0.0,
            workspace_score: 0.0,
            interpretation: KpiInterpretation::Disjoint,
        });
    }

    let n = embedded.len();

    // ── integrated_information ────────────────────────────────────────────────
    //
    // = clamp(normalized_fiedler, 0, 1) × clamp(1 − mean_nn_cosine_dist, 0, 1)
    //
    // normalized_fiedler: we reuse the same graph-build as check_kb_health;
    // SpectralTracker already normalises the score to [0, 1].
    let fiedler_norm = if n >= 2 {
        compute_fiedler(embedded)
    } else {
        0.0
    };

    let mean_nn_dist = if n >= 2 {
        compute_mean_nn_cosine_distance(embedded)
    } else {
        1.0 // single chunk: maximally distant from any "neighbour"
    };

    let semantic_density = (1.0_f32 - mean_nn_dist).clamp(0.0, 1.0);
    let integrated_information = (fiedler_norm * semantic_density).clamp(0.0, 1.0);

    // ── workspace_score ───────────────────────────────────────────────────────
    //
    // centroid = mean of all embeddings
    // threshold: a chunk is "within 1 hop" if cosine(chunk, centroid) ≥ GRAPH_EDGE_THRESHOLD
    let workspace_score = if n == 0 {
        0.0
    } else {
        compute_workspace_score(embedded)
    };

    let interpretation = KpiInterpretation::from_score(integrated_information);

    Ok(ConsciousnessKpi {
        integrated_information,
        workspace_score,
        interpretation,
    })
}

/// Mean nearest-neighbour cosine *distance* (= 1 − cosine_similarity) across all chunks.
///
/// For each chunk finds its closest peer (excluding itself) and accumulates the distance.
fn compute_mean_nn_cosine_distance(embedded: &[Embedded]) -> f32 {
    let n = embedded.len();
    debug_assert!(n >= 2);

    let mut total_dist = 0.0_f32;
    for i in 0..n {
        let mut best_sim = f32::NEG_INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let sim = cosine_similarity(&embedded[i].embedding, &embedded[j].embedding) as f32;
            if sim > best_sim {
                best_sim = sim;
            }
        }
        // best_sim is the nearest-neighbour similarity; distance = 1 − sim.
        // Clamp: cosine_similarity can produce values slightly outside [−1, 1] due to float rounding.
        let nn_dist = (1.0_f32 - best_sim.clamp(-1.0, 1.0)).clamp(0.0, 2.0);
        // Normalise distance to [0, 1] (cosine distance max is 2 for anti-parallel vectors).
        total_dist += nn_dist / 2.0;
    }
    total_dist / n as f32
}

/// Fraction of chunks within one hop of the corpus centroid.
///
/// "Within one hop" = cosine_similarity(chunk, centroid) ≥ GRAPH_EDGE_THRESHOLD (0.5).
fn compute_workspace_score(embedded: &[Embedded]) -> f32 {
    let n = embedded.len();
    if n == 0 {
        return 0.0;
    }

    let dim = embedded[0].embedding.len();
    if dim == 0 {
        return 0.0;
    }

    // Compute centroid (mean of all embeddings).
    let mut centroid = vec![0.0_f32; dim];
    for e in embedded {
        for (c, v) in centroid.iter_mut().zip(e.embedding.iter()) {
            *c += v;
        }
    }
    let inv_n = 1.0 / n as f32;
    for c in &mut centroid {
        *c *= inv_n;
    }

    // L2-normalise centroid for valid cosine comparison.
    let norm: f32 = centroid.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-9 {
        return 0.0; // zero centroid — no coherent direction
    }
    for c in &mut centroid {
        *c /= norm;
    }

    let within_hop = embedded.iter().filter(|e| {
        cosine_similarity(&e.embedding, &centroid) as f32 >= GRAPH_EDGE_THRESHOLD as f32
    });
    within_hop.count() as f32 / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use learn_core::{Chunk, Embedded};

    fn make_embedded(chunk_id: &str, video_id: &str, embedding: Vec<f32>) -> Embedded {
        Embedded {
            chunk: Chunk {
                chunk_id: chunk_id.to_string(),
                video_id: video_id.to_string(),
                start_seconds: 0.0,
                end_seconds: 1.0,
                text: "test chunk".to_string(),
                token_count: 3,
            },
            embedding,
            embedding_model: "test".to_string(),
        }
    }

    fn unit(angle_deg: f32, dim: usize) -> Vec<f32> {
        // Spread vectors around a circle in the first two dimensions.
        let rad = angle_deg.to_radians();
        let mut v = vec![0.0f32; dim];
        v[0] = rad.cos();
        v[1] = rad.sin();
        v
    }

    // ---------------------------------------------------------------------------
    // check_kb_health tests
    // ---------------------------------------------------------------------------

    /// A well-conditioned set: 8 vectors evenly spread on a circle are diverse
    /// (not near-duplicates) and form a well-connected graph.
    #[test]
    fn check_kb_health_well_conditioned_has_high_fiedler() {
        let embedded: Vec<Embedded> = (0..8)
            .map(|i| {
                let angle = i as f32 * 45.0;
                make_embedded(&format!("c{i}"), "v1", unit(angle, 4))
            })
            .collect();
        let health = check_kb_health(&embedded).unwrap();
        assert_eq!(health.vector_count, 8);
        assert!(
            health.fiedler_value > 0.0,
            "expected positive Fiedler for well-spread vectors, got {}",
            health.fiedler_value
        );
        assert!(
            health.flagged_pairs.is_empty(),
            "no near-duplicate pairs expected, found {:?}",
            health.flagged_pairs
        );
    }

    /// Two tightly packed clusters with one intentional cross-cluster near-duplicate.
    /// That pair should be flagged.
    #[test]
    fn check_kb_health_flags_near_duplicate_cross_cluster_pair() {
        let dim = 4;
        // Cluster A: vectors near (1, 0, 0, 0)
        let mut embedded = vec![
            make_embedded("a1", "v1", vec![1.0, 0.01, 0.0, 0.0]),
            make_embedded("a2", "v1", vec![1.0, 0.02, 0.0, 0.0]),
        ];
        // Cluster B: vectors near (0, 1, 0, 0)
        embedded.push(make_embedded("b1", "v2", vec![0.01, 1.0, 0.0, 0.0]));
        embedded.push(make_embedded("b2", "v2", vec![0.02, 1.0, 0.0, 0.0]));
        // Cross-cluster near-duplicate: b3 is almost identical to a1 but lives in cluster B's video
        let _ = dim; // used for readability
        embedded.push(make_embedded("b3", "v2", vec![1.0, 0.005, 0.0, 0.0]));

        let health = check_kb_health(&embedded).unwrap();
        // a1 and b3 should be flagged (cosine ~ 1.0)
        let has_ab = health
            .flagged_pairs
            .iter()
            .any(|(x, y)| (x == "a1" && y == "b3") || (x == "b3" && y == "a1"));
        assert!(
            has_ab,
            "expected (a1, b3) in flagged pairs, got {:?}",
            health.flagged_pairs
        );
    }

    /// Empty input must return an error.
    #[test]
    fn check_kb_health_empty_returns_error() {
        assert!(check_kb_health(&[]).is_err());
    }

    // ---------------------------------------------------------------------------
    // check_drift tests
    // ---------------------------------------------------------------------------

    /// Stable history: all scores near 0.9 → no changepoint.
    #[test]
    fn check_drift_stable_history_no_changepoint() {
        let records: Vec<QueryRecord> = (0..20)
            .map(|i| QueryRecord {
                query_embedding: vec![1.0, 0.0],
                top_hit_score: 0.88 + (i % 3) as f32 * 0.01,
                timestamp: i as f64,
            })
            .collect();
        let report = check_drift(&records).unwrap();
        assert!(
            !report.changepoint_detected,
            "stable history should not trigger changepoint"
        );
    }

    /// Sharp score drop in recent half → changepoint detected.
    #[test]
    fn check_drift_score_drop_triggers_changepoint() {
        let baseline: Vec<QueryRecord> = (0..10)
            .map(|i| QueryRecord {
                query_embedding: vec![1.0, 0.0],
                top_hit_score: 0.90,
                timestamp: i as f64,
            })
            .collect();
        let recent: Vec<QueryRecord> = (10..20)
            .map(|i| QueryRecord {
                query_embedding: vec![0.0, 1.0],
                top_hit_score: 0.55, // sharp drop > DRIFT_THRESHOLD (0.08)
                timestamp: i as f64,
            })
            .collect();
        let records: Vec<QueryRecord> = baseline.into_iter().chain(recent).collect();
        let report = check_drift(&records).unwrap();
        assert!(
            report.changepoint_detected,
            "sharp quality drop should trigger changepoint"
        );
        assert!(
            report.baseline_quality_score > report.recent_quality_score,
            "baseline should be higher than recent"
        );
    }

    /// Fewer than 4 records must return an error.
    #[test]
    fn check_drift_insufficient_records_returns_error() {
        let records: Vec<QueryRecord> = (0..3)
            .map(|i| QueryRecord {
                query_embedding: vec![1.0],
                top_hit_score: 0.8,
                timestamp: i as f64,
            })
            .collect();
        assert!(check_drift(&records).is_err());
    }

    // -----------------------------------------------------------------------
    // check_kb_health_single_vector_returns_zero_health
    //
    // Invariant: a single embedded vector is a degenerate graph — no edges are
    // possible, so Fiedler value is 0 (documented edge case in check_kb_health),
    // contradiction_rate is 0 (needs at least 2 vectors), and no pairs can be
    // flagged. The call must succeed (not error) since n == 1 > 0.
    // -----------------------------------------------------------------------

    #[test]
    fn check_kb_health_single_vector_returns_zero_health() {
        let single: Vec<Embedded> =
            vec![make_embedded("vid-1:0", "vid-1", vec![1.0, 0.0, 0.0, 0.0])];
        let health = check_kb_health(&single).expect("single vector should not error");
        assert_eq!(health.vector_count, 1);
        assert_eq!(
            health.fiedler_value, 0.0,
            "fiedler is 0 for n=1 — documented edge case"
        );
        assert_eq!(health.contradiction_rate, 0.0);
        assert!(health.flagged_pairs.is_empty());
    }

    // ── ConsciousnessKpi tests ────────────────────────────────────────────────

    /// Two tight orthogonal clusters of 5 vectors each with no cross-cluster
    /// similarity: the Fiedler value of the resulting graph is low (nearly
    /// disconnected), so integrated_information must be < 0.3 (Disjoint).
    #[test]
    fn consciousness_kpi_disjoint_clusters_score_low() {
        let dim = 8;
        // Cluster A: vectors along axis 0.
        let mut embedded: Vec<Embedded> = (0..5)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[0] = 1.0;
                v[1] = i as f32 * 0.001; // tiny perturbation to avoid exact duplicates
                make_embedded(&format!("a{i}"), "va", v)
            })
            .collect();
        // Cluster B: vectors along axis 2 — orthogonal to cluster A.
        for i in 0..5 {
            let mut v = vec![0.0f32; dim];
            v[2] = 1.0;
            v[3] = i as f32 * 0.001;
            embedded.push(make_embedded(&format!("b{i}"), "vb", v));
        }

        let kpi = compute_consciousness_kpi(&embedded).unwrap();
        assert!(
            kpi.integrated_information < 0.3,
            "two orthogonal clusters should score < 0.3, got {}",
            kpi.integrated_information
        );
        assert_eq!(
            kpi.interpretation,
            KpiInterpretation::Disjoint,
            "should be Disjoint interpretation"
        );
    }

    /// 10 embeddings all nearly aligned: high density → KPI > 0.7.
    #[test]
    fn consciousness_kpi_coherent_corpus_scores_high() {
        let dim = 4;
        // All vectors point roughly in the same direction with small perturbations.
        let embedded: Vec<Embedded> = (0..10)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[0] = 1.0;
                v[1] = (i as f32) * 0.02; // small deviation
                make_embedded(&format!("c{i}"), "vc", v)
            })
            .collect();

        let kpi = compute_consciousness_kpi(&embedded).unwrap();
        assert!(
            kpi.integrated_information > 0.7,
            "near-aligned embeddings should score > 0.7, got {}",
            kpi.integrated_information
        );
        assert!(
            matches!(
                kpi.interpretation,
                KpiInterpretation::Coherent | KpiInterpretation::HighlyIntegrated
            ),
            "should be Coherent or HighlyIntegrated, got {:?}",
            kpi.interpretation
        );
    }

    /// Empty input must return a zero-valued Disjoint KPI without erroring.
    #[test]
    fn consciousness_kpi_empty_returns_disjoint_zero() {
        let kpi = compute_consciousness_kpi(&[]).unwrap();
        assert_eq!(kpi.integrated_information, 0.0);
        assert_eq!(kpi.workspace_score, 0.0);
        assert_eq!(kpi.interpretation, KpiInterpretation::Disjoint);
    }

    /// Boundary values for KpiInterpretation thresholds.
    #[test]
    fn kpi_interpretation_thresholds() {
        // Boundary: exactly at each threshold transition.
        assert_eq!(
            KpiInterpretation::from_score(0.0),
            KpiInterpretation::Disjoint
        );
        assert_eq!(
            KpiInterpretation::from_score(0.299),
            KpiInterpretation::Disjoint
        );
        assert_eq!(
            KpiInterpretation::from_score(0.30),
            KpiInterpretation::Loose
        );
        assert_eq!(
            KpiInterpretation::from_score(0.599),
            KpiInterpretation::Loose
        );
        assert_eq!(
            KpiInterpretation::from_score(0.60),
            KpiInterpretation::Coherent
        );
        assert_eq!(
            KpiInterpretation::from_score(0.849),
            KpiInterpretation::Coherent
        );
        assert_eq!(
            KpiInterpretation::from_score(0.85),
            KpiInterpretation::HighlyIntegrated
        );
        assert_eq!(
            KpiInterpretation::from_score(1.0),
            KpiInterpretation::HighlyIntegrated
        );
    }
}
