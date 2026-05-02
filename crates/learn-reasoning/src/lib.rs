//! `learn-reasoning` — JSONL-backed trajectory store with cosine retrieval.
//!
//! # Storage design
//!
//! Each topic's bank lives at `<kb_root>/_reasoning/<topic>.rbank`.  The file
//! is newline-delimited JSON: one `Trajectory` per line.  On `record`, a new
//! line is appended and the file is flushed.  On `retrieve`, all lines are
//! deserialized and ranked by cosine similarity against the query embedding,
//! filtered by `mode`, then truncated to `k`.
//!
//! JSONL + linear scan was chosen over `LearnIndex`/RvfStore because:
//! - RvfStore requires the embedding dimension at file-creation time;
//!   trajectories may come from different models across a topic's lifetime.
//! - RvfStore returns only `(id, distance)`, requiring a separate sidecar to
//!   reconstruct payloads — two files to keep in sync.
//! - Expected trajectory counts (tens to low hundreds) make linear scan correct
//!   and auditable.  A future compaction step can promote to HNSW when needed.

#![deny(unsafe_code)]

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};

use camino::{Utf8Path, Utf8PathBuf};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use learn_core::{LearnError, Result, Topic};

// ──────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────

/// Whether the trajectory came from an Ask query or an Apply action.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrajectoryMode {
    Ask,
    Apply,
}

impl TrajectoryMode {
    fn as_str(self) -> &'static str {
        match self {
            TrajectoryMode::Ask => "ask",
            TrajectoryMode::Apply => "apply",
        }
    }
}

/// One recorded past trajectory, persisted verbatim in the `.rbank` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    /// Stable 16-hex-char ID derived from `(task, topic, mode, timestamp)`.
    pub trajectory_id: String,
    pub topic: String,
    /// The user's question or apply-task text.
    pub task: String,
    pub mode: TrajectoryMode,
    /// Dense embedding of `task`; used for cosine retrieval.
    pub query_embedding: Vec<f32>,
    /// Chunk IDs that fed the synthesis step.
    pub used_chunk_ids: Vec<String>,
    /// 0.0–1.0 quality signal.
    pub outcome_score: f32,
    /// For Apply trajectories: short summary of the produced artifact.
    pub artifact_summary: String,
    /// Unix seconds at record time.
    pub timestamp_secs: i64,
}

/// Summary statistics for `learn status`.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryStats {
    pub trajectory_count: usize,
    pub avg_outcome_score: f32,
}

// ──────────────────────────────────────────────────────────────────
// ID derivation  (PINNED — do NOT change the SHA-256 recipe)
// ──────────────────────────────────────────────────────────────────

/// Derive a stable 16-hex-char trajectory id.
///
/// Recipe: `SHA-256( task NUL topic NUL mode_str NUL timestamp_decimal )`
/// Take the first 8 bytes (16 hex chars) of the digest.
pub fn derive_trajectory_id(
    task: &str,
    topic: &str,
    mode: TrajectoryMode,
    timestamp_secs: i64,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(task.as_bytes());
    hasher.update(b"\x00");
    hasher.update(topic.as_bytes());
    hasher.update(b"\x00");
    hasher.update(mode.as_str().as_bytes());
    hasher.update(b"\x00");
    hasher.update(timestamp_secs.to_string().as_bytes());
    let digest = hasher.finalize();
    format!(
        "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    )
}

// ──────────────────────────────────────────────────────────────────
// ReasoningBank
// ──────────────────────────────────────────────────────────────────

/// Persistent trajectory store for a single topic.
///
/// Backed by `<kb_root>/_reasoning/<topic>.rbank` (newline-delimited JSON).
pub struct ReasoningBank {
    path: Utf8PathBuf,
}

impl ReasoningBank {
    /// Open or create the reasoning bank for `topic` under `kb_root`.
    ///
    /// Creates `<kb_root>/_reasoning/` if it does not exist.
    pub fn open(kb_root: &Utf8Path, topic: Topic) -> Result<Self> {
        let dir = kb_root.join("_reasoning");
        fs::create_dir_all(&dir).map_err(LearnError::Io)?;
        let path = dir.join(format!("{}.rbank", topic.as_str()));
        // Touch the file if it doesn't exist yet (ensures it is writable).
        if !path.as_std_path().exists() {
            File::create(&path).map_err(LearnError::Io)?;
        }
        Ok(Self { path })
    }

    /// Append a trajectory to the bank file.
    pub fn record(&mut self, t: &Trajectory) -> Result<()> {
        let line = serde_json::to_string(t)?;
        let mut file = OpenOptions::new()
            .append(true)
            .open(&self.path)
            .map_err(LearnError::Io)?;
        writeln!(file, "{line}").map_err(LearnError::Io)?;
        file.flush().map_err(LearnError::Io)?;
        Ok(())
    }

    /// Return the top-`k` past trajectories most similar to `query_embedding`,
    /// restricted to trajectories with the given `mode`.
    ///
    /// Returns an empty `Vec` (not `Err`) when the bank is empty or no
    /// trajectories match the mode filter.
    pub fn retrieve(
        &self,
        query_embedding: &[f32],
        mode: TrajectoryMode,
        k: usize,
    ) -> Result<Vec<Trajectory>> {
        let trajectories = self.load_all()?;
        if trajectories.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let query_norm = l2_norm(query_embedding);

        let mut scored: Vec<(f32, Trajectory)> = trajectories
            .into_iter()
            .filter(|t| t.mode == mode)
            .filter_map(|t| {
                if t.query_embedding.len() != query_embedding.len() {
                    return None;
                }
                let score = cosine_similarity(query_embedding, &t.query_embedding, query_norm);
                Some((score, t))
            })
            .collect();

        // Highest similarity first.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored.into_iter().map(|(_, t)| t).collect())
    }

    /// Return aggregate stats without loading embeddings into an HNSW; all
    /// data is already in JSONL so we just scan once.
    pub fn stats(&self) -> TrajectoryStats {
        let trajectories = self.load_all().unwrap_or_default();
        let count = trajectories.len();
        let avg = if count == 0 {
            0.0
        } else {
            let sum: f32 = trajectories.iter().map(|t| t.outcome_score).sum();
            sum / count as f32
        };
        TrajectoryStats {
            trajectory_count: count,
            avg_outcome_score: avg,
        }
    }

    // ── private ────────────────────────────────────────────────────

    fn load_all(&self) -> Result<Vec<Trajectory>> {
        let file = File::open(&self.path).map_err(LearnError::Io)?;
        let reader = BufReader::new(file);
        let mut out = Vec::new();
        for (lineno, line) in reader.lines().enumerate() {
            let line = line.map_err(LearnError::Io)?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let t: Trajectory = serde_json::from_str(trimmed).map_err(|e| {
                LearnError::Retrieve(format!("rbank parse error at line {}: {e}", lineno + 1))
            })?;
            out.push(t);
        }
        Ok(out)
    }
}

// ──────────────────────────────────────────────────────────────────
// Math helpers
// ──────────────────────────────────────────────────────────────────

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity in [-1, 1].  Returns 0.0 for zero vectors.
fn cosine_similarity(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let b_norm = l2_norm(b);
    if a_norm == 0.0 || b_norm == 0.0 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (a_norm * b_norm)
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use camino::Utf8PathBuf;
    use tempfile::TempDir;

    fn kb_root(tmp: &TempDir) -> Utf8PathBuf {
        Utf8PathBuf::from_path_buf(tmp.path().to_path_buf()).expect("tempdir path is valid UTF-8")
    }

    fn make_trajectory(
        task: &str,
        topic: &str,
        mode: TrajectoryMode,
        embedding: Vec<f32>,
        score: f32,
    ) -> Trajectory {
        let ts = 1_700_000_000i64;
        Trajectory {
            trajectory_id: derive_trajectory_id(task, topic, mode, ts),
            topic: topic.to_owned(),
            task: task.to_owned(),
            mode,
            query_embedding: embedding,
            used_chunk_ids: vec!["c1".into()],
            outcome_score: score,
            artifact_summary: String::new(),
            timestamp_secs: ts,
        }
    }

    // ── 1. record then retrieve finds similar ──────────────────────

    #[test]
    fn record_then_retrieve_finds_similar() {
        let tmp = TempDir::new().unwrap();
        let root = kb_root(&tmp);
        let topic = Topic::new("test-topic").unwrap();

        let mut bank = ReasoningBank::open(&root, topic.clone()).unwrap();

        // Two trajectories: one near [1,0,0], one near [0,1,0].
        let t_near = make_trajectory(
            "near task",
            topic.as_str(),
            TrajectoryMode::Ask,
            vec![1.0, 0.0, 0.0],
            0.9,
        );
        let t_far = make_trajectory(
            "far task",
            topic.as_str(),
            TrajectoryMode::Ask,
            vec![0.0, 1.0, 0.0],
            0.8,
        );
        bank.record(&t_near).unwrap();
        bank.record(&t_far).unwrap();

        // Query close to [1,0,0].
        let results = bank
            .retrieve(&[0.9, 0.1, 0.0], TrajectoryMode::Ask, 2)
            .unwrap();
        assert_eq!(
            results[0].trajectory_id, t_near.trajectory_id,
            "most similar trajectory should be rank 0"
        );
    }

    // ── 2. retrieve filters by mode ────────────────────────────────

    #[test]
    fn retrieve_filters_by_mode() {
        let tmp = TempDir::new().unwrap();
        let root = kb_root(&tmp);
        let topic = Topic::new("mode-filter").unwrap();

        let mut bank = ReasoningBank::open(&root, topic.clone()).unwrap();

        let ask_t = make_trajectory(
            "ask task",
            topic.as_str(),
            TrajectoryMode::Ask,
            vec![1.0, 0.0],
            0.9,
        );
        let apply_t = make_trajectory(
            "apply task",
            topic.as_str(),
            TrajectoryMode::Apply,
            vec![1.0, 0.0],
            0.8,
        );
        bank.record(&ask_t).unwrap();
        bank.record(&apply_t).unwrap();

        // Querying with Ask should not surface the Apply trajectory.
        let results = bank.retrieve(&[1.0, 0.0], TrajectoryMode::Ask, 10).unwrap();
        assert!(
            results.iter().all(|t| t.mode == TrajectoryMode::Ask),
            "Ask query must not surface Apply trajectories"
        );
        assert_eq!(results.len(), 1);

        // And vice versa.
        let apply_results = bank
            .retrieve(&[1.0, 0.0], TrajectoryMode::Apply, 10)
            .unwrap();
        assert_eq!(apply_results.len(), 1);
        assert_eq!(apply_results[0].mode, TrajectoryMode::Apply);
    }

    // ── 3. persists across reopen ──────────────────────────────────

    #[test]
    fn persists_across_reopen() {
        let tmp = TempDir::new().unwrap();
        let root = kb_root(&tmp);
        let topic = Topic::new("persist-topic").unwrap();

        let recorded_id = {
            let mut bank = ReasoningBank::open(&root, topic.clone()).unwrap();
            let t = make_trajectory(
                "persist task",
                topic.as_str(),
                TrajectoryMode::Ask,
                vec![1.0, 0.0, 0.0],
                0.95,
            );
            let id = t.trajectory_id.clone();
            bank.record(&t).unwrap();
            // bank drops here, file handle closed.
            id
        };

        // Reopen — new ReasoningBank instance reads from disk.
        let bank2 = ReasoningBank::open(&root, topic).unwrap();
        let results = bank2
            .retrieve(&[1.0, 0.0, 0.0], TrajectoryMode::Ask, 1)
            .unwrap();
        assert_eq!(results.len(), 1, "trajectory must survive reopen");
        assert_eq!(results[0].trajectory_id, recorded_id);
    }

    // ── 4. trajectory_id deterministic ────────────────────────────

    #[test]
    fn trajectory_id_deterministic() {
        let id1 = derive_trajectory_id("my task", "cooking", TrajectoryMode::Ask, 1_700_000_000);
        let id2 = derive_trajectory_id("my task", "cooking", TrajectoryMode::Ask, 1_700_000_000);
        assert_eq!(id1, id2, "same inputs must produce same id");
        assert_eq!(id1.len(), 16, "id must be 16 hex chars");

        // Different inputs → different id.
        let id3 = derive_trajectory_id("other task", "cooking", TrajectoryMode::Ask, 1_700_000_000);
        assert_ne!(id1, id3);
    }

    // ── 5. empty bank returns empty vec ───────────────────────────

    #[test]
    fn empty_bank_returns_empty_vec() {
        let tmp = TempDir::new().unwrap();
        let root = kb_root(&tmp);
        let topic = Topic::new("empty-bank").unwrap();

        let bank = ReasoningBank::open(&root, topic).unwrap();
        let results = bank
            .retrieve(&[0.0, 1.0, 0.0], TrajectoryMode::Ask, 5)
            .unwrap();
        assert!(
            results.is_empty(),
            "fresh bank must return empty vec, not Err"
        );
    }

    // ── 6. stats reports avg score ────────────────────────────────

    #[test]
    fn stats_reports_avg_score() {
        let tmp = TempDir::new().unwrap();
        let root = kb_root(&tmp);
        let topic = Topic::new("stats-topic").unwrap();

        let mut bank = ReasoningBank::open(&root, topic.clone()).unwrap();

        let scores = [0.4f32, 0.6, 0.8];
        for (i, &score) in scores.iter().enumerate() {
            let t = Trajectory {
                trajectory_id: derive_trajectory_id(
                    &format!("task {i}"),
                    topic.as_str(),
                    TrajectoryMode::Ask,
                    i as i64,
                ),
                topic: topic.as_str().to_owned(),
                task: format!("task {i}"),
                mode: TrajectoryMode::Ask,
                query_embedding: vec![i as f32, 0.0],
                used_chunk_ids: vec![],
                outcome_score: score,
                artifact_summary: String::new(),
                timestamp_secs: i as i64,
            };
            bank.record(&t).unwrap();
        }

        let stats = bank.stats();
        assert_eq!(stats.trajectory_count, 3);
        let expected_avg = (0.4 + 0.6 + 0.8) / 3.0f32;
        assert!(
            (stats.avg_outcome_score - expected_avg).abs() < 1e-5,
            "avg_outcome_score {:.4} != expected {:.4}",
            stats.avg_outcome_score,
            expected_avg
        );
    }
}
