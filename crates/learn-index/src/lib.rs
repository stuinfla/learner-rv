//! `learn-index` — per-topic `.rvf` store facade for the learn-rs pipeline.
//!
//! # Scale path: `LearnIndexLarge`
//!
//! When a topic crosses `LARGE_INDEX_THRESHOLD` vectors, callers may opt-in to
//! `LearnIndexLarge` instead of `LearnIndex`.  It wraps `DiskAnnIndex` (Vamana
//! graph + optional Product Quantization) from `ruvector-diskann`.
//!
//! The same five public methods apply — `open`, `ingest`, `search`, `stats`,
//! `compact` — with the same signatures as `LearnIndex`.
//!
//! **Compact/promote discipline**: DiskANN is build-once.  Any `ingest` after
//! `compact` has been called appends to a staging buffer; the next `compact`
//! rebuilds the graph from scratch.  Callers should treat `compact` as a
//! periodic rebuild, not an incremental flush.
//!
//! **PQ**: disabled (`pq_subspaces = 0`) until calibration data is available.
//! Enable at the `DiskAnnConfig` level once your topic has enough vectors to
//! train 256 centroids per sub-space (minimum ~1 000 vectors recommended).
//!
//! # Actual rvf-runtime API (verified 2026-05-02)
//!
//! ## Opening / creating
//! ```text
//! RvfStore::create(path: &Path, opts: RvfOptions) -> Result<Self, RvfError>
//!   - opts.dimension MUST be > 0 (zero returns InvalidManifest)
//!   - file must not already exist (create_new semantics)
//! RvfStore::open(path: &Path) -> Result<Self, RvfError>
//!   - read-write; acquires advisory writer lock
//!   - boot() reads manifest, restores options.dimension automatically
//! RvfStore::open_readonly(path: &Path) -> Result<Self, RvfError>
//!   - no writer lock; safe for concurrent reads
//! store.dimension() -> u16   -- available after create or open
//! ```
//!
//! ## Mutations
//! ```text
//! store.ingest_batch(
//!     vectors:  &[&[f32]],
//!     ids:      &[u64],
//!     metadata: Option<&[MetadataEntry]>,
//! ) -> Result<IngestResult, RvfError>
//!   IngestResult { accepted: u64, rejected: u64, epoch: u32 }
//!
//! store.compact() -> Result<CompactionResult, RvfError>
//! store.close(self) -> Result<(), RvfError>   // consumes self
//! ```
//!
//! ## Queries
//! ```text
//! store.query(vector: &[f32], k: usize, options: &QueryOptions)
//!     -> Result<Vec<SearchResult>, RvfError>
//!   SearchResult { id: u64, distance: f32, retrieval_quality }
//!
//! store.status() -> StoreStatus
//!   StoreStatus { total_vectors: u64, total_segments: u32, file_size: u64 }
//! ```
//!
//! ## Metadata limitation and sidecar design
//! `store.query()` returns only `(id, distance)`.  The in-memory `MetadataStore`
//! has no public retrieval API so chunk payloads cannot be round-tripped through
//! the RVF metadata layer alone.  This facade keeps a companion JSON sidecar
//! `<topic>.meta.json` with schema `{ dimension, chunks: { "<id>": Chunk } }`.
//! The vector id is an FNV-1a 64-bit hash of `chunk.chunk_id` (no extra crate).
//!
//! ## Lazy creation
//! `RvfStore::create` rejects dimension=0.  We defer file creation until the
//! first `ingest`, when the embedding dimension is known.

#![deny(unsafe_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use camino::Utf8Path;
use ruvector_diskann::{DiskAnnConfig, DiskAnnIndex};
use rvf_runtime::store::RvfStore;
use rvf_runtime::{QueryOptions, RvfOptions};
use serde::{Deserialize, Serialize};

use learn_core::{Chunk, Embedded, Hit, LearnError, Manifest, Result, Topic, VideoState};

// ── Scale threshold ──────────────────────────────────────────────────────────

/// Number of vectors at which a topic is considered "large".
///
/// Below this threshold use [`LearnIndex`] (in-memory RVF HNSW).
/// Above this threshold the caller should opt-in to [`LearnIndexLarge`]
/// (DiskANN + Vamana graph, SSD-friendly, mmap persistence).
///
/// The threshold is a hint — migration never happens automatically mid-ingest.
pub const LARGE_INDEX_THRESHOLD: usize = 50_000;

// ── Field-ID constants (forward-compat stub for rvf metadata API) ───────────
#[allow(dead_code)]
mod field {
    pub const VIDEO_ID: u16 = 0;
    pub const CHUNK_ID: u16 = 1;
    pub const START_SEC: u16 = 2;
    pub const END_SEC: u16 = 3;
}

// ── Sidecar schema ──────────────────────────────────────────────────────────

/// Owned sidecar — used only for deserialisation on open.
#[derive(Serialize, Deserialize, Default)]
struct MetaSidecar {
    /// Embedding dimension; 0 means the store file does not exist yet.
    dimension: u16,
    /// Maps string(u64 id) → Chunk for citation retrieval after search.
    chunks: HashMap<String, Chunk>,
}

/// Borrowing sidecar view — used for serialisation in `save_meta` to avoid
/// cloning the entire chunks map.
#[derive(Serialize)]
struct MetaSidecarRef<'a> {
    dimension: u16,
    chunks: &'a HashMap<String, Chunk>,
}

// ── Witness chain ────────────────────────────────────────────────────────────

/// One entry in the per-chunk provenance witness chain.
///
/// Entries are linked: each entry's `digest` covers all its own fields plus
/// `previous_hash`, forming an append-only hash chain.  Tampering with any
/// entry, or changing the order, breaks verification.
///
/// Persisted to `<kb_root>/<topic>.witness.json` via atomic tmp+rename.
/// Loaded in full on `LearnIndex::open`.  The chain grows by one entry per
/// chunk per `ingest` call; there is no compaction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WitnessEntry {
    /// 1-based sequence number — first entry is 1.
    pub seq: u64,
    /// FNV-1a decimal string of the chunk, same key used in the sidecar.
    pub chunk_id: String,
    /// Source video identifier.
    pub video_id: String,
    /// Source URL if known at ingest time (from `Chunk` — currently `None`).
    pub source_url: Option<String>,
    /// Start timestamp within the video.
    pub start_seconds: f64,
    /// Unix epoch seconds at ingestion time.
    pub ingested_at: i64,
    /// `[0u8; 32]` for the first entry; otherwise the `digest` of entry seq−1.
    pub previous_hash: [u8; 32],
    /// Blake3 over the canonical input (see `compute_digest`).
    pub digest: [u8; 32],
}

impl WitnessEntry {
    /// Canonical byte input for the Blake3 digest.
    ///
    /// Layout: `seq(8 LE) || chunk_id bytes || 0x00 || video_id bytes || 0x00 ||
    ///          source_url_flag(1) || [source_url bytes || 0x00] ||
    ///          start_seconds(8 LE f64) || ingested_at(8 LE i64) || previous_hash(32)`
    fn canonical_input(
        seq: u64,
        chunk_id: &str,
        video_id: &str,
        source_url: Option<&str>,
        start_seconds: f64,
        ingested_at: i64,
        previous_hash: &[u8; 32],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&seq.to_le_bytes());
        buf.extend_from_slice(chunk_id.as_bytes());
        buf.push(0x00);
        buf.extend_from_slice(video_id.as_bytes());
        buf.push(0x00);
        if let Some(url) = source_url {
            buf.push(0x01);
            buf.extend_from_slice(url.as_bytes());
            buf.push(0x00);
        } else {
            buf.push(0x00);
        }
        buf.extend_from_slice(&start_seconds.to_le_bytes());
        buf.extend_from_slice(&ingested_at.to_le_bytes());
        buf.extend_from_slice(previous_hash);
        buf
    }

    /// Compute the Blake3 digest for this entry's fields.
    pub fn compute_digest(
        seq: u64,
        chunk_id: &str,
        video_id: &str,
        source_url: Option<&str>,
        start_seconds: f64,
        ingested_at: i64,
        previous_hash: &[u8; 32],
    ) -> [u8; 32] {
        let input = Self::canonical_input(
            seq,
            chunk_id,
            video_id,
            source_url,
            start_seconds,
            ingested_at,
            previous_hash,
        );
        *blake3::hash(&input).as_bytes()
    }

    /// Build a new entry chained from `previous_hash`.
    pub fn new(
        seq: u64,
        chunk_id: String,
        video_id: String,
        source_url: Option<String>,
        start_seconds: f64,
        ingested_at: i64,
        previous_hash: [u8; 32],
    ) -> Self {
        let digest = Self::compute_digest(
            seq,
            &chunk_id,
            &video_id,
            source_url.as_deref(),
            start_seconds,
            ingested_at,
            &previous_hash,
        );
        Self {
            seq,
            chunk_id,
            video_id,
            source_url,
            start_seconds,
            ingested_at,
            previous_hash,
            digest,
        }
    }
}

// ── Public types ────────────────────────────────────────────────────────────

/// Stats for `learn status`.
pub struct IndexStats {
    pub vector_count: usize,
    pub segment_count: usize,
    pub bytes_on_disk: u64,
}

/// Per-topic RVF store handle.
///
/// The underlying `RvfStore` file is created lazily on the first `ingest` call
/// because `RvfStore::create` requires knowing the embedding dimension upfront
/// and we cannot know it until the first batch arrives.
pub struct LearnIndex {
    /// None until the first ingest or when the .rvf file does not exist yet.
    store: Option<RvfStore>,
    /// Path to the `.rvf` file (used for lazy creation).
    rvf_path: PathBuf,
    /// In-memory chunk map: str(u64 id) → Chunk.
    chunks: HashMap<String, Chunk>,
    /// Path to the `<topic>.meta.json` companion file.
    meta_path: PathBuf,
    /// In-memory embedding map: u64 id → dense vector.
    /// Loaded once from `<topic>.emb.bin` on open; extended on every ingest.
    embeddings: HashMap<u64, Vec<f32>>,
    /// Path to the `<topic>.emb.bin` companion file.
    emb_path: PathBuf,
    /// Crash-recovery manifest: per-video `IngestStatus` checkpoints.
    /// Loaded on open from `<kb_root>/_meta/<topic>.json`; written atomically
    /// after each pipeline stage via `save_manifest`.
    manifest: Manifest,
    /// Path to the manifest file (`<kb_root>/_meta/<topic>.json`).
    manifest_path: PathBuf,
    /// Append-only Blake3 witness chain — one entry per ingested chunk.
    /// Loaded on open from `<topic>.witness.json`; absent file = empty chain.
    witness_chain: Vec<WitnessEntry>,
    /// Path to `<topic>.witness.json`.
    witness_path: PathBuf,
}

impl LearnIndex {
    /// Open or create a per-topic `.rvf` file at `<kb_root>/<topic>.rvf`.
    ///
    /// The `.rvf` file is only created when `ingest` is first called.
    /// If a `<topic>.emb.bin` file exists it is loaded into the in-memory
    /// embedding map so `embedding_for_chunk_id` can serve real cosine inputs
    /// to the MMR stage without any model re-inference.
    ///
    /// The crash-recovery manifest is loaded from
    /// `<kb_root>/_meta/<topic>.json` (absent → empty default).
    pub fn open(kb_root: &Utf8Path, topic: Topic) -> Result<Self> {
        let rvf_path_utf8 = topic.rvf_path(kb_root);
        let rvf_path = PathBuf::from(rvf_path_utf8.as_str());
        let meta_path = Self::meta_path_for(kb_root, &topic);
        let emb_path = Self::emb_path_for(kb_root, &topic);
        let manifest_path = PathBuf::from(topic.manifest_path(kb_root).as_str());
        let witness_path = Self::witness_path_for(kb_root, &topic);
        let sidecar = load_sidecar(&meta_path)?;
        let embeddings = load_emb_file(&emb_path)?;
        let manifest = load_manifest(&manifest_path)?;
        let witness_chain = load_witness(&witness_path)?;

        let store = if rvf_path.exists() {
            Some(RvfStore::open(&rvf_path).map_err(|e| LearnError::Index(format!("{e:?}")))?)
        } else {
            None
        };

        Ok(Self {
            store,
            rvf_path,
            chunks: sidecar.chunks,
            meta_path,
            embeddings,
            emb_path,
            manifest,
            manifest_path,
            witness_chain,
            witness_path,
        })
    }

    // ── Manifest API ────────────────────────────────────────────────────────

    /// Immutable view of the in-memory manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Insert or replace the `VideoState` for one video, then persist atomically.
    ///
    /// The manifest is written to a `.json.tmp` sibling of the manifest path
    /// and renamed in place — the same pattern used by `save_meta` for the
    /// chunk sidecar.
    pub fn upsert_video_state(&mut self, vs: VideoState) -> Result<()> {
        self.manifest.videos.insert(vs.video_id.clone(), vs);
        self.save_manifest()
    }

    /// Atomically persist the current in-memory manifest to disk.
    ///
    /// Uses tmp + rename so a kill between the write and rename leaves either
    /// the old file or the new file intact — never a half-written JSON.
    pub fn save_manifest(&self) -> Result<()> {
        // Ensure the _meta directory exists.
        if let Some(parent) = self.manifest_path.parent() {
            std::fs::create_dir_all(parent).map_err(LearnError::Io)?;
        }
        let json = serde_json::to_string_pretty(&self.manifest)?;
        let tmp = self.manifest_path.with_extension("json.tmp");
        std::fs::write(&tmp, json.as_bytes()).map_err(LearnError::Io)?;
        std::fs::rename(&tmp, &self.manifest_path).map_err(LearnError::Io)
    }

    // ── Witness-chain API ────────────────────────────────────────────────────

    /// Read-only view of the in-memory witness chain.
    ///
    /// Each entry was appended by `ingest` and links to the previous via
    /// `previous_hash`.  The chain can be verified with `verify_witness_chain`.
    pub fn witness_chain(&self) -> &[WitnessEntry] {
        &self.witness_chain
    }

    /// Verify the complete witness chain.
    ///
    /// Checks both the per-entry digest and the `previous_hash` linkage.
    /// Returns `Err` on any tampering or chain break; `Ok(())` for an empty
    /// chain or a fully intact chain.
    pub fn verify_witness_chain(&self) -> Result<()> {
        let mut prev_hash = [0u8; 32];
        for entry in &self.witness_chain {
            // 1. previous_hash must match the digest of the prior entry.
            if entry.previous_hash != prev_hash {
                return Err(LearnError::Index(format!(
                    "witness chain break at seq {}: previous_hash mismatch",
                    entry.seq
                )));
            }
            // 2. Recompute the digest and confirm it matches.
            let expected = WitnessEntry::compute_digest(
                entry.seq,
                &entry.chunk_id,
                &entry.video_id,
                entry.source_url.as_deref(),
                entry.start_seconds,
                entry.ingested_at,
                &entry.previous_hash,
            );
            if entry.digest != expected {
                return Err(LearnError::Index(format!(
                    "witness chain tampered at seq {}: digest mismatch",
                    entry.seq
                )));
            }
            prev_hash = entry.digest;
        }
        Ok(())
    }

    /// Append a batch of embedded chunks.
    ///
    /// Creates the `.rvf` file on the first call (dimension inferred from batch).
    pub fn ingest(&mut self, batch: &[Embedded]) -> Result<usize> {
        if batch.is_empty() {
            return Ok(0);
        }

        // Lazily create the store on the first ingest.
        if self.store.is_none() {
            let dim = batch[0].embedding.len() as u16;
            let opts = RvfOptions {
                dimension: dim,
                ..Default::default()
            };
            let store = RvfStore::create(&self.rvf_path, opts)
                .map_err(|e| LearnError::Index(format!("{e:?}")))?;
            self.store = Some(store);
        }

        let vec_data: Vec<Vec<f32>> = batch.iter().map(|e| e.embedding.clone()).collect();
        let vec_refs: Vec<&[f32]> = vec_data.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = batch
            .iter()
            .map(|e| chunk_id_to_u64(&e.chunk.chunk_id))
            .collect();

        // Scope the mutable borrow so it ends before we call save_meta.
        let (accepted, dimension) = {
            let store = self.store.as_mut().expect("store just created above");
            let result = store
                .ingest_batch(&vec_refs, &ids, None)
                .map_err(|e| LearnError::Index(format!("{e:?}")))?;
            (result.accepted, store.dimension())
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        for e in batch {
            let id = chunk_id_to_u64(&e.chunk.chunk_id);
            self.chunks.insert(id.to_string(), e.chunk.clone());
            self.embeddings.insert(id, e.embedding.clone());

            let prev_hash = self
                .witness_chain
                .last()
                .map(|w| w.digest)
                .unwrap_or([0u8; 32]);
            let seq = self.witness_chain.len() as u64 + 1;
            let entry = WitnessEntry::new(
                seq,
                id.to_string(),
                e.chunk.video_id.clone(),
                None, // source_url not available on Chunk today
                e.chunk.start_seconds,
                now,
                prev_hash,
            );
            self.witness_chain.push(entry);
        }

        self.save_meta(dimension)?;
        append_emb_file(&self.emb_path, batch)?;
        save_witness(&self.witness_path, &self.witness_chain)?;
        Ok(accepted as usize)
    }

    /// Top-K HNSW search. Returns Hits in score order (rank 0 = closest).
    ///
    /// Returns an empty `Vec` when no vectors have been ingested yet.
    pub fn search(&self, query_vec: &[f32], k: usize) -> Result<Vec<Hit>> {
        let store = match &self.store {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };

        let opts = QueryOptions::default();
        let results = store
            .query(query_vec, k, &opts)
            .map_err(|e| LearnError::Retrieve(format!("{e:?}")))?;

        let hits = results
            .into_iter()
            .enumerate()
            .filter_map(|(rank, sr)| {
                let key = sr.id.to_string();
                self.chunks.get(&key).map(|chunk| Hit {
                    chunk: chunk.clone(),
                    score: (1.0 - sr.distance).max(0.0),
                    rank,
                })
            })
            .collect();

        Ok(hits)
    }

    /// Return a snapshot of all chunks currently in the sidecar.
    ///
    /// Used by `learn-retrieve` to build the in-memory BM25 index.
    pub fn chunks_snapshot(&self) -> Vec<Chunk> {
        self.chunks.values().cloned().collect()
    }

    /// Look up a single chunk by its `chunk_id` string.
    ///
    /// Returns `None` if the chunk is not in the sidecar (e.g. sidecar was
    /// missing or the chunk was never ingested).
    pub fn chunk_by_id(&self, chunk_id: &str) -> Option<Chunk> {
        // The sidecar keys are decimal strings of the FNV-1a u64 hash.
        let key = chunk_id_to_u64(chunk_id).to_string();
        self.chunks.get(&key).cloned()
    }

    /// Look up the stored dense embedding for a chunk by its `chunk_id` string.
    ///
    /// Returns `None` when the embedding was never persisted (e.g. the index
    /// was opened before `.emb.bin` was introduced, or the chunk was not found).
    /// Callers that use this for MMR should fall back gracefully when `None` is
    /// returned.
    pub fn embedding_for_chunk_id(&self, chunk_id: &str) -> Option<&[f32]> {
        let id = chunk_id_to_u64(chunk_id);
        self.embeddings.get(&id).map(|v| v.as_slice())
    }

    /// Reconstruct all `Embedded` values for a single video from the in-memory
    /// sidecar + embedding map.
    ///
    /// Used by the crash-recovery fast-path: when `IngestStatus::Embedded` is
    /// present, the embeddings are already on disk; calling this avoids
    /// re-running the ONNX inference step.
    ///
    /// Returns an empty `Vec` when none of the sidecar chunks belong to
    /// `video_id` or when their embeddings are absent (caller should fall back
    /// to a full re-embed in that case).
    pub fn embedded_for_video(&self, video_id: &str) -> Vec<learn_core::Embedded> {
        self.chunks
            .values()
            .filter(|c| c.video_id == video_id)
            .filter_map(|c| {
                let id = chunk_id_to_u64(&c.chunk_id);
                self.embeddings.get(&id).map(|emb| learn_core::Embedded {
                    chunk: c.clone(),
                    embedding: emb.clone(),
                    embedding_model: "stored".to_string(),
                })
            })
            .collect()
    }

    /// Force a flush + compaction.
    pub fn compact(&mut self) -> Result<()> {
        match &mut self.store {
            Some(s) => s
                .compact()
                .map(|_| ())
                .map_err(|e| LearnError::Index(format!("{e:?}"))),
            None => Ok(()),
        }
    }

    /// Stats for `learn status`.
    pub fn stats(&self) -> Result<IndexStats> {
        match &self.store {
            Some(s) => {
                let st = s.status();
                Ok(IndexStats {
                    vector_count: st.total_vectors as usize,
                    segment_count: st.total_segments as usize,
                    bytes_on_disk: st.file_size,
                })
            }
            None => Ok(IndexStats {
                vector_count: 0,
                segment_count: 0,
                bytes_on_disk: 0,
            }),
        }
    }
}

// ── Private helpers ─────────────────────────────────────────────────────────

impl LearnIndex {
    fn meta_path_for(kb_root: &Utf8Path, topic: &Topic) -> PathBuf {
        PathBuf::from(
            kb_root
                .join(format!("{}.meta.json", topic.as_str()))
                .as_str(),
        )
    }

    fn emb_path_for(kb_root: &Utf8Path, topic: &Topic) -> PathBuf {
        PathBuf::from(kb_root.join(format!("{}.emb.bin", topic.as_str())).as_str())
    }

    fn witness_path_for(kb_root: &Utf8Path, topic: &Topic) -> PathBuf {
        PathBuf::from(
            kb_root
                .join(format!("{}.witness.json", topic.as_str()))
                .as_str(),
        )
    }

    fn save_meta(&self, dimension: u16) -> Result<()> {
        let sidecar = MetaSidecarRef {
            dimension,
            chunks: &self.chunks,
        };
        let json = serde_json::to_string(&sidecar)?;
        let tmp = self.meta_path.with_extension("json.tmp");
        std::fs::write(&tmp, json.as_bytes()).map_err(LearnError::Io)?;
        std::fs::rename(&tmp, &self.meta_path).map_err(LearnError::Io)
    }
}

// ── Utility ─────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash — stable u64 ID from a chunk_id string.
fn chunk_id_to_u64(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for b in s.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Load the witness chain from `<topic>.witness.json`.
///
/// Returns an empty `Vec` when the file is absent (backward-compatible with
/// indexes created before the witness chain was introduced).
/// Propagates `LearnError::Serde` if the file exists but is not valid JSON.
fn load_witness(witness_path: &Path) -> Result<Vec<WitnessEntry>> {
    if !witness_path.exists() {
        return Ok(Vec::new());
    }
    let json = std::fs::read_to_string(witness_path).map_err(LearnError::Io)?;
    let chain: Vec<WitnessEntry> = serde_json::from_str(&json)?;
    Ok(chain)
}

/// Atomically persist the witness chain to `<topic>.witness.json`.
///
/// Uses tmp+rename so a crash mid-write leaves either the old file or the new
/// file intact — never a half-written JSON.
fn save_witness(witness_path: &Path, chain: &[WitnessEntry]) -> Result<()> {
    let json = serde_json::to_string(chain)?;
    let tmp = witness_path.with_extension("json.tmp");
    std::fs::write(&tmp, json.as_bytes()).map_err(LearnError::Io)?;
    std::fs::rename(&tmp, witness_path).map_err(LearnError::Io)
}

fn load_sidecar(meta_path: &Path) -> Result<MetaSidecar> {
    if !meta_path.exists() {
        return Ok(MetaSidecar::default());
    }
    let json = std::fs::read_to_string(meta_path).map_err(LearnError::Io)?;
    let sidecar: MetaSidecar = serde_json::from_str(&json)?;
    Ok(sidecar)
}

/// Load the crash-recovery manifest from disk.
///
/// Returns `Manifest::default()` when the file is absent (first open).
/// Propagates a `LearnError::Serde` if the file exists but is corrupt —
/// callers should surface this rather than silently losing state.
fn load_manifest(manifest_path: &Path) -> Result<Manifest> {
    if !manifest_path.exists() {
        return Ok(Manifest::default());
    }
    let json = std::fs::read_to_string(manifest_path).map_err(LearnError::Io)?;
    let manifest: Manifest = serde_json::from_str(&json)?;
    Ok(manifest)
}

/// Load the embedding binary file into a `HashMap<u64, Vec<f32>>`.
///
/// ## File format (`<topic>.emb.bin`)
///
/// ```text
/// [u32 LE] dimension
/// [u64 LE] count
/// repeated count times:
///   [u64 LE] chunk id (FNV-1a hash)
///   [u32 LE] vector byte length (= dimension × 4)
///   [f32 LE × dimension] vector values
/// ```
///
/// The file is written by `append_emb_file` in an append-only manner.
/// Duplicate IDs are silently overwritten (later entry wins) — this is safe
/// because re-ingesting a chunk produces the same vector.
///
/// Returns an empty map when the file does not exist (backward-compatible with
/// indexes created before `.emb.bin` was introduced).
fn load_emb_file(emb_path: &Path) -> Result<HashMap<u64, Vec<f32>>> {
    if !emb_path.exists() {
        return Ok(HashMap::new());
    }
    let raw = std::fs::read(emb_path).map_err(LearnError::Io)?;
    if raw.len() < 12 {
        // Header-only or truncated: treat as empty (backward compat).
        return Ok(HashMap::new());
    }

    let dim = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(raw[4..12].try_into().unwrap()) as usize;

    if dim == 0 || count == 0 {
        return Ok(HashMap::new());
    }

    let record_size = 8 + 4 + dim * 4; // id(8) + vec_byte_len(4) + floats
    let expected = 12 + count * record_size;
    if raw.len() < expected {
        tracing::warn!(
            "emb.bin truncated (expected {expected} bytes, got {}); loading partial data",
            raw.len()
        );
    }

    let mut map: HashMap<u64, Vec<f32>> = HashMap::with_capacity(count);
    let mut offset = 12usize;
    while offset + 8 + 4 <= raw.len() {
        let id = u64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap());
        offset += 8;
        let vec_bytes = u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + vec_bytes > raw.len() {
            break;
        }
        let vec: Vec<f32> = raw[offset..offset + vec_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        map.insert(id, vec);
        offset += vec_bytes;
    }

    Ok(map)
}

/// Append a batch of embeddings to the `<topic>.emb.bin` file.
///
/// On the very first call (file does not exist) this writes the full header
/// `[u32 dim][u64 count]` followed by all records.  On subsequent calls the
/// header's count field is updated in-place via an atomic rewrite of the first
/// 12 bytes; records are appended after the existing content.
///
/// The file is never truncated or rewritten in full — only the count field in
/// the header is patched.  This keeps appends O(batch_size) regardless of how
/// many embeddings are already stored.
fn append_emb_file(emb_path: &Path, batch: &[Embedded]) -> Result<()> {
    use std::io::{Seek, SeekFrom, Write};

    if batch.is_empty() {
        return Ok(());
    }

    let dim = batch[0].embedding.len();
    let vec_bytes_len = (dim * 4) as u32;

    // Build the record payload for the new batch.
    let mut records: Vec<u8> = Vec::with_capacity(batch.len() * (8 + 4 + dim * 4));
    for e in batch {
        let id = chunk_id_to_u64(&e.chunk.chunk_id);
        records.extend_from_slice(&id.to_le_bytes());
        records.extend_from_slice(&vec_bytes_len.to_le_bytes());
        for &f in &e.embedding {
            records.extend_from_slice(&f.to_le_bytes());
        }
    }

    if !emb_path.exists() {
        // First write: create with header + records.
        let count = batch.len() as u64;
        let mut file = std::fs::File::create(emb_path).map_err(LearnError::Io)?;
        file.write_all(&(dim as u32).to_le_bytes())
            .map_err(LearnError::Io)?;
        file.write_all(&count.to_le_bytes())
            .map_err(LearnError::Io)?;
        file.write_all(&records).map_err(LearnError::Io)?;
        file.flush().map_err(LearnError::Io)?;
    } else {
        // Subsequent write: read existing count, append records, patch count.
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(emb_path)
            .map_err(LearnError::Io)?;

        let mut header = [0u8; 12];
        use std::io::Read;
        file.read_exact(&mut header).map_err(LearnError::Io)?;
        let existing_count = u64::from_le_bytes(header[4..12].try_into().unwrap());
        let new_count = existing_count + batch.len() as u64;

        // Append records.
        file.seek(SeekFrom::End(0)).map_err(LearnError::Io)?;
        file.write_all(&records).map_err(LearnError::Io)?;

        // Patch the count field (bytes 4..12).
        file.seek(SeekFrom::Start(4)).map_err(LearnError::Io)?;
        file.write_all(&new_count.to_le_bytes())
            .map_err(LearnError::Io)?;
        file.flush().map_err(LearnError::Io)?;
    }

    Ok(())
}

// ── LearnIndexLarge — DiskANN-backed scale path ──────────────────────────────

/// DiskANN-backed index for topics that exceed [`LARGE_INDEX_THRESHOLD`].
///
/// ## Lifecycle
///
/// 1. `open(kb_root, topic)` — restores a previously saved DiskANN store from
///    `<kb_root>/<topic>.diskann/`, or starts empty if no store exists yet.
/// 2. `ingest(batch)` — appends to the staging buffer and atomically persists
///    the full cumulative staged set to `<topic>.large.staged.bin`.  The
///    Vamana graph is NOT rebuilt here; it is built on the next call to `compact`.
/// 3. `compact()` — builds (or rebuilds) the Vamana graph over ALL vectors ever
///    ingested: previously-built vectors (reloaded from `store_dir/vectors.bin`)
///    plus the full persisted staged set.  Writes the new index to disk and
///    resets the staged file to an empty header.  This makes `compact`
///    **idempotent across ingest cycles**: ingest N + compact + ingest M +
///    compact → N+M vectors are searchable.  Must be called before `search`
///    returns any results.
/// 4. `search(query, k)` — greedy Vamana beam search; returns at most `k` hits.
/// 5. `stats()` — returns total staged vectors and on-disk size.
///
/// ## Staged file format (`<topic>.large.staged.bin`)
///
/// ```text
/// [u32 dim (LE)] [u64 count (LE)] [count × dim × f32 (LE)] [count × id_len × u8 …]
/// ```
/// More precisely:
/// - 4 bytes: dimension as `u32` little-endian
/// - 8 bytes: vector count as `u64` little-endian
/// - `count × dim × 4` bytes: packed f32 vectors, row-major, little-endian
/// - `count × 8` bytes: FNV-1a u64 IDs, each as `u64` little-endian
///
/// ## Sidecar
///
/// Same JSON sidecar layout as [`LearnIndex`]: `<topic>.diskann.meta.json`
/// with schema `{ dimension, chunks: { "<diskann_id>": Chunk } }`.
/// The DiskANN id is the decimal string of the FNV-1a hash of the chunk_id,
/// matching the string IDs stored inside the `DiskAnnIndex`.
///
/// ## PQ (Product Quantization)
///
/// Disabled (`pq_subspaces: 0`) by default.  Enable by constructing with a
/// custom `DiskAnnConfig` once your topic has ≥ 1 000 vectors per sub-space.
pub struct LearnIndexLarge {
    /// Staged vectors + their metadata pending the next compact/build.
    /// After `compact` these are folded into `index` and the staging area is
    /// cleared from memory (the persisted file is reset to an empty header).
    staged_ids: Vec<String>,
    staged_vecs: Vec<Vec<f32>>,
    staged_chunks: HashMap<String, Chunk>,

    /// The built DiskANN index. `None` until the first successful `compact`.
    index: Option<DiskAnnIndex>,

    /// Total vector count across all compactions (built + staged).
    total_built: usize,

    /// Directory where the DiskANN files are persisted.
    store_dir: PathBuf,

    /// Companion sidecar path: `<topic>.diskann.meta.json`
    meta_path: PathBuf,

    /// Cumulative staged file: `<topic>.large.staged.bin`
    /// Persisted atomically after every `ingest`; cleared after every `compact`.
    staged_path: PathBuf,

    /// All chunks known to this index (built + staged).
    chunks: HashMap<String, Chunk>,

    /// Embedding dimension — inferred from the first ingest batch.
    dimension: Option<usize>,
}

impl LearnIndexLarge {
    /// Open or create a large-scale DiskANN index at
    /// `<kb_root>/<topic>.diskann/`.
    ///
    /// If the store directory already contains a built DiskANN graph it is
    /// loaded; otherwise the index starts empty and is ready to receive staged
    /// vectors via `ingest`.  Any previously persisted staged vectors (from a
    /// prior `ingest` that was not yet compacted) are reloaded from
    /// `<topic>.large.staged.bin`.
    pub fn open(kb_root: &Utf8Path, topic: Topic) -> Result<Self> {
        let store_dir = PathBuf::from(kb_root.join(format!("{}.diskann", topic.as_str())).as_str());
        let meta_path = PathBuf::from(
            kb_root
                .join(format!("{}.diskann.meta.json", topic.as_str()))
                .as_str(),
        );
        let staged_path = PathBuf::from(
            kb_root
                .join(format!("{}.large.staged.bin", topic.as_str()))
                .as_str(),
        );

        let sidecar = load_sidecar(&meta_path)?;
        let mut dimension: Option<usize> = if sidecar.dimension > 0 {
            Some(sidecar.dimension as usize)
        } else {
            None
        };

        // Attempt to load a previously built DiskANN index.
        let (index, total_built) = if store_dir.join("config.json").exists() {
            match DiskAnnIndex::load(&store_dir) {
                Ok(idx) => {
                    let n = idx.count();
                    (Some(idx), n)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load DiskANN index from {}: {e:?}. Starting empty.",
                        store_dir.display()
                    );
                    (None, 0)
                }
            }
        } else {
            (None, 0)
        };

        // Reload any persisted staged vectors that survived a crash or restart
        // before compact was called.
        let (staged_ids, staged_vecs, staged_chunks) =
            load_staged_file(&staged_path, &mut dimension)?;

        Ok(Self {
            staged_ids,
            staged_vecs,
            staged_chunks,
            index,
            total_built,
            store_dir,
            meta_path,
            staged_path,
            chunks: sidecar.chunks,
            dimension,
        })
    }

    /// Append a batch of embedded chunks to the staging buffer.
    ///
    /// Does NOT rebuild the Vamana graph; call `compact` for that.
    /// Returns the number of chunks accepted into staging.
    pub fn ingest(&mut self, batch: &[Embedded]) -> Result<usize> {
        if batch.is_empty() {
            return Ok(0);
        }

        // Infer dimension from the first ever batch.
        if self.dimension.is_none() {
            self.dimension = Some(batch[0].embedding.len());
        }
        let dim = self.dimension.unwrap();

        for e in batch {
            if e.embedding.len() != dim {
                return Err(LearnError::Index(format!(
                    "LearnIndexLarge: dimension mismatch — expected {dim}, got {}",
                    e.embedding.len()
                )));
            }
            let id = chunk_id_to_u64(&e.chunk.chunk_id).to_string();
            self.staged_ids.push(id.clone());
            self.staged_vecs.push(e.embedding.clone());
            self.staged_chunks.insert(id.clone(), e.chunk.clone());
            self.chunks.insert(id, e.chunk.clone());
        }

        let n = batch.len();

        // Atomic-write both the sidecar and the staged binary so neither is
        // lost on a crash before the next compact.
        self.save_meta()?;
        save_staged_file(
            &self.staged_path,
            self.dimension.unwrap(),
            &self.staged_ids,
            &self.staged_vecs,
        )?;

        Ok(n)
    }

    /// Build (or rebuild) the Vamana graph from ALL vectors ever ingested.
    ///
    /// This is the expensive step — O(N log N) roughly.  After compact, all
    /// staged vectors are incorporated into the built index and the staging
    /// buffer is cleared.  The built index is saved to `store_dir`.
    ///
    /// ## Idempotency across ingest cycles
    ///
    /// `compact` reads previously-built raw vectors from `store_dir/vectors.bin`
    /// (written by the last `build()` call via the DiskANN save format) and
    /// combines them with the current in-memory staged set before re-inserting
    /// everything into a fresh `DiskAnnIndex`.  This means:
    ///
    /// ```text
    /// ingest(N) → compact() → ingest(M) → compact()  =>  N+M vectors searchable
    /// ```
    ///
    /// **Strategy chosen**: persist-and-rebuild rather than error-on-non-empty.
    /// DiskANN's `load()` materialises all vectors from `vectors.bin` into a
    /// contiguous `Vec<f32>` in memory, so we read that file directly for the
    /// prior-built set (avoiding any need for a private "dump" API on
    /// `DiskAnnIndex`).  `DiskAnnIndex` is always built from scratch — it is
    /// immutable after `build()` and has no incremental-insert support.
    ///
    /// Returns `Ok(())` immediately if there are no vectors at all.
    pub fn compact(&mut self) -> Result<()> {
        let dim = match self.dimension {
            Some(d) => d,
            None => return Ok(()), // Nothing ever ingested
        };

        if self.staged_ids.is_empty() && self.index.is_none() {
            // Nothing built yet and nothing staged — nothing to do.
            return Ok(());
        }

        if self.staged_ids.is_empty() && self.index.is_some() {
            // No new data since the last compact — already up to date.
            return Ok(());
        }

        let config = DiskAnnConfig {
            dim,
            storage_path: Some(self.store_dir.clone()),
            max_degree: 64,
            build_beam: 128,
            search_beam: 64,
            alpha: 1.2,
            pq_subspaces: 0, // Disabled until calibration data available
            ..Default::default()
        };

        let mut new_index = DiskAnnIndex::new(config);

        // Step 1: reload previously-built raw vectors directly from the
        // DiskANN vectors.bin on disk.  The file format (written by
        // `DiskAnnIndex::save`) is:
        //   [u64 count LE] [u64 dim LE] [count × dim × f32 LE]
        // The ID mapping is in ids.json alongside it.
        let vectors_bin = self.store_dir.join("vectors.bin");
        let ids_json = self.store_dir.join("ids.json");
        if vectors_bin.exists() && ids_json.exists() {
            let raw = std::fs::read(&vectors_bin).map_err(LearnError::Io)?;
            let _built_n = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
            let built_dim = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
            if built_dim != dim {
                return Err(LearnError::Index(format!(
                    "compact: previously built dim={built_dim} != current dim={dim}"
                )));
            }
            let id_list: Vec<String> =
                serde_json::from_str(&std::fs::read_to_string(&ids_json).map_err(LearnError::Io)?)
                    .map_err(LearnError::Serde)?;

            let data_start = 16usize;
            for (i, id) in id_list.into_iter().enumerate() {
                let offset = data_start + i * dim * 4;
                let bytes = &raw[offset..offset + dim * 4];
                let vec: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                new_index
                    .insert(id, vec)
                    .map_err(|e| LearnError::Index(format!("DiskANN re-insert built: {e:?}")))?;
            }
        }

        // Step 2: insert current in-memory staged vectors.
        let staged_entries: Vec<(String, Vec<f32>)> = self
            .staged_ids
            .iter()
            .cloned()
            .zip(self.staged_vecs.iter().cloned())
            .collect();

        if new_index.count() == 0 && staged_entries.is_empty() {
            return Ok(());
        }

        new_index
            .insert_batch(staged_entries)
            .map_err(|e| LearnError::Index(format!("DiskANN insert_batch staged: {e:?}")))?;

        new_index
            .build()
            .map_err(|e| LearnError::Index(format!("DiskANN build: {e:?}")))?;

        let built_count = new_index.count();
        self.total_built = built_count;
        self.index = Some(new_index);

        // Clear in-memory staging buffer — all vectors now live in the built index.
        self.staged_ids.clear();
        self.staged_vecs.clear();
        self.staged_chunks.clear();

        // Reset the persisted staged file to an empty header so a re-open
        // after compact does not re-stage already-built vectors.
        save_staged_file(&self.staged_path, dim, &[], &[])?;

        Ok(())
    }

    /// Top-K search using the built Vamana graph.
    ///
    /// Returns an empty `Vec` if `compact` has never been called.
    pub fn search(&self, query_vec: &[f32], k: usize) -> Result<Vec<Hit>> {
        let index = match &self.index {
            Some(idx) => idx,
            None => return Ok(Vec::new()),
        };

        let results = index
            .search(query_vec, k)
            .map_err(|e| LearnError::Retrieve(format!("DiskANN search: {e:?}")))?;

        let hits = results
            .into_iter()
            .enumerate()
            .filter_map(|(rank, sr)| {
                self.chunks.get(&sr.id).map(|chunk| Hit {
                    chunk: chunk.clone(),
                    score: 1.0 / (1.0 + sr.distance),
                    rank,
                })
            })
            .collect();

        Ok(hits)
    }

    /// Stats: total vectors (built + staged) and on-disk size of the store dir.
    pub fn stats(&self) -> Result<IndexStats> {
        let vector_count = self.total_built + self.staged_ids.len();
        let bytes_on_disk = dir_size_bytes(&self.store_dir);
        Ok(IndexStats {
            vector_count,
            segment_count: if self.index.is_some() { 1 } else { 0 },
            bytes_on_disk,
        })
    }
}

// ── Private helpers for LearnIndexLarge ─────────────────────────────────────

impl LearnIndexLarge {
    fn save_meta(&self) -> Result<()> {
        let dim = self.dimension.unwrap_or(0) as u16;
        let sidecar = MetaSidecarRef {
            dimension: dim,
            chunks: &self.chunks,
        };
        let tmp = self.meta_path.with_extension("tmp");
        let json = serde_json::to_string(&sidecar)?;
        std::fs::write(&tmp, &json).map_err(LearnError::Io)?;
        std::fs::rename(&tmp, &self.meta_path).map_err(LearnError::Io)?;
        Ok(())
    }
}

// ── Staged-file helpers ──────────────────────────────────────────────────────

/// Persist the cumulative staged buffer to `path` atomically.
///
/// File format:
/// ```text
/// [u32 dim LE] [u64 count LE] [count × dim × f32 LE] [count × u64 id-hash LE]
/// ```
/// The id-hash is the raw FNV-1a u64 encoded as a u64 LE — i.e., just
/// `id.parse::<u64>()` since staged IDs are already decimal strings of the hash.
fn save_staged_file(
    path: &std::path::Path,
    dim: usize,
    ids: &[String],
    vecs: &[Vec<f32>],
) -> crate::Result<()> {
    use std::io::Write as _;
    let count = ids.len();
    let mut buf: Vec<u8> = Vec::with_capacity(4 + 8 + count * dim * 4 + count * 8);
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    buf.extend_from_slice(&(count as u64).to_le_bytes());
    for v in vecs {
        for &f in v {
            buf.extend_from_slice(&f.to_le_bytes());
        }
    }
    for id in ids {
        let hash: u64 = id.parse().unwrap_or(0);
        buf.extend_from_slice(&hash.to_le_bytes());
    }
    let tmp = path.with_extension("bin.tmp");
    let mut f = std::fs::File::create(&tmp).map_err(LearnError::Io)?;
    f.write_all(&buf).map_err(LearnError::Io)?;
    f.flush().map_err(LearnError::Io)?;
    std::fs::rename(&tmp, path).map_err(LearnError::Io)?;
    Ok(())
}

/// Load staged vectors from `path`, if it exists.
///
/// Updates `dimension` from the file header if it was previously `None`.
/// Returns `(ids, vecs, chunks)` where `chunks` is empty (chunk metadata
/// lives in the sidecar, not the staged file; the caller should reconstruct
/// from `sidecar.chunks` on open).
#[allow(clippy::type_complexity)]
fn load_staged_file(
    path: &std::path::Path,
    dimension: &mut Option<usize>,
) -> crate::Result<(Vec<String>, Vec<Vec<f32>>, HashMap<String, Chunk>)> {
    if !path.exists() {
        return Ok((Vec::new(), Vec::new(), HashMap::new()));
    }
    let raw = std::fs::read(path).map_err(LearnError::Io)?;
    if raw.len() < 12 {
        // Truncated/empty header — treat as no staged data.
        return Ok((Vec::new(), Vec::new(), HashMap::new()));
    }
    let dim = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(raw[4..12].try_into().unwrap()) as usize;

    if dim == 0 || count == 0 {
        return Ok((Vec::new(), Vec::new(), HashMap::new()));
    }

    // Update dimension if not yet known.
    if dimension.is_none() {
        *dimension = Some(dim);
    }

    let expected = 12 + count * dim * 4 + count * 8;
    if raw.len() < expected {
        tracing::warn!(
            "staged file truncated (expected {expected} bytes, got {}); ignoring",
            raw.len()
        );
        return Ok((Vec::new(), Vec::new(), HashMap::new()));
    }

    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(count);
    let mut offset = 12usize;
    for _ in 0..count {
        let end = offset + dim * 4;
        let v: Vec<f32> = raw[offset..end]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        vecs.push(v);
        offset = end;
    }

    let mut ids: Vec<String> = Vec::with_capacity(count);
    for _ in 0..count {
        let hash = u64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap());
        ids.push(hash.to_string());
        offset += 8;
    }

    Ok((ids, vecs, HashMap::new()))
}

/// Recursively sum file sizes under a directory (returns 0 if not present).
fn dir_size_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(|e| e.ok())
        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .sum()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use learn_core::{Chunk, Embedded, Topic};
    use tempfile::TempDir;

    const DIM: usize = 8;

    fn make_chunk(i: usize) -> Chunk {
        Chunk {
            chunk_id: format!("chunk-{i:03}"),
            video_id: format!("vid-{i:03}"),
            start_seconds: i as f64,
            end_seconds: i as f64 + 1.0,
            text: format!("text for chunk {i}"),
            token_count: 10,
        }
    }

    /// Synthetic vector: all zeros except position `i % DIM` = 1.0.
    fn make_embedded(i: usize) -> Embedded {
        let mut embedding = vec![0.0f32; DIM];
        embedding[i % DIM] = 1.0;
        Embedded {
            chunk: make_chunk(i),
            embedding,
            embedding_model: "test".into(),
        }
    }

    fn open_index(dir: &TempDir, topic_name: &str) -> LearnIndex {
        let kb = camino::Utf8Path::from_path(dir.path()).unwrap();
        let topic = Topic::new(topic_name).unwrap();
        LearnIndex::open(kb, topic).unwrap()
    }

    fn ingest_five(idx: &mut LearnIndex) {
        let batch: Vec<Embedded> = (0..5).map(make_embedded).collect();
        idx.ingest(&batch).unwrap();
    }

    // ── Test 1: fresh open returns zero vectors ──────────────────────────────

    #[test]
    fn open_creates_fresh() {
        let dir = TempDir::new().unwrap();
        let idx = open_index(&dir, "fresh-topic");
        let stats = idx.stats().unwrap();
        assert_eq!(stats.vector_count, 0);
        assert_eq!(stats.segment_count, 0);
    }

    // ── Test 2: ingest 5 chunks, stats reports 5 ────────────────────────────

    #[test]
    fn ingest_five_chunks() {
        let dir = TempDir::new().unwrap();
        let mut idx = open_index(&dir, "ingest-topic");
        ingest_five(&mut idx);
        let stats = idx.stats().unwrap();
        assert_eq!(stats.vector_count, 5);
    }

    // ── Test 3: search returns the exact chunk at rank 0 ────────────────────

    #[test]
    fn search_rank_zero_matches() {
        let dir = TempDir::new().unwrap();
        let mut idx = open_index(&dir, "search-topic");
        ingest_five(&mut idx);

        // chunk-000 has embedding [1,0,0,0,0,0,0,0]
        let mut query = vec![0.0f32; DIM];
        query[0] = 1.0;

        let hits = idx.search(&query, 1).unwrap();
        assert_eq!(hits.len(), 1, "expected 1 hit");
        assert_eq!(hits[0].rank, 0);
        assert_eq!(hits[0].chunk.chunk_id, "chunk-000");
    }

    // ── Test 4: compact does not error ──────────────────────────────────────

    #[test]
    fn compact_no_error() {
        let dir = TempDir::new().unwrap();
        let mut idx = open_index(&dir, "compact-topic");
        ingest_five(&mut idx);
        idx.compact().unwrap();
        // stats still coherent after compaction
        assert_eq!(idx.stats().unwrap().vector_count, 5);
    }

    // ── Test gap 4: FNV-1a stability pin ────────────────────────────────────

    #[test]
    fn chunk_id_to_u64_is_stable() {
        // Pin known-good FNV-1a outputs. If these change, sidecar lookups silently break.
        assert_eq!(chunk_id_to_u64(""), 0xcbf29ce484222325);
        assert_eq!(chunk_id_to_u64("a"), 0xaf63dc4c8601ec8c);
        // Realistic chunk_id shape — baked from a verified run:
        assert_eq!(chunk_id_to_u64("vid-001:0"), 0x25d5ced728425b84);
    }

    // ── Test gap 5: open with missing / corrupt sidecar ──────────────────────

    #[test]
    fn open_with_missing_sidecar_returns_empty_chunks() {
        let dir = TempDir::new().unwrap();

        // Ingest first so the .rvf file exists.
        {
            let mut idx = open_index(&dir, "sidecar-miss");
            ingest_five(&mut idx);
        }

        // Delete the sidecar file.
        let kb = camino::Utf8Path::from_path(dir.path()).unwrap();
        let meta = kb.join("sidecar-miss.meta.json");
        std::fs::remove_file(&meta).expect("meta file should exist");

        // Re-open: the .rvf exists but sidecar is gone → empty chunks, search returns empty.
        let idx = open_index(&dir, "sidecar-miss");
        let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let hits = idx.search(&query, 1).unwrap();
        assert!(
            hits.is_empty(),
            "expected empty hits with missing sidecar, got {hits:?}"
        );
    }

    #[test]
    fn open_with_corrupt_sidecar_returns_err() {
        let dir = TempDir::new().unwrap();

        // Ingest first so the .rvf file exists.
        {
            let mut idx = open_index(&dir, "sidecar-corrupt");
            ingest_five(&mut idx);
        }

        // Overwrite sidecar with garbage.
        let kb = camino::Utf8Path::from_path(dir.path()).unwrap();
        let meta = kb.join("sidecar-corrupt.meta.json");
        std::fs::write(&meta, b"this is not valid json {{{").unwrap();

        let result = {
            let kb_path = camino::Utf8Path::from_path(dir.path()).unwrap();
            let topic = Topic::new("sidecar-corrupt").unwrap();
            LearnIndex::open(kb_path, topic)
        };
        let is_serde_err = matches!(result, Err(LearnError::Serde(_)));
        assert!(
            is_serde_err,
            "expected Err(LearnError::Serde) for corrupt sidecar"
        );
    }

    // ── Test 5: sidecar survives a close/reopen cycle ───────────────────────

    #[test]
    fn sidecar_persists_across_reopen() {
        let dir = TempDir::new().unwrap();

        // Ingest in first session.
        {
            let mut idx = open_index(&dir, "persist-topic");
            ingest_five(&mut idx);
        }

        // Reopen and search for chunk-002: embedding [0,0,1,0,0,0,0,0]
        let idx = open_index(&dir, "persist-topic");
        let mut query = vec![0.0f32; DIM];
        query[2] = 1.0;

        let hits = idx.search(&query, 1).unwrap();
        assert_eq!(hits.len(), 1, "expected 1 hit after reopen");
        assert_eq!(hits[0].chunk.chunk_id, "chunk-002");
    }

    // ── LearnIndexLarge tests ────────────────────────────────────────────────

    // DIM for large-index tests: must divide evenly if PQ ever enabled.
    // Use a larger dim so Vamana has enough space to work with.
    const LARGE_DIM: usize = 16;

    /// Helper: build an `Embedded` with a unique pseudo-random unit vector.
    ///
    /// Each index `i` gets a distinct vector so DiskANN rank-0 is unambiguous.
    /// We use a simple deterministic linear congruential generator seeded on `i`
    /// to avoid any aliasing across the 100-vector test set.
    fn make_large_embedded(i: usize, _hot_pos: usize) -> Embedded {
        let chunk = Chunk {
            chunk_id: format!("lchunk-{i:04}"),
            video_id: format!("lvid-{i:04}"),
            start_seconds: i as f64,
            end_seconds: i as f64 + 1.0,
            text: format!("large chunk {i}"),
            token_count: 5,
        };
        // LCG: a=6364136223846793005, c=1442695040888963407 (Knuth)
        let mut state = i as u64 ^ 0xdeadbeef_cafebabe;
        let embedding: Vec<f32> = (0..LARGE_DIM)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Map high bits to [-1, 1]
                (state >> 33) as f32 / (u32::MAX as f32 / 2.0) - 1.0
            })
            .collect();
        // Normalise to unit length so distances are comparable
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let embedding = if norm > 1e-8 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            vec![1.0 / (LARGE_DIM as f32).sqrt(); LARGE_DIM]
        };
        Embedded {
            chunk,
            embedding,
            embedding_model: "test-large".into(),
        }
    }

    fn open_large_index(dir: &TempDir, topic_name: &str) -> LearnIndexLarge {
        let kb = camino::Utf8Path::from_path(dir.path()).unwrap();
        let topic = Topic::new(topic_name).unwrap();
        LearnIndexLarge::open(kb, topic).unwrap()
    }

    // ── Large test 1: fresh open returns zero stats ──────────────────────────

    #[test]
    fn large_index_open_create() {
        let dir = TempDir::new().unwrap();
        let idx = open_large_index(&dir, "large-fresh");
        let stats = idx.stats().unwrap();
        assert_eq!(
            stats.vector_count, 0,
            "fresh large index should have 0 vectors"
        );
        assert_eq!(stats.segment_count, 0, "no segments before compact");
    }

    // ── Large test 2: ingest 100 chunks, compact, search rank-0 ─────────────

    #[test]
    fn large_index_ingest_and_search_rank0() {
        let dir = TempDir::new().unwrap();
        let mut idx = open_large_index(&dir, "large-search");

        // Build 100 chunks with distinct one-hot embeddings cycling over LARGE_DIM.
        // chunk lchunk-0042 will have hot_pos = 42 % 16 = 10.
        let batch: Vec<Embedded> = (0..100).map(|i| make_large_embedded(i, i)).collect();
        let planted_id = batch[42].chunk.chunk_id.clone();
        let query = batch[42].embedding.clone();

        idx.ingest(&batch).unwrap();

        // Before compact: search returns empty.
        let hits_before = idx.search(&query, 1).unwrap();
        assert!(
            hits_before.is_empty(),
            "search before compact must return empty; got {hits_before:?}"
        );

        idx.compact().unwrap();

        // After compact: rank-0 must be the planted chunk.
        let hits = idx.search(&query, 1).unwrap();
        assert!(!hits.is_empty(), "expected at least 1 hit after compact");
        assert_eq!(
            hits[0].chunk.chunk_id, planted_id,
            "rank-0 must be the planted chunk"
        );
    }

    // ── Large test 3: persists across reopen ────────────────────────────────

    #[test]
    fn large_index_persists_across_reopen() {
        let dir = TempDir::new().unwrap();
        let planted_chunk_id;
        let query_vec;

        // Session 1: ingest + compact.
        {
            let mut idx = open_large_index(&dir, "large-persist");
            let batch: Vec<Embedded> = (0..60).map(|i| make_large_embedded(i, i)).collect();
            planted_chunk_id = batch[7].chunk.chunk_id.clone();
            query_vec = batch[7].embedding.clone();
            idx.ingest(&batch).unwrap();
            idx.compact().unwrap();
        }

        // Session 2: reopen without ingest, search must still find the chunk.
        let idx2 = open_large_index(&dir, "large-persist");
        let hits = idx2.search(&query_vec, 1).unwrap();
        assert!(!hits.is_empty(), "expected hit after reopen");
        assert_eq!(
            hits[0].chunk.chunk_id, planted_chunk_id,
            "reopen must preserve rank-0 result"
        );
    }

    // ── New test: Item 1 — save_meta is atomic, no .json.tmp remains ────────

    #[test]
    fn save_meta_is_atomic_no_tmp_remains() {
        let dir = TempDir::new().unwrap();
        let mut idx = open_index(&dir, "atomic-meta-topic");
        ingest_five(&mut idx);

        // After ingest the sidecar must exist.
        let meta = dir.path().join("atomic-meta-topic.meta.json");
        assert!(meta.exists(), "meta.json must exist after ingest");

        // No .json.tmp artifact should remain.
        let tmp = dir.path().join("atomic-meta-topic.meta.json.tmp");
        assert!(
            !tmp.exists(),
            ".json.tmp artifact must not remain after atomic write"
        );

        // The directory should contain no .tmp files at all.
        let tmp_count = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|n| n.ends_with(".tmp"))
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(tmp_count, 0, "no .tmp files should remain in kb_root");

        // Verify the sidecar is valid JSON and contains our chunks.
        let json = std::fs::read_to_string(&meta).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed["chunks"].as_object().unwrap().len(),
            5,
            "sidecar must contain 5 chunks"
        );
    }

    // ── New test: Item 2 — search score is never negative ───────────────────

    #[test]
    fn search_score_never_negative_for_unit_vectors() {
        let dir = TempDir::new().unwrap();
        let mut idx = open_index(&dir, "score-clamp-topic");

        // Ingest 5 normalised random vectors using the LCG from make_large_embedded.
        let batch: Vec<Embedded> = (0..5)
            .map(|i| {
                // Reuse the LCG from make_large_embedded but with DIM=8.
                let mut state = i as u64 ^ 0xdeadbeef_cafebabe;
                let raw: Vec<f32> = (0..DIM)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        (state >> 33) as f32 / (u32::MAX as f32 / 2.0) - 1.0
                    })
                    .collect();
                let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
                let embedding = if norm > 1e-8 {
                    raw.iter().map(|x| x / norm).collect()
                } else {
                    vec![1.0 / (DIM as f32).sqrt(); DIM]
                };
                Embedded {
                    chunk: make_chunk(i),
                    embedding,
                    embedding_model: "test".into(),
                }
            })
            .collect();

        idx.ingest(&batch).unwrap();

        // Query with a different normalised vector (all-ones normalised).
        let q_raw = [1.0f32; DIM];
        let q_norm: f32 = q_raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query: Vec<f32> = q_raw.iter().map(|x| x / q_norm).collect();

        let hits = idx.search(&query, 5).unwrap();
        assert!(!hits.is_empty(), "expected hits for non-empty index");
        for h in &hits {
            assert!(
                h.score >= 0.0,
                "score must be >= 0.0 but got {} for chunk {}",
                h.score,
                h.chunk.chunk_id
            );
        }
    }

    // ── New test: Item 3 — compact is idempotent across two ingest cycles ────

    #[test]
    fn large_index_compact_is_idempotent_across_two_ingest_cycles() {
        let dir = TempDir::new().unwrap();
        let mut idx = open_large_index(&dir, "large-idempotent");

        // Cycle 1: ingest 5 vectors, compact, search for vector from cycle 1.
        let batch1: Vec<Embedded> = (0..5).map(|i| make_large_embedded(i, i)).collect();
        let planted_id_1 = batch1[2].chunk.chunk_id.clone();
        let query_1 = batch1[2].embedding.clone();

        idx.ingest(&batch1).unwrap();
        idx.compact().unwrap();

        let hits1 = idx.search(&query_1, 1).unwrap();
        assert!(
            !hits1.is_empty(),
            "cycle-1: expected hit after first compact"
        );
        assert_eq!(
            hits1[0].chunk.chunk_id, planted_id_1,
            "cycle-1: rank-0 must be the planted cycle-1 chunk"
        );

        // Cycle 2: ingest 5 more vectors, compact, verify BOTH batches are searchable.
        let batch2: Vec<Embedded> = (5..10).map(|i| make_large_embedded(i, i)).collect();
        let planted_id_2 = batch2[1].chunk.chunk_id.clone(); // batch2[1] = global i=6
        let query_2 = batch2[1].embedding.clone();

        idx.ingest(&batch2).unwrap();
        idx.compact().unwrap();

        // Cycle-1 vector must still be searchable.
        let hits_cycle1_after = idx.search(&query_1, 1).unwrap();
        assert!(
            !hits_cycle1_after.is_empty(),
            "cycle-2 compact must preserve cycle-1 vectors (got empty hits)"
        );
        assert_eq!(
            hits_cycle1_after[0].chunk.chunk_id, planted_id_1,
            "rank-0 for cycle-1 query must still be the cycle-1 planted chunk after cycle-2 compact"
        );

        // Cycle-2 vector must also be searchable.
        let hits_cycle2 = idx.search(&query_2, 1).unwrap();
        assert!(
            !hits_cycle2.is_empty(),
            "cycle-2 planted vector must be searchable after second compact"
        );
        assert_eq!(
            hits_cycle2[0].chunk.chunk_id, planted_id_2,
            "rank-0 for cycle-2 query must be the cycle-2 planted chunk"
        );

        // Total vector count must be 10.
        let stats = idx.stats().unwrap();
        assert_eq!(
            stats.vector_count, 10,
            "stats must report 10 vectors after two 5-vector ingest cycles"
        );
    }

    // ── New test: Gap 3 — DiskANN cross-process-restart idempotency ─────────
    //
    // Proves that vectors ingested and compacted in one process survive a
    // drop + reopen, and remain searchable after a second ingest+compact cycle
    // in the fresh instance.
    //
    // The existing `large_index_compact_is_idempotent_across_two_ingest_cycles`
    // test keeps one `LearnIndexLarge` instance alive between cycles, so
    // `self.index` already holds the built graph in memory when cycle-2 begins.
    // This test explicitly drops the instance after cycle-1 and opens a NEW one
    // from the same `kb_root`, proving the disk-reload path is exercised for
    // the set-A vectors.

    #[test]
    fn large_index_compact_idempotent_across_process_restart() {
        let dir = TempDir::new().unwrap();

        // ---- Cycle 1 ("process 1"): ingest set A, compact, drop ----
        let set_a: Vec<Embedded> = (0..5).map(|i| make_large_embedded(i, i)).collect();
        let planted_a_id = set_a[2].chunk.chunk_id.clone();
        let query_a = set_a[2].embedding.clone();

        {
            let mut idx = open_large_index(&dir, "restart-topic");
            idx.ingest(&set_a).unwrap();
            idx.compact().unwrap();
            // Explicit drop: simulates process exit.  After this line the
            // DiskANN files are on disk; the in-memory index is gone.
        }

        // ---- Cycle 2 ("process 2"): open fresh, ingest set B, compact ----
        let set_b: Vec<Embedded> = (5..10).map(|i| make_large_embedded(i, i)).collect();
        let planted_b_id = set_b[1].chunk.chunk_id.clone(); // global i=6
        let query_b = set_b[1].embedding.clone();

        {
            let mut idx2 = open_large_index(&dir, "restart-topic");
            idx2.ingest(&set_b).unwrap();
            idx2.compact().unwrap();
        }

        // ---- Verification: open a third instance just for querying ----
        let idx3 = open_large_index(&dir, "restart-topic");

        // Set-A vector must survive the drop+reopen+second-compact cycle.
        let hits_a = idx3.search(&query_a, 1).unwrap();
        assert!(
            !hits_a.is_empty(),
            "set-A vector must be searchable after drop + reopen + second compact"
        );
        assert_eq!(
            hits_a[0].chunk.chunk_id, planted_a_id,
            "rank-0 for set-A query must be the set-A planted chunk"
        );

        // Set-B vector must also be searchable.
        let hits_b = idx3.search(&query_b, 1).unwrap();
        assert!(
            !hits_b.is_empty(),
            "set-B vector must be searchable after second compact"
        );
        assert_eq!(
            hits_b[0].chunk.chunk_id, planted_b_id,
            "rank-0 for set-B query must be the set-B planted chunk"
        );

        // Total count must equal 10, computed from a fresh open (not from the
        // in-memory total of any prior instance).
        let stats = idx3.stats().unwrap();
        assert_eq!(
            stats.vector_count, 10,
            "fresh open after two 5-vector compact cycles must report 10 vectors"
        );
    }

    // ── Crash-recovery manifest tests (Phase 3E) ─────────────────────────────

    /// Opening a fresh kb_root returns an empty manifest with no error.
    #[test]
    fn manifest_loads_empty_on_fresh_open() {
        let dir = TempDir::new().unwrap();
        let idx = open_index(&dir, "manifest-fresh");
        assert!(
            idx.manifest().videos.is_empty(),
            "fresh open must return empty manifest"
        );
    }

    /// After `upsert_video_state` + drop + reopen, the state is present.
    #[test]
    fn upsert_video_state_persists_across_reopen() {
        use learn_core::{IngestStatus, VideoState};

        let dir = TempDir::new().unwrap();
        let vs = VideoState {
            video_id: "vid-persist".to_string(),
            status: IngestStatus::Acquired,
            fetched_at: Some("2026-01-01T00:00:00Z".to_string()),
            indexed_at: None,
            chunk_count: 0,
            error: None,
        };

        // Session 1: upsert + drop.
        {
            let mut idx = open_index(&dir, "manifest-persist");
            idx.upsert_video_state(vs.clone()).unwrap();
        }

        // Session 2: reopen and check.
        let idx2 = open_index(&dir, "manifest-persist");
        let recovered = idx2
            .manifest()
            .videos
            .get("vid-persist")
            .expect("video state must survive reopen");
        assert_eq!(recovered.status, IngestStatus::Acquired);
        assert_eq!(
            recovered.fetched_at.as_deref(),
            Some("2026-01-01T00:00:00Z")
        );
    }

    /// `save_manifest` writes via tmp + rename; no `.json.tmp` artifact remains.
    #[test]
    fn save_manifest_is_atomic_no_tmp_remains() {
        use learn_core::{IngestStatus, VideoState};

        let dir = TempDir::new().unwrap();
        let mut idx = open_index(&dir, "manifest-atomic");
        idx.upsert_video_state(VideoState {
            video_id: "vid-atomic".to_string(),
            status: IngestStatus::Embedded,
            fetched_at: None,
            indexed_at: None,
            chunk_count: 3,
            error: None,
        })
        .unwrap();

        // The manifest JSON file must exist.
        let meta_dir = dir.path().join("_meta");
        let manifest_file = meta_dir.join("manifest-atomic.json");
        assert!(
            manifest_file.exists(),
            "manifest file must exist after upsert"
        );

        // No .json.tmp artifact should remain.
        let tmp_file = meta_dir.join("manifest-atomic.json.tmp");
        assert!(
            !tmp_file.exists(),
            ".json.tmp must not remain after atomic write"
        );

        // Verify no stray .tmp files anywhere under the kb_root.
        let tmp_count = walkdir_tmp_count(dir.path());
        assert_eq!(tmp_count, 0, "no .tmp files should remain in kb_root");
    }

    /// Advancing through status transitions leaves the last status on disk.
    #[test]
    fn manifest_records_status_transitions() {
        use learn_core::{IngestStatus, VideoState};

        let dir = TempDir::new().unwrap();
        let mut idx = open_index(&dir, "manifest-transitions");

        let statuses = [
            IngestStatus::Pending,
            IngestStatus::Acquired,
            IngestStatus::Transcribed,
            IngestStatus::Chunked,
            IngestStatus::Embedded,
            IngestStatus::Indexed,
        ];

        for &status in &statuses {
            idx.upsert_video_state(VideoState {
                video_id: "vid-transitions".to_string(),
                status,
                fetched_at: None,
                indexed_at: None,
                chunk_count: 0,
                error: None,
            })
            .unwrap();
        }

        // Final status on disk must be Indexed.
        let idx2 = open_index(&dir, "manifest-transitions");
        let vs = idx2
            .manifest()
            .videos
            .get("vid-transitions")
            .expect("video state must be present after all transitions");
        assert_eq!(
            vs.status,
            IngestStatus::Indexed,
            "final persisted status must be Indexed"
        );
    }

    /// Helper: count `.tmp` files anywhere under `root`.
    fn walkdir_tmp_count(root: &std::path::Path) -> usize {
        fn recurse(dir: &std::path::Path) -> usize {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return 0;
            };
            entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    let p = e.path();
                    if p.is_dir() {
                        recurse(&p)
                    } else if p.extension().and_then(|ex| ex.to_str()) == Some("tmp") {
                        1
                    } else {
                        0
                    }
                })
                .sum()
        }
        recurse(root)
    }

    // ── Witness chain tests ──────────────────────────────────────────────────

    /// Ingest 5 chunks — the witness chain must have exactly 5 entries.
    #[test]
    fn witness_chain_appends_one_per_ingest() {
        let dir = TempDir::new().unwrap();
        let mut idx = open_index(&dir, "witness-count");
        ingest_five(&mut idx);
        assert_eq!(
            idx.witness_chain().len(),
            5,
            "expected 5 witness entries after ingesting 5 chunks"
        );
    }

    /// A fresh chain with correct digests must verify without error.
    #[test]
    fn witness_chain_verify_fresh_index_passes() {
        let dir = TempDir::new().unwrap();
        let mut idx = open_index(&dir, "witness-verify-fresh");
        ingest_five(&mut idx);
        idx.verify_witness_chain()
            .expect("fresh witness chain must verify cleanly");
    }

    /// Corrupting one entry's `digest` must make `verify_witness_chain` return Err.
    #[test]
    fn witness_chain_detects_tampering() {
        let dir = TempDir::new().unwrap();
        let mut idx = open_index(&dir, "witness-tamper");
        ingest_five(&mut idx);

        // Corrupt seq-3's digest.
        idx.witness_chain[2].digest[0] ^= 0xFF;

        let result = idx.verify_witness_chain();
        assert!(
            result.is_err(),
            "verify must return Err when a digest is corrupted"
        );
    }

    /// Corrupting one entry's `previous_hash` must break the chain linkage check.
    #[test]
    fn witness_chain_detects_chain_break() {
        let dir = TempDir::new().unwrap();
        let mut idx = open_index(&dir, "witness-chain-break");
        ingest_five(&mut idx);

        // Corrupt seq-2's previous_hash (changes the link, not the digest).
        idx.witness_chain[1].previous_hash[0] ^= 0xFF;

        let result = idx.verify_witness_chain();
        assert!(
            result.is_err(),
            "verify must return Err when previous_hash linkage is broken"
        );
    }

    /// After a drop + reopen cycle the witness chain must still verify.
    #[test]
    fn witness_chain_persists_across_open_close() {
        let dir = TempDir::new().unwrap();

        // Session 1: ingest and drop.
        {
            let mut idx = open_index(&dir, "witness-persist");
            ingest_five(&mut idx);
            assert_eq!(idx.witness_chain().len(), 5);
        }

        // Session 2: reopen and verify.
        let idx2 = open_index(&dir, "witness-persist");
        assert_eq!(
            idx2.witness_chain().len(),
            5,
            "witness chain must survive a drop + reopen"
        );
        idx2.verify_witness_chain()
            .expect("reloaded witness chain must verify cleanly");
    }

    // ── Existing tests unchanged marker ─────────────────────────────────────
    // The tests above (open_creates_fresh, ingest_five_chunks, search_rank_zero_matches,
    // compact_no_error, chunk_id_to_u64_is_stable, sidecar_persists_across_reopen)
    // are the original LearnIndex tests.  They are run by the same `cargo test -p
    // learn-index` invocation and must remain green.

    // ── FNV-1a stability pin (explicit alias to satisfy spec naming) ─────────

    #[test]
    fn chunk_id_to_u64_stability_pin_unchanged() {
        // Same golden values as chunk_id_to_u64_is_stable — duplicate pin ensures
        // the spec test name passes even if the original test is renamed.
        assert_eq!(chunk_id_to_u64(""), 0xcbf29ce484222325);
        assert_eq!(chunk_id_to_u64("a"), 0xaf63dc4c8601ec8c);
        assert_eq!(chunk_id_to_u64("vid-001:0"), 0x25d5ced728425b84);
    }

    // ── New: embedding_for_chunk_id_returns_known_value ──────────────────────
    //
    // Ingest 3 chunks with known embeddings.  Assert that
    // `embedding_for_chunk_id` returns the exact vector stored.

    #[test]
    fn embedding_for_chunk_id_returns_known_value() {
        let dir = TempDir::new().unwrap();
        let mut idx = open_index(&dir, "emb-known");

        let batch: Vec<Embedded> = (0..3).map(make_embedded).collect();
        idx.ingest(&batch).unwrap();

        // chunk-001 → make_embedded(1) → embedding[1] = 1.0, rest 0.0
        let emb = idx
            .embedding_for_chunk_id("chunk-001")
            .expect("embedding must exist for chunk-001");
        assert_eq!(emb.len(), DIM, "dimension must match");
        assert!(
            (emb[1] - 1.0_f32).abs() < 1e-6,
            "position 1 must be 1.0, got {}",
            emb[1]
        );
        for (j, &v) in emb.iter().enumerate() {
            if j != 1 {
                assert!(v.abs() < 1e-6, "position {j} must be 0.0, got {v}");
            }
        }
    }

    // ── New: embedding_persists_across_open_close ─────────────────────────────
    //
    // Ingest, drop the index (simulates process exit), reopen, assert the
    // embedding round-trips correctly through the `.emb.bin` file.

    #[test]
    fn embedding_persists_across_open_close() {
        let dir = TempDir::new().unwrap();

        // Session 1: ingest.
        {
            let mut idx = open_index(&dir, "emb-persist");
            let batch: Vec<Embedded> = (0..5).map(make_embedded).collect();
            idx.ingest(&batch).unwrap();
        }

        // Session 2: reopen, assert embedding for chunk-002 survived.
        let idx2 = open_index(&dir, "emb-persist");
        let emb = idx2
            .embedding_for_chunk_id("chunk-002")
            .expect("embedding must survive reopen");

        assert_eq!(emb.len(), DIM);
        // chunk-002 → make_embedded(2) → embedding[2] = 1.0, rest 0.0
        assert!(
            (emb[2] - 1.0_f32).abs() < 1e-6,
            "position 2 must be 1.0 after reopen"
        );
    }
}
