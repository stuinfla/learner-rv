# DDD-001 — Learn-RV Bounded Contexts

**Status:** Living document
**Date:** 2026-05-02
**Companion to:** [ADR-001 — Elite Roadmap](../adr/ADR-001-elite-roadmap.md)

## Why this exists

The Learn-RV codebase is twelve crates wide and growing. As capabilities land, the risk of cross-context coupling grows with it. This document maps the eight bounded contexts, names the aggregates inside each, and pins the ubiquitous language so future contributors (human or agent) speak the same words for the same things.

## Ubiquitous language

| Term | Meaning |
|---|---|
| **Topic** | A user-named slug under which a knowledge base accumulates (e.g. `french-cooking`, `indexed-arbitrage`). One `.rvf` file per topic. |
| **Source** | Anything that can be ingested: a YouTube URL, a playlist URL, a `ytsearch:<query>` pseudo-scheme, a channel `@handle`, a local video file. |
| **VideoRef** | The canonical reference to one acquired video (id, url, title, channel, duration). |
| **Acquired** | The artifact set returned by acquisition: VideoRef + paths to captions VTT and/or audio. |
| **Transcript** | A list of timestamped `Segment`s (start_seconds, end_seconds, text). Source is either `Captions` or `WhisperLocal`. |
| **Chunk** | A semantically-coherent slice of transcript text (~300 tokens, with overlap), suitable for embedding. |
| **Embedded** | A `Chunk` with its dense embedding vector attached. |
| **Hit** | A retrieved `Chunk` plus its score and rank. |
| **Answer** | The synthesizer's output: text + citations + abstain flag. |
| **Citation** | A pointer to a source video at a specific timestamp. |
| **Trajectory** | A past successful query/apply outcome stored in ReasoningBank for adapting future similar tasks. |
| **Curriculum** | An ordered list of `CurriculumPick`s — videos chosen by `learn study` autonomous discovery. |
| **Claim** | A declarative statement extracted from a transcript chunk. Has a deterministic `claim_id` (SHA-256). |
| **Entity** | A named thing referenced by claims (Person, Organization, Concept, Paper, Product, Place). |
| **Relation** | A directed edge between two entities (Cites, Refutes, BuildsOn, Mentions, EmployedBy). |
| **GoldenItem** | One row of a per-topic Q&A regression suite. |
| **EvalReport** | The output of running golden Q&A against a topic. |
| **KbHealth** | Per-topic health snapshot from `learn-coherence` — Fiedler eigenvalue (graph connectivity proxy) plus contradiction rate. |
| **DriftReport** | Output of `learn-coherence` drift detection — CUSUM-style changepoint over the embedding distribution of recent ingests. |

## The seven bounded contexts

Each context owns its own crate(s) and exposes a small public surface to the rest of the system. Cross-context calls go through `learn-core` types only.

### Context 1 — Acquisition

**Crate:** `learn-acquire`
**Aggregate root:** `Acquired { video, captions_vtt, audio_mp3, raw_dir }`
**Domain events:** `SourceValidated`, `VideoFetched`, `CaptionsParsed`, `AcquisitionFailed`
**External integrations:** `yt-dlp` (subprocess), `ffmpeg` (subprocess for audio extraction in Phase 2D-plus)

Responsibilities:
- Validate the source string before any subprocess invocation (URL, ytsearch, local path, @handle).
- Invoke yt-dlp captions-first; fall back to audio download only if captions absent.
- Parse VTT into `Vec<Segment>`.
- Persist raw artifacts under `<kb_root>/_raw/<topic>/<video_id>/`.

Anti-corruption: nothing inside this context speaks RVF or HNSW. The output is plain `Acquired` and `Vec<Segment>`.

### Context 2 — Transcription

**Crate:** `learn-asr`
**Aggregate root:** `Transcript { video_id, language, source: TranscriptSource, segments }`
**Domain events:** `WhisperLoaded`, `AudioDecoded`, `TranscriptionCompleted`, `ModelMissing`
**External integrations:** `whisper-rs` → `whisper.cpp` (Metal on Apple Silicon, CPU elsewhere), `ffmpeg` for audio decode

Responsibilities:
- Load a ggml whisper model on demand. Fetch `ggml-base.en.bin` if absent.
- Decode mp3 → 16 kHz mono f32 PCM via ffmpeg.
- Run Whisper inference and produce a `Transcript` with `source = WhisperLocal`.

Boundary: this context is **only** invoked when Acquisition couldn't produce captions. The default ingest path stays captions-only.

### Context 3 — Intelligence (chunking + embedding + indexing + learning)

**Crates:** `learn-chunk`, `learn-embed`, `learn-index`, `learn-coherence`, `learn-reasoning`
**Aggregate roots:** `Vec<Chunk>` (transient), `Embedder` (long-lived per topic), `LearnIndex` (handle to one `.rvf`)
**Domain events:** `ChunksProduced`, `EmbeddingsComputed`, `RvfIngested`, `FeedbackRecorded`, `AdapterPersisted`, `WitnessChainExtended`
**External integrations:** `ort` → ONNX Runtime → BGE-large-en-v1.5, `ruvector-sona` → MicroLoRA, `rvf-runtime` → RvfStore, `ruvector-diskann` → DiskAnnIndex (large-scale path)

Responsibilities:
- Chunker: turn a `Transcript` into `Vec<Chunk>` with sentence-aware boundaries, ~300-token target, 50-token overlap, runt-tail merge.
- Embedder: tokenize each chunk, run BGE-large ONNX session, produce 1024-dim CLS vector. Apply SONA MicroLoRA delta if the per-topic adapter is non-zero. Persist adapter weights to `~/.cache/learn-rs/adapters/<topic>/lora.json` on `record_feedback`.
- Index: append `Embedded` batches to the per-topic `.rvf`. Maintain JSON sidecar mapping `chunk_id_u64 → Chunk` (and Phase 3B onward: a `.emb.bin` companion mapping `chunk_id_u64 → Vec<f32>` for proper MMR cosine).
- Auto-promote to `LearnIndexLarge` (DiskANN) when chunk count exceeds 50 000.
- `learn-coherence` — KB health (`KbHealth` with Fiedler eigenvalue + contradiction rate) and drift detection (`DriftReport` with CUSUM-style changepoint).
- `learn-reasoning` — `ReasoningBank` trajectory store (`Trajectory`, `derive_trajectory_id`), JSONL persistence, cosine retrieval over past successful queries.

**Witness chain (planned, not yet implemented in `learn-index`):** the underlying `rvf-runtime` crate provides `WitnessBuilder` and `GovernancePolicy` types for cryptographically anchoring inserts to source URL + timestamp. `learn-index` does not yet wire them — the sidecar stores `Chunk` data with `video_id` and `start_seconds` but does not produce a witness entry. ADR-001 Phase 4B tracks this as part of the formal-proofs work.

### Context 4 — Retrieval

**Crate:** `learn-retrieve`
**Aggregate root:** `Retriever { index, embedder, reranker, bm25 }`
**Domain events:** `QueryEmbedded`, `DenseHitsRetrieved`, `SparseHitsRetrieved`, `Fused`, `Reranked`, `MmrApplied`
**External integrations:** `tantivy` (in-memory BM25), `learn-embed::Reranker` (cross-encoder)

Responsibilities:
- Embed the query through the same `Embedder` used for ingest (so adapter sharpening applies to queries too).
- Run dense top-50 (HNSW via `LearnIndex::search`) and sparse top-50 (in-memory tantivy rebuilt on demand from the sidecar).
- Reciprocal Rank Fusion (k=60) across the two ranked lists.
- Optional cross-encoder rerank over the top 50 using `bge-reranker-base`.
- MMR with λ=0.7 and source-cap (≤ 3 chunks per video) for the final top-K.
- (Phase 3B) MMR uses real cosine over persisted embeddings, not score proxy.

### Context 5 — Synthesis

**Crate:** `learn-synth`
**Aggregate roots:** `dyn Synthesizer` (trait), `AnthropicSynthesizer`, `RuvllmSynthesizer`
**Domain events:** `AimdsScannedInbound`, `LlmInvoked`, `LlmResponded`, `AimdsScannedOutbound`, `Abstained`, `AnswerProduced`
**External integrations:** Anthropic Messages API (cloud), `ruvllm` → BitNet/MoE/MicroLoRA on Metal (sovereignty), `npx @ruflo/aidefence` (AIMDS, optional)

Responsibilities:
- Select implementation: `LEARN_SYNTH_LOCAL=1` → `RuvllmSynthesizer`; otherwise `AnthropicSynthesizer`.
- AIMDS guard: scan inbound (user question/task) before LLM call; scan outbound (model answer) after. On block in either direction, return `Answer { abstained: true }`.
- Render the `ASK_USER_TEMPLATE` or `APPLY_USER_TEMPLATE` with the `Hit` context block injected.
- Detect abstain in the model's text ("KB doesn't cover this" or "ABSTAIN") and set the flag.
- Build `Citation`s from the `Hit` set — every fact the user sees is anchored to a video timestamp.

### Context 6 — Curation (autonomous discovery)

**Crate:** `learn-discover`
**Aggregate root:** `Curriculum { topic_description, depth, picks }`
**Domain events:** `CandidatesHarvested`, `CandidatesScored`, `CaptionsGated`, `CurriculumProposed`, `IngestionTriggered`
**External integrations:** `yt-dlp ytsearch{N}:` (subprocess), Anthropic Messages API (curation prompt)

Responsibilities:
- Harvest a candidate pool (quick=30, medium=60, deep=150) via `ytsearch{N}:`.
- Apply the 5-factor heuristic rubric (title alignment, channel authority, recency, duration sanity, captions).
- Caption gate: full info-fetch on top 2× surface count to confirm caption availability.
- Send the gated shortlist to Claude with the curation prompt; receive ranked picks with sub-topic clustering and rationale.
- (Phase 2.5 deliverable) Surface to user, await confirmation, then drive ingestion through Context 3.

### Context 7 — Knowledge Graph

**Crate:** `learn-graph`
**Aggregate roots:** `LearnGraph` (handle to one redb-backed graph)
**Domain events:** `EntityUpserted`, `ClaimInserted`, `RelationInserted`, `CommunitiesDetected`, `PageRankComputed`
**External integrations:** `ruvector-graph` → redb-backed graph store

Responsibilities:
- Maintain the per-topic graph at `<kb_root>/_graph/<topic>.graphdb`.
- Deterministic `claim_id` derivation: SHA-256 over `text \x00 claimant \x00 video_id \x00 chunk_id`, first 16 hex chars.
- Louvain communities (with side-table optimization for scale), PageRank, BFS shortest path.
- Legacy JSON format detected on first open, migrated, source file removed.

### Context 8 — Evaluation (cross-context)

**Crate:** `learn-eval`
**Aggregate root:** `GoldenSet { topic, version, items }` → `EvalReport`
**Domain events:** `GoldenLoaded`, `GoldenValidated`, `EvalRun`, `ReportProduced`
**External integrations:** consumes Retrieval and Synthesis through trait abstractions

Responsibilities:
- Load `<kb_root>/<topic>/eval/golden.yaml` via serde_yaml.
- Run each item through `Retriever::search` + `Synthesizer::ask`/`apply`.
- Score: expected substring present (any-of), forbidden substring absent (none-of), citation count ≥ minimum, abstain acceptable for "shouldn't know X" cases.
- Aggregate to `EvalReport { passed, failed, abstained, items, aggregate_score, started_at, finished_at }`.

## Anti-corruption layers

Every cross-context call goes through `learn-core` types — there is no direct call from `learn-acquire` into `learn-index`, no direct call from `learn-graph` into `learn-retrieve`. The CLI (`learn-cli`) is the orchestrator that strings the contexts together; it is the only place where every context's public surface is in scope.

This keeps each context independently testable, independently mockable for `learn-eval`, and independently swappable (e.g. swapping `learn-asr` from `whisper-rs` to a Python sidecar would not require touching `learn-chunk`).

**Acknowledged exception — Retrieval/Intelligence shared kernel.** `learn-retrieve` directly imports `learn-embed::Embedder` / `Reranker` and `learn-index::LearnIndex` rather than mediating through `learn-core` types. This is a deliberate shared-kernel pattern: Retrieval, Embedding, and Indexing form a coherent ingest+query cluster where the round-trip through learn-core would cost performance with no architectural gain. The "every cross-context call goes through `learn-core` types only" rule applies to all OTHER context pairs.

## Aggregate consistency rules

- **One writer per Topic (after first ingest).** `RvfStore` (the substrate inside `LearnIndex`) acquires an OS-level advisory writer lock once the `.rvf` file exists. `LearnIndex::open` itself does not acquire an application-level lock — when the `.rvf` does not yet exist, no lock is held. Concurrent ingest from two processes against a brand-new topic is a known race; the second process will succeed-or-fail depending on filesystem ordering. Once any `ingest` call has created the file, the substrate's lock prevents a second concurrent writer.
- **Append-only writes for vector data.** Re-ingesting a topic appends new RVF segments; old segments are untouched. A crashed write is still readable up to the last committed segment.
- **Deterministic identifiers.** `claim_id` (SHA-256 of canonical input) and `chunk_id_u64` (FNV-1a of `chunk_id` string) are stable across processes and versions. Any change to the recipe is a breaking change documented in ADR-001.
- **Witness chain integrity.** Every chunk insert produces a witness entry. The chain is the audit trail for citations.

## Map to ADR-001 phases

| ADR phase | DDD context(s) primarily affected |
|---|---|
| Phase 0 — scaffold | none yet |
| Phase 1 — ingest crates | Acquisition, Transcription, Intelligence |
| Phase 1.5 — capability adoptions | Intelligence (SONA), KG (ruvector-graph), Synthesis (ruvllm) |
| Phase 2A — capability adoptions | Intelligence (ReasoningBank), Retrieval (hybrid) |
| Phase 2B — fix-pack | KG, Intelligence, Acquisition, Synthesis |
| Phase 2C — CLI wiring | none new — orchestrates the existing eight |
| Phase 2D — first cited answer | Retrieval, Synthesis (end-to-end check) |
| Phase 2E — Anthropic real | Synthesis |
| Phase 2.5 — `learn study` | Curation |
| Phase 3A — 10 CLI subcommands | KG (who-said, timeline, compare), Intelligence (compact, list, status), Curation (watch), Evaluation (eval) |
| Phase 3B — sidecar embeddings + MMR cosine | Intelligence + Retrieval |
| Phase 3C — AIMDS sidecar | Synthesis |
| Phase 3D — eval harness | Evaluation |
| Phase 3E — manifest resume | Intelligence |
| Phase 4A — consciousness KPI | Intelligence (new sub-context: KPI) |
| Phase 4B — verified formal proofs | Intelligence (chunker), KG (claim_id) |
| Phase 4C — final QA panel | all |

## Validation

`npx ruflo ddd-validate` (Phase 4D, **aspirational** — the `ddd` subcommand does not exist in the installed Ruflo CLI as of 2026-05-02; this command is the planned interface):
- No cross-context import violations (e.g. `learn-acquire` directly importing from `learn-index`).
- No aggregate root field is mutated outside its owning context.
- Domain event names are unique across the system.
- Public surface of each context lives in `learn-core` types or types defined in the context's own crate.

## Change log

- 2026-05-02 — initial draft, all seven bounded contexts mapped, ubiquitous language pinned, anti-corruption layers and aggregate consistency rules captured.
