# ADR-001 — Learn-RV Elite Roadmap

**Status:** accepted (phases 0–2E) | proposed (phases 2.5, 3, 4, 5)
**Date Started:** 2026-05-02
**Owner:** Stuart Kerr
**Decision Authority:** Stuart Kerr (commissioner) + Ruflo agent panel (verification)

## Context

Learn-RV is a pure-Rust knowledge-base ingestion CLI. Source video → captions → semantic chunks → BGE-large embeddings → per-topic `.rvf` file → cited answers. The substrate is RuVector (RVF format, HNSW, witness chain) — Stuart's own technology stack, not a third-party dependency. The goal is an *elite-tier* system that learns over time (SONA), self-checks for drift (CoherenceMonitor), surfaces sub-topics autonomously (Louvain communities), scales to billion-vector recall (DiskANN), and answers entirely on-device when sovereignty is required (ruvllm).

This ADR records every phase of the build so progress survives any context loss, mid-session crash, or hand-off.

## Decision

Build to the **Elite bar** in 14 phases. Each phase has four gates that must close before it counts as done: **Written**, **Ruflo-QA'd**, **Tested**, **Confirmed**.

- **Written** — code lands, file exists, public surface matches design memos.
- **Ruflo-QA'd** — passed the four-mandate panel: makes-sense / is-working / makes-a-difference / tests-legitimate.
- **Tested** — `cargo build --workspace && cargo clippy -D warnings && cargo test --workspace` green.
- **Confirmed** — observable user-facing behaviour matches the README claim, verified by either an integration test running real code paths or a smoke test against a real input.

## Key technical decisions

The decisions captured here are the non-obvious choices made during the build that future contributors should not have to re-derive.

| Decision | Chosen approach | Rejected alternative | Rationale |
|---|---|---|---|
| Vector substrate | RuVector RVF format (path-deps to `~/RuVector_Clean/crates/rvf-*`) | Postgres + pgvector / Pinecone / SQLite-vec | Stuart's CLAUDE.md Rule 1 mandates RVF; Rule 17 mandates Rust-first when RuVector is principal. RVF gives append-only writes, witness chain, and HNSW native to the format with zero daemon. |
| Graph store | `ruvector-graph` redb backend | Continue JSON-on-disk from Phase 1 | redb gives ACID and crash-safety; JSON file required full-rewrite per mutation. |
| Cypher | Omitted from public surface | Wire upstream `ruvector-graph::cypher::*` | Upstream Cypher modules are non-functional stubs. Louvain/PageRank/shortest_path implemented from scratch on the adjacency API. |
| BM25 persistence | In-memory tantivy, rebuilt at query time from sidecar | On-disk persistent tantivy index | Sidecar is authoritative; dual writable artifact creates sync bugs. Revisit if a topic exceeds ~20K chunks. |
| Embedding persistence | `<topic>.emb.bin` companion file added in Phase 3B | Inline in `.rvf` sidecar at ingest time | Keeps JSON sidecar lean (human-readable); separate file makes the embedding load O(N) single read. Required to unblock proper MMR cosine and `differentiableSearch`. |
| Retrieval ACL exception | `learn-retrieve` directly imports `learn-embed` and `learn-index` | Mediate every call through `learn-core` types | Retrieval, Embedding, and Indexing form a coherent shared kernel; an ACL between them would force every Hit lookup to round-trip through learn-core, costing performance for no architectural gain. |
| Synthesizer trait + dispatch | Trait with two impls, `LEARN_SYNTH_LOCAL` env var picks at runtime | Compile-time feature flag | Users can flip cloud↔local without rebuilding. |
| Topic slug derivation (Phase 1) | Simple URL-trailing-segment heuristic | LLM-based naming | Phase 1 placeholder. Phase 2 swaps to yt-dlp metadata + LLM-derived semantic name. |
| Whisper backend | `whisper-rs` with Metal feature | `faster-whisper` (Python) / `mlx-whisper` | Pure Rust, no Python sidecar; whisper.cpp via bindgen, Metal native on M-series. |
| Embedder dim choice | 1024 (BGE-large-en-v1.5) | 384 (BGE-small) / 768 (BGE-base) | Higher dim improves separation as topics grow; M3 Max can run 1024-dim ONNX inference at acceptable latency. |
| AIMDS sidecar (npm) | Fail-soft when `@ruflo/aidefence` is absent (404 on public npm) | Block all queries until AIMDS available | Package not yet published; users wire `LEARN_AIMDS_BIN` to enable scanning. `LEARN_AIMDS_REQUIRED=1` flips to fail-closed. |
| Cross-platform | M-series-only at v1 | Universal binary at v1 | Stuart's explicit prioritization: M-series Mac → Intel Mac → Linux → Windows. Cross-compile is Phase 5. |

## The Plan

### Phase 0 — Workspace scaffold

- [x] **Written** — 12-crate Cargo workspace, Ruflo init, contracts in `learn-core`
- [x] **Ruflo-QA'd** — initial review pass
- [x] **Tested** — `cargo build && test` green, 23 tests
- [x] **Confirmed** — clean baseline, no architectural debt

### Phase 1 — Ingest path crates

- [x] **Written** — `learn-acquire`, `learn-asr` (whisper-rs Metal), `learn-chunk`, `learn-embed` (BGE-large ONNX), `learn-index` (RVF), `learn-graph` (JSON-on-disk initial)
- [x] **Ruflo-QA'd** — code review verdict 84/100, test audit passed
- [x] **Tested** — 88 tests passing
- [x] **Confirmed** — every crate has hermetic tests; integration smoke `#[ignore]` until model files present

### Phase 1.5 — Five RuVector capability adoptions

- [x] **Written** — SONA self-learning, CoherenceMonitor + SemanticDriftDetector, ruvector-graph (Louvain + PageRank), DiskANN scale path, ruvllm sovereignty backend
- [x] **Ruflo-QA'd** — 4 of 5 catalog claims caught and adapted (AdaptiveEmbedder, RaBitQ, CoherenceMonitor, ReasoningBank schema)
- [x] **Tested** — 116 tests passing
- [x] **Confirmed** — five new capabilities sit cleanly behind learn-core contracts

### Phase 2A — Two more capability adoptions

- [x] **Written** — ReasoningBank (JSONL trajectory store), hybrid retrieval (tantivy 0.22 BM25 + RRF + MMR placeholder)
- [x] **Ruflo-QA'd** — verified
- [x] **Tested** — 119 tests passing
- [x] **Confirmed** — Retriever publicly exposes `search()`

### Phase 2B — QA-verdict fix-pack (13 items)

- [x] **Written** — SONA persistence, Louvain HashMap side-table, sidecar atomic writes, score-formula clamp, URL pre-validation, 3 invariant tests, 2 fmt hunks, JSON-deletion contract assertion, FNV-1a stability pin, shortest_path tests, single-vector edge-case test, LEARN_SYNTH_LOCAL empty-value test, sona_delta passthrough test
- [x] **Ruflo-QA'd** — four-agent panel verdict (sense / working / tests / value)
- [x] **Tested** — 133 tests passing, 0 failed
- [x] **Confirmed** — wave-A QA panel returned all four verdicts; vacuous-test pattern caught and 2 P0 tests strengthened

### Phase 2B-test-strengthening — close vacuous-green tests

- [x] **Written** — `apply_sona_delta` promoted to `pub(crate)`, two SONA tests rewritten to call production paths, cross-restart DiskANN idempotency test added
- [x] **Ruflo-QA'd** — review confirmed: tests now exercise real production glue
- [x] **Tested** — 135 tests passing
- [x] **Confirmed** — production paths exercised, no parallel-implementation bypass

### Phase 2C — CLI wiring (the make-or-break turn)

- [x] **Written** — `Cmd::Ingest`, `Cmd::Ask`, `Cmd::Apply` wired to real pipeline calls in `learn-cli/src/main.rs`. Error envelope translates LearnError variants to user-readable stderr.
- [x] **Ruflo-QA'd** — QA blockers closed 2026-05-02: `--depth` wired to retriever `k` (quick=5, medium=10, deep=20); `--since` and `--with_frames` emit explicit user-facing warnings (no silent drop); `learn-asr` and `anyhow` unused deps removed from `learn-cli/Cargo.toml`; `--limit` was already wired.
- [x] **Tested** — 137 tests passing, build clean, clippy clean, release binary builds
- [x] **Confirmed** — smoke test ran: `learn ingest "https://youtu.be/QZMljuD10sU" --topic claude-skills` → 31 chunks ingested into a real `~/Docs/KB/claude-skills.rvf` (125 KB binary HNSW + 41 KB sidecar)

### Phase 2D — First real cited answer

- [x] **Written** — N/A (verifies Phase 2C end-to-end against real models)
- [x] **Ruflo-QA'd** — N/A (smoke test, not new code)
- [x] **Tested** — `learn ingest "https://youtu.be/QZMljuD10sU" --topic claude-skills` → 31 chunks ingested into `~/Docs/KB/claude-skills.rvf` (125 KB binary HNSW + 41 KB sidecar)
- [x] **Confirmed** — `learn ask claude-skills "<q>"` returns a real cited answer with `[1][2][3]` citation markers anchored to the chunks; verified end-to-end through Anthropic Messages API on 2026-05-02.

**Depends on:** Phase 2E (AnthropicSynthesizer real). To resume after context loss: confirm `cargo build --release` is green, set `ANTHROPIC_API_KEY` in shell, run `./target/release/learn ask claude-skills "what does this teach"`, expect text + citations.

### Phase 2E — AnthropicSynthesizer real implementation

- [x] **Written** — `crates/learn-synth/src/lib.rs` `AnthropicSynthesizer::ask` and `apply` make real `reqwest` calls to `https://api.anthropic.com/v1/messages` with `claude-opus-4-7` (override via `LEARN_ANTHROPIC_MODEL`); inbound + outbound AIMDS scan envelope preserved; exponential backoff on 429/503.
- [ ] **Ruflo-QA'd** — pending (will fold into the final Phase 4C panel)
- [x] **Tested** — 24 tests in `learn-synth` (8 AIMDS + 16 lib + 5 new for Anthropic), 0 failed
- [x] **Confirmed** — verified by Phase 2D smoke test on 2026-05-02

### Phase 2.5 — `learn study` autonomous curriculum discovery

- [x] **Written** — `learn-discover` real implementation landed (harvest + 5-factor scoring + caption gate + Claude curation; heuristic fallback when API key absent)
- [ ] **Ruflo-QA'd** — pending Phase 4C
- [x] **Tested** — 14 hermetic tests in learn-discover, 0 failed
- [ ] **Confirmed** — pending the `Cmd::Study` re-wire to call the real implementation (cross-agent re-wire turn)

### Phase 3A — 10 remaining CLI subcommands

- [x] **Written** — 10 subcommands wired in `learn-cli/src/commands.rs`
- [ ] **Ruflo-QA'd** — pending Phase 4C
- [x] **Tested** — 19 tests in learn-cli, 0 failed
- [ ] **Confirmed** — pending the `Cmd::Eval` and `Cmd::Study` re-wires (cross-agent staleness cleanup)

### Phase 3B — Persist embeddings in sidecar + proper MMR cosine

- [x] **Written** — `<topic>.emb.bin` companion, MMR over real cosine
- [ ] **Ruflo-QA'd** — pending Phase 4C
- [x] **Tested** — green
- [ ] **Confirmed** — pending

### Phase 3C — AIMDS sidecar wiring on query path

- [x] **Written** — `learn-synth/src/aimds.rs` with inbound + outbound scan envelopes
- [ ] **Ruflo-QA'd** — pending
- [x] **Tested** — 12 hermetic AIMDS tests passing; sidecar binary `npx @ruflo/aidefence` is **not on npm publicly**, code path returns `Skipped` by default and the user wires `LEARN_AIMDS_BIN` to enable it. Honest gap; documented.
- [ ] **Confirmed** — production wiring against a real AIMDS binary is a future Stuart-side task

### Phase 3D — Eval harness with golden Q&A regression

- [x] **Written** — `learn-eval` crate with `GoldenSet`, `run_eval`, `EvalReport`
- [ ] **Ruflo-QA'd** — pending Phase 4C
- [x] **Tested** — green
- [ ] **Confirmed** — pending `learn eval <topic>` end-to-end run against a golden YAML

### Phase 3E — Manifest resume + atomic writes everywhere

- [ ] **Written** — `LearnIndex::save_meta` already atomic; manifest crash-recovery resume logic still pending
- [ ] **Ruflo-QA'd** —
- [ ] **Tested** —
- [ ] **Confirmed** — kill `learn ingest` mid-flight, re-run, picks up at the right point

### Phase 4A — `ruvector-consciousness` integrated-information KPI

- [ ] **Written** — surface in `learn status`
- [ ] **Ruflo-QA'd** —
- [ ] **Tested** —
- [ ] **Confirmed** — `learn status <topic>` prints a KPI score derived from the corpus

### Phase 4B — `ruvector-verified` formal proofs

- [ ] **Written** — SAT/SMT proofs over chunker invariants and `claim_id` derivation
- [ ] **Ruflo-QA'd** —
- [ ] **Tested** —
- [ ] **Confirmed** — `cargo verified` (or equivalent) emits a proof certificate

### Phase 4C — Final four-agent QA panel against full elite state

- [ ] **Written** — N/A (review only)
- [ ] **Ruflo-QA'd** — sense / working / makes-a-difference / tests-legitimate
- [ ] **Tested** — full workspace gate green
- [ ] **Confirmed** — verdict says "Elite ships"

### Phase 4D — `ruflo-adr:adr-index` + `ruflo-ddd:ddd-validate`

- [ ] **Written** — formalize this ADR + accompanying DDD docs in AgentDB index
- [ ] **Ruflo-QA'd** —
- [ ] **Tested** — `npx ruflo adr-validate` clean, `npx ruflo ddd-validate` clean
- [ ] **Confirmed** — ADRs and bounded contexts pass static governance

### Phase 5 — Cross-platform builds (deferred per Stuart's M-series-first preference)

- [ ] **Written** — release.yml CI matrix for x86_64-apple-darwin, x86_64-pc-windows-msvc, x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu
- [ ] **Ruflo-QA'd** —
- [ ] **Tested** — all 5 targets in CI
- [ ] **Confirmed** — released binaries on GitHub for all 5 targets

## Tracked deviations and design caveats

These are real items that landed but with caveats Stuart should know about:

- **DiskANN private-format coupling.** `LearnIndexLarge::compact` reads `vectors.bin` directly with hand-written byte parsing. If `ruvector-diskann` changes its save format, this will silently corrupt or panic. Caveat documented in code; track for Phase 4 hardening.
- **`validate_source` UX gap.** `@handle` and `ytsearch:` schemes pass the validator but `Url::parse` later rejects them. Caveat: `learn ingest` against a channel handle currently errors with a confusing message; channel-ingest needs a separate code path. Track for Phase 5.
- **AIMDS package not on public npm.** `@ruflo/aidefence` returns 404 when fetched via npx. The wiring is correct; the binary just isn't published. User wires `LEARN_AIMDS_BIN` to enable real scanning.
- **`differentiableSearch` exists but unwired.** `ruvector-gnn` exposes the function but it takes raw embedding vectors and `ruvector-gnn` isn't a workspace dep. Phase 3B unlocks this once embeddings persist in the sidecar; ruvector-gnn dep can be added then.
- **Cypher omitted from `learn-graph`.** Upstream `ruvector-graph::cypher::*` modules are non-functional stubs. Cypher waits on upstream. Louvain/PageRank/shortest_path are implemented from scratch on top of the adjacency API.
- **Phase 2C flag wiring (2026-05-02).** `--depth` (Ask) wired to retriever k-count. `--limit` (Ingest) was already wired via `run_ingest_with_limit`. `--since` and `--with_frames` (Ingest) warn-and-ignore: `acquire_url` takes no date-filter or frame-extraction parameters; these flags are real future work, not silent no-ops.

## Active in-flight agents (as of 2026-05-02 18:30 EDT)

- Phase 2E AnthropicSynthesizer real implementation (ruflo-core:coder)
- Phase 2.5 learn-discover autonomous curriculum (ruflo-goals:deep-researcher)
- Phase 3A 10 remaining CLI subcommands (ruflo-core:coder)
- Phase 3B sidecar embeddings + MMR cosine (ruflo-ruvector:vector-engineer)
- Phase 3D eval harness (ruflo-testgen:tester) — landed (lib.rs visible)

## Verification commands

```bash
cd "/Users/stuartkerr/Code/Video watcher skill/learn-rs"
cargo build --release
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
./target/release/learn ingest "https://youtu.be/QZMljuD10sU" --topic claude-skills
./target/release/learn ask claude-skills "what does this teach?"
```

## Persistence

This ADR is committed to the project repo at `learn-rs/docs/adr/ADR-001-elite-roadmap.md`. It survives context loss, session crash, machine restart, and hand-off. When picking up work after any interruption: read this file first, find the highest unchecked checkbox, resume from there.

## Related documents

- `docs/phase2-design.md` — Phase 2 retrieval + synthesis design memo
- `docs/phase25-design.md` — Phase 2.5 autonomous curriculum design memo
- `docs/ddd/DDD-001-bounded-contexts.md` — Domain-driven design map of the seven bounded contexts (this ADR's companion)
- `~/.claude/projects/-Users-stuartkerr-Code-Video-watcher-skill/memory/MEMORY.md` — durable Claude memory index for the project

## Change log

- 2026-05-02 — initial draft, Phases 0 through 2C-test-strengthening checked, all later phases unchecked.
- 2026-05-02 (later) — Phase 2D/2E confirmed end-to-end; Phase 2.5, 3A, 3B, 3C, 3D landed; 4 of 5 wave-B agents reported in; final test count 187 passed, 0 failed (after the Phase-2C-test-strengthening agent and the gate-fix agent close). Anthropic real cited answer verified against QZMljuD10sU. ADR + DDD P0 edits applied per Ruflo doc-QA verdicts.
