# Phase 2 Design Memo

## Capability outcome

`learn ask <topic> "<question>"` returns a paragraph-length answer grounded exclusively in the indexed video corpus, with every factual claim hyperlinked to the precise timestamp in the source video. `learn apply <topic> "<task>"` produces a structured artifact — recipe, strategy, plan, or code skeleton — assembled from the corpus, with the same citation discipline. Both commands abstain gracefully when the knowledge base does not cover the question rather than hallucinating; the user gets a clear "KB doesn't cover this" message instead of a plausible-sounding fabrication.

---

## Architecture

### Hybrid retrieval

**Crate choice.** Tantivy 0.22 was the placeholder documented in `learn-retrieve/Cargo.toml`; the current stable release is **0.26.1**. Use `tantivy = "0.26"`. The API is backward-compatible with the 0.22 stub comment.

**Persistence decision: in-memory, rebuilt at query time.** The corpus is small and bounded per topic (a KB of a few hundred video chunks is typically under 10 MB of text). An in-memory Tantivy `Index` built from the sidecar's `HashMap<String, Chunk>` takes under 100 ms even for 5,000 chunks. A persistent on-disk Tantivy index alongside the `.rvf` introduces a second writable artifact that can diverge from the sidecar on crash or partial ingest. The sidecar is already the authoritative chunk store; re-deriving BM25 from it on each query is correct-by-construction and eliminates sync bugs. This is the right tradeoff until corpus size proves the assumption wrong (>50K chunks would warrant reconsideration).

**RRF fusion formula.** Given rank lists `R_dense` and `R_bm25`, each item's fused score is:

```
rrf(d) = 1/(k + rank_dense(d)) + 1/(k + rank_bm25(d)),  k = 60
```

Items absent from one list get `rank = N + 1` (just beyond the tail). Sort descending by `rrf(d)`, take top 50 as input to the reranker.

**Function signatures for `learn-retrieve`:**

```rust
pub struct Retriever {
    index: LearnIndex,
    embedder: Embedder,
}

impl Retriever {
    pub fn new(index: LearnIndex, embedder: Embedder) -> Self;

    /// Hybrid dense+BM25 search with RRF fusion.
    /// Returns up to `k` hits, ranked by fused score.
    pub fn search(&mut self, query: &str, k: usize) -> Result<Vec<Hit>>;

    /// Rebuild the in-memory BM25 index from chunks in `index`.
    fn build_bm25(&self) -> Result<tantivy::Index>;

    /// Compute RRF scores and return fused, sorted hits.
    fn rrf_fuse(dense: &[Hit], bm25: &[Hit], k_rrf: u32) -> Vec<Hit>;
}
```

`search` calls `build_bm25`, runs both legs, calls `rrf_fuse`, applies MMR + source-cap (see below), and returns the final top-`k` list. `build_bm25` is cheap enough to call per query; no persistent state needed.

---

### Reranker

**Model.** `BAAI/bge-reranker-base` is a cross-encoder (~278 M params) already stubbed in `learn-embed/src/lib.rs` as `Reranker::score_pairs(query, docs) -> Result<Vec<f32>>`. The stub is complete: it tokenizes `"<query>[SEP]<doc>"`, runs the ONNX session, and extracts the scalar logit. No changes to `learn-embed` are needed beyond model download.

**ORT compatibility.** `learn-embed` already pins `ort = "2.0.0-rc.12"` and the reranker stub uses the identical `Session::builder()` + `session.run(inputs![...])` pattern as the embedder. Loading bge-reranker-base via the same path is confirmed safe.

**Integration shape.** After RRF returns the top-50 candidate hits, `learn-retrieve` passes control to the reranker:

1. Extract `chunk.text` from each hit as a `&str` slice.
2. Call `reranker.score_pairs(query, &docs)` — returns `Vec<f32>` of logits in input order.
3. Zip scores with hits, sort descending by logit, take top 10.
4. These 10 hits enter MMR.

The reranker lives in `learn-embed` (already stubbed). `learn-retrieve` takes it as a constructor parameter alongside `Embedder`.

---

### MMR + source diversity

**Algorithm.** After reranking, apply MMR over the top-10 candidates to produce the final context window (target 6 chunks):

```
selected = []
candidates = reranked_top_10

while len(selected) < 6 and candidates not empty:
    best = argmax over c in candidates of:
        lambda * relevance(c) - (1 - lambda) * max(sim(c, s) for s in selected)
    selected.append(best)
    candidates.remove(best)
```

- `lambda = 0.7` (relevance-weighted; user asked a specific question, not exploring).
- `relevance(c)` = the reranker logit, normalized to [0, 1] across the candidate set.
- `sim(c, s)` = cosine similarity between the 1024-dim BGE-large embeddings already on each `Embedded`/`Hit`. No extra computation required.
- When `selected` is empty, `max(sim(...))` is defined as 0.

**Source cap.** Before MMR, deduplicate by `video_id`: keep only the 3 highest-scoring hits per `video_id` from the reranker output. This happens before the MMR loop, applied to the top-10 input. With 6 target slots and a 3-per-video cap, no single video can dominate.

**Rationale.** A single verbose video would otherwise fill all 6 context slots with overlapping content from nearby timestamps. The cap forces the synthesizer to draw on at least two distinct sources, improving citation diversity without degrading relevance.

---

### Synthesis

**Model.** `claude-opus-4-7` per Stuart's global config (confirmed: `claude-sonnet-4-6` is the current session model; the CLAUDE.md references `claude-opus-4-7` as Stuart's default for synthesis tasks). The coder must inject the model name from an env var `LEARN_ANTHROPIC_MODEL` defaulting to `claude-opus-4-7` so it can be overridden without a rebuild.

**HTTP client.** `reqwest = { version = "0.13", features = ["json", "rustls-tls", "stream"] }` with streaming response enabled for long `apply` outputs.

**Abstain threshold.** Abstain when **either** condition holds:
- The top-1 hit score after reranking is below **0.3** (logit, raw — bge-reranker-base logits are typically in [-5, 5]; 0.3 sits at the low-confidence boundary based on published cross-encoder calibration data).
- Fewer than **2** hits survive the source-cap + MMR filter with score above 0.0.

Rationale: a single borderline hit risks citation hallucination. Two independent hits with non-trivial scores represent a real coverage signal.

**`ask` mode prompt (verbatim):**

```
You are a precise research assistant. Your ONLY knowledge source is the
transcript excerpts below. Do not use outside knowledge under any
circumstances.

TOPIC: {topic}
QUESTION: {question}

TRANSCRIPT EXCERPTS:
{chunks_block}

Instructions:
- Answer the question using ONLY the excerpts above.
- Cite every claim inline as [{title} @ {MM:SS}]({url}&t={start_s}s).
- If the excerpts do not contain enough information, respond with exactly:
  "KB doesn't cover this."
- Do not speculate, hedge with "likely", or fill gaps with general knowledge.
- Keep the answer under 300 words unless the question requires more.
```

**`apply` mode prompt (verbatim):**

```
You are a precise assistant that produces grounded artifacts. Your ONLY
knowledge source is the transcript excerpts below. Do not use outside
knowledge under any circumstances.

TOPIC: {topic}
TASK: {task}
OUTPUT FORMAT: {format}

TRANSCRIPT EXCERPTS:
{chunks_block}

Instructions:
- Produce the requested artifact using ONLY the excerpts above.
- Every substantive claim or step must be cited as [{title} @ {MM:SS}]({url}&t={start_s}s).
- If the excerpts do not contain enough information to complete the task,
  respond with exactly: "KB doesn't cover this."
- Do not invent steps, ingredients, parameters, or code that are not
  grounded in the excerpts.
```

`{chunks_block}` format per chunk:
```
[{title} @ {MM:SS}] {url}&t={start_s}s
{chunk.text}
```

---

### AIMDS

Per Rule 12, every user-facing LLM surface requires AIMDS. `learn-synth` calls the Anthropic API and exposes both inbound user text and outbound model output — it is in scope.

**Integration points:**

1. **Inbound scan** — before the API call, `learn-synth` shells out to the AIMDS Node sidecar: `npx @ruflo/aidefence scan --input "<question_or_task>" --threshold medium`. If the exit code is non-zero (blocked), `learn-synth` returns `Answer { abstained: true, text: "KB doesn't cover this.", citations: [] }` without calling the Anthropic API.

2. **Outbound scan** — after receiving the model's full response (streaming collected into a buffer), scan the assembled text: `npx @ruflo/aidefence scan --input "<model_response>" --threshold medium`. If blocked, replace the response with the abstain string before returning to the caller.

**Block behavior.** Both inbound and outbound blocks surface as `Answer { abstained: true }`. The CLI layer prints the abstain message; no stack trace, no partial output.

**Invocation.** `learn-synth` uses `std::process::Command` to invoke `npx @ruflo/aidefence`. This is a synchronous shell-out within an async Tokio context; wrap in `tokio::task::spawn_blocking` to avoid blocking the executor thread.

---

## Crate dependency graph

| Crate | Direct deps added in Phase 2 |
|---|---|
| `learn-retrieve` | `tantivy = "0.26"`, `learn-embed`, `learn-index`, `learn-core` |
| `learn-synth` | `reqwest = { version = "0.13", features = ["json", "rustls-tls", "stream"] }`, `tokio` (workspace), `learn-core` |
| `learn-embed` | no new deps — `Reranker::score_pairs` stub is already wired |
| `learn-cli` | no new deps — already imports `learn-retrieve` and `learn-synth` |

`learn-synth` does NOT need `serde_json` as a new dep — it is already in the workspace `[workspace.dependencies]` and can be added to `learn-synth/Cargo.toml` from the workspace alias.

---

## Implementation plan for the coder agent

1. **`crates/learn-retrieve/src/lib.rs`** — Implement `Retriever::new`, `Retriever::search`, `build_bm25` (in-memory Tantivy index from sidecar chunks), `rrf_fuse`. Add `tantivy = "0.26"` to `crates/learn-retrieve/Cargo.toml`. Unit tests: empty corpus returns empty; known chunk ranks in top-3.

2. **`crates/learn-retrieve/src/mmr.rs`** — Implement `apply_mmr(candidates: Vec<Hit>, lambda: f32, target: usize) -> Vec<Hit>` and `apply_source_cap(hits: &mut Vec<Hit>, max_per_video: usize)`. Unit tests: cap logic, diversity selection.

3. **`crates/learn-embed/src/download.rs`** (extend) — Add `ensure_reranker_model() -> Result<Utf8PathBuf>` that downloads `BAAI/bge-reranker-base` ONNX + tokenizer to `~/.cache/learn-rs/models/bge-reranker-base/`. Mirror the existing `ensure_default_model` pattern.

4. **`crates/learn-synth/src/lib.rs`** — Implement `Synthesizer::new(api_key: String, model: String)`, `Synthesizer::ask(topic, question, hits) -> Result<Answer>`, `Synthesizer::apply(topic, task, format, hits) -> Result<Answer>`. Includes AIMDS inbound/outbound scans via `spawn_blocking` shell-out, abstain threshold logic, and streaming reqwest call. Add `reqwest`, `tokio`, `serde_json` to `crates/learn-synth/Cargo.toml`.

5. **`crates/learn-synth/src/prompt.rs`** — Render the `ask` and `apply` prompt templates into `String` from a `Vec<Hit>` + metadata. Pure function, no I/O. Unit test: citation URL format is `{url}&t={start_s}s`.

6. **`crates/learn-synth/src/aimds.rs`** — `scan_text(text: &str, threshold: &str) -> Result<bool>` wrapping the `npx @ruflo/aidefence` shell-out. Returns `true` if safe, `false` if blocked. Unit test: mock exit codes.

7. **`crates/learn-cli/src/main.rs`** — Wire `Cmd::Ask` and `Cmd::Apply` to `Retriever` + `Synthesizer`. Print `Answer.text`; if `abstained`, print to stderr and exit 0. The KB root resolves `LEARN_KB_ROOT` env var or defaults to `~/Docs/KB`.

---

## Risks

1. **Tantivy in-memory rebuild latency at scale.** The in-memory BM25 design is correct for hundreds of chunks but rebuilding a Tantivy `Index` per query becomes perceptible (~500 ms) above ~20K chunks. Mitigation: cache the `tantivy::Index` inside `Retriever` and invalidate it only when `LearnIndex::ingest` is called. The sidecar's `HashMap` length can be compared across calls as a cheap staleness check.

2. **AIMDS sidecar startup cost.** `npx @ruflo/aidefence` cold-starts Node.js on every scan (~200-400 ms on M-series). Two scans per query adds up to 800 ms of latency that has nothing to do with retrieval quality. Mitigation: implement a long-lived HTTP sidecar mode — start the AIMDS process once at CLI startup and POST to a local port. Fall back to process spawn if the port is not listening.

3. **BGE-reranker-base output logit calibration.** The 0.3 abstain threshold is an estimate. bge-reranker-base logits are uncalibrated across domains; a cooking KB will have a different effective range than a finance KB. A threshold that correctly abstains on off-topic questions in one domain may over-abstain in another. Mitigation: expose the threshold as `LEARN_ABSTAIN_THRESHOLD` env var. Add a `learn eval` golden-set test that measures abstain false-positive rate against a known on-topic golden Q&A file.

4. **Streaming response + AIMDS outbound scan interaction.** Streaming the Anthropic response and then scanning the full buffer before returning breaks the streaming UX: the user sees nothing until the scan completes. Mitigation: buffer internally, scan, then flush to stdout in one write for `ask`. For `apply` (potentially long), surface a progress indicator ("synthesizing...") before the buffer flush.

5. **`embed_text` called inside `search` on a mutable `Embedder`.** `Retriever::search` takes `&mut self` because embedding the query mutates the ONNX session's internal state. This prevents concurrent queries against the same `Retriever` instance. Mitigation: document the single-threaded constraint in the type's doc comment. The CLI is inherently single-query-at-a-time; concurrency is not a Phase 2 requirement. If it becomes one, an `Arc<Mutex<Retriever>>` wrapper is the correct Phase 3 fix.
