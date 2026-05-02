# Phase 2.5 Design Memo: Autonomous Curriculum

## Capability outcome

`learn study "advanced Rust async programming"` walks the user through a complete, opinionated learning path — no URLs to hunt, no playlist curation — by searching YouTube at the requested depth, scoring each candidate on channel authority, duration, recency, and title-to-topic alignment, sending the top scorers to Claude Opus for sub-topic clustering and prerequisite ordering, presenting a ranked table for one-click approval, then ingesting all approved videos into a single `async-rust.rvf` knowledge base. The result is a topic KB populated with grounded, cited knowledge, ready for `learn ask` or `learn apply` immediately after the run.

---

## Pipeline

```
learn study "<description>" [--depth quick|medium|deep] [--auto]
           |
           v
  [1] HARVEST
  yt-dlp ytsearch{N}:<description> --dump-json --flat-playlist --skip-download
  N: quick=30, medium=60, deep=150
  Raw JSON lines → Vec<Candidate>
           |
           v
  [2] HEURISTIC SCORE (pure-Rust, no network)
  channel_authority + duration_sanity + recency_bonus + title_alignment (BGE-large)
  → sort descending → keep top 2× surface count (to give Claude room)
           |
           v
  [3] CAPTION PRE-FILTER  ← full info fetch, batched, concurrency=4
  yt-dlp --dump-json --skip-download per candidate
  drop: automatic_captions + subtitles both empty for any "en*" key
  survivors ≥ surface_count required; if not, relax to top-2× anyway
           |
           v
  [4] CURATION (Claude Opus API call)
  POST /v1/messages with candidate JSON + topic description
  Response: [{video_id, sub_topic, rationale, rank, prerequisite_ranks}]
           |
           v
  [5] CONFIRM (unless --auto)
  Render tabular summary; prompt via dialoguer::Confirm
           |
           v
  [6] INGEST (learn-acquire per video, tokio bounded channel)
  acquire_url → chunk → embed → index → manifest entry
           |
           v
  <slug>.rvf  (topic KB ready for ask/apply)
```

---

## Candidate harvest

**Command (Step 1):**

```
yt-dlp "ytsearch{N}:<topic_description>" \
  --dump-json \
  --flat-playlist \
  --skip-download \
  --no-warnings
```

This emits one JSON line per candidate. Confirmed fields from live probe (yt-dlp 2026.03.17):

| Field | Type | Notes |
|---|---|---|
| `id` | String | YouTube video ID |
| `title` | String | Video title |
| `channel` | String | Uploader display name |
| `channel_id` | String | Stable channel identifier |
| `view_count` | u64 | Total views at search time |
| `duration` | f64 | Seconds (float) |
| `upload_date` | Option\<String\> | YYYYMMDD or null |
| `subtitles` | null | Always null in flat mode |
| `automatic_captions` | null | Always null in flat mode |

Caption fields are always null in `--flat-playlist` mode (confirmed above). A separate full info fetch is required to gate on caption availability.

**Search pool by depth:**

| Depth | Search pool (N) | Surface count | Claude input cap |
|---|---|---|---|
| quick | 30 | 5 | top 10 by score |
| medium | 60 | 10 | top 20 by score |
| deep | 150 | 25 | top 50 by score |

**Caption gating strategy:** After the heuristic score pass, take the top `2 × surface_count` candidates and run `yt-dlp --dump-json --skip-download` on each at concurrency 4 (see Step 3). Check: `subtitles` or `automatic_captions` has any key starting with `en`. Discard candidates that pass neither. If fewer than `surface_count` survive, fall back to the scored list without the caption gate rather than aborting — flag each un-gated video with `has_captions: false` in the manifest so downstream indexing can route them to ASR instead.

---

## Scoring rubric

Each factor produces a value in [0.0, 1.0]. Final score is a weighted sum; weights shown are defaults and exposed as config fields on `ScoringConfig`.

| Factor | Weight | Formula |
|---|---|---|
| `title_alignment` | 0.35 | cosine\_sim(embed(title), embed(topic\_description)) using BGE-large via `Embedder::embed_text` |
| `channel_authority` | 0.25 | min(log10(view\_count + 1) / 8.0, 1.0) — log-capped; raw view count is the only signal available from flat search |
| `recency_bonus` | 0.20 | applied with `recency_bias` from `StudyDepth`; age\_days = today - upload\_date; score = exp(−age\_days × recency\_bias / 365.0); missing date ⇒ 0.5 |
| `duration_sanity` | 0.15 | triangle: 0.0 at 0 s, 1.0 at 600 s–3600 s (ideal band), 0.0 at 7200 s. When `allow_long_form = true`, extend upper limit to 7200 s before decay |
| `has_captions` | 0.05 | 1.0 if en captions confirmed, 0.5 if unknown (pre-filter stage), 0.0 if confirmed absent |

**Formula:** `score = 0.35*title_alignment + 0.25*channel_authority + 0.20*recency_bonus + 0.15*duration_sanity + 0.05*has_captions`

Title alignment requires spinning up `Embedder` once per `discover` call; the topic embedding is computed once and reused across all candidate titles.

---

## Curation prompt

**Anthropic API call:** POST to `https://api.anthropic.com/v1/messages`, model `claude-opus-4-7`, `max_tokens: 2048`, no streaming.

**Decision: call Anthropic directly, not via Ruflo's `goal-plan` MCP tool.** Reasoning: `goal-plan` is a GOAP planner optimised for agent action sequences over a state space. Curriculum selection is a one-shot classification and ranking problem over a fixed JSON list — no state transitions, no re-planning loop. The overhead of spawning an MCP call, serialising through the Ruflo layer, and waiting for a round-trip adds latency with no benefit. Call the Anthropic SDK directly with `reqwest`. Reserve `goal-plan` for a later phase if the user wants multi-session learning progression tracking.

**Verbatim system prompt (stored as a `const &str` in `learn-discover/src/curate.rs`):**

```
You are a curriculum designer. The user wants to learn about a specific topic.
You are given a list of YouTube video candidates with metadata and heuristic scores.
Your job: select the best videos that together form a coherent, progressive curriculum.

Rules:
1. Prioritise breadth: cover distinct sub-topics, not the same ground twice.
2. Order from prerequisites and fundamentals to advanced material.
3. Deduplicate: if two videos cover identical ground, keep the higher-scored one.
4. Prefer videos with confirmed captions (has_captions: true).
5. Output ONLY valid JSON — no prose before or after.
```

**Verbatim user message template:**

```
Topic: {{topic_description}}

Select exactly {{surface_count}} videos from the candidates below.
Return a JSON array with this schema per element:
[
  {
    "video_id": "<string>",
    "sub_topic": "<one short phrase describing what this video covers>",
    "rationale": "<one sentence: why this video belongs in the curriculum>",
    "rank": <1-based integer, 1 = watch first>,
    "prerequisite_ranks": [<list of rank integers that should be watched before this one>]
  }
]

Candidates (JSON array):
{{candidates_json}}
```

`candidates_json` is the top-N scored candidates serialised as a compact JSON array with fields: `video_id`, `title`, `channel`, `duration_seconds`, `upload_date`, `score`, `has_captions`.

---

## Confirmation UX

**Interaction shape:**

After curation returns, render to stdout:

```
Curriculum for: "advanced Rust async programming"  [deep — 25 videos]

Rank  Title                                    Channel         Duration  Sub-topic            Rationale
────  ───────────────────────────────────────  ──────────────  ────────  ───────────────────  ──────────────────────────────
   1  Async Rust from scratch                  Jon Gjengset    1h 12m    Foundations          Best intro to the async model…
   2  Tokio internals explained                ...
```

Rendered using Rust's standard `format!` / `println!` — not a TUI library. The table is plain markdown-compatible text so it copies cleanly.

Then: `Proceed with ingestion? [y/N]: ` via `dialoguer::Confirm::new()`.

`--auto` skips the table print and the confirm prompt entirely and logs `auto=true` via `tracing::info!`.

**Flag-only flow over a full TUI:** dialoguer is one crate, statically linked, zero runtime cost. A flag-only flow (require `--confirm` to proceed) is worse UX because the user cannot review the curriculum before committing. `dialoguer::Confirm` is the minimal-surface choice.

---

## Ingestion handoff

After confirmation, iterate `curriculum.picks` in rank order:

- **Parallelism:** `tokio::sync::Semaphore` with 3 permits (not a channel, not a thread pool). Three concurrent `acquire_url` calls is enough to saturate a typical home connection without hammering yt-dlp rate limits. The semaphore is held for the full acquire → chunk → embed → index sequence per video.
- **Per-video error handling:** each video runs in a `tokio::task::spawn` behind the semaphore. On error, log `tracing::warn!` with the video ID and error, set `VideoState.status = IngestStatus::Failed` with the error string, and continue. Do not abort the whole curriculum on a single failure.
- **Manifest semantics:** `Topic::manifest_path` already resolves to `<kb_root>/_meta/<slug>.json`. Before ingestion begins, load or create the manifest. After each video completes (success or failure), write a `VideoState` entry: `video_id`, `status`, `fetched_at` (RFC3339), `indexed_at` (RFC3339 on success), `chunk_count`, and `error`. Write the manifest after every video, not only at the end, so partial progress survives a crash.
- **Idempotency:** check the manifest before acquiring; if `status == IngestStatus::Indexed` for a given `video_id`, skip acquisition (respect existing `--force` pattern from `Cmd::Ingest`).

---

## Crate dependencies

New additions to `learn-discover/Cargo.toml`:

| Crate | Rationale |
|---|---|
| `reqwest` (features: `json`, `rustls-tls`) | Anthropic API call (POST /v1/messages); already a transitive dep in the ecosystem, rustls avoids OpenSSL |
| `learn-embed` (workspace path) | `Embedder::embed_text` for title-alignment scoring; model already downloaded by the time `study` runs if `ingest` has been used |
| `dialoguer` | `Confirm::new()` for the y/N prompt; statically compiled, zero system deps |
| `tokio` (already workspace) | `Semaphore`, `spawn`, `process::Command` |

`learn-acquire` is already a dependency of `learn-cli`; `learn-discover` calls `acquire_url` via the CLI orchestration layer (not a direct crate dep) to keep the dependency graph acyclic. The CLI's `Cmd::Study` arm drives the ingest loop.

---

## Implementation plan for the coder agent

1. **`crates/learn-discover/Cargo.toml`** — add `reqwest`, `learn-embed`, `dialoguer` to `[dependencies]`.
2. **`crates/learn-discover/src/harvest.rs`** (new) — `async fn harvest(topic: &str, pool_n: usize) -> Result<Vec<Candidate>>` — shells out to yt-dlp flat-playlist, parses NDJSON lines into `Candidate` structs.
3. **`crates/learn-discover/src/score.rs`** (new) — `fn score(candidates: &mut [Candidate], topic_embed: &[f32], depth: &StudyDepth)` — applies the five-factor formula, sorts in place.
4. **`crates/learn-discover/src/caption_gate.rs`** (new) — `async fn gate_captions(candidates: &mut [Candidate], concurrency: usize)` — fetches full info JSON per candidate behind a semaphore, writes `has_captions` field.
5. **`crates/learn-discover/src/curate.rs`** (new) — `async fn curate(candidates: &[Candidate], topic: &str, surface_count: usize) -> Result<Vec<CurriculumPick>>` — builds and POSTs the Anthropic prompt, parses response JSON.
6. **`crates/learn-discover/src/lib.rs`** — replace the stub `discover` fn: wire harvest → score → caption_gate → curate → return `Curriculum`.
7. **`crates/learn-cli/src/main.rs`** — expand `Cmd::Study` arm: call `discover`, render table, `dialoguer::Confirm`, then iterate picks with semaphore, calling `acquire_url` + downstream pipeline, writing manifest after each.
8. **`crates/learn-core/src/lib.rs`** — add `LearnError::Discover(String)` variant and a `ScoringConfig` struct (weights + pool sizes).
9. **Integration test (`crates/learn-discover/tests/harvest_smoke.rs`)** — `#[ignore]` test that runs harvest for a single search result and checks the `Candidate` fields are populated.

---

## Risks

1. **yt-dlp rate limiting at deep depth.** Fetching caption metadata for up to 100 individual videos (2 × 50 for deep) fires 100 sequential or concurrent yt-dlp processes against YouTube. YouTube 429s are silent — yt-dlp reports non-zero exit but often no HTTP status code. Mitigation: cap caption-gate concurrency at 4, add a 500ms back-off on consecutive non-zero exits, and fall back to un-gated scoring if >30% of candidates fail the fetch.

2. **Anthropic cost at deep depth.** The curation prompt at deep sends 50 candidates as JSON (~3–5 KB) plus system prompt (~400 tokens). Each deep run costs roughly 5K input tokens + 2K output tokens on Opus ≈ $0.15–0.20 per call. Not alarming alone, but if `--auto` is scripted in a loop this compounds. Mitigation: log the estimated token count before the API call and surface it in the confirmation table.

3. **BGE-large cold start on first `study` call.** `Embedder::load` triggers an ONNX Runtime model download (~1.3 GB) on first use if the model cache is empty. This blocks the harvest pipeline for several minutes with no progress indicator. Mitigation: run `ensure_default_model()` in a `tokio::task::spawn_blocking` before the harvest starts, print a one-line "downloading embedding model..." status to stderr.

4. **No subscriber count in flat-playlist output.** The `channel_authority` factor falls back to `view_count` only because `subscriber_count` is not included in `ytsearch` flat-playlist results. This means a viral 10-minute video can outscore a well-established technical channel's methodical series. Mitigation: document the limitation in `ScoringConfig`; a future pass can do a channel-level info fetch for the top 10 distinct `channel_id`s to enrich the authority score.

5. **Upload date null rate.** Live probe shows `upload_date` is null in flat-playlist results for at least some entries. With `recency_bonus` weighted at 0.20, a null date collapses to 0.5 (neutral), which means newer channels posting without date metadata are neither penalised nor rewarded. This is acceptable but should be noted in `ScoringConfig` docs so the user understands why recency feels weak on some queries.
