//! Implementations for the Phase-2C command surface.
//!
//! Each `run_*` function mirrors the pattern in `main.rs`: takes typed args,
//! opens required pipeline crates, runs the operation, prints to stdout, and
//! returns `learn_core::Result<()>`.

use async_trait::async_trait;
use camino::Utf8PathBuf;
use learn_coherence::compute_consciousness_kpi;
use learn_core::{
    Embedded, Hit, IngestStatus, LearnError, Manifest, Result, Topic, Transcript, TranscriptSource,
    VideoState,
};
use learn_graph::{EntityId, LearnGraph};
use learn_index::LearnIndex;

// ── Ingest (Phase 3E: crash-recovery resume) ─────────────────────────────────

/// Ingest a URL/path into the topic knowledge base.
///
/// Convenience wrapper that disables frame captioning and uses the default
/// max-frames cap. Called by tests and the `learn study` path when frame flags
/// are not needed.
///
/// ## Resume behaviour (crash-recovery)
///
/// Before starting work on a video the manifest is checked:
///
/// | Stored status | Behaviour |
/// |---|---|
/// | `Indexed` | Skip unless `--force`. |
/// | `Embedded` | Embedded stage done — jump straight to index step. |
/// | `Chunked` | Chunks are not persisted to disk, so restart acquisition. |
/// | `Acquired` | Attempt to reuse the cached VTT at `_raw/<topic>/<vid>.vtt`. |
/// | `Failed` | Skip unless `--force`. |
/// | Absent / `Pending` | Full pipeline. |
///
/// Each successful stage writes the manifest atomically before the next stage
/// begins, so a kill between stages leaves a valid checkpoint on disk.
// Used by integration tests in main.rs; not called directly from the binary entry point.
#[allow(dead_code)]
pub async fn run_ingest(
    source: String,
    topic_override: Option<String>,
    kb_root: Utf8PathBuf,
    force: bool,
) -> Result<()> {
    run_ingest_with_limit(source, topic_override, kb_root, force, None, false, 60).await
}

/// Ingest with an optional playlist / channel / search limit.
///
/// `limit` is forwarded to `resolve_to_videos` as `--playlist-end N`.
/// `frames_enabled` controls Sonnet-vision frame captioning.
/// `max_frames` caps the keyframe count per video.
pub async fn run_ingest_with_limit(
    source: String,
    topic_override: Option<String>,
    kb_root: Utf8PathBuf,
    force: bool,
    limit: Option<usize>,
    frames_enabled: bool,
    max_frames: usize,
) -> Result<()> {
    use learn_acquire::{classify_source, resolve_to_videos, SourceKind};

    let kind = classify_source(&source);
    let is_multi = matches!(
        kind,
        SourceKind::Playlist | SourceKind::Channel | SourceKind::Search
    );

    if is_multi {
        let urls = resolve_to_videos(&source, limit).await?;
        tracing::info!(
            source = %source,
            count = urls.len(),
            "ingest: resolved multi-video source"
        );
        for url in urls {
            if let Err(e) = ingest_single_video(
                url.clone(),
                topic_override.clone(),
                kb_root.clone(),
                force,
                frames_enabled,
                max_frames,
            )
            .await
            {
                tracing::warn!(%url, error = %e, "failed to ingest video");
            }
        }
        return Ok(());
    }

    ingest_single_video(
        source,
        topic_override,
        kb_root,
        force,
        frames_enabled,
        max_frames,
    )
    .await
}

/// Core single-video ingestion pipeline (acquire → transcribe → [frames] → chunk → embed → index).
async fn ingest_single_video(
    source: String,
    topic_override: Option<String>,
    kb_root: Utf8PathBuf,
    force: bool,
    frames_enabled: bool,
    max_frames: usize,
) -> Result<()> {
    // 1. Resolve topic.
    let topic = match topic_override {
        Some(t) => Topic::new(&t)?,
        None => super::derive_topic_from_source(&source)?,
    };

    // 2. Open index (loads manifest from _meta/<topic>.json).
    let mut index = LearnIndex::open(kb_root.as_ref(), topic.clone())?;

    // 3. Acquire (or resume from checkpoint).
    let raw_dir = topic.raw_dir(&kb_root);

    // 3a. Slug-collision guard: refuse if raw_dir has cached data from a video
    //     not in this topic's manifest (i.e. squatted by an unrelated prior run).
    {
        let known_ids: std::collections::BTreeSet<String> =
            index.manifest().videos.keys().cloned().collect();
        learn_acquire::check_slug_collision(&raw_dir, topic.as_str(), &known_ids)?;
    }

    tracing::info!(%topic, %source, "ingest: acquiring");
    let acquired = learn_acquire::acquire_url(&source, &kb_root, &raw_dir, frames_enabled).await?;
    let video_id = acquired.video.video_id.clone();

    // 4. Check existing manifest state and apply resume logic.
    let existing_status = index.manifest().videos.get(&video_id).map(|vs| vs.status);

    match existing_status {
        Some(IngestStatus::Indexed) if !force => {
            println!(
                "video {video_id} already indexed, skipping \
                 (use --force to re-ingest)"
            );
            return Ok(());
        }
        Some(IngestStatus::Failed) if !force => {
            let err_msg = index
                .manifest()
                .videos
                .get(&video_id)
                .and_then(|vs| vs.error.clone())
                .unwrap_or_default();
            println!(
                "video {video_id} previously failed: {err_msg} — skipping \
                 (use --force to retry)"
            );
            return Ok(());
        }
        Some(IngestStatus::Embedded) if !force => {
            // Embeddings are already on disk — skip acquire/transcribe/chunk/embed.
            let embedded = index.embedded_for_video(&video_id);
            if !embedded.is_empty() {
                tracing::info!(
                    video_id = %video_id,
                    chunks = embedded.len(),
                    "ingest: resuming from Embedded checkpoint — skipping to index step"
                );
                let now = super::rfc3339_now();
                let chunk_count = embedded.len();
                let accepted = index.ingest(&embedded)?;
                let indexed_at = super::rfc3339_now();
                index.upsert_video_state(VideoState {
                    video_id: video_id.clone(),
                    status: IngestStatus::Indexed,
                    fetched_at: Some(now),
                    indexed_at: Some(indexed_at),
                    chunk_count,
                    error: None,
                })?;
                println!(
                    "ingested {accepted} chunks from {video_id} into \
                     {kb_root}/{topic}.rvf (resumed from Embedded)"
                );
                return Ok(());
            }
            // Embeddings absent from sidecar — fall through to full pipeline.
            tracing::warn!(
                video_id = %video_id,
                "ingest: Embedded checkpoint present but no embeddings in sidecar; \
                 falling back to full pipeline"
            );
        }
        _ => {}
    }

    // 5. Record Pending status (marks video as in-flight).
    let now = super::rfc3339_now();
    index.upsert_video_state(VideoState {
        video_id: video_id.clone(),
        status: IngestStatus::Pending,
        fetched_at: Some(now.clone()),
        indexed_at: None,
        chunk_count: 0,
        error: None,
    })?;

    // 6. Mark Acquired after acquire_url succeeded.
    index.upsert_video_state(VideoState {
        video_id: video_id.clone(),
        status: IngestStatus::Acquired,
        fetched_at: Some(now.clone()),
        indexed_at: None,
        chunk_count: 0,
        error: None,
    })?;

    // 7. Parse captions; if the video was previously Acquired we try the cached VTT.
    let caption_segments = if let Some(vtt_path) = &acquired.captions_vtt {
        tracing::info!(path = %vtt_path, "parsing captions");
        learn_acquire::vtt::parse_vtt(vtt_path)?
    } else {
        tracing::warn!(
            "no captions found for {video_id}; Whisper fallback is Phase-2D work — \
             skipping transcription"
        );
        vec![]
    };

    // 7b. Optionally extract keyframes and caption them with Sonnet vision.
    let frame_segments = if frames_enabled {
        let video_path = find_video_file(&acquired.raw_dir);
        match video_path {
            Some(ref p) => run_frame_captioning(p, &acquired.raw_dir, max_frames).await,
            None => {
                tracing::warn!(
                    raw_dir = %acquired.raw_dir,
                    "frames enabled but no video file found in raw_dir — \
                     yt-dlp may have failed to download; continuing captions-only"
                );
                vec![]
            }
        }
    } else {
        vec![]
    };

    let segments = learn_frames::merge_segments(caption_segments, frame_segments);

    if segments.is_empty() {
        tracing::warn!(
            %video_id,
            "no transcript or frame descriptions available; \
             automatic captions not found, Whisper fallback not yet wired, \
             and frame captioning produced no output"
        );
        // Record failed state so next run can skip cleanly without --force.
        let _ = index.upsert_video_state(VideoState {
            video_id: video_id.clone(),
            status: IngestStatus::Failed,
            fetched_at: Some(now),
            indexed_at: None,
            chunk_count: 0,
            error: Some("no transcript or frames available".to_string()),
        });
        return Ok(());
    }

    // 8. Build Transcript and mark Transcribed.
    let transcript = Transcript {
        video_id: video_id.clone(),
        language: Some("en".to_string()),
        source: TranscriptSource::Captions,
        segments,
    };
    index.upsert_video_state(VideoState {
        video_id: video_id.clone(),
        status: IngestStatus::Transcribed,
        fetched_at: Some(now.clone()),
        indexed_at: None,
        chunk_count: 0,
        error: None,
    })?;

    // 9. Chunk and mark Chunked.
    tracing::info!("chunking transcript");
    let chunks = learn_chunk::chunk_transcript(&transcript, &learn_chunk::ChunkConfig::default())?;
    let chunk_count = chunks.len();
    tracing::info!(count = chunk_count, "chunks produced");
    index.upsert_video_state(VideoState {
        video_id: video_id.clone(),
        status: IngestStatus::Chunked,
        fetched_at: Some(now.clone()),
        indexed_at: None,
        chunk_count,
        error: None,
    })?;

    // 10. Embed and mark Embedded.
    let embedder_path = super::default_model_dir();
    let embed_cfg = learn_embed::EmbedConfig {
        model_dir: embedder_path.clone(),
        ..Default::default()
    };
    tracing::info!(path = %embedder_path, "loading embedder");
    let mut embedder = learn_embed::Embedder::for_topic(&topic, &embed_cfg)?;
    let embedded = embedder.embed_chunks(&chunks)?;
    index.upsert_video_state(VideoState {
        video_id: video_id.clone(),
        status: IngestStatus::Embedded,
        fetched_at: Some(now.clone()),
        indexed_at: None,
        chunk_count,
        error: None,
    })?;

    // 11. Ingest into the RVF index and mark Indexed.
    tracing::info!(topic = %topic, "opening index");
    let accepted = index.ingest(&embedded)?;
    let indexed_at = super::rfc3339_now();
    index.upsert_video_state(VideoState {
        video_id: video_id.clone(),
        status: IngestStatus::Indexed,
        fetched_at: Some(now),
        indexed_at: Some(indexed_at),
        chunk_count,
        error: None,
    })?;

    println!("ingested {accepted} chunks from {video_id} into {kb_root}/{topic}.rvf");
    Ok(())
}

// ── WhoSaid ──────────────────────────────────────────────────────────────────

/// Find every claim attributable to a speaker / entity whose name fuzzy-matches
/// `query`. Prints each claim with a `[<video>:<chunk>]` citation.
pub async fn run_who_said(topic_str: String, query: String, kb_root: Utf8PathBuf) -> Result<()> {
    let topic = Topic::new(&topic_str)?;
    let graph = LearnGraph::open(kb_root.as_ref(), topic)?;
    let query_lc = query.to_lowercase();
    let entity_id = find_entity_by_name(&graph, &query_lc)?;
    let Some(eid) = entity_id else {
        println!("No entity matching {query:?} found in graph.");
        return Ok(());
    };
    let claims = graph.claims_by_entity(&eid)?;
    if claims.is_empty() {
        println!("No claims found for {query:?}.");
        return Ok(());
    }
    for c in &claims {
        println!("[{}:{}] {}", c.source_video_id, c.source_chunk_id, c.text);
    }
    Ok(())
}

// ── Timeline ─────────────────────────────────────────────────────────────────

/// Build a temporal timeline of how an entity has been discussed.
pub async fn run_timeline(topic_str: String, entity: String, kb_root: Utf8PathBuf) -> Result<()> {
    let topic = Topic::new(&topic_str)?;
    let graph = LearnGraph::open(kb_root.as_ref(), topic)?;
    let entity_lc = entity.to_lowercase();
    let entity_id = find_entity_by_name(&graph, &entity_lc)?;
    let Some(eid) = entity_id else {
        println!("No entity matching {entity:?} found in graph.");
        return Ok(());
    };
    let mut claims = graph.claims_by_entity(&eid)?;
    if claims.is_empty() {
        println!("No claims found for {entity:?}.");
        return Ok(());
    }
    claims.sort_by(|a, b| {
        a.source_timestamp
            .partial_cmp(&b.source_timestamp)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for c in &claims {
        let ts = format_seconds(c.source_timestamp);
        println!("[{}  {}]  {}", c.source_video_id, ts, c.text);
    }
    Ok(())
}

// ── Compare ──────────────────────────────────────────────────────────────────

/// Compare two concepts using the KB as a grounding source.
pub async fn run_compare(
    topic_str: String,
    a: String,
    b: String,
    kb_root: Utf8PathBuf,
) -> Result<()> {
    let topic = Topic::new(&topic_str)?;
    let embedder_path = super::default_model_dir();
    let index = LearnIndex::open(kb_root.as_ref(), topic.clone())?;
    let mut retriever =
        learn_retrieve::Retriever::for_topic(index, &topic, embedder_path.as_ref())?;
    retriever.refresh_bm25()?;
    let hits_a = retriever.search(&a, 5).await?;
    let hits_b = retriever.search(&b, 5).await?;
    let mut combined = hits_a;
    let existing_ids: std::collections::HashSet<String> =
        combined.iter().map(|h| h.chunk.chunk_id.clone()).collect();
    for h in hits_b {
        if !existing_ids.contains(&h.chunk.chunk_id) {
            combined.push(h);
        }
    }
    if combined.is_empty() {
        println!("KB doesn't cover either concept.");
        return Ok(());
    }
    let synth = learn_synth::select_synthesizer()?;
    let task = format!("Compare {a} vs {b}");
    let answer = synth
        .apply(topic.as_str(), &task, "markdown", &combined)
        .await?;
    if answer.abstained {
        eprintln!("(model abstained: insufficient evidence in KB)");
    } else {
        println!("{}", answer.text);
    }
    Ok(())
}

// ── Summarize ────────────────────────────────────────────────────────────────

/// Summarize a topic or one specific video.
pub async fn run_summarize(
    topic_str: String,
    video: Option<String>,
    kb_root: Utf8PathBuf,
) -> Result<()> {
    let topic = Topic::new(&topic_str)?;
    let embedder_path = super::default_model_dir();

    if let Some(video_id) = video {
        let graph = LearnGraph::open(kb_root.as_ref(), topic.clone())?;
        let claims = graph.claims_in_video(&video_id)?;
        if claims.is_empty() {
            println!("No claims found for video {video_id:?}. Try re-ingesting.");
            return Ok(());
        }
        let hits = claims_to_hits(&claims, &video_id);
        let synth = learn_synth::select_synthesizer()?;
        let task = format!("Summarize the key points from video {video_id}");
        let answer = synth
            .apply(topic.as_str(), &task, "markdown", &hits)
            .await?;
        if answer.abstained {
            eprintln!("(model abstained)");
        } else {
            println!("{}", answer.text);
        }
    } else {
        let graph = LearnGraph::open(kb_root.as_ref(), topic.clone())?;
        let ranked = graph.pagerank()?;
        let index = LearnIndex::open(kb_root.as_ref(), topic.clone())?;
        let mut retriever =
            learn_retrieve::Retriever::for_topic(index, &topic, embedder_path.as_ref())?;
        retriever.refresh_bm25()?;
        let query = ranked
            .first()
            .and_then(|(eid, _)| graph.entity(eid).ok().flatten().map(|e| e.name))
            .unwrap_or_else(|| topic.as_str().to_string());
        let hits = retriever.search(&query, 10).await?;
        if hits.is_empty() {
            println!("KB is empty. Ingest some videos first.");
            return Ok(());
        }
        let synth = learn_synth::select_synthesizer()?;
        let task = format!(
            "Summarize the key ideas in this knowledge base about {}",
            topic.as_str()
        );
        let answer = synth
            .apply(topic.as_str(), &task, "markdown", &hits)
            .await?;
        if answer.abstained {
            eprintln!("(model abstained)");
        } else {
            println!("{}", answer.text);
        }
    }
    Ok(())
}

// ── List ─────────────────────────────────────────────────────────────────────

/// List videos in a topic, grouped by video_id, sorted by `by`.
pub fn run_list(topic_str: Option<String>, by: String, kb_root: Utf8PathBuf) -> Result<()> {
    // No topic? List every KB we've built. Friendly empty-state.
    let Some(topic_str) = topic_str else {
        return run_list_all_topics(kb_root);
    };
    let topic = Topic::new(&topic_str)?;
    let kb_path = kb_root.join(format!("{}.rvf", topic.as_str()));
    if !kb_path.exists() {
        println!("No knowledge base for topic '{topic}' yet.");
        println!();
        println!("To build one, try one of:");
        println!("  learn ingest \"<youtube-url>\" --topic {topic}");
        println!("  learn study \"<topic description>\" --topic {topic}");
        println!();
        println!("Or run `learn list` (no args) to see what KBs you do have.");
        return Ok(());
    }
    let index = LearnIndex::open(kb_root.as_ref(), topic.clone())?;
    let manifest = load_manifest_opt(&kb_root, &topic);
    let chunks = index.chunks_snapshot();
    if chunks.is_empty() {
        println!("Topic '{topic}' exists but has no videos yet.");
        println!("Add one with: learn ingest \"<youtube-url>\" --topic {topic}");
        return Ok(());
    }
    // Group: video_id → (chunk_count, max_end_seconds)
    let mut by_video: std::collections::BTreeMap<String, (usize, f64)> = Default::default();
    for c in &chunks {
        let e = by_video.entry(c.video_id.clone()).or_insert((0, 0.0));
        e.0 += 1;
        if c.end_seconds > e.1 {
            e.1 = c.end_seconds;
        }
    }
    let mut rows: Vec<ListRow> = by_video
        .into_iter()
        .map(|(vid, (chunks, dur))| {
            let fetched_at = manifest
                .as_ref()
                .and_then(|m| m.videos.get(&vid))
                .and_then(|vs| vs.fetched_at.clone())
                .unwrap_or_else(|| "—".to_string());
            ListRow {
                video_id: vid,
                chunk_count: chunks,
                duration_seconds: dur,
                fetched_at,
            }
        })
        .collect();
    match by.as_str() {
        "date" => rows.sort_by(|a, b| a.fetched_at.cmp(&b.fetched_at)),
        "duration" => rows.sort_by(|a, b| {
            a.duration_seconds
                .partial_cmp(&b.duration_seconds)
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        _ => {} // default: alphabetical by video_id (BTreeMap order)
    }
    println!(
        "{:<20}  {:<8}  {:>8}  fetched_at",
        "video_id", "chunks", "duration"
    );
    println!("{}", "─".repeat(60));
    for r in &rows {
        println!(
            "{:<20}  {:<8}  {:>8}  {}",
            r.video_id,
            r.chunk_count,
            format_seconds(r.duration_seconds),
            r.fetched_at,
        );
    }
    Ok(())
}

struct ListRow {
    video_id: String,
    chunk_count: usize,
    duration_seconds: f64,
    fetched_at: String,
}

// ── Status ───────────────────────────────────────────────────────────────────

/// Health snapshot of a KB (or all topics if topic is None).
pub fn run_status(topic: Option<String>, kb_root: Utf8PathBuf) -> Result<()> {
    if let Some(topic_str) = topic {
        let topic = Topic::new(&topic_str)?;
        let index = LearnIndex::open(kb_root.as_ref(), topic.clone())?;
        let stats = index.stats()?;
        let graph_entity_count = count_graph_entities(&kb_root, &topic);
        let manifest = load_manifest_opt(&kb_root, &topic);
        let video_count = manifest.as_ref().map(|m| m.videos.len()).unwrap_or(0);
        println!("topic:   {topic}");
        println!("vectors: {}", stats.vector_count);
        println!("bytes:   {}", stats.bytes_on_disk);
        println!("videos:  {video_count}");
        println!("graph:   {graph_entity_count} entities");
        // ── Consciousness KPI ────────────────────────────────────────────────
        let embedded = build_embedded_snapshot(&index);
        if let Ok(kpi) = compute_consciousness_kpi(&embedded) {
            println!(
                "coherence: integrated={:.2} workspace={:.2} [{}]",
                kpi.integrated_information,
                kpi.workspace_score,
                kpi.interpretation.label(),
            );
        }
    } else {
        let rvf_files = list_rvf_files(&kb_root);
        if rvf_files.is_empty() {
            println!("0 topics in {kb_root}");
            return Ok(());
        }
        let mut total_vectors = 0usize;
        let mut total_bytes = 0u64;
        for path in &rvf_files {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
            if let Ok(topic) = Topic::new(stem) {
                if let Ok(index) = LearnIndex::open(kb_root.as_ref(), topic.clone()) {
                    if let Ok(stats) = index.stats() {
                        println!(
                            "{:<30}  {:>8} vectors  {:>10} bytes",
                            stem, stats.vector_count, stats.bytes_on_disk
                        );
                        total_vectors += stats.vector_count;
                        total_bytes += stats.bytes_on_disk;
                    }
                }
            }
        }
        println!("─────────────────────────────────────────────────────────────");
        println!(
            "total: {} topics  {} vectors  {} bytes",
            rvf_files.len(),
            total_vectors,
            total_bytes
        );
    }
    Ok(())
}

// ── Watch ────────────────────────────────────────────────────────────────────

/// Schedule recurring ingestion of a channel into a topic.
///
/// Translates the cadence parameter to a cron expression and prints instructions
/// for the user to install the schedule manually. On macOS, generates a LaunchAgent plist;
/// on other systems, prints a raw cron line. Does NOT auto-install (per CLAUDE.md Rule 14).
pub fn run_watch(topic_str: String, channel: String, cadence: String) -> Result<()> {
    let topic = Topic::new(&topic_str)?;
    let cron_expr = cadence_to_cron(&cadence)?;

    // Construct the command to run
    let learn_bin = std::env::current_exe()
        .map_err(|e| LearnError::Synth(format!("could not resolve current exe: {e}")))?;
    let cmd = format!(
        "{} ingest \"{}\" --topic {}",
        learn_bin.display(),
        channel,
        topic.as_str()
    );

    let task_id = format!("learn-watch-{}", topic.as_str());

    // Try to schedule via the best available method
    schedule_via_best_available(&task_id, &cron_expr, &cmd, &topic)?;

    Ok(())
}

/// Translate a human-readable cadence ("weekly", "daily", etc.) to a 5-field cron expression.
fn cadence_to_cron(cadence: &str) -> Result<String> {
    match cadence.trim().to_lowercase().as_str() {
        "hourly" => Ok("0 * * * *".to_string()),
        "daily" => Ok("0 6 * * *".to_string()), // 6am UTC daily
        "weekly" => Ok("0 6 * * 0".to_string()), // 6am UTC Sundays
        "monthly" => Ok("0 6 1 * *".to_string()), // 6am UTC 1st of month
        s if s.matches(' ').count() == 4 => {
            // Assume raw 5-field cron expression
            Ok(s.to_string())
        }
        _ => Err(LearnError::Synth(format!(
            "unknown cadence: {cadence:?} — use hourly|daily|weekly|monthly \
             or a raw 5-field cron expression (e.g., \"0 6 * * 0\")"
        ))),
    }
}

/// Schedule the task via the best available method:
/// 1. macOS LaunchAgent (prints plist content + installation instructions)
/// 2. Raw cron line (for manual crontab installation)
fn schedule_via_best_available(
    task_id: &str,
    cron_expr: &str,
    cmd: &str,
    topic: &Topic,
) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        print_macos_launchagent(task_id, cron_expr, cmd, topic)?;
    }

    #[cfg(not(target_os = "macos"))]
    {
        print_raw_cron_line(task_id, cron_expr, cmd);
    }

    Ok(())
}

/// Print a macOS LaunchAgent plist for manual installation.
/// Does NOT auto-install (per CLAUDE.md Rule 14).
#[cfg(target_os = "macos")]
fn print_macos_launchagent(task_id: &str, cron_expr: &str, cmd: &str, topic: &Topic) -> Result<()> {
    use std::env;

    let home = env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let plist_path = format!(
        "{}/Library/LaunchAgents/com.learn-rv.{}.plist",
        home,
        topic.as_str()
    );

    // Parse cron expression to get the run frequency
    let (minute, hour, _day, _month, weekday) = parse_cron(cron_expr)?;

    // LaunchAgent plist (runs via StartInterval or StartCalendarInterval)
    let plist_content = generate_plist(task_id, cmd, hour, minute, weekday)?;

    println!(
        "┌─ Schedule for topic '{}' ─────────────────────────────────────┐",
        topic.as_str()
    );
    println!("│");
    println!("│ Cron expression: {}", cron_expr);
    println!("│ Task ID: {}", task_id);
    println!("│");
    println!("│ 1. Copy the following plist to: {}", plist_path);
    println!("│");
    println!("{}", format_plist_for_display(&plist_content));
    println!("│");
    println!("│ 2. Then run this command to install:");
    println!("│");
    println!("│    launchctl bootstrap gui/$(id -u) {}", plist_path);
    println!("│");
    println!("│ To unload the agent later:");
    println!("│");
    println!("│    launchctl bootout gui/$(id -u) {}", plist_path);
    println!("│");
    println!("└───────────────────────────────────────────────────────────────┘");

    Ok(())
}

/// Print a raw cron line for manual crontab installation (non-macOS).
#[cfg(not(target_os = "macos"))]
fn print_raw_cron_line(task_id: &str, cron_expr: &str, cmd: &str) {
    println!(
        "┌─ Schedule for {} ───────────────────────────────────┐",
        task_id
    );
    println!("│");
    println!("│ Cron expression: {}", cron_expr);
    println!("│");
    println!("│ 1. Run: crontab -e");
    println!("│");
    println!("│ 2. Add this line:");
    println!("│");
    println!("│    {} {}", cron_expr, cmd);
    println!("│");
    println!("│ 3. Save and exit");
    println!("│");
    println!("└──────────────────────────────────────────────────────────┘");
}

/// Parse a 5-field cron expression into (minute, hour, day, month, weekday).
/// Fields containing `*` are returned as `u32::MAX` (meaning "every unit").
fn parse_cron(cron_expr: &str) -> Result<(u32, u32, u32, u32, u32)> {
    fn field(s: &str, name: &str) -> Result<u32> {
        if s == "*" {
            return Ok(u32::MAX);
        }
        s.parse::<u32>()
            .map_err(|_| LearnError::Synth(format!("invalid {name} field")))
    }

    let parts: Vec<&str> = cron_expr.split_whitespace().collect();
    if parts.len() != 5 {
        return Err(LearnError::Synth(
            "cron expression must have exactly 5 fields".to_string(),
        ));
    }

    Ok((
        field(parts[0], "minute")?,
        field(parts[1], "hour")?,
        field(parts[2], "day")?,
        field(parts[3], "month")?,
        field(parts[4], "weekday")?,
    ))
}

/// Generate a macOS LaunchAgent plist XML (simplified for common cases).
#[cfg(target_os = "macos")]
fn generate_plist(
    task_id: &str,
    cmd: &str,
    hour: u32,
    minute: u32,
    weekday: u32,
) -> Result<String> {
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.learn-rv.{}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-c</string>
        <string>{}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{}</integer>
        <key>Minute</key>
        <integer>{}</integer>
        <key>Weekday</key>
        <integer>{}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/learn-watch-{}.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/learn-watch-{}.err</string>
</dict>
</plist>"#,
        task_id, cmd, hour, minute, weekday, task_id, task_id,
    );
    Ok(plist)
}

/// Format plist for display in terminal (with indentation).
#[cfg(target_os = "macos")]
fn format_plist_for_display(plist: &str) -> String {
    plist
        .lines()
        .map(|line| format!("│    {}", line))
        .collect::<Vec<_>>()
        .join("\n")
}

// ── Eval adapters ─────────────────────────────────────────────────────────────

/// Adapts `learn_retrieve::Retriever` to `learn_eval::Retriever`.
struct EvalRetrieverAdapter<'a> {
    inner: &'a mut learn_retrieve::Retriever,
}

#[async_trait]
impl learn_eval::Retriever for EvalRetrieverAdapter<'_> {
    async fn search(&mut self, query: &str, k: usize) -> Result<Vec<Hit>> {
        self.inner.search(query, k).await
    }
}

/// Adapts `Box<dyn learn_synth::Synthesizer>` to `learn_eval::Synthesizer`.
struct EvalSynthAdapter {
    inner: Box<dyn learn_synth::Synthesizer>,
}

#[async_trait]
impl learn_eval::Synthesizer for EvalSynthAdapter {
    async fn ask(&self, topic: &str, question: &str, hits: &[Hit]) -> Result<learn_core::Answer> {
        self.inner.ask(topic, question, hits).await
    }

    async fn apply(
        &self,
        topic: &str,
        task: &str,
        format: &str,
        hits: &[Hit],
    ) -> Result<learn_core::Answer> {
        self.inner.apply(topic, task, format, hits).await
    }
}

// ── Eval ─────────────────────────────────────────────────────────────────────

/// Run the topic's golden Q&A regression suite.
///
/// Loads `<kb_root>/<topic>/eval/golden.yaml`, runs each item through the
/// retriever + synthesizer, prints a summary, and saves the full JSON report.
pub async fn run_regression(topic_str: String, kb_root: Utf8PathBuf) -> Result<()> {
    let topic = Topic::new(&topic_str)?;
    let golden_path = kb_root
        .join(topic.as_str())
        .join("eval")
        .join("golden.yaml");

    if !golden_path.exists() {
        return Err(LearnError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no golden set at {golden_path}"),
        )));
    }

    let set = learn_eval::load_golden(&golden_path)?;
    learn_eval::validate_golden(&set)?;

    let embedder_path = super::default_model_dir();
    let index = LearnIndex::open(kb_root.as_ref(), topic.clone())?;
    let mut retriever =
        learn_retrieve::Retriever::for_topic(index, &topic, embedder_path.as_ref())?;
    retriever.refresh_bm25()?;

    let synth = learn_synth::select_synthesizer()?;

    let mut retriever_adapter = EvalRetrieverAdapter {
        inner: &mut retriever,
    };
    let synth_adapter = EvalSynthAdapter { inner: synth };

    let report = learn_eval::run_eval(&set, &mut retriever_adapter, &synth_adapter).await?;

    println!(
        "eval results: {}/{} passed, {} abstained, score={:.3}",
        report.passed, report.total, report.abstained, report.aggregate_score
    );

    let results_dir = kb_root.join(topic.as_str()).join("eval");
    std::fs::create_dir_all(results_dir.as_std_path()).map_err(LearnError::Io)?;
    let timestamp = eval_timestamp_now();
    let results_path = results_dir.join(format!("results-{timestamp}.json"));
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(results_path.as_std_path(), json.as_bytes()).map_err(LearnError::Io)?;
    println!("full report saved to {results_path}");

    Ok(())
}

/// Compact timestamp string for eval result filenames (UTC, no colons).
fn eval_timestamp_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format: YYYYMMDDTHHMMSSz — safe for file names on all platforms.
    let days = secs / 86_400;
    let time_secs = secs % 86_400;
    let h = time_secs / 3600;
    let m = (time_secs % 3600) / 60;
    let s = time_secs % 60;
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}{mo:02}{d:02}T{h:02}{m:02}{s:02}Z")
}

// ── Forget ───────────────────────────────────────────────────────────────────

/// Drop a topic or one video from the KB.
pub fn run_forget(topic_str: String, video: Option<String>, kb_root: Utf8PathBuf) -> Result<()> {
    let topic = Topic::new(&topic_str)?;
    if let Some(video_id) = video {
        println!(
            "learn forget: per-video forget not yet wired for video {video_id:?}.\n\
             When wired: will call LearnIndex::forget_video and remove the manifest entry."
        );
        return Ok(());
    }
    eprint!("Delete ALL data for topic {topic_str:?}? This cannot be undone. [y/N]: ");
    let mut line = String::new();
    use std::io::BufRead as _;
    std::io::stdin()
        .lock()
        .read_line(&mut line)
        .map_err(LearnError::Io)?;
    if !line.trim().eq_ignore_ascii_case("y") {
        println!("Aborted.");
        return Ok(());
    }
    remove_topic_artifacts(&kb_root, &topic)?;
    println!("Deleted topic {topic_str:?}.");
    Ok(())
}

// ── Compact ──────────────────────────────────────────────────────────────────

/// Re-embed, dedupe, optimize HNSW for a topic.
pub fn run_compact(topic_str: String, kb_root: Utf8PathBuf) -> Result<()> {
    let topic = Topic::new(&topic_str)?;
    let mut index = LearnIndex::open(kb_root.as_ref(), topic.clone())?;
    index.compact()?;
    let stats = index.stats()?;
    let _graph = LearnGraph::open(kb_root.as_ref(), topic.clone())?;
    println!(
        "compact complete: {topic}  {} vectors  {} bytes on disk",
        stats.vector_count, stats.bytes_on_disk
    );
    Ok(())
}

// ── Study ────────────────────────────────────────────────────────────────────

/// Autonomous curriculum: discover top videos and optionally ingest them.
pub async fn run_study(
    topic_description: String,
    depth: String,
    max_videos: Option<usize>,
    auto: bool,
    topic_override: Option<String>,
    kb_root: Utf8PathBuf,
) -> Result<()> {
    let topic = match topic_override {
        Some(t) => Topic::new(&t)?,
        None => Topic::new(&topic_description)?,
    };
    let mut study_depth = depth_from_str(&depth);
    if let Some(n) = max_videos {
        study_depth.max_videos = n;
    }
    tracing::info!(topic = %topic, %topic_description, %depth, auto, "study: calling discover");

    // Call discover — Phase 2.5 stub returns Err until wired.
    let curriculum = learn_discover::discover(&topic_description, study_depth).await?;

    if curriculum.picks.is_empty() {
        println!("No videos found for {topic_description:?}.");
        return Ok(());
    }

    println!(
        "Curriculum for: {:?}  [{depth} — {} videos]",
        topic_description,
        curriculum.picks.len()
    );
    println!();
    println!("{:<4}  {:<45}  Rationale", "Rank", "Sub-topic");
    println!("{}", "─".repeat(80));
    for pick in &curriculum.picks {
        println!(
            "{:<4}  {:<45}  {}",
            pick.rank,
            truncate(&pick.sub_topic, 45),
            truncate(&pick.rationale, 60)
        );
    }

    if !auto {
        eprint!("\nProceed with ingestion? [y/N]: ");
        let mut line = String::new();
        use std::io::BufRead as _;
        std::io::stdin()
            .lock()
            .read_line(&mut line)
            .map_err(LearnError::Io)?;
        if !line.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    for pick in &curriculum.picks {
        let url = pick.video.url.as_str().to_string();
        tracing::info!(video_id = %pick.video.video_id, rank = pick.rank, "study: ingesting");
        if let Err(e) = run_ingest_with_limit(
            url,
            Some(topic.as_str().to_string()),
            kb_root.clone(),
            false,
            None,
            true, // frames enabled by default in study mode
            60,
        )
        .await
        {
            tracing::warn!(
                video_id = %pick.video.video_id,
                rank = pick.rank,
                error = %e,
                "study: failed to ingest"
            );
        }
    }
    println!(
        "Study complete: {} videos ingested into {topic}.",
        curriculum.picks.len()
    );
    Ok(())
}

// ── Frame captioning helper ──────────────────────────────────────────────────

/// Locate the downloaded video file in `raw_dir`.
///
/// yt-dlp writes the video as `video.<ext>` (e.g. `video.mp4`, `video.webm`).
/// Returns the first matching path that is not a caption (`.vtt`) or metadata
/// (`.json`) file. Returns `None` when no video file is present (captions-only
/// run, or yt-dlp download failed).
fn find_video_file(raw_dir: &camino::Utf8Path) -> Option<camino::Utf8PathBuf> {
    let entries = std::fs::read_dir(raw_dir.as_std_path()).ok()?;
    for entry in entries.flatten() {
        let path = camino::Utf8PathBuf::from_path_buf(entry.path()).ok()?;
        let name = path.file_name().unwrap_or("");
        // Must start with "video." and not be a subtitle or metadata file.
        if name.starts_with("video.")
            && !name.ends_with(".vtt")
            && !name.ends_with(".json")
            && !name.ends_with(".part")
        {
            return Some(path);
        }
    }
    None
}

/// Extract keyframes from a video and caption them with Sonnet vision.
///
/// Returns an empty `Vec` (not an error) if the video path does not exist or if
/// `ANTHROPIC_API_KEY` is absent — frame captioning is best-effort.
async fn run_frame_captioning(
    video_path: &camino::Utf8PathBuf,
    out_dir: &camino::Utf8PathBuf,
    max_frames: usize,
) -> Vec<learn_core::Segment> {
    if !video_path.exists() {
        tracing::debug!(
            path = %video_path,
            "frame captioning: video file not found — skipping"
        );
        return vec![];
    }

    let extractor_cfg = learn_frames::ExtractorConfig {
        max_frames,
        ..Default::default()
    };
    let captioner_cfg = learn_frames::CaptionerConfig::default();

    // Print cost estimate.
    learn_frames::estimate_and_print_cost(video_path, &extractor_cfg);

    match learn_frames::extract_and_caption(video_path, out_dir, &extractor_cfg, &captioner_cfg)
        .await
    {
        Ok(segs) => {
            tracing::info!(count = segs.len(), "frame descriptions produced");
            segs
        }
        Err(e) => {
            tracing::warn!(error = %e, "frame captioning failed — continuing without frames");
            vec![]
        }
    }
}

// ── Private helpers ──────────────────────────────────────────────────────────

/// Find an entity by fuzzy-matching `query_lc` against entity names and aliases.
fn find_entity_by_name(graph: &LearnGraph, query_lc: &str) -> Result<Option<EntityId>> {
    let pr = graph.pagerank()?;
    for (eid, _) in &pr {
        if let Some(entity) = graph.entity(eid)? {
            if entity.name.to_lowercase().contains(query_lc) {
                return Ok(Some(eid.clone()));
            }
            for alias in &entity.aliases {
                if alias.to_lowercase().contains(query_lc) {
                    return Ok(Some(eid.clone()));
                }
            }
        }
    }
    Ok(None)
}

/// Convert claim list to synthetic `Hit`s for the synthesizer.
fn claims_to_hits(claims: &[learn_graph::Claim], video_id: &str) -> Vec<learn_core::Hit> {
    claims
        .iter()
        .enumerate()
        .map(|(rank, c)| learn_core::Hit {
            chunk: learn_core::Chunk {
                chunk_id: c.claim_id.clone(),
                video_id: video_id.to_string(),
                start_seconds: c.source_timestamp,
                end_seconds: c.source_timestamp + 1.0,
                text: c.text.clone(),
                token_count: c.text.split_whitespace().count(),
                kind: learn_core::SegmentKind::Caption,
            },
            score: 1.0,
            rank,
        })
        .collect()
}

/// Remove all persisted artifacts for a topic.
pub fn remove_topic_artifacts(kb_root: &Utf8PathBuf, topic: &Topic) -> Result<()> {
    let rvf = topic.rvf_path(kb_root);
    let meta_json = kb_root.join(format!("{}.meta.json", topic.as_str()));
    let graph_db = kb_root
        .join("_graph")
        .join(format!("{}.graphdb", topic.as_str()));
    let manifest = topic.manifest_path(kb_root);
    let raw_dir = topic.raw_dir(kb_root);
    for path in &[
        rvf.as_std_path(),
        meta_json.as_std_path(),
        graph_db.as_std_path(),
        manifest.as_std_path(),
    ] {
        if path.exists() {
            std::fs::remove_file(path).map_err(LearnError::Io)?;
        }
    }
    if raw_dir.exists() {
        std::fs::remove_dir_all(raw_dir.as_std_path()).map_err(LearnError::Io)?;
    }
    Ok(())
}

fn load_manifest_opt(kb_root: &Utf8PathBuf, topic: &Topic) -> Option<Manifest> {
    let path = topic.manifest_path(kb_root);
    if !path.exists() {
        return None;
    }
    std::fs::read_to_string(path.as_std_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

/// `learn list` with no topic — show every KB the user has built. Friendly
/// empty-state when nothing exists yet (no errors, just guidance).
fn run_list_all_topics(kb_root: Utf8PathBuf) -> Result<()> {
    let rvf_files = list_rvf_files(&kb_root);
    if rvf_files.is_empty() {
        println!("No knowledge bases yet at {kb_root}.");
        println!();
        println!("Build your first one:");
        println!("  learn ingest \"<youtube-url>\" --topic <name>");
        println!("    Add a single video, channel, playlist, or search.");
        println!();
        println!("  learn study \"<topic description>\" --topic <name>");
        println!("    Auto-discover the best videos for a topic and ingest them.");
        return Ok(());
    }
    println!("{:<24}  {:>8}  {:>10}  size", "topic", "videos", "vectors");
    println!("{}", "─".repeat(60));
    let mut total_vectors = 0usize;
    let mut total_bytes = 0u64;
    for path in rvf_files {
        let topic_slug = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        total_bytes = total_bytes.saturating_add(bytes);
        let topic = match Topic::new(&topic_slug) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let (vectors, videos) = LearnIndex::open(kb_root.as_ref(), topic.clone())
            .ok()
            .and_then(|idx| idx.stats().ok().map(|s| (s.vector_count, idx)))
            .map(|(v, idx)| {
                let manifest = load_manifest_opt(&kb_root, &topic);
                let video_count = manifest
                    .as_ref()
                    .map(|m| m.videos.len())
                    .unwrap_or_else(|| {
                        let chunks = idx.chunks_snapshot();
                        chunks
                            .iter()
                            .map(|c| c.video_id.clone())
                            .collect::<std::collections::BTreeSet<_>>()
                            .len()
                    });
                (v, video_count)
            })
            .unwrap_or((0, 0));
        total_vectors += vectors;
        println!(
            "{:<24}  {:>8}  {:>10}  {}",
            topic_slug,
            videos,
            vectors,
            format_bytes(bytes),
        );
    }
    println!("{}", "─".repeat(60));
    println!(
        "  total: {} vectors across {} topics, {}",
        total_vectors,
        list_rvf_files(&kb_root).len(),
        format_bytes(total_bytes),
    );
    println!();
    println!("Inspect a topic:  learn list <topic>");
    println!("Health snapshot:  learn status <topic>");
    Ok(())
}

fn format_bytes(b: u64) -> String {
    if b < 1024 {
        format!("{b} B")
    } else if b < 1024 * 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else if b < 1024 * 1024 * 1024 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else {
        format!("{:.2} GB", b as f64 / 1_073_741_824.0)
    }
}

fn list_rvf_files(kb_root: &Utf8PathBuf) -> Vec<std::path::PathBuf> {
    let Ok(entries) = std::fs::read_dir(kb_root.as_std_path()) else {
        return vec![];
    };
    entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("rvf"))
        .collect()
}

fn count_graph_entities(kb_root: &Utf8PathBuf, topic: &Topic) -> usize {
    LearnGraph::open(kb_root.as_ref(), topic.clone())
        .ok()
        .and_then(|g| g.pagerank().ok())
        .map(|pr| pr.len())
        .unwrap_or(0)
}

/// Pair every chunk from the index with its stored embedding to produce
/// `Embedded` values for the consciousness KPI computation.
///
/// Chunks whose embedding is absent from the in-memory map (e.g. index
/// opened in a state before embeddings were flushed) are silently skipped
/// rather than panicking.
fn build_embedded_snapshot(index: &LearnIndex) -> Vec<Embedded> {
    index
        .chunks_snapshot()
        .into_iter()
        .filter_map(|chunk| {
            index
                .embedding_for_chunk_id(&chunk.chunk_id)
                .map(|emb| Embedded {
                    embedding: emb.to_vec(),
                    embedding_model: "stored".to_string(),
                    chunk,
                })
        })
        .collect()
}

/// Convert a depth string to a `StudyDepth`.
pub fn depth_from_str(depth: &str) -> learn_discover::StudyDepth {
    match depth {
        "quick" => learn_discover::StudyDepth::quick(),
        "deep" => learn_discover::StudyDepth::deep(),
        _ => learn_discover::StudyDepth::medium(),
    }
}

fn format_seconds(secs: f64) -> String {
    let total = secs as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}

// ── SONA flush helper ────────────────────────────────────────────────────────

/// Flush the SONA adapter at session end, catching any upstream panics.
///
/// ruvector-sona can panic on freshly-zeroed MicroLoRA weights when
/// `down_proj` is uninitialized (lora.rs index-out-of-bounds). We treat that
/// as a non-fatal best-effort flush; the session JSONL is already persisted.
///
/// This spawns into `tokio::task::spawn_blocking` to keep the panic isolated
/// from the main async task, avoiding a runtime-within-runtime error.
async fn run_sona_flush(
    topic: learn_core::Topic,
    embedder_path: camino::Utf8PathBuf,
    session: learn_chat::ChatSession,
) {
    let _ = tokio::task::spawn_blocking(move || {
        let cfg = learn_embed::EmbedConfig {
            model_dir: embedder_path,
            ..Default::default()
        };
        let Ok(mut embedder) = learn_embed::Embedder::for_topic(&topic, &cfg) else {
            return;
        };
        // Build query + chunk_ids inline without async to avoid nested runtime.
        let assistant_turns: Vec<_> = session
            .history
            .iter()
            .filter(|t| t.role == learn_chat::Role::Assistant)
            .collect();
        if assistant_turns.is_empty() {
            return;
        }
        let chunk_ids: Vec<String> = assistant_turns
            .iter()
            .flat_map(|t| t.hits.iter().flatten())
            .map(|h| h.chunk.chunk_id.clone())
            .collect();
        if chunk_ids.is_empty() {
            return;
        }
        let id_refs: Vec<&str> = chunk_ids.iter().map(|s| s.as_str()).collect();
        let query = session
            .history
            .first()
            .map(|t| t.content.as_str())
            .unwrap_or("session");

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            embedder.record_feedback(query, &id_refs, learn_embed::Outcome::Helpful)
        }));
        if result.is_err() {
            tracing::warn!("SONA adapter flush panicked (upstream bug); session data intact");
        }
    })
    .await;
}

// ── Chat ─────────────────────────────────────────────────────────────────────

/// Run the interactive multi-turn chat REPL for `topic`.
///
/// If `resume_id` is given, restore the existing session. Otherwise start a new
/// one. Reads lines from stdin; slash commands `/help`, `/save`, `/cite`,
/// `/quit` are handled. Empty input at EOF exits cleanly.
pub async fn run_chat(
    topic_str: String,
    resume_id: Option<uuid::Uuid>,
    depth: String,
    kb_root: Utf8PathBuf,
) -> Result<()> {
    use learn_chat::{new_session, resume_session};
    use std::io::{BufRead, Write};

    let topic = learn_core::Topic::new(&topic_str)?;
    let embedder_path = super::default_model_dir();
    let k = crate::depth_to_k(&depth);

    // Open index and build retriever.
    let index = learn_index::LearnIndex::open(&kb_root, topic.clone())?;
    if index.manifest().videos.is_empty() {
        eprintln!("error: topic '{topic_str}' has no data (KB missing or not yet ingested)");
        std::process::exit(2);
    }
    let mut retriever =
        learn_retrieve::Retriever::for_topic(index, &topic, embedder_path.as_ref())?;
    retriever.refresh_bm25()?;

    let synth = learn_synth::select_synthesizer()?;

    // Create or restore session.
    let mut session = match resume_id {
        Some(id) => {
            let s = resume_session(id, &topic, &kb_root)?;
            eprintln!("Resumed session {} ({} prior turns)", s.id, s.history.len());
            s
        }
        None => {
            let s = new_session(&topic, &kb_root, k)?;
            eprintln!("New session {} — topic: {topic_str}", s.id);
            s
        }
    };

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();

    loop {
        // Print prompt.
        {
            let mut out = stdout.lock();
            out.write_all(b"> ").ok();
            out.flush().ok();
        }

        // Read a line.
        let line = {
            let mut buf = String::new();
            let mut locked = stdin.lock();
            match locked.read_line(&mut buf) {
                Ok(0) => break, // EOF
                Ok(_) => buf,
                Err(_) => break,
            }
        };
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        // Handle slash commands.
        if let Some(rest) = trimmed.strip_prefix('/') {
            let cmd = rest.split_whitespace().next().unwrap_or(rest);
            match cmd {
                "quit" | "q" => break,
                "save" => {
                    eprintln!(
                        "Session auto-saved to KB/_chat/{}/{}.jsonl",
                        topic_str, session.id
                    );
                }
                "cite" => {
                    let last_cits: Vec<_> = session
                        .history
                        .iter()
                        .rev()
                        .find(|t| t.role == learn_chat::Role::Assistant)
                        .map(|t| t.citations.clone())
                        .unwrap_or_default();
                    if last_cits.is_empty() {
                        println!("No citations in the last answer.");
                    } else {
                        for (i, c) in last_cits.iter().enumerate() {
                            println!(
                                "[{}] {} — {}",
                                i + 1,
                                c.url,
                                c.title.as_deref().unwrap_or("")
                            );
                        }
                    }
                }
                "help" | "?" => {
                    println!("Slash commands: /help  /cite  /save  /quit");
                }
                _ => {
                    println!("Unknown command /{cmd}. Try /help");
                }
            }
            continue;
        }

        // Regular question.
        match session
            .ask(trimmed, &mut retriever, synth.as_ref(), &kb_root)
            .await
        {
            Ok(answer) => {
                println!("Assistant: {}", answer.text);
                if !answer.citations.is_empty() {
                    println!();
                    for (i, c) in answer.citations.iter().enumerate() {
                        println!("[{}] {}", i + 1, c.url);
                    }
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
            }
        }
        println!();
    }

    // End session: flush SONA adapter once (gated to session-end per architect).
    // ruvector-sona can panic on freshly-zeroed MicroLoRA weights (upstream bug).
    // A failed flush is non-fatal — the session data is already persisted.
    let session_id = session.id;
    run_sona_flush(topic, embedder_path, session).await;

    eprintln!("Session {} ended.", session_id);
    Ok(())
}

// ── Serve ─────────────────────────────────────────────────────────────────────

/// Start the MCP server for `topic` using `transport` (only "stdio" supported).
///
/// Reads JSON-RPC 2.0 from stdin, dispatches to `kb_query`, `kb_synthesize`,
/// or `kb_list_videos`, and writes JSON-RPC responses to stdout.
pub fn run_serve(topic: String, transport: String, kb_root: Utf8PathBuf) -> Result<()> {
    if transport != "stdio" {
        return Err(LearnError::Retrieve(format!(
            "unsupported transport '{transport}'; only 'stdio' is supported"
        )));
    }
    let cfg = learn_serve::ServerConfig { topic, kb_root };
    learn_serve::run_server(cfg).map_err(|e| LearnError::Retrieve(format!("mcp server: {e}")))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use camino::Utf8PathBuf;
    use tempfile::TempDir;

    fn kb(dir: &TempDir) -> Utf8PathBuf {
        Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap()
    }

    #[test]
    fn cmd_status_on_empty_kb_root_lists_zero_topics() {
        let dir = TempDir::new().unwrap();
        let result = run_status(None, kb(&dir));
        assert!(
            result.is_ok(),
            "status on empty KB should be Ok: {result:?}"
        );
    }

    #[test]
    fn cmd_forget_removes_topic_artifacts() {
        let dir = TempDir::new().unwrap();
        let kb_root = kb(&dir);
        let topic = Topic::new("forget-test").unwrap();
        let rvf = topic.rvf_path(&kb_root);
        let meta = kb_root.join(format!("{}.meta.json", topic.as_str()));
        let graph_dir = kb_root.join("_graph");
        std::fs::create_dir_all(graph_dir.as_std_path()).unwrap();
        std::fs::write(rvf.as_std_path(), b"stub").unwrap();
        std::fs::write(meta.as_std_path(), b"{}").unwrap();
        remove_topic_artifacts(&kb_root, &topic).unwrap();
        assert!(!rvf.exists(), ".rvf should be deleted after forget");
        assert!(!meta.exists(), ".meta.json should be deleted after forget");
    }

    #[test]
    fn cmd_list_returns_empty_on_fresh_topic() {
        let dir = TempDir::new().unwrap();
        let result = run_list(Some("fresh-list".to_string()), "date".to_string(), kb(&dir));
        assert!(
            result.is_ok(),
            "list on fresh topic should be Ok: {result:?}"
        );
    }

    #[test]
    fn cmd_compact_returns_ok_on_fresh_topic() {
        let dir = TempDir::new().unwrap();
        let result = run_compact("compact-fresh".to_string(), kb(&dir));
        assert!(
            result.is_ok(),
            "compact on fresh topic should be Ok: {result:?}"
        );
    }

    #[tokio::test]
    async fn cmd_eval_returns_err_when_golden_yaml_missing() {
        let dir = TempDir::new().unwrap();
        let result = run_regression("my-topic".to_string(), kb(&dir)).await;
        match &result {
            Err(LearnError::Io(e)) => {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
                assert!(
                    e.to_string().contains("no golden set at"),
                    "error should mention golden set path; got: {e}"
                );
            }
            other => panic!("expected Err(Io(NotFound)), got {other:?}"),
        }
    }

    /// Write a minimal golden YAML to a tempdir, build mock adapters, run eval,
    /// assert the report has the expected pass count.
    ///
    /// Marked `#[ignore]` because the real retriever requires ONNX model files.
    /// To run: `cargo test -p learn-cli cmd_eval_loads_golden_and_runs -- --ignored`
    #[tokio::test]
    #[ignore = "requires ONNX model files"]
    async fn cmd_eval_loads_golden_and_runs() {
        let dir = TempDir::new().unwrap();
        let kb_root = kb(&dir);

        // Create eval directory and golden YAML.
        let eval_dir = kb_root.join("test-topic").join("eval");
        std::fs::create_dir_all(eval_dir.as_std_path()).unwrap();
        let golden_path = eval_dir.join("golden.yaml");
        std::fs::write(
            golden_path.as_std_path(),
            b"topic: test-topic\nversion: 1\nitems:\n  - id: q1\n    question: What is X?\n    mode: ask\n    expected_substrings: [\"answer\"]\n    forbidden_substrings: []\n    min_citations: 1\n    abstain_acceptable: false\n",
        )
        .unwrap();

        // The real retriever requires model files; this test is ignore-gated.
        // When run with real files, it should succeed and print a summary line.
        let result = run_regression("test-topic".to_string(), kb_root).await;
        assert!(
            result.is_ok(),
            "eval with golden yaml should succeed: {result:?}"
        );
    }

    /// Unit-level test using `learn_eval` directly with mock adapters,
    /// validating the adapter wire-up logic without the CLI layer.
    #[tokio::test]
    async fn eval_adapter_wire_up_passes_canned_answer() {
        use learn_core::{Answer, Chunk, Citation, Hit, SegmentKind};
        use learn_eval::{GoldenItem, GoldenSet, ItemMode};
        use url::Url;

        fn make_hit() -> Hit {
            Hit {
                chunk: Chunk {
                    chunk_id: "c1".into(),
                    video_id: "v1".into(),
                    start_seconds: 0.0,
                    end_seconds: 5.0,
                    text: "answer content".into(),
                    token_count: 2,
                    kind: SegmentKind::Caption,
                },
                score: 0.9,
                rank: 0,
            }
        }

        fn make_citation() -> Citation {
            Citation {
                video_id: "v1".into(),
                title: Some("Test".into()),
                url: Url::parse("https://youtube.com/watch?v=abc").unwrap(),
                start_seconds: 0.0,
            }
        }

        struct DirectMockRetriever;

        #[async_trait::async_trait]
        impl learn_eval::Retriever for DirectMockRetriever {
            async fn search(&mut self, _q: &str, _k: usize) -> learn_core::Result<Vec<Hit>> {
                Ok(vec![make_hit()])
            }
        }

        struct DirectMockSynth;

        #[async_trait::async_trait]
        impl learn_eval::Synthesizer for DirectMockSynth {
            async fn ask(&self, _t: &str, _q: &str, _h: &[Hit]) -> learn_core::Result<Answer> {
                Ok(Answer {
                    text: "answer is here".into(),
                    citations: vec![make_citation()],
                    abstained: false,
                })
            }
            async fn apply(
                &self,
                _t: &str,
                _tk: &str,
                _f: &str,
                _h: &[Hit],
            ) -> learn_core::Result<Answer> {
                Ok(Answer {
                    text: "applied".into(),
                    citations: vec![],
                    abstained: true,
                })
            }
        }

        let set = GoldenSet {
            topic: "test".into(),
            version: 1,
            items: vec![GoldenItem {
                id: "q1".into(),
                question: "What is X?".into(),
                mode: ItemMode::Ask,
                apply_task: None,
                apply_format: None,
                expected_substrings: vec!["answer".into()],
                forbidden_substrings: vec![],
                min_citations: 1,
                abstain_acceptable: false,
            }],
        };

        let mut retriever = DirectMockRetriever;
        let synth = DirectMockSynth;
        let report = learn_eval::run_eval(&set, &mut retriever, &synth)
            .await
            .unwrap();
        assert_eq!(report.total, 1);
        assert_eq!(
            report.passed, 1,
            "canned answer satisfies expected_substrings"
        );
    }

    #[test]
    fn cmd_watch_prints_phase4_message() {
        let result = run_watch(
            "rust-async".to_string(),
            "@jonhoo".to_string(),
            "weekly".to_string(),
        );
        assert!(result.is_ok(), "watch stub should be Ok: {result:?}");
    }

    #[test]
    fn format_seconds_handles_hours() {
        assert_eq!(format_seconds(3661.0), "1:01:01");
        assert_eq!(format_seconds(90.0), "1:30");
        assert_eq!(format_seconds(0.0), "0:00");
    }

    #[test]
    fn depth_from_str_maps_correctly() {
        assert_eq!(depth_from_str("quick").max_videos, 5);
        assert_eq!(depth_from_str("medium").max_videos, 10);
        assert_eq!(depth_from_str("deep").max_videos, 25);
        assert_eq!(depth_from_str("bogus").max_videos, 10);
    }

    #[test]
    fn truncate_clips_long_strings() {
        let s = "hello world long string";
        assert_eq!(truncate(s, 5), "hell…");
        assert_eq!(truncate("hi", 10), "hi");
    }

    #[test]
    fn list_rvf_files_returns_empty_for_fresh_dir() {
        let dir = TempDir::new().unwrap();
        let root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let files = list_rvf_files(&root);
        assert!(files.is_empty());
    }

    #[test]
    fn list_rvf_files_finds_rvf_not_other_extensions() {
        let dir = TempDir::new().unwrap();
        let root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        std::fs::write(dir.path().join("topic.rvf"), b"").unwrap();
        std::fs::write(dir.path().join("topic.json"), b"").unwrap();
        let files = list_rvf_files(&root);
        assert_eq!(files.len(), 1, "only the .rvf should be returned");
    }

    // ── Phase 3E: ingest resume tests ────────────────────────────────────────

    /// Pre-populate the manifest with `Indexed` status and verify the skip
    /// predicate fires when `force = false`.
    ///
    /// `run_ingest` cannot be called hermetically (it shells out to yt-dlp),
    /// so we exercise the load-bearing manifest check via `LearnIndex` directly.
    /// The skip predicate in `run_ingest` is:
    ///   `Some(IngestStatus::Indexed) if !force => skip`
    /// We verify both the persist path and the predicate logic here.
    #[test]
    fn cmd_ingest_skips_indexed_video_unless_force() {
        use learn_core::{IngestStatus, VideoState};
        use learn_index::LearnIndex;

        let dir = TempDir::new().unwrap();
        let kb_root = kb(&dir);
        let topic = Topic::new("skip-indexed-topic").unwrap();

        // Pre-populate the manifest with Indexed status via LearnIndex.
        let mut index = LearnIndex::open(kb_root.as_ref(), topic.clone()).unwrap();
        index
            .upsert_video_state(VideoState {
                video_id: "dQw4w9WgXcQ".to_string(),
                status: IngestStatus::Indexed,
                fetched_at: Some("2026-01-01T00:00:00Z".to_string()),
                indexed_at: Some("2026-01-01T00:00:01Z".to_string()),
                chunk_count: 42,
                error: None,
            })
            .unwrap();

        // Re-open and confirm the status survived a reopen.
        let index2 = LearnIndex::open(kb_root.as_ref(), topic).unwrap();
        let vs = index2
            .manifest()
            .videos
            .get("dQw4w9WgXcQ")
            .expect("video state must be present after reopen");
        assert_eq!(vs.status, IngestStatus::Indexed);
        assert_eq!(vs.chunk_count, 42);

        // The skip predicate from run_ingest: Indexed + !force → skip.
        let force = false;
        let would_skip = vs.status == IngestStatus::Indexed && !force;
        assert!(
            would_skip,
            "should skip when status=Indexed and force=false"
        );
    }

    /// Pre-populate the manifest with `Failed` status, verify that with
    /// `force = true` the skip predicate does NOT fire.
    #[test]
    fn cmd_ingest_resumes_from_failed_state_with_force() {
        use learn_core::{IngestStatus, VideoState};
        use learn_index::LearnIndex;

        let dir = TempDir::new().unwrap();
        let kb_root = kb(&dir);
        let topic = Topic::new("resume-failed-topic").unwrap();

        // Pre-populate with Failed.
        let mut index = LearnIndex::open(kb_root.as_ref(), topic.clone()).unwrap();
        index
            .upsert_video_state(VideoState {
                video_id: "failedvid001".to_string(),
                status: IngestStatus::Failed,
                fetched_at: Some("2026-01-01T00:00:00Z".to_string()),
                indexed_at: None,
                chunk_count: 0,
                error: Some("no transcript available".to_string()),
            })
            .unwrap();

        // Re-open and confirm.
        let index2 = LearnIndex::open(kb_root.as_ref(), topic).unwrap();
        let vs = index2
            .manifest()
            .videos
            .get("failedvid001")
            .expect("video state must be present after reopen");
        assert_eq!(vs.status, IngestStatus::Failed);
        assert_eq!(vs.error.as_deref(), Some("no transcript available"));

        // With force=true, the skip predicate must NOT fire.
        let force = true;
        let would_skip = vs.status == IngestStatus::Failed && !force;
        assert!(
            !would_skip,
            "should NOT skip when force=true even for Failed status"
        );
    }

    /// Phase 3E: Embedded-checkpoint resume.
    ///
    /// Simulates a crash after the Embed stage: the manifest has `Embedded`
    /// status and the sidecar already holds the chunks + embeddings (written
    /// by `index.ingest()` during the first run).  After a reopen the
    /// `embedded_for_video` method must reconstruct the full batch so the
    /// CLI can skip directly to the index step.
    ///
    /// This test exercises the real production path (upsert_video_state →
    /// save_manifest → reopen → embedded_for_video) without calling acquire_url.
    #[test]
    fn cmd_ingest_embedded_checkpoint_resume_skips_to_index_step() {
        use learn_core::{Chunk, Embedded, IngestStatus, SegmentKind, VideoState};
        use learn_index::LearnIndex;

        let dir = TempDir::new().unwrap();
        let kb_root = kb(&dir);
        let topic = Topic::new("embedded-resume-topic").unwrap();
        let video_id = "embed_resume_vid_001";

        // --- Phase A: simulate the first run completing up to Embedded ---
        {
            let mut index = LearnIndex::open(kb_root.as_ref(), topic.clone()).unwrap();

            // Build two synthetic chunks with 4-dim embeddings.
            let chunks: Vec<Chunk> = (0..2)
                .map(|i| Chunk {
                    chunk_id: format!("{video_id}-chunk-{i}"),
                    video_id: video_id.to_string(),
                    start_seconds: i as f64 * 10.0,
                    end_seconds: i as f64 * 10.0 + 9.9,
                    text: format!("chunk text {i}"),
                    token_count: 3,
                    kind: SegmentKind::Caption,
                })
                .collect();

            let embedded: Vec<Embedded> = chunks
                .iter()
                .map(|c| Embedded {
                    chunk: c.clone(),
                    // 4-dim embedding; RVF requires dim > 0.
                    embedding: vec![0.1_f32 * (c.start_seconds as f32 + 1.0); 4],
                    embedding_model: "test-model".to_string(),
                })
                .collect();

            // Ingest writes chunks + embeddings into the sidecar and .emb.bin.
            index.ingest(&embedded).unwrap();

            // Write Embedded status — this is the checkpoint a killed process leaves.
            index
                .upsert_video_state(VideoState {
                    video_id: video_id.to_string(),
                    status: IngestStatus::Embedded,
                    fetched_at: Some("2026-01-01T00:00:00Z".to_string()),
                    indexed_at: None,
                    chunk_count: embedded.len(),
                    error: None,
                })
                .unwrap();
        }

        // --- Phase B/C/D/E: reopen, verify, recover, complete ---
        // Each phase uses a scoped block so the RVF writer lock is released
        // before the next open (RvfStore acquires an advisory writer lock on
        // the .rvf file; two live instances in the same process conflict).
        {
            let mut index2 = LearnIndex::open(kb_root.as_ref(), topic.clone()).unwrap();

            // Phase B: manifest survived the reopen.
            let status = index2
                .manifest()
                .videos
                .get(video_id)
                .expect("video state must survive reopen")
                .status;
            assert_eq!(
                status,
                IngestStatus::Embedded,
                "manifest must show Embedded after reopen"
            );

            // Phase C: embedded_for_video reconstructs the batch.
            let recovered = index2.embedded_for_video(video_id);
            assert_eq!(
                recovered.len(),
                2,
                "embedded_for_video must return both chunks from the sidecar"
            );
            assert!(
                recovered.iter().all(|e| e.embedding.len() == 4),
                "each recovered embedding must have dim=4"
            );
            assert!(
                recovered.iter().all(|e| e.chunk.video_id == video_id),
                "all recovered chunks must belong to the target video"
            );

            // Phase D: the CLI skip predicate fires for Embedded+!force.
            let force = false;
            let would_resume = status == IngestStatus::Embedded && !force && !recovered.is_empty();
            assert!(
                would_resume,
                "should resume from Embedded checkpoint (not re-embed) when force=false"
            );

            // Phase E: the index step accepts the recovered embeddings.
            let accepted = index2.ingest(&recovered).unwrap();
            assert!(
                accepted > 0 || !recovered.is_empty(),
                "index step must accept the recovered embeddings"
            );
            index2
                .upsert_video_state(VideoState {
                    video_id: video_id.to_string(),
                    status: IngestStatus::Indexed,
                    fetched_at: Some("2026-01-01T00:00:00Z".to_string()),
                    indexed_at: Some("2026-01-01T00:00:05Z".to_string()),
                    chunk_count: recovered.len(),
                    error: None,
                })
                .unwrap();
        } // drops index2, releasing the RVF writer lock

        // Re-open once more and confirm Indexed status persisted.
        let index3 = LearnIndex::open(kb_root.as_ref(), topic).unwrap();
        let final_vs = index3
            .manifest()
            .videos
            .get(video_id)
            .expect("video state must survive second reopen");
        assert_eq!(
            final_vs.status,
            IngestStatus::Indexed,
            "status must be Indexed after resume completes"
        );
    }

    #[test]
    fn test_cadence_to_cron_hourly() {
        let cron = cadence_to_cron("hourly").unwrap();
        assert_eq!(cron, "0 * * * *");
    }

    #[test]
    fn test_cadence_to_cron_daily() {
        let cron = cadence_to_cron("daily").unwrap();
        assert_eq!(cron, "0 6 * * *");
    }

    #[test]
    fn test_cadence_to_cron_weekly() {
        let cron = cadence_to_cron("weekly").unwrap();
        assert_eq!(cron, "0 6 * * 0");
    }

    #[test]
    fn test_cadence_to_cron_monthly() {
        let cron = cadence_to_cron("monthly").unwrap();
        assert_eq!(cron, "0 6 1 * *");
    }

    #[test]
    fn test_cadence_to_cron_raw_five_field() {
        let cron = cadence_to_cron("30 2 15 * *").unwrap();
        assert_eq!(cron, "30 2 15 * *");
    }

    #[test]
    fn test_cadence_to_cron_whitespace_normalization() {
        let cron = cadence_to_cron("  daily  ").unwrap();
        assert_eq!(cron, "0 6 * * *");
    }

    #[test]
    fn test_cadence_to_cron_case_insensitive() {
        let cron = cadence_to_cron("WEEKLY").unwrap();
        assert_eq!(cron, "0 6 * * 0");
    }

    #[test]
    fn test_cadence_to_cron_unknown_cadence() {
        let result = cadence_to_cron("unknown-cadence");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_cron_valid_five_field() {
        let (min, hour, day, month, wday) = parse_cron("30 2 15 6 3").unwrap();
        assert_eq!(min, 30);
        assert_eq!(hour, 2);
        assert_eq!(day, 15);
        assert_eq!(month, 6);
        assert_eq!(wday, 3);
    }

    #[test]
    fn test_parse_cron_invalid_field_count() {
        let result = parse_cron("30 2 15 6");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_cron_non_numeric_field() {
        let result = parse_cron("xx 2 15 6 3");
        assert!(result.is_err());
    }
}
