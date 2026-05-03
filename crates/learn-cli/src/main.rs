//! `learn` — point at a video source, build a knowledge base, query it.

mod commands;
mod doctor;

use camino::Utf8PathBuf;
use clap::{Parser, Subcommand};
use learn_core::Topic;
use std::process;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(name = "learn", version, about = "Pure-Rust video knowledge-base CLI")]
struct Cli {
    /// Override the KB root (default: ~/Docs/KB).
    #[arg(long, env = "LEARN_KB_ROOT", global = true)]
    kb_root: Option<Utf8PathBuf>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Tactical entry: ingest a known source (URL, playlist, channel, search,
    /// or local file) into a knowledge base. The topic name is auto-derived
    /// from the source's metadata unless `--topic` is set.
    Ingest {
        /// URL, playlist URL, channel @handle, "ytsearch20:<query>", or local path.
        source: String,
        /// Override the auto-derived topic slug.
        #[arg(long)]
        topic: Option<String>,
        #[arg(long)]
        since: Option<String>,
        #[arg(long)]
        limit: Option<usize>,
        /// Disable Sonnet-vision frame captioning (enabled by default).
        #[arg(long)]
        no_frames: bool,
        /// Explicitly enable Sonnet-vision frame captioning (default behaviour; symmetric with --no-frames).
        #[arg(long, conflicts_with = "no_frames")]
        with_frames: bool,
        /// Maximum number of keyframes to extract per video (default: 60).
        #[arg(long, default_value = "60")]
        max_frames: usize,
        #[arg(long)]
        force: bool,
    },
    /// Ask a question against a topic KB; answers come with citations.
    Ask {
        topic: String,
        question: String,
        #[arg(long, default_value = "deep")]
        depth: String,
    },
    /// Use the topic KB as the prior to *produce* something — a recipe, a
    /// strategy, a plan, code. Grounded in the corpus, fully cited.
    Apply {
        topic: String,
        task: String,
        /// Output file (default: stdout).
        #[arg(long)]
        out: Option<Utf8PathBuf>,
        /// Output format: markdown | json | code.
        #[arg(long, default_value = "markdown")]
        format: String,
    },
    /// Autonomous curriculum: from a natural-language topic, find the top
    /// videos worth watching, then ingest them into a per-topic .rvf.
    /// Uses Ruflo GOAP + deep-research for ranking.
    Study {
        /// Natural-language topic description.
        topic_description: String,
        /// quick (5 videos) | medium (10) | deep (25).
        #[arg(long, default_value = "medium")]
        depth: String,
        /// Override the auto-picked count.
        #[arg(long)]
        max_videos: Option<usize>,
        /// Skip the human-confirm step and ingest immediately.
        #[arg(long)]
        auto: bool,
        /// Override the slug (default: derived from topic_description).
        #[arg(long)]
        topic: Option<String>,
    },
    /// Find every claim attributable to a speaker or entity.
    WhoSaid { topic: String, query: String },
    /// Build a temporal timeline of how an entity/topic has been discussed.
    Timeline { topic: String, entity: String },
    /// Compare two things using the KB.
    Compare { topic: String, a: String, b: String },
    /// Summarize a topic or one specific video.
    Summarize {
        topic: String,
        #[arg(long)]
        video: Option<String>,
    },
    /// List all knowledge bases (no args), or videos in one topic.
    List {
        /// Topic name. Omit to see every KB you've built.
        topic: Option<String>,
        #[arg(long, default_value = "date")]
        by: String,
    },
    /// Health snapshot of a KB (or all topics if omitted).
    Status { topic: Option<String> },
    /// Schedule recurring ingestion of a channel into a topic.
    Watch {
        topic: String,
        channel: String,
        #[arg(long, default_value = "weekly")]
        cadence: String,
    },
    /// Run the topic's golden Q&A regression suite.
    Eval { topic: String },
    /// Drop a topic or one video.
    Forget {
        topic: String,
        #[arg(long)]
        video: Option<String>,
    },
    /// Re-embed, dedupe, optimize HNSW.
    Compact { topic: String },
    /// First-60-seconds environment diagnostic: check deps, storage, network,
    /// version, and config.  Exit 0 when all required checks pass.
    Doctor,
    /// Interactive multi-turn chat session grounded in a topic KB.
    Chat {
        topic: String,
        /// Resume an existing session by UUID.
        #[arg(long)]
        resume: Option<Uuid>,
        /// Retrieval depth: quick (5) | medium (10) | deep (20).
        #[arg(long, default_value = "medium")]
        depth: String,
    },
    /// Start an MCP server exposing the KB as a tool surface for Claude Code.
    ///
    /// The server reads JSON-RPC 2.0 from stdin and writes responses to stdout.
    /// Add it to your Claude Code MCP config and the kb_query / kb_synthesize /
    /// kb_list_videos tools become available in any Claude Code session.
    Serve {
        /// Topic whose KB to expose.
        topic: String,
        /// Transport: only "stdio" is supported (default).
        #[arg(long, default_value = "stdio")]
        transport: String,
    },
}

/// Print the friendly orientation block and exit 0.
/// Called when the binary is invoked with no arguments.
fn print_orientation() -> ! {
    println!(
        r#"Learn-RV — Your tool for building intelligent knowledge bases, stored in RuVector.

▶ 30-second quickstart
  learn ingest "<youtube-url>" --topic <name>     Add a video, channel, playlist, or search
  learn ask <topic> "<question>"                  Cited answer from the KB
  learn list                                      See every KB you've built

▶ Going deeper
  learn study <topic> --depth medium              Autonomous curriculum (auto-discover videos)
  learn apply <topic> "<task>"                    Generate a cited artifact (recipe, plan, code)
  learn watch <topic> --cadence weekly            Schedule recurring channel ingestion

▶ All 15 commands:    learn --help
▶ Per-command flags:  learn <command> --help

▶ In Claude Code, you don't type any of this.
  Just say what you want: "build me a KB on french cooking technique"
  Claude finds the learn-rv skill and runs the right commands for you.

KB location:    ~/Docs/KB/<topic>.rvf
Skill manifest: ~/.claude/skills/learn-rv/SKILL.md
Repo:           https://github.com/stuinfla/learner-rv"#
    );
    std::process::exit(0);
}

#[tokio::main]
async fn main() {
    if std::env::args().count() == 1 {
        print_orientation();
    }
    init_tracing();
    let cli = Cli::parse();
    let kb_root = resolve_kb_root(cli.kb_root);

    // Doctor is handled separately: it has its own exit-code logic.
    if matches!(cli.cmd, Cmd::Doctor) {
        let ok = doctor::run_doctor(kb_root.as_std_path()).await;
        process::exit(if ok { 0 } else { 1 });
    }

    let result = match cli.cmd {
        Cmd::Ingest {
            source,
            topic,
            since,
            limit,
            no_frames,
            with_frames: _,
            max_frames,
            force,
        } => {
            if since.is_some() {
                tracing::warn!("--since is not yet implemented and will be ignored");
            }
            let frames_enabled = !no_frames;
            commands::run_ingest_with_limit(
                source,
                topic,
                kb_root,
                force,
                limit,
                frames_enabled,
                max_frames,
            )
            .await
        }
        Cmd::Ask {
            topic,
            question,
            depth,
        } => run_ask(topic, question, depth_to_k(&depth), kb_root).await,
        Cmd::Apply {
            topic,
            task,
            out,
            format,
        } => run_apply(topic, task, out, format, kb_root).await,
        Cmd::Study {
            topic_description,
            depth,
            max_videos,
            auto,
            topic,
        } => commands::run_study(topic_description, depth, max_videos, auto, topic, kb_root).await,
        Cmd::WhoSaid { topic, query } => commands::run_who_said(topic, query, kb_root).await,
        Cmd::Timeline { topic, entity } => commands::run_timeline(topic, entity, kb_root).await,
        Cmd::Compare { topic, a, b } => commands::run_compare(topic, a, b, kb_root).await,
        Cmd::Summarize { topic, video } => commands::run_summarize(topic, video, kb_root).await,
        Cmd::List { topic, by } => commands::run_list(topic, by, kb_root),
        Cmd::Status { topic } => commands::run_status(topic, kb_root),
        Cmd::Watch {
            topic,
            channel,
            cadence,
        } => commands::run_watch(topic, channel, cadence),
        Cmd::Eval { topic } => commands::run_regression(topic, kb_root).await,
        Cmd::Forget { topic, video } => commands::run_forget(topic, video, kb_root),
        Cmd::Compact { topic } => commands::run_compact(topic, kb_root),
        Cmd::Chat {
            topic,
            resume,
            depth,
        } => commands::run_chat(topic, resume, depth, kb_root).await,
        Cmd::Serve { topic, transport } => commands::run_serve(topic, transport, kb_root),
        // Doctor is dispatched above; this arm is unreachable but required by exhaustiveness.
        Cmd::Doctor => unreachable!("doctor dispatched before match"),
    };

    if let Err(e) = result {
        print_error(&e);
        process::exit(1);
    }
}

// ── Command implementations ──────────────────────────────────────────────────

/// Exit codes for `learn ask`:
/// - 0 = real cited answer returned
/// - 1 = error (network, API, parse) — propagated via `Err(LearnError)`
/// - 2 = topic doesn't exist (KB missing / no videos ever ingested)
/// - 3 = topic exists but no relevant data (real abstain)
async fn run_ask(
    topic_str: String,
    question: String,
    k: usize,
    kb_root: Utf8PathBuf,
) -> learn_core::Result<()> {
    let topic = Topic::new(&topic_str)?;
    let embedder_path = default_model_dir();

    // 1. Open index.
    let index = learn_index::LearnIndex::open(&kb_root, topic.clone())?;

    // 2. Exit 2 if topic has never had any videos ingested (KB missing).
    if index.manifest().videos.is_empty() {
        eprintln!("error: topic '{topic_str}' has no data (KB missing or not yet ingested)");
        process::exit(2);
    }

    // 3. Build retriever with per-topic SONA adapter.
    let mut retriever =
        learn_retrieve::Retriever::for_topic(index, &topic, embedder_path.as_ref())?;

    // 4. Build BM25 index.
    retriever.refresh_bm25()?;

    // 5. Search with depth-derived k.
    let hits = retriever.search(&question, k).await?;

    if hits.is_empty() {
        eprintln!("(no relevant chunks found for this question in topic '{topic_str}')");
        process::exit(3);
    }

    // 6. Synthesize.
    let synth = learn_synth::select_synthesizer()?;
    let answer = synth.ask(topic.as_str(), &question, &hits).await?;

    if answer.abstained {
        eprintln!("(model abstained: insufficient evidence in KB)");
        process::exit(3);
    }

    println!("{}", answer.text);
    Ok(())
}

async fn run_apply(
    topic_str: String,
    task: String,
    out: Option<Utf8PathBuf>,
    format: String,
    kb_root: Utf8PathBuf,
) -> learn_core::Result<()> {
    let topic = Topic::new(&topic_str)?;
    let embedder_path = default_model_dir();

    // 1. Open index.
    let index = learn_index::LearnIndex::open(&kb_root, topic.clone())?;

    // 2. Build retriever with per-topic SONA adapter.
    let mut retriever =
        learn_retrieve::Retriever::for_topic(index, &topic, embedder_path.as_ref())?;
    retriever.refresh_bm25()?;

    // 3. Search using task text as query.
    let hits = retriever.search(&task, 10).await?;

    if hits.is_empty() {
        println!("KB doesn't cover this.");
        return Ok(());
    }

    // 4. Synthesize.
    let synth = learn_synth::select_synthesizer()?;
    let answer = synth.apply(topic.as_str(), &task, &format, &hits).await?;

    let text = if answer.abstained {
        eprintln!("(model abstained: insufficient evidence in KB)");
        return Ok(());
    } else {
        answer.text
    };

    // 5. Write output.
    match out {
        Some(path) => {
            std::fs::write(path.as_std_path(), text.as_bytes())
                .map_err(learn_core::LearnError::Io)?;
        }
        None => print!("{text}"),
    }

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Map `--depth` string to retriever k-count.
///
/// - "quick"  →  5 results
/// - "deep"   → 20 results
/// - "medium" or anything else → 10 results (default)
fn depth_to_k(depth: &str) -> usize {
    match depth {
        "quick" => 5,
        "deep" => 20,
        _ => 10,
    }
}

/// Resolve KB root: flag → env → ~/Docs/KB.
fn resolve_kb_root(flag: Option<Utf8PathBuf>) -> Utf8PathBuf {
    if let Some(p) = flag {
        return p;
    }
    if let Ok(env) = std::env::var("LEARN_KB_ROOT") {
        if !env.is_empty() {
            return Utf8PathBuf::from(env);
        }
    }
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    Utf8PathBuf::from_path_buf(home.join("Docs").join("KB"))
        .unwrap_or_else(|_| Utf8PathBuf::from("./Docs/KB"))
}

/// Default BGE-large model directory.
pub(crate) fn default_model_dir() -> Utf8PathBuf {
    if let Ok(env) = std::env::var("LEARN_EMBED_MODEL_DIR") {
        if !env.is_empty() {
            return Utf8PathBuf::from(env);
        }
    }
    let cache = dirs::cache_dir().unwrap_or_else(|| std::path::PathBuf::from(".cache"));
    Utf8PathBuf::from_path_buf(
        cache
            .join("learn-rs")
            .join("models")
            .join("bge-large-en-v15"),
    )
    .unwrap_or_else(|_| Utf8PathBuf::from(".cache/learn-rs/models/bge-large-en-v15"))
}

/// Produce a minimal RFC 3339 timestamp (UTC, second precision).
///
/// Does not require `chrono`. Uses `SystemTime` + Hatcher's civil-from-days
/// algorithm (all arithmetic in `i64` to match the signed-integer requirement).
/// Format: `YYYY-MM-DDTHH:MM:SSZ`.
pub(crate) fn rfc3339_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let days: i64 = secs / 86_400;
    let time_secs = secs % 86_400;
    let h = time_secs / 3600;
    let m = (time_secs % 3600) / 60;
    let s = time_secs % 60;
    // Hatcher civil_from_days (Gregorian, signed).
    let z: i64 = days + 719_468;
    let era: i64 = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe: i64 = z - era * 146_097;
    let yoe: i64 = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y: i64 = yoe + era * 400;
    let doy: i64 = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp: i64 = (5 * doy + 2) / 153;
    let d: i64 = doy - (153 * mp + 2) / 5 + 1;
    let mo: i64 = if mp < 10 { mp + 3 } else { mp - 9 };
    let y: i64 = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Print a `LearnError` as a user-friendly stderr message.
fn print_error(err: &learn_core::LearnError) {
    use learn_core::LearnError::*;
    match err {
        Acquire(msg) => eprintln!("error: failed to acquire source: {msg}"),
        Embed(msg) => eprintln!(
            "error: embedder unavailable. Place model.onnx + tokenizer.json at \
             ~/.cache/learn-rs/models/bge-large-en-v15/ or set $LEARN_EMBED_MODEL_DIR.\n\
             details: {msg}"
        ),
        Synth(msg) => eprintln!(
            "error: synthesis failed: {msg}\n  \
             Set ANTHROPIC_API_KEY for cloud, or LEARN_SYNTH_LOCAL=1 + GGUF model at \
             ~/.cache/learn-rs/models/ruvllm-default.gguf for local."
        ),
        Topic(msg) => eprintln!("error: invalid topic: {msg}"),
        Io(e) => eprintln!("error: io: {e}"),
        Serde(e) => eprintln!("error: serde: {e}"),
        Chunk(msg) => eprintln!("error: chunk: {msg}"),
        Index(msg) => eprintln!("error: index: {msg}"),
        Retrieve(msg) => eprintln!("error: retrieve: {msg}"),
        Apply(msg) => eprintln!("error: apply: {msg}"),
        Graph(msg) => eprintln!("error: graph: {msg}"),
        Transcribe(msg) => eprintln!("error: transcribe: {msg}"),
    }
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    fmt().with_env_filter(filter).with_target(false).init();
}

/// Phase-1 placeholder topic derivation from a source string.
///
/// - URL with `v=` query param → video id
/// - URL → last path segment
/// - URL with no useful path → host
/// - local path → file stem
pub(crate) fn derive_topic_from_source(source: &str) -> learn_core::Result<Topic> {
    let raw = if let Ok(u) = url::Url::parse(source) {
        u.query_pairs()
            .find_map(|(k, v)| (k == "v").then(|| v.into_owned()))
            .or_else(|| {
                u.path_segments().and_then(|s| {
                    s.filter(|seg| !seg.is_empty())
                        .next_back()
                        .map(str::to_owned)
                })
            })
            .unwrap_or_else(|| u.host_str().unwrap_or("untitled").to_owned())
    } else {
        std::path::Path::new(source)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("untitled")
            .to_owned()
    };
    Topic::new(&raw)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── derive_topic_from_source (pre-existing, must stay green) ─────────────

    #[test]
    fn derive_youtube_short_url() {
        let t = derive_topic_from_source("https://youtu.be/QZMljuD10sU").unwrap();
        assert_eq!(t.as_str(), "qzmljud10su");
    }

    #[test]
    fn derive_youtube_v_query() {
        let t =
            derive_topic_from_source("https://www.youtube.com/watch?v=QZMljuD10sU&si=abc").unwrap();
        assert_eq!(t.as_str(), "qzmljud10su");
    }

    #[test]
    fn derive_local_path() {
        let t = derive_topic_from_source("/tmp/My Cooking Lecture.mp4").unwrap();
        assert_eq!(t.as_str(), "my-cooking-lecture");
    }

    #[test]
    fn derive_falls_back_to_host() {
        let t = derive_topic_from_source("https://example.com/").unwrap();
        assert_eq!(t.as_str(), "example-com");
    }

    #[test]
    fn derive_returns_err_when_slug_normalizes_empty() {
        let r = derive_topic_from_source("https://example.com/!!!");
        assert!(r.is_err(), "expected Err for all-punctuation path segment");
    }

    // ── New test 1: ingest body rejects dash-prefix source immediately ────────

    #[tokio::test]
    async fn cmd_ingest_returns_acquire_error_on_dash_prefix_source() {
        let dir = tempfile::tempdir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let result = commands::run_ingest(
            "--malicious".to_string(),
            Some("test-topic".to_string()),
            kb_root,
            false,
        )
        .await;
        assert!(
            matches!(result, Err(learn_core::LearnError::Acquire(_))),
            "expected Err(LearnError::Acquire) for dash-prefix source, got: {result:?}"
        );
    }

    // ── New test 2: ingest propagates Embed error when model dir is absent ────

    #[tokio::test]
    #[ignore = "requires ONNX model files"]
    async fn ingest_creates_topic_dirs() {
        // This test exercises the full ingest pipeline with a fixture VTT.
        // It requires ONNX model files to be present — mark ignored for CI.
        //
        // When run with real model files, point LEARN_EMBED_MODEL_DIR at them
        // and provide a real yt-dlp-accessible URL.
        //
        // The hermetic version below tests that the Embed error propagates
        // correctly when the model dir does not exist.
    }

    /// Hermetic version: use a non-existent model dir and verify the
    /// Embed error is returned with the expected-path message.
    ///
    /// This test does NOT need yt-dlp or ONNX files. It verifies that when
    /// acquisition succeeds but the embedder path is absent, we get a clean
    /// `LearnError::Embed` (not a panic, not an Io error).
    ///
    /// We exercise this by calling run_ingest with a fixture VTT path as the
    /// source AND a real kb_root, but with no model files — the Embed step
    /// must fire the correct error.
    ///
    /// Because acquire_url calls yt-dlp, we cannot make this fully hermetic
    /// without mocking the subprocess. Instead we directly test the
    /// embed-missing-model path via `Embedder::for_topic` with a bad path.
    #[test]
    fn embed_error_propagates_with_expected_path_message() {
        use learn_embed::{EmbedConfig, Embedder};
        let dir = tempfile::tempdir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let topic = learn_core::Topic::new("test-topic").unwrap();
        // Point at a path guaranteed not to exist.
        let absent_model_dir = kb_root.join("no-models-here");
        let cfg = EmbedConfig {
            model_dir: absent_model_dir.clone(),
            ..Default::default()
        };
        let result = Embedder::for_topic(&topic, &cfg);
        let is_embed_err = matches!(result, Err(learn_core::LearnError::Embed(_)));
        assert!(
            is_embed_err,
            "expected Err(LearnError::Embed) for absent model dir"
        );
        if let Err(learn_core::LearnError::Embed(msg)) = result {
            // The error message should contain the model path.
            assert!(
                msg.contains("model") || msg.contains("onnx") || msg.contains("load"),
                "error message should mention model loading; got: {msg}"
            );
        }
    }

    // ── run_ask exit-code discriminator tests ─────────────────────────────────

    /// When a topic has an empty manifest (no videos ever ingested), the index
    /// is considered "KB missing". The test verifies the manifest discriminator
    /// works correctly — i.e. `LearnIndex::open` on a fresh dir yields an empty
    /// manifest, which is the trigger for exit 2.
    #[test]
    fn ask_exit2_discriminator_empty_manifest_on_fresh_topic() {
        let dir = tempfile::tempdir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let topic = learn_core::Topic::new("nonexistent-topic").unwrap();
        let index = learn_index::LearnIndex::open(&kb_root, topic).unwrap();
        assert!(
            index.manifest().videos.is_empty(),
            "fresh topic must have empty manifest — discriminator for exit 2"
        );
    }

    /// When slug-collision guard fires, ingest returns LearnError::Acquire
    /// with the blocking message before calling yt-dlp.
    #[tokio::test]
    async fn cmd_ingest_slug_collision_returns_acquire_error() {
        let dir = tempfile::tempdir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();

        // Pre-create the raw_dir with a video.info.json for a video NOT in manifest.
        let raw_dir = kb_root.join("_raw").join("my-topic");
        std::fs::create_dir_all(raw_dir.as_std_path()).unwrap();
        std::fs::write(
            raw_dir.join("video.info.json").as_std_path(),
            r#"{"id":"squatter_xyz","title":"unrelated video"}"#,
        )
        .unwrap();

        // The manifest is empty so "squatter_xyz" is NOT a known video.
        let result = commands::run_ingest_with_limit(
            "https://example.com/my-topic".to_string(),
            Some("my-topic".to_string()),
            kb_root,
            false, // force=false
            None,
            false,
            60,
        )
        .await;

        assert!(
            matches!(result, Err(learn_core::LearnError::Acquire(_))),
            "slug collision must return Err(LearnError::Acquire), got: {result:?}"
        );
        if let Err(learn_core::LearnError::Acquire(msg)) = result {
            assert!(
                msg.contains("squatter_xyz"),
                "error should name the conflicting video; got: {msg}"
            );
            assert!(
                msg.contains("learn forget"),
                "error should suggest learn forget; got: {msg}"
            );
        }
    }
}
