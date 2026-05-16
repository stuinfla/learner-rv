//! HTTP server for `learn ui` — serves the dashboard and REST API.

use axum::{
    extract::{Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{sse::Event, sse::KeepAlive, IntoResponse, Sse},
    routing::{get, post},
    Json, Router,
};
use camino::Utf8PathBuf;
use chrono::DateTime;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tower_http::cors::CorsLayer;

static UI_HTML: &str = include_str!("../ui/index.html");

// ── Shared state ──────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub kb_root: Utf8PathBuf,
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn build_router(kb_root: Utf8PathBuf) -> Router {
    let state = Arc::new(AppState { kb_root });
    Router::new()
        .route("/", get(serve_ui))
        .route("/api/health", get(health))
        .route("/api/topics", get(list_topics))
        .route("/api/status", get(status))
        .route("/api/ask", post(ask))
        .route("/api/ingest/progress", get(ingest_progress))
        .route("/api/seed/discover", post(seed_discover))
        .route("/api/seed/configure", post(seed_configure))
        .with_state(state)
        .layer(CorsLayer::permissive())
}

/// Start the HTTP server. Blocks until the process is killed.
pub async fn run(kb_root: Utf8PathBuf, port: u16) -> anyhow::Result<()> {
    let app = build_router(kb_root);
    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("learn-rv dashboard → http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Handlers ─────────────────────────────────────────────────────────────────

async fn serve_ui() -> impl IntoResponse {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        "text/html; charset=utf-8".parse().unwrap(),
    );
    (headers, UI_HTML)
}

async fn health() -> Json<Value> {
    Json(json!({"ok": true}))
}

#[derive(Serialize)]
struct TopicEntry {
    slug: String,
    video_count: usize,
    chunks: u64,
    size_kb: u64,
    updated_at: String,
}

async fn list_topics(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut topics: Vec<TopicEntry> = Vec::new();

    if let Ok(mut rd) = tokio::fs::read_dir(state.kb_root.as_std_path()).await {
        while let Ok(Some(entry)) = rd.next_entry().await {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) != Some("rvf") {
                continue;
            }
            let slug = match p.file_stem().and_then(|s| s.to_str()) {
                Some(s) if !s.is_empty() && !s.starts_with('_') => s.to_string(),
                _ => continue,
            };
            let meta = entry.metadata().await.ok();
            let size_kb = meta.as_ref().map(|m| m.len() / 1024).unwrap_or(0);
            let updated_at = meta
                .as_ref()
                .and_then(|m| m.modified().ok())
                .map(|t| {
                    let secs = t
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    DateTime::from_timestamp(secs as i64, 0)
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_default()
                })
                .unwrap_or_default();

            let (video_count, chunks) = read_topic_stats(&state.kb_root, &slug).await;
            topics.push(TopicEntry { slug, video_count, chunks, size_kb, updated_at });
        }
    }

    topics.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Json(json!({"topics": topics}))
}

async fn read_topic_stats(kb_root: &Utf8PathBuf, slug: &str) -> (usize, u64) {
    let manifest_path = kb_root.join(format!("{slug}.manifest.json"));
    if let Ok(bytes) = tokio::fs::read(&manifest_path).await {
        if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
            let videos = v["videos"].as_array().map(|a| a.len()).unwrap_or(0);
            let chunks = v["total_chunks"].as_u64().unwrap_or(0);
            return (videos, chunks);
        }
    }
    (0, 0)
}

async fn status(State(state): State<Arc<AppState>>) -> Json<Value> {
    let seed_addr: Option<String> = std::env::var("LEARN_SEED_ADDRESS").ok().or_else(|| {
        let path = dirs::config_dir()
            .unwrap_or_default()
            .join("learn-rs/config.json");
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| v["seed"]["address"].as_str().map(str::to_string))
            .filter(|s| !s.is_empty())
    });

    // Lightweight TCP probe instead of HTTP client dep
    let seed_connected = if let Some(ref addr) = seed_addr {
        let addr_with_port = if addr.contains(':') {
            addr.clone()
        } else {
            format!("{addr}:80")
        };
        tokio::time::timeout(
            Duration::from_millis(800),
            tokio::net::TcpStream::connect(&addr_with_port),
        )
        .await
        .ok()
        .and_then(|r| r.ok())
        .is_some()
    } else {
        false
    };

    Json(json!({
        "model": "all-MiniLM-L6-v2",
        "kb_root": state.kb_root.as_str(),
        "seed": { "connected": seed_connected, "ip": seed_addr }
    }))
}

#[derive(Deserialize)]
struct AskBody {
    question: String,
    topic: String,
}

async fn ask(
    State(state): State<Arc<AppState>>,
    Json(body): Json<AskBody>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let output = tokio::process::Command::new("learn")
        .args(["ask", &body.topic, &body.question, "--kb-root", state.kb_root.as_str()])
        .output()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))))?;

    if output.status.success() {
        let text = String::from_utf8_lossy(&output.stdout).to_string();
        let mut citations = Vec::new();
        let mut answer_lines = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('·') || trimmed.starts_with("  ·") {
                citations.push(trimmed.trim_start_matches(['·', ' ']).to_string());
            } else {
                answer_lines.push(line);
            }
        }
        Ok(Json(json!({
            "answer": answer_lines.join("\n").trim(),
            "citations": citations
        })))
    } else {
        let err = String::from_utf8_lossy(&output.stderr).to_string();
        Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": err}))))
    }
}

#[derive(Deserialize)]
struct IngestQuery {
    source: String,
    #[serde(default)]
    topic: String,
}

async fn ingest_progress(
    State(state): State<Arc<AppState>>,
    Query(q): Query<IngestQuery>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<String>(64);
    let kb_root  = state.kb_root.to_string();
    let source   = q.source.clone();
    let topic    = q.topic.clone();

    tokio::spawn(async move {
        let send = |msg: &str, level: &str, pct: u8, done: bool| {
            let _ = tx.try_send(
                json!({"message": msg, "level": level, "progress": pct, "done": done}).to_string(),
            );
        };

        send("Starting ingest pipeline…", "info", 2, false);

        let mut args = vec!["ingest", source.as_str(), "--kb-root", kb_root.as_str()];
        if !topic.is_empty() {
            args.extend(["--topic", topic.as_str()]);
        }

        let mut child = match tokio::process::Command::new("learn")
            .args(&args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                send(&format!("Failed to start: {e}"), "warn", 0, true);
                return;
            }
        };

        if let Some(stderr) = child.stderr.take() {
            use tokio::io::{AsyncBufReadExt, BufReader};
            let mut lines = BufReader::new(stderr).lines();
            let mut pct = 5u8;
            while let Ok(Some(line)) = lines.next_line().await {
                let level = if line.contains("error") || line.contains("Error") {
                    "warn"
                } else if line.contains("Done") || line.contains("indexed") {
                    "success"
                } else if line.contains("…") || line.contains("Embedding") || line.contains("Captioning") {
                    "active"
                } else {
                    "info"
                };
                pct = (pct + 7).min(95);
                send(&line, level, pct, false);
            }
        }

        match child.wait().await {
            Ok(s) if s.success() => {
                send("Ingest complete.", "success", 97, false);
                // Stream the Seed push if configured
                stream_seed_push(&send, &topic, &kb_root).await;
            }
            Ok(_)  => send("Finished with errors — check `learn doctor`.", "warn", 100, true),
            Err(e) => send(&format!("Process error: {e}"), "warn", 100, true),
        }
    });

    let stream = ReceiverStream::new(rx).map(|data| Ok(Event::default().data(data)));
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("ping"))
}

/// If a Seed is configured, push the topic to it and stream progress events.
async fn stream_seed_push(
    send: &impl Fn(&str, &str, u8, bool),
    topic: &str,
    kb_root: &str,
) {
    let seed_addr: Option<String> = std::env::var("LEARN_SEED_ADDRESS").ok().or_else(|| {
        let path = dirs::config_dir()
            .unwrap_or_default()
            .join("learn-rs/config.json");
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v["seed"]["address"].as_str().map(str::to_string))
            .filter(|s| !s.is_empty())
    });

    let Some(addr) = seed_addr else {
        send("Seed not configured — skipping push", "warn", 100, true);
        return;
    };

    // Derive the topic slug if not explicitly provided
    let topic_arg = if topic.is_empty() {
        // Without a slug we cannot push — auto_push in the CLI handles this case
        send(&format!("Stored locally · push with: learn push <topic> --seed {addr}"), "info", 100, true);
        return;
    } else {
        topic.to_string()
    };

    send(&format!("Pushing to Cognitum Seed {addr}…"), "active", 98, false);

    let result = tokio::process::Command::new("learn")
        .args(["push", &topic_arg, "--seed", &addr, "--kb-root", kb_root])
        .output()
        .await;

    match result {
        Ok(o) if o.status.success() => {
            send(&format!("Synced to Seed {addr}"), "success", 100, true);
        }
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            send(&format!("Push failed: {}", err.lines().next().unwrap_or("unknown error")), "warn", 100, true);
        }
        Err(e) => {
            send(&format!("Push error: {e}"), "warn", 100, true);
        }
    }
}

// ── Seed discovery & config ──────────────────────────────────────────────────

#[derive(Deserialize, Default)]
struct DiscoverBody {
    #[serde(default = "default_discover_timeout")]
    timeout_secs: u64,
}
fn default_discover_timeout() -> u64 { 3 }

async fn seed_discover(body: Option<Json<DiscoverBody>>) -> Json<Value> {
    let timeout = body.map(|Json(b)| b.timeout_secs).unwrap_or(3).clamp(1, 10);

    let task = tokio::task::spawn_blocking(move || -> Vec<String> {
        use mdns_sd::{ServiceDaemon, ServiceEvent};

        let Ok(daemon) = ServiceDaemon::new() else { return vec![]; };
        let Ok(receiver) = daemon.browse("_cognitum._tcp.local.") else { return vec![]; };

        let deadline = std::time::Instant::now() + Duration::from_secs(timeout);
        let mut found: Vec<String> = Vec::new();
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() { break; }
            match receiver.recv_timeout(remaining) {
                Ok(ServiceEvent::ServiceResolved(info)) => {
                    let addr = info
                        .get_addresses_v4()
                        .into_iter()
                        .next()
                        .map(|a| a.to_string())
                        .unwrap_or_else(|| info.get_hostname().trim_end_matches('.').to_owned());
                    if !found.contains(&addr) { found.push(addr); }
                }
                Ok(_) | Err(_) => {}
            }
        }
        found
    });

    let addrs = task.await.unwrap_or_default();
    Json(json!({ "found": addrs }))
}

#[derive(Deserialize)]
struct ConfigureBody {
    address: String,
    #[serde(default)]
    auto_push: bool,
}

async fn seed_configure(
    Json(body): Json<ConfigureBody>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let path = dirs::config_dir()
        .unwrap_or_default()
        .join("learn-rs/config.json");

    if let Some(dir) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))));
        }
    }

    // Preserve any unknown fields by reading-modify-writing.
    let mut current: Value = std::fs::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_else(|| json!({}));
    let seed = current
        .as_object_mut()
        .expect("just created as object")
        .entry("seed")
        .or_insert_with(|| json!({}));
    seed["address"] = json!(body.address);
    seed["auto_push"] = json!(body.auto_push);

    let bytes = serde_json::to_vec_pretty(&current)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))))?;
    std::fs::write(&path, &bytes)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))))?;

    Ok(Json(json!({"ok": true, "path": path.to_string_lossy()})))
}
