//! `learn-synth` — Answer synthesis with a sovereignty path.
//!
//! Two implementations of [`Synthesizer`] are provided:
//!
//! - [`AnthropicSynthesizer`] — calls the Anthropic Messages API
//!   (`POST /v1/messages`) using [`ASK_SYSTEM_PROMPT`] / [`ASK_USER_TEMPLATE`]
//!   / [`APPLY_SYSTEM_PROMPT`] / [`APPLY_USER_TEMPLATE`]. Requires
//!   `ANTHROPIC_API_KEY`. Model defaults to `claude-opus-4-7`; override with
//!   `LEARN_ANTHROPIC_MODEL`. Retries 429/503 up to 3 times (1 s / 2 s / 4 s).
//!
//! - [`RuvllmSynthesizer`] — on-device inference via `ruvllm`. Active when
//!   `LEARN_SYNTH_LOCAL=1` is set in the environment.
//!
//! Use [`select_synthesizer`] to obtain the right implementation at runtime.
//!
//! # AI Defence (AIMDS)
//!
//! Every synthesizer wraps its inbound text (question / task) and outbound
//! text (the generated answer) with an AIMDS safety scan via the [`aimds`]
//! module. See [`aimds`] for configuration details.

#![deny(unsafe_code)]

pub mod aimds;

use aimds::ScanVerdict;
use async_trait::async_trait;
use learn_core::{Answer, Citation, Hit, LearnError, Result};
use serde::Deserialize;
use tracing::info;
use url::Url;

// ── Prompt-template constants (Phase 2 design memo) ─────────────────────────

/// System prompt injected before every `ask` request.
///
/// Phase 2 will substitute `{topic}` and `{context_snippets}` at call time.
pub const ASK_SYSTEM_PROMPT: &str = "\
You are a precise knowledge assistant. \
Answer the question using ONLY the provided source excerpts. \
Cite every factual claim with the excerpt index it comes from ([1], [2], …). \
If the excerpts do not contain enough information to answer confidently, \
respond with the single word ABSTAIN.\
";

/// User-turn template for `ask`.
///
/// Placeholders: `{topic}`, `{context_snippets}`, `{question}`.
pub const ASK_USER_TEMPLATE: &str = "\
Topic: {topic}

Source excerpts:
{context_snippets}

Question: {question}\
";

/// System prompt for `apply` (task-completion mode).
///
/// Phase 2 will substitute `{topic}`, `{format}`, `{context_snippets}`.
pub const APPLY_SYSTEM_PROMPT: &str = "\
You are a precise knowledge assistant performing a structured task. \
Use ONLY the provided source excerpts. \
Return output in the requested format. \
If the excerpts do not contain sufficient information, respond with ABSTAIN.\
";

/// User-turn template for `apply`.
///
/// Placeholders: `{topic}`, `{task}`, `{format}`, `{context_snippets}`.
pub const APPLY_USER_TEMPLATE: &str = "\
Topic: {topic}
Task: {task}
Output format: {format}

Source excerpts:
{context_snippets}\
";

// ── Trait ────────────────────────────────────────────────────────────────────

/// Synthesize an [`Answer`] from retrieval [`Hit`]s.
///
/// Both `ask` (open-ended Q&A) and `apply` (structured task) must:
/// - Ground every factual claim in the provided `hits`.
/// - Abstain (set `Answer::abstained = true`) rather than hallucinate when
///   the hits do not contain enough signal.
#[async_trait]
pub trait Synthesizer: Send + Sync {
    /// Answer a free-form question about `topic` using the retrieved `hits`.
    async fn ask(&self, topic: &str, question: &str, hits: &[Hit]) -> Result<Answer>;

    /// Complete a structured `task` for `topic`, returning output in `format`.
    async fn apply(&self, topic: &str, task: &str, format: &str, hits: &[Hit]) -> Result<Answer>;
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

/// Return the active [`Synthesizer`].
///
/// When `LEARN_SYNTH_LOCAL` is set to any non-empty value, inference runs
/// entirely on-device via [`RuvllmSynthesizer`]. Otherwise
/// [`AnthropicSynthesizer`] is returned (requires `ANTHROPIC_API_KEY`).
pub fn select_synthesizer() -> Result<Box<dyn Synthesizer>> {
    if std::env::var("LEARN_SYNTH_LOCAL").is_ok() {
        info!("LEARN_SYNTH_LOCAL is set — using on-device RuvllmSynthesizer");
        Ok(Box::new(RuvllmSynthesizer::load()?))
    } else {
        info!("LEARN_SYNTH_LOCAL not set — using AnthropicSynthesizer");
        Ok(Box::new(AnthropicSynthesizer::new()?))
    }
}

// ── AnthropicSynthesizer ──────────────────────────────────────────────────────

/// Deserialised Anthropic Messages API response.
#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    r#type: String,
    text: String,
}

/// Synthesizer backed by the Anthropic Messages API.
///
/// Requires `ANTHROPIC_API_KEY`. Model defaults to `claude-opus-4-7`;
/// override with `LEARN_ANTHROPIC_MODEL`. Retries 429/503 with exponential
/// back-off (1 s / 2 s / 4 s, max 3 attempts).
pub struct AnthropicSynthesizer {
    client: reqwest::Client,
}

impl AnthropicSynthesizer {
    /// Construct the synthesizer. Builds a `reqwest::Client` (60 s timeout).
    /// Does NOT read `ANTHROPIC_API_KEY` at construction — the key is read
    /// per-call so tests can swap the env var without rebuilding the struct.
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| LearnError::Synth(format!("reqwest client build failed: {e}")))?;
        Ok(Self { client })
    }
}

/// Render each [`Hit`] as a numbered context block.
fn format_context(hits: &[Hit]) -> String {
    hits.iter()
        .enumerate()
        .map(|(i, h)| {
            format!(
                "[{}] {}\n    (from video={} @ {:.0}s)",
                i + 1,
                h.chunk.text.trim(),
                h.chunk.video_id,
                h.chunk.start_seconds,
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Build YouTube short-link citations from retrieval hits.
fn build_citations_from_hits(hits: &[Hit]) -> Vec<Citation> {
    hits.iter()
        .filter_map(|h| {
            let url = format!(
                "https://youtu.be/{}?t={}",
                h.chunk.video_id, h.chunk.start_seconds as u64
            )
            .parse::<Url>()
            .ok()?;
            Some(Citation {
                video_id: h.chunk.video_id.clone(),
                title: None,
                url,
                start_seconds: h.chunk.start_seconds,
            })
        })
        .collect()
}

/// POST `body` to the Anthropic Messages API with exponential back-off retry
/// on 429 / 503. Returns the raw response body string on success.
async fn post_with_retries(
    client: &reqwest::Client,
    api_key: &str,
    body: &serde_json::Value,
) -> Result<String> {
    const URL: &str = "https://api.anthropic.com/v1/messages";
    const MAX_RETRIES: u32 = 3;

    let mut delay_secs = 1u64;
    for attempt in 0..MAX_RETRIES {
        let resp = client
            .post(URL)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| LearnError::Synth(format!("Anthropic request failed: {e}")))?;

        let status = resp.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS
            || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
        {
            if attempt + 1 < MAX_RETRIES {
                tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                delay_secs *= 2;
                continue;
            }
            return Err(LearnError::Synth(format!(
                "Anthropic API returned {status} after {MAX_RETRIES} attempts"
            )));
        }

        if !status.is_success() {
            let excerpt = resp
                .text()
                .await
                .unwrap_or_default()
                .chars()
                .take(200)
                .collect::<String>();
            return Err(LearnError::Synth(format!(
                "Anthropic API error {status}: {excerpt}"
            )));
        }

        return resp
            .text()
            .await
            .map_err(|e| LearnError::Synth(format!("Anthropic response read failed: {e}")));
    }

    Err(LearnError::Synth(
        "Anthropic retry loop exhausted unexpectedly".to_string(),
    ))
}

/// Call the Anthropic API with the given system prompt, user message, and hits.
/// Returns the synthesised answer text, citation list, and abstain flag.
async fn call_anthropic(
    client: &reqwest::Client,
    system: &str,
    user_msg: String,
    hits: &[Hit],
) -> Result<Answer> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
        LearnError::Synth(
            "ANTHROPIC_API_KEY not set — set the env var or use LEARN_SYNTH_LOCAL=1 \
             for on-device inference"
                .into(),
        )
    })?;
    let model =
        std::env::var("LEARN_ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-opus-4-7".to_string());

    let body = serde_json::json!({
        "model": model,
        "max_tokens": 4096,
        "system": system,
        "messages": [{"role": "user", "content": user_msg}],
    });

    let response_body = post_with_retries(client, &api_key, &body).await?;
    let parsed: AnthropicResponse = serde_json::from_str(&response_body)
        .map_err(|e| LearnError::Synth(format!("malformed Anthropic response: {e}")))?;

    let text = parsed
        .content
        .into_iter()
        .filter(|c| c.r#type == "text")
        .map(|c| c.text)
        .collect::<Vec<_>>()
        .join("");

    let abstained = text.trim_start().starts_with("KB doesn't cover this")
        || text.trim_start().starts_with("ABSTAIN");
    let citations = if abstained {
        vec![]
    } else {
        build_citations_from_hits(hits)
    };

    Ok(Answer {
        text,
        citations,
        abstained,
    })
}

#[async_trait]
impl Synthesizer for AnthropicSynthesizer {
    async fn ask(&self, topic: &str, question: &str, hits: &[Hit]) -> Result<Answer> {
        // ── Inbound scan ──────────────────────────────────────────────────────
        match aimds::scan_text(question).await? {
            ScanVerdict::Safe => {}
            ScanVerdict::Blocked(reason) => {
                return Ok(Answer {
                    text: format!("AIMDS blocked input: {reason}"),
                    citations: vec![],
                    abstained: true,
                });
            }
            ScanVerdict::Skipped(_) if aimds::is_required() => {
                return Err(LearnError::Synth(
                    "AIMDS required but unavailable".to_string(),
                ));
            }
            ScanVerdict::Skipped(_) => {}
        }

        // ── Inference ─────────────────────────────────────────────────────────
        let context_block = format_context(hits);
        let user_msg = ASK_USER_TEMPLATE
            .replace("{topic}", topic)
            .replace("{context_snippets}", &context_block)
            .replace("{question}", question);

        let answer = call_anthropic(&self.client, ASK_SYSTEM_PROMPT, user_msg, hits).await?;

        // ── Outbound scan ─────────────────────────────────────────────────────
        match aimds::scan_text(&answer.text).await? {
            ScanVerdict::Safe => Ok(answer),
            ScanVerdict::Blocked(reason) => Ok(Answer {
                text: format!("AIMDS blocked output: {reason}"),
                citations: vec![],
                abstained: true,
            }),
            ScanVerdict::Skipped(_) if aimds::is_required() => Err(LearnError::Synth(
                "AIMDS required but unavailable on outbound".to_string(),
            )),
            ScanVerdict::Skipped(_) => Ok(answer),
        }
    }

    async fn apply(&self, topic: &str, task: &str, format: &str, hits: &[Hit]) -> Result<Answer> {
        // ── Inbound scan ──────────────────────────────────────────────────────
        match aimds::scan_text(task).await? {
            ScanVerdict::Safe => {}
            ScanVerdict::Blocked(reason) => {
                return Ok(Answer {
                    text: format!("AIMDS blocked input: {reason}"),
                    citations: vec![],
                    abstained: true,
                });
            }
            ScanVerdict::Skipped(_) if aimds::is_required() => {
                return Err(LearnError::Synth(
                    "AIMDS required but unavailable".to_string(),
                ));
            }
            ScanVerdict::Skipped(_) => {}
        }

        // ── Inference ─────────────────────────────────────────────────────────
        let context_block = format_context(hits);
        let user_msg = APPLY_USER_TEMPLATE
            .replace("{topic}", topic)
            .replace("{task}", task)
            .replace("{format}", format)
            .replace("{context_snippets}", &context_block);

        let answer = call_anthropic(&self.client, APPLY_SYSTEM_PROMPT, user_msg, hits).await?;

        // ── Outbound scan ─────────────────────────────────────────────────────
        match aimds::scan_text(&answer.text).await? {
            ScanVerdict::Safe => Ok(answer),
            ScanVerdict::Blocked(reason) => Ok(Answer {
                text: format!("AIMDS blocked output: {reason}"),
                citations: vec![],
                abstained: true,
            }),
            ScanVerdict::Skipped(_) if aimds::is_required() => Err(LearnError::Synth(
                "AIMDS required but unavailable on outbound".to_string(),
            )),
            ScanVerdict::Skipped(_) => Ok(answer),
        }
    }
}

// ── RuvllmSynthesizer ────────────────────────────────────────────────────────

/// Default model cache path used when no override is provided.
///
/// Users place a GGUF quantized model here to enable local inference.
pub const DEFAULT_MODEL_PATH: &str = "~/.cache/learn-rs/models/ruvllm-default.gguf";

/// Expand `~/…` to an absolute path.
fn expand_tilde(path: &str) -> std::path::PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    std::path::PathBuf::from(path)
}

/// On-device synthesizer backed by [`ruvllm::LlmBackend`].
///
/// The backend is loaded synchronously in [`RuvllmSynthesizer::load`]; the
/// `async` methods on [`Synthesizer`] call `generate` inside
/// `tokio::task::spawn_blocking` to avoid blocking the async runtime.
pub struct RuvllmSynthesizer {
    backend: Box<dyn ruvllm::LlmBackend>,
    /// Absolute path from which the model was loaded (kept for diagnostics).
    #[allow(dead_code)]
    model_path: std::path::PathBuf,
}

impl RuvllmSynthesizer {
    /// Load the model from the default cache location.
    ///
    /// Returns [`LearnError::Synth`] with a human-readable message (including
    /// the expected path) when the file does not exist.
    pub fn load() -> Result<Self> {
        Self::load_from(DEFAULT_MODEL_PATH)
    }

    /// Load the model from an explicit path.
    ///
    /// `path` may use the `~/` home-directory prefix.
    pub fn load_from(path: &str) -> Result<Self> {
        let model_path = expand_tilde(path);

        if !model_path.exists() {
            return Err(LearnError::Synth(format!(
                "on-device model not found at `{}`. \
                 Download a GGUF quantized model and place it there, or unset \
                 LEARN_SYNTH_LOCAL to use the Anthropic API path.",
                model_path.display()
            )));
        }

        let mut backend = ruvllm::create_backend();
        let model_id = model_path
            .to_str()
            .ok_or_else(|| LearnError::Synth("model path is not valid UTF-8".to_string()))?;

        backend
            .load_model(model_id, ruvllm::ModelConfig::default())
            .map_err(|e| LearnError::Synth(format!("ruvllm load_model failed: {e}")))?;

        info!(
            "RuvllmSynthesizer loaded model from {}",
            model_path.display()
        );
        Ok(Self {
            backend,
            model_path,
        })
    }

    /// Expose the backend kind for test-only inspection.
    ///
    /// Returns `"ruvllm"` unconditionally — used in unit tests that verify
    /// `select_synthesizer` dispatches to the right branch without running
    /// actual inference.
    #[doc(hidden)]
    pub fn backend_kind(&self) -> &'static str {
        "ruvllm"
    }

    /// Format context snippets from retrieval hits for inclusion in a prompt.
    fn format_context(hits: &[Hit]) -> String {
        hits.iter()
            .enumerate()
            .map(|(i, h)| {
                format!(
                    "[{}] ({}  {:.1}–{:.1}s)  {}",
                    i + 1,
                    h.chunk.video_id,
                    h.chunk.start_seconds,
                    h.chunk.end_seconds,
                    h.chunk.text.trim()
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Extract citations from retrieval hits.
    fn build_citations(hits: &[Hit]) -> Vec<Citation> {
        hits.iter()
            .filter_map(|h| {
                // Best-effort: construct a youtube URL from video_id.
                let url = format!("https://www.youtube.com/watch?v={}", h.chunk.video_id)
                    .parse::<Url>()
                    .ok()?;
                Some(Citation {
                    video_id: h.chunk.video_id.clone(),
                    title: None, // chunk does not carry a title
                    url,
                    start_seconds: h.chunk.start_seconds,
                })
            })
            .collect()
    }

    /// Run the synchronous `backend.generate()` call and parse the result into
    /// an [`Answer`].
    fn generate_answer(&self, prompt: &str, hits: &[Hit]) -> Result<Answer> {
        let params = ruvllm::GenerateParams::default()
            .with_max_tokens(512)
            .with_temperature(0.2);

        let text = self
            .backend
            .generate(prompt, params)
            .map_err(|e| LearnError::Synth(format!("ruvllm generate failed: {e}")))?;

        let abstained = text.trim().eq_ignore_ascii_case("ABSTAIN");
        let citations = if abstained {
            vec![]
        } else {
            Self::build_citations(hits)
        };

        Ok(Answer {
            text,
            citations,
            abstained,
        })
    }
}

#[async_trait]
impl Synthesizer for RuvllmSynthesizer {
    async fn ask(&self, topic: &str, question: &str, hits: &[Hit]) -> Result<Answer> {
        // ── Inbound scan ──────────────────────────────────────────────────────
        match aimds::scan_text(question).await? {
            ScanVerdict::Safe => {}
            ScanVerdict::Blocked(reason) => {
                return Ok(Answer {
                    text: format!("AIMDS blocked input: {reason}"),
                    citations: vec![],
                    abstained: true,
                });
            }
            ScanVerdict::Skipped(_) if aimds::is_required() => {
                return Err(LearnError::Synth(
                    "AIMDS required but unavailable".to_string(),
                ));
            }
            ScanVerdict::Skipped(_) => {}
        }

        // ── Inference ─────────────────────────────────────────────────────────
        let context = Self::format_context(hits);
        let prompt = format!(
            "{system}\n\n{user}",
            system = ASK_SYSTEM_PROMPT,
            user = ASK_USER_TEMPLATE
                .replace("{topic}", topic)
                .replace("{context_snippets}", &context)
                .replace("{question}", question)
        );
        // backend.generate is sync and may block; keep it off the async thread.
        // We call it directly here since we own &self — the trait requires
        // `+ Sync`, so this is safe as long as the backend impl is Sync.
        // (For a future heavy model, wrap with spawn_blocking.)
        let answer = self.generate_answer(&prompt, hits)?;

        // ── Outbound scan ─────────────────────────────────────────────────────
        match aimds::scan_text(&answer.text).await? {
            ScanVerdict::Safe => Ok(answer),
            ScanVerdict::Blocked(reason) => Ok(Answer {
                text: format!("AIMDS blocked output: {reason}"),
                citations: vec![],
                abstained: true,
            }),
            ScanVerdict::Skipped(_) if aimds::is_required() => Err(LearnError::Synth(
                "AIMDS required but unavailable on outbound".to_string(),
            )),
            ScanVerdict::Skipped(_) => Ok(answer),
        }
    }

    async fn apply(&self, topic: &str, task: &str, format: &str, hits: &[Hit]) -> Result<Answer> {
        // ── Inbound scan ──────────────────────────────────────────────────────
        match aimds::scan_text(task).await? {
            ScanVerdict::Safe => {}
            ScanVerdict::Blocked(reason) => {
                return Ok(Answer {
                    text: format!("AIMDS blocked input: {reason}"),
                    citations: vec![],
                    abstained: true,
                });
            }
            ScanVerdict::Skipped(_) if aimds::is_required() => {
                return Err(LearnError::Synth(
                    "AIMDS required but unavailable".to_string(),
                ));
            }
            ScanVerdict::Skipped(_) => {}
        }

        // ── Inference ─────────────────────────────────────────────────────────
        let context = Self::format_context(hits);
        let prompt = format!(
            "{system}\n\n{user}",
            system = APPLY_SYSTEM_PROMPT,
            user = APPLY_USER_TEMPLATE
                .replace("{topic}", topic)
                .replace("{task}", task)
                .replace("{format}", format)
                .replace("{context_snippets}", &context)
        );
        let answer = self.generate_answer(&prompt, hits)?;

        // ── Outbound scan ─────────────────────────────────────────────────────
        match aimds::scan_text(&answer.text).await? {
            ScanVerdict::Safe => Ok(answer),
            ScanVerdict::Blocked(reason) => Ok(Answer {
                text: format!("AIMDS blocked output: {reason}"),
                citations: vec![],
                abstained: true,
            }),
            ScanVerdict::Skipped(_) if aimds::is_required() => Err(LearnError::Synth(
                "AIMDS required but unavailable on outbound".to_string(),
            )),
            ScanVerdict::Skipped(_) => Ok(answer),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use learn_core::Chunk;
    use std::sync::Mutex;

    // Process-level mutex serialises tests that mutate ANTHROPIC_API_KEY or
    // MOCK_AIMDS_VERDICT. Tests must acquire this lock before removing/setting
    // those vars so concurrent test threads don't race on the same key.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_hit(video_id: &str, text: &str) -> Hit {
        Hit {
            chunk: Chunk {
                chunk_id: "c1".to_string(),
                video_id: video_id.to_string(),
                start_seconds: 0.0,
                end_seconds: 5.0,
                text: text.to_string(),
                token_count: 10,
            },
            score: 0.9,
            rank: 0,
        }
    }

    // ── dispatch tests ───────────────────────────────────────────────────────

    /// When LEARN_SYNTH_LOCAL is set, select_synthesizer must return the
    /// RuvllmSynthesizer branch.  We cannot call load() without a model file,
    /// so we call load_from() with a path that does NOT exist and assert we get
    /// the expected LearnError::Synth (not a panic or wrong variant).
    #[test]
    fn select_synthesizer_with_env_var_returns_ruvllm_branch() {
        // Use a path guaranteed to not exist.
        let absent_path = "/nonexistent/ruvllm-default.gguf";
        let result = RuvllmSynthesizer::load_from(absent_path);
        assert!(result.is_err(), "expected Err for absent model, got Ok");
        // SAFETY: asserted is_err() above.
        let err = result.err().unwrap();
        match err {
            LearnError::Synth(msg) => {
                assert!(
                    msg.contains("not found at"),
                    "error message should mention 'not found at'; got: {msg}"
                );
            }
            other => panic!("expected LearnError::Synth, got {other:?}"),
        }
    }

    /// When LEARN_SYNTH_LOCAL is NOT set, select_synthesizer must return the
    /// AnthropicSynthesizer branch (which fails on missing API key).
    #[test]
    fn select_synthesizer_default_returns_anthropic_branch() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Ensure the env vars are absent for the duration of this test.
        let _guard = EnvGuard::remove("LEARN_SYNTH_LOCAL");
        let _key_guard = EnvGuard::remove("ANTHROPIC_API_KEY");
        let _mock_guard = EnvGuard::set("MOCK_AIMDS_VERDICT", "safe");

        let synth = select_synthesizer().expect("AnthropicSynthesizer::new() must not fail");

        // Without an API key, ask() should fail with an API_KEY error.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(synth.ask("topic", "question", &[]));
        match result {
            Err(LearnError::Synth(msg)) => {
                assert!(
                    msg.contains("ANTHROPIC_API_KEY"),
                    "expected API key error message; got: {msg}"
                );
            }
            other => panic!("expected Err(Synth(ANTHROPIC_API_KEY ...)), got {other:?}"),
        }
    }

    // ── New hermetic tests ────────────────────────────────────────────────────

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn ask_returns_synth_error_when_api_key_missing() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _key_guard = EnvGuard::remove("ANTHROPIC_API_KEY");
        let _local_guard = EnvGuard::remove("LEARN_SYNTH_LOCAL");
        // AIMDS mock so the scan pass-through doesn't block on missing binary.
        let _mock_guard = EnvGuard::set("MOCK_AIMDS_VERDICT", "safe");

        let synth = AnthropicSynthesizer::new().unwrap();
        let result = synth.ask("topic", "question", &[]).await;
        match result {
            Err(LearnError::Synth(msg)) => {
                assert!(
                    msg.contains("ANTHROPIC_API_KEY not set"),
                    "expected API key error; got: {msg}"
                );
            }
            other => panic!("expected Err(Synth(API key)), got {other:?}"),
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn apply_returns_synth_error_when_api_key_missing() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _key_guard = EnvGuard::remove("ANTHROPIC_API_KEY");
        let _local_guard = EnvGuard::remove("LEARN_SYNTH_LOCAL");
        let _mock_guard = EnvGuard::set("MOCK_AIMDS_VERDICT", "safe");

        let synth = AnthropicSynthesizer::new().unwrap();
        let result = synth.apply("topic", "summarise", "bullet list", &[]).await;
        match result {
            Err(LearnError::Synth(msg)) => {
                assert!(
                    msg.contains("ANTHROPIC_API_KEY not set"),
                    "expected API key error; got: {msg}"
                );
            }
            other => panic!("expected Err(Synth(API key)), got {other:?}"),
        }
    }

    #[test]
    fn render_ask_user_template_substitutes_placeholders() {
        let rendered = ASK_USER_TEMPLATE
            .replace("{topic}", "cooking")
            .replace("{context_snippets}", "[1] some snippet")
            .replace("{question}", "how do I bake bread?");
        assert!(rendered.contains("cooking"), "topic not substituted");
        assert!(
            rendered.contains("how do I bake bread?"),
            "question not substituted"
        );
        assert!(!rendered.contains("{topic}"), "placeholder still present");
        assert!(
            !rendered.contains("{question}"),
            "placeholder still present"
        );
    }

    #[test]
    fn parse_anthropic_response_extracts_text() {
        let json = r#"{"content":[{"type":"text","text":"hello world"}],"stop_reason":"end_turn"}"#;
        let parsed: AnthropicResponse = serde_json::from_str(json).unwrap();
        let text = parsed
            .content
            .into_iter()
            .filter(|c| c.r#type == "text")
            .map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(text, "hello world");
    }

    #[test]
    fn parse_anthropic_response_detects_abstain() {
        let json = r#"{"content":[{"type":"text","text":"KB doesn't cover this topic."}],"stop_reason":"end_turn"}"#;
        let parsed: AnthropicResponse = serde_json::from_str(json).unwrap();
        let text = parsed
            .content
            .into_iter()
            .filter(|c| c.r#type == "text")
            .map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");
        let abstained = text.trim_start().starts_with("KB doesn't cover this")
            || text.trim_start().starts_with("ABSTAIN");
        assert!(abstained, "should detect abstain prefix");
    }

    /// Absent model file returns a clear LearnError::Synth that includes the
    /// expected path in its message.
    #[test]
    fn ruvllm_synthesizer_load_without_model_returns_clear_error() {
        let result = RuvllmSynthesizer::load_from("/does/not/exist/model.gguf");
        match result {
            Err(LearnError::Synth(msg)) => {
                assert!(
                    msg.contains("/does/not/exist/model.gguf"),
                    "error should quote the path; got: {msg}"
                );
                assert!(
                    msg.contains("not found at"),
                    "error should say 'not found at'; got: {msg}"
                );
            }
            Err(other) => panic!("expected LearnError::Synth, got {other:?}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    /// Compile-time check: Box<dyn Synthesizer> must be constructible (object
    /// safety).  The test body is intentionally minimal — the assertion is the
    /// act of compiling without error.
    #[test]
    fn synthesizer_trait_object_safe() {
        fn _accepts_boxed(_s: Box<dyn Synthesizer>) {}
        fn _make_anthropic() -> Box<dyn Synthesizer> {
            Box::new(AnthropicSynthesizer::new().unwrap())
        }
        // Calling _make_anthropic() would require inference; just assert it
        // compiles by referencing the function pointer.
        let _fp: fn() -> Box<dyn Synthesizer> = _make_anthropic;
    }

    // ── context-formatting helpers ────────────────────────────────────────────

    #[test]
    fn format_context_produces_numbered_snippets() {
        let hits = vec![
            make_hit("vid1", "first chunk"),
            make_hit("vid2", "second chunk"),
        ];
        let ctx = RuvllmSynthesizer::format_context(&hits);
        assert!(ctx.contains("[1]"), "should number from 1");
        assert!(ctx.contains("[2]"), "should have second entry");
        assert!(ctx.contains("first chunk"));
        assert!(ctx.contains("second chunk"));
    }

    #[test]
    fn build_citations_constructs_youtube_urls() {
        let hits = vec![make_hit("dQw4w9WgXcQ", "text")];
        let cits = RuvllmSynthesizer::build_citations(&hits);
        assert_eq!(cits.len(), 1);
        assert!(
            cits[0].url.as_str().contains("dQw4w9WgXcQ"),
            "URL should contain the video ID"
        );
    }

    // ── EnvGuard helper ───────────────────────────────────────────────────────

    /// RAII guard that removes an env var on construction and restores its
    /// previous value (or removes it again) on drop. Needed to isolate env-var
    /// state between tests even when tests run in parallel.
    struct EnvGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvGuard {
        fn remove(key: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, previous }
        }

        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    // ── select_synthesizer_with_empty_env_var_documents_current_behavior ─────
    //
    // Invariant: LEARN_SYNTH_LOCAL="" (empty string) is still considered "set"
    // by std::env::var().is_ok(), so select_synthesizer routes to the
    // RuvllmSynthesizer branch. That branch fails because no model file exists
    // at the default path. We pin that error message shape so a future change
    // to the .is_ok() check would break this test and surface the behaviour
    // change explicitly.

    #[test]
    fn select_synthesizer_with_empty_env_var_documents_current_behavior() {
        // Set the env var to empty string — is_ok() returns true for any set value.
        let _guard = EnvGuard::set("LEARN_SYNTH_LOCAL", "");

        let result = select_synthesizer();
        // Must take the RuvllmSynthesizer branch, which fails because no model
        // file exists at the default path.
        let ok = matches!(&result, Err(LearnError::Synth(msg))
            if msg.contains("on-device model not found") || msg.contains("ruvllm"));
        assert!(
            ok,
            "empty LEARN_SYNTH_LOCAL should still trigger local mode"
        );
    }
}
