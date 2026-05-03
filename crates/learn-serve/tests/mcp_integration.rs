//! Integration test: start `learn serve` as a subprocess, send JSON-RPC
//! messages over stdin, assert the three tool names appear in `tools/list`.
//!
//! Marked `#[ignore]` because it requires the `learn` binary to be installed.
//! Run with: `cargo test -p learn-serve -- --ignored`

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Helper: spawn `learn serve <topic>` with `LEARN_KB_ROOT` set, write
/// newline-delimited JSON-RPC lines to stdin, close stdin, wait up to
/// `wait_ms` ms, and return the full stdout.
fn run_serve_with_messages(kb_root: &str, topic: &str, messages: &[&str], wait_ms: u64) -> String {
    let mut child = Command::new("learn")
        .args(["serve", topic])
        .env("LEARN_KB_ROOT", kb_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn `learn serve` — is the binary installed?");

    let stdin = child.stdin.as_mut().unwrap();
    for msg in messages {
        writeln!(stdin, "{msg}").unwrap();
    }
    drop(child.stdin.take());

    std::thread::sleep(Duration::from_millis(wait_ms));

    let output = child
        .wait_with_output()
        .expect("failed to wait for child process");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Assert that a running `learn serve` subprocess responds to the MCP
/// handshake and exposes the three expected tool names.
#[test]
#[ignore = "requires `learn` binary in PATH and a writable /tmp dir"]
fn mcp_tools_list_contains_three_tools() {
    let kb_root = "/tmp/verified-demo-kb";
    // Ensure minimal manifest exists so the server can open the topic.
    std::fs::create_dir_all(format!("{kb_root}/_meta")).unwrap();
    let manifest = serde_json::json!({
        "topic": "verified-demo",
        "videos": {
            "dQw4w9WgXcQ": {
                "video_id": "dQw4w9WgXcQ",
                "status": "indexed",
                "fetched_at": "2026-05-02T00:00:00Z",
                "indexed_at": "2026-05-02T00:00:00Z",
                "chunk_count": 1,
                "error": null
            }
        }
    });
    std::fs::write(
        format!("{kb_root}/_meta/verified-demo.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let mut child = Command::new("learn")
        .args(["serve", "verified-demo"])
        .env("LEARN_KB_ROOT", kb_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn `learn serve` — is the binary installed?");

    // Write the two MCP requests
    let stdin = child.stdin.as_mut().unwrap();
    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2024-11-05","capabilities":{{}},"clientInfo":{{"name":"test","version":"1.0"}}}}}}"#
    )
    .unwrap();
    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{{}}}}"#
    )
    .unwrap();
    drop(child.stdin.take()); // close stdin → server sees EOF and exits

    // Give the server up to 500 ms to respond
    std::thread::sleep(Duration::from_millis(500));

    let output = child
        .wait_with_output()
        .expect("failed to wait for child process");
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check tool names appear in the tools/list response
    assert!(
        stdout.contains("kb_query"),
        "tools/list must include kb_query; got: {stdout}"
    );
    assert!(
        stdout.contains("kb_synthesize"),
        "tools/list must include kb_synthesize; got: {stdout}"
    );
    assert!(
        stdout.contains("kb_list_videos"),
        "tools/list must include kb_list_videos; got: {stdout}"
    );
}

/// End-to-end wire test for `kb_query` and `kb_synthesize`.
///
/// Requires:
///   - `learn` binary installed (`cargo install --path crates/learn-cli`)
///   - `LEARN_KB_ROOT` pointing at a KB with a `verified-demo` topic that has
///     at least one indexed video (default: `~/Docs/KB`)
///   - `ANTHROPIC_API_KEY` set (for `kb_synthesize`)
#[test]
#[ignore = "requires `learn` binary, real KB data, and ANTHROPIC_API_KEY"]
fn mcp_integration_kb_query_kb_synthesize() {
    let kb_root = std::env::var("LEARN_KB_ROOT")
        .unwrap_or_else(|_| format!("{}/Docs/KB", std::env::var("HOME").unwrap_or_default()));
    let topic = "verified-demo";

    // ── Step 1: kb_query ──────────────────────────────────────────────────────
    let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#;
    let query = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"kb_query","arguments":{"question":"what is discussed?","k":3}}}"#;

    let stdout = run_serve_with_messages(&kb_root, topic, &[init, query], 4000);

    // Find the tools/call response for id=2
    let query_line = stdout
        .lines()
        .find(|l| l.contains("\"id\":2"))
        .unwrap_or_else(|| panic!("no response with id=2; full output: {stdout}"));

    let response: serde_json::Value =
        serde_json::from_str(query_line).expect("response must be valid JSON");

    let content_text = response["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("result.content[0].text missing; got: {response}"));

    let tool_result: serde_json::Value =
        serde_json::from_str(content_text).expect("content text must be valid JSON");

    let hits = tool_result["hits"]
        .as_array()
        .unwrap_or_else(|| panic!("kb_query response must have 'hits' array; got: {tool_result}"));

    assert!(
        !hits.is_empty(),
        "kb_query must return ≥1 hit; got empty array"
    );

    let first_hit = &hits[0];
    assert!(
        first_hit.get("video_id").is_some(),
        "hit must have 'video_id'; got: {first_hit}"
    );
    assert!(
        first_hit.get("start_seconds").is_some(),
        "hit must have 'start_seconds'; got: {first_hit}"
    );
    assert!(
        first_hit.get("text").is_some(),
        "hit must have 'text'; got: {first_hit}"
    );
    assert!(
        first_hit.get("score").is_some(),
        "hit must have 'score'; got: {first_hit}"
    );

    // ── Step 2: kb_synthesize using hits from Step 1 ─────────────────────────
    let hits_json = serde_json::to_string(hits).unwrap();
    let synth_args = format!(r#"{{"question":"summarize","hits":{hits_json}}}"#);
    let synth_call = format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{{"name":"kb_synthesize","arguments":{}}}}}"#,
        synth_args
    );

    let stdout2 = run_serve_with_messages(&kb_root, topic, &[init, &synth_call], 15000);

    let synth_line = stdout2
        .lines()
        .find(|l| l.contains("\"id\":3"))
        .unwrap_or_else(|| panic!("no response with id=3; full output: {stdout2}"));

    let synth_response: serde_json::Value =
        serde_json::from_str(synth_line).expect("synth response must be valid JSON");

    let synth_text = synth_response["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("synth result.content[0].text missing; got: {synth_response}"));

    let synth_result: serde_json::Value =
        serde_json::from_str(synth_text).expect("synth content text must be valid JSON");

    assert!(
        synth_result.get("answer").is_some(),
        "kb_synthesize response must have 'answer'; got: {synth_result}"
    );
    assert!(
        synth_result["answer"]
            .as_str()
            .map(|s| !s.is_empty())
            .unwrap_or(false),
        "kb_synthesize 'answer' must be a non-empty string; got: {synth_result}"
    );
    assert!(
        synth_result.get("citations").is_some(),
        "kb_synthesize response must have 'citations'; got: {synth_result}"
    );
    assert!(
        synth_result["citations"].is_array(),
        "kb_synthesize 'citations' must be an array; got: {synth_result}"
    );
}
