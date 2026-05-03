//! Integration test: start `learn serve` as a subprocess, send JSON-RPC
//! messages over stdin, assert the three tool names appear in `tools/list`.
//!
//! Marked `#[ignore]` because it requires the `learn` binary to be installed.
//! Run with: `cargo test -p learn-serve -- --ignored`

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

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
