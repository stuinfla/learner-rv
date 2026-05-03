//! JSON-RPC 2.0 + MCP wire protocol handler.
//!
//! Reads newline-delimited JSON from stdin, dispatches to tool handlers,
//! writes newline-delimited JSON responses to stdout.
//!
//! MCP methods implemented:
//!   initialize, notifications/initialized, tools/list, tools/call
//!
//! Any unknown method returns a JSON-RPC MethodNotFound error.

use std::io::{BufRead, Write};

use camino::Utf8PathBuf;
use serde_json::{json, Value};

use crate::tools::{handle_kb_list_videos, handle_kb_query, handle_kb_synthesize};

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "learn-rv";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Configuration passed to [`run_server`].
pub struct ServerConfig {
    pub topic: String,
    pub kb_root: Utf8PathBuf,
}

/// Run the MCP server loop on stdin/stdout until EOF.
///
/// Each line on stdin is a JSON-RPC 2.0 request; each response is written
/// as a single JSON line to stdout.  Notifications (no `id`) receive no reply.
pub fn run_server(cfg: ServerConfig) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let msg: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                write_response(
                    &mut out,
                    &json_rpc_error(Value::Null, -32700, &format!("Parse error: {e}")),
                )?;
                continue;
            }
        };

        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        let method = msg
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .to_string();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        // Notifications have no id — send no reply.
        let is_notification = msg.get("id").is_none();

        let response = dispatch(&cfg, &method, params, id.clone());

        if !is_notification {
            write_response(&mut out, &response)?;
        }
    }

    Ok(())
}

fn dispatch(cfg: &ServerConfig, method: &str, params: Value, id: Value) -> Value {
    match method {
        "initialize" => handle_initialize(id),
        "notifications/initialized" => Value::Null, // handled via is_notification above
        "tools/list" => handle_tools_list(id),
        "tools/call" => handle_tools_call(cfg, params, id),
        _ => json_rpc_error(id, -32601, &format!("Method not found: {method}")),
    }
}

fn handle_initialize(id: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION
            },
            "capabilities": {
                "tools": {}
            }
        }
    })
}

fn handle_tools_list(id: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "tools": [
                {
                    "name": "kb_query",
                    "description": "Hybrid retrieval against the topic KB. Returns ranked chunks with timestamps.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "question": { "type": "string", "description": "The query to search for" },
                            "k": { "type": "number", "description": "Number of results (default 10)" }
                        },
                        "required": ["question"]
                    }
                },
                {
                    "name": "kb_synthesize",
                    "description": "Synthesize a cited answer from provided hits using Anthropic. Requires ANTHROPIC_API_KEY.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "question": { "type": "string", "description": "Question to answer" },
                            "hits": {
                                "type": "array",
                                "description": "Hits from kb_query",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "video_id": { "type": "string" },
                                        "start_seconds": { "type": "number" },
                                        "text": { "type": "string" },
                                        "score": { "type": "number" }
                                    }
                                }
                            }
                        },
                        "required": ["question", "hits"]
                    }
                },
                {
                    "name": "kb_list_videos",
                    "description": "List all videos indexed in this topic KB.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }
    })
}

fn handle_tools_call(cfg: &ServerConfig, params: Value, id: Value) -> Value {
    let name = params
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .to_string();
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

    let result = match name.as_str() {
        "kb_query" => handle_kb_query(cfg, &arguments),
        "kb_synthesize" => handle_kb_synthesize(cfg, &arguments),
        "kb_list_videos" => handle_kb_list_videos(cfg),
        unknown => Err(anyhow::anyhow!("Unknown tool: {unknown}")),
    };

    match result {
        Ok(content) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [{ "type": "text", "text": content }]
            }
        }),
        Err(e) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [{ "type": "text", "text": format!("error: {e}") }],
                "isError": true
            }
        }),
    }
}

fn json_rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message }
    })
}

fn write_response(out: &mut impl Write, response: &Value) -> anyhow::Result<()> {
    if *response == Value::Null {
        return Ok(());
    }
    let line = serde_json::to_string(response)?;
    writeln!(out, "{line}")?;
    out.flush()?;
    Ok(())
}
