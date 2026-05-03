//! Append-only witness chain for MCP tool calls.
//!
//! Each `kb_query` and `kb_synthesize` invocation records an entry in
//! `<kb_root>/<topic>.mcp.witness.json` with a Blake3 hash over the
//! request+response content, chained from the previous entry's digest.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// One entry in the MCP witness chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpWitnessEntry {
    pub seq: u64,
    pub tool: String,
    pub request_hash: String,
    pub response_hash: String,
    pub called_at: i64,
    pub previous_digest: String,
    pub digest: String,
}

/// Append a witness entry for a tool call.
///
/// `request_json` and `response_json` are hashed; the chain is extended.
/// Writes atomically via tmp+rename. Silently no-ops on any I/O error
/// (witness chain is best-effort for MCP calls).
pub fn append_witness(witness_path: &Path, tool: &str, request_json: &str, response_json: &str) {
    if let Err(e) = try_append(witness_path, tool, request_json, response_json) {
        tracing::warn!("witness append failed (non-fatal): {e}");
    }
}

fn try_append(
    path: &Path,
    tool: &str,
    request_json: &str,
    response_json: &str,
) -> anyhow::Result<()> {
    let mut chain = load_chain(path);
    let seq = chain.len() as u64 + 1;
    let called_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let previous_digest = chain
        .last()
        .map(|e: &McpWitnessEntry| e.digest.clone())
        .unwrap_or_else(hex_zero);

    let req_hash = blake3_hex(request_json.as_bytes());
    let resp_hash = blake3_hex(response_json.as_bytes());

    let canonical = format!("{seq}{tool}{req_hash}{resp_hash}{called_at}{previous_digest}");
    let digest = blake3_hex(canonical.as_bytes());

    chain.push(McpWitnessEntry {
        seq,
        tool: tool.to_string(),
        request_hash: req_hash,
        response_hash: resp_hash,
        called_at,
        previous_digest,
        digest,
    });

    atomic_write(path, &serde_json::to_string_pretty(&chain)?)?;
    Ok(())
}

fn load_chain(path: &Path) -> Vec<McpWitnessEntry> {
    if !path.exists() {
        return Vec::new();
    }
    let data = std::fs::read_to_string(path).unwrap_or_default();
    serde_json::from_str(&data).unwrap_or_default()
}

fn atomic_write(path: &Path, content: &str) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp: PathBuf = path.with_extension("json.tmp");
    std::fs::write(&tmp, content.as_bytes())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn blake3_hex(data: &[u8]) -> String {
    let hash = blake3::hash(data);
    hash.to_hex().to_string()
}

fn hex_zero() -> String {
    "0".repeat(64)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn witness_chain_grows_and_links() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.witness.json");

        append_witness(&path, "kb_query", r#"{"question":"a"}"#, r#"{"hits":[]}"#);
        append_witness(&path, "kb_query", r#"{"question":"b"}"#, r#"{"hits":[]}"#);

        let chain: Vec<McpWitnessEntry> =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].seq, 1);
        assert_eq!(chain[1].seq, 2);
        assert_eq!(chain[0].previous_digest, hex_zero());
        assert_eq!(chain[1].previous_digest, chain[0].digest);
    }

    #[test]
    fn witness_digest_is_deterministic_for_same_input() {
        let a = blake3_hex(b"hello");
        let b = blake3_hex(b"hello");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }
}
