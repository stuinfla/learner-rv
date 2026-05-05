//! `learn push <topic> --seed <address>` — transfer a `.rvf` KB to a
//! Cognitum One Seed device over LAN.
//!
//! Requirements: Anthropic API key + Ruflo (RuVector ecosystem).
//! The Seed must expose the standard Cognitum HTTP API on port 80.

#![deny(unsafe_code)]

use camino::Utf8PathBuf;
use learn_core::LearnError;
use std::time::Duration;

/// Resolve the seed address: return the provided address directly, or discover
/// via mDNS if none is given.  When multiple Seeds are found and `seed_index`
/// is `Some(n)` (1-based), the n-th result is chosen without prompting.
///
/// Exposed for unit testing.
pub(crate) async fn resolve_seed_address(
    seed: Option<String>,
    seed_index: Option<usize>,
) -> learn_core::Result<String> {
    match seed {
        Some(addr) => Ok(addr),
        None => discover_via_mdns(seed_index).await,
    }
}

/// Construct the `.rvf` file path for a topic under `kb_root`.
///
/// Exposed for unit testing.
pub(crate) fn rvf_path_for_topic(topic: &str, kb_root: &Utf8PathBuf) -> Utf8PathBuf {
    kb_root.join(format!("{topic}.rvf"))
}

/// Browse for `_cognitum._tcp.local.` with a 5-second timeout.
/// Returns the address of the single device found, or errors on 0 or 2+.
/// When multiple Seeds are found and `seed_index` is `Some(n)` (1-based),
/// the n-th result is chosen without an interactive prompt.
async fn discover_via_mdns(seed_index: Option<usize>) -> learn_core::Result<String> {
    use mdns_sd::{ServiceDaemon, ServiceEvent};

    let daemon = ServiceDaemon::new()
        .map_err(|e| LearnError::Acquire(format!("failed to start mDNS daemon: {e}")))?;

    let receiver = daemon
        .browse("_cognitum._tcp.local.")
        .map_err(|e| LearnError::Acquire(format!("mDNS browse failed: {e}")))?;

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut found: Vec<String> = Vec::new();

    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match receiver.recv_timeout(remaining) {
            Ok(ServiceEvent::ServiceResolved(info)) => {
                // Use the first IPv4 address if available, else the hostname.
                let addr = info
                    .get_addresses_v4()
                    .into_iter()
                    .next()
                    .map(|a| a.to_string())
                    .unwrap_or_else(|| info.get_hostname().trim_end_matches('.').to_owned());
                found.push(addr);
            }
            Ok(_) => {}
            Err(_) => break, // timeout or channel closed
        }
    }

    // Stop the browse to free resources; ignore errors (best-effort cleanup).
    let _ = daemon.stop_browse("_cognitum._tcp.local.");

    match found.len() {
        0 => Err(LearnError::Acquire(
            "no Cognitum Seed found on the network — use `--seed <address>` to specify one manually.".into(),
        )),
        1 => Ok(found.remove(0)),
        _ => {
            // Multiple devices found.
            if let Some(idx) = seed_index {
                // Non-interactive: use the pre-chosen index.
                if idx == 0 || idx > found.len() {
                    return Err(LearnError::Acquire(format!(
                        "--seed-index {idx} out of range (found {} Seeds) — use `--seed <address>` to specify one manually.",
                        found.len()
                    )));
                }
                return Ok(found.remove(idx - 1));
            }
            // Interactive fallback.
            eprintln!("Multiple Cognitum Seeds found:");
            for (i, addr) in found.iter().enumerate() {
                eprintln!("  {}: {addr}", i + 1);
            }
            eprintln!("Tip: re-run with `--seed-index N` to skip this prompt.");
            eprint!("Enter number: ");
            let mut line = String::new();
            std::io::stdin()
                .read_line(&mut line)
                .map_err(LearnError::Io)?;
            let choice: usize = line.trim().parse().unwrap_or(0);
            if choice == 0 || choice > found.len() {
                Err(LearnError::Acquire(
                    "invalid selection — use `--seed <address>` to specify a device manually."
                        .into(),
                ))
            } else {
                Ok(found.remove(choice - 1))
            }
        }
    }
}

/// Push a topic KB (`.rvf`) to a Cognitum One Seed over LAN.
///
/// # Requirements
/// - Anthropic API key (for the broader learn-rv ecosystem).
/// - Ruflo installed (RuVector ecosystem hard requirement).
/// - The Seed device must be reachable and running the Cognitum HTTP API.
pub async fn run_push(
    topic: String,
    seed: Option<String>,
    seed_index: Option<usize>,
    kb_root: Utf8PathBuf,
) -> learn_core::Result<()> {
    // 1. Resolve seed address.
    let address = resolve_seed_address(seed, seed_index).await?;

    // 2. Find the .rvf file.
    let rvf_path = rvf_path_for_topic(&topic, &kb_root);
    if !rvf_path.exists() {
        return Err(LearnError::Acquire(format!(
            "topic '{topic}' not found at {rvf_path}\n  \
             Run `learn ingest <source> --topic {topic}` to build it first."
        )));
    }

    // 3. Read the file.
    let file_bytes = std::fs::read(rvf_path.as_std_path()).map_err(LearnError::Io)?;
    let file_size = file_bytes.len();
    let filename = format!("{topic}.rvf");

    println!("pushing {filename} to {address}…");

    // 4. POST to /api/v1/store/ingest as multipart.
    let client = reqwest::Client::new();
    let part = reqwest::multipart::Part::bytes(file_bytes)
        .file_name(filename.clone())
        .mime_str("application/octet-stream")
        .map_err(|e| LearnError::Acquire(format!("failed to build multipart part: {e}")))?;
    let form = reqwest::multipart::Form::new().part("file", part);

    let ingest_url = format!("http://{address}/api/v1/store/ingest");
    let response = client
        .post(&ingest_url)
        .multipart(form)
        .send()
        .await
        .map_err(|e| {
            let hint = if e.is_connect() {
                " — Seed not reachable or API not running (check Seed is on and has RVF API enabled)"
            } else if e.is_timeout() {
                " — connection timed out (check network or increase --timeout)"
            } else {
                ""
            };
            LearnError::Acquire(format!("HTTP POST to {ingest_url} failed: {e}{hint}"))
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(LearnError::Acquire(format!(
            "Seed rejected the upload (HTTP {status}): {body}"
        )));
    }

    println!("✓ pushed ({file_size} bytes) — verifying…");

    // 5. GET status.
    let status_url = format!("http://{address}/api/v1/store/status/{topic}");
    let status_response = client
        .get(&status_url)
        .send()
        .await
        .map_err(|e| LearnError::Acquire(format!("HTTP GET {status_url} failed: {e}")))?;

    let status_body = status_response
        .text()
        .await
        .map_err(|e| LearnError::Acquire(format!("failed to read status response: {e}")))?;

    println!("{status_body}");
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resolve_seed_address_uses_provided_address() {
        let result = resolve_seed_address(Some("192.168.1.42".to_string()), None).await;
        assert_eq!(result.unwrap(), "192.168.1.42");
    }

    #[tokio::test]
    async fn resolve_seed_address_uses_mdns_hostname() {
        let result = resolve_seed_address(Some("cognitum.local".to_string()), None).await;
        assert_eq!(result.unwrap(), "cognitum.local");
    }

    #[test]
    #[cfg(unix)]
    fn rvf_path_for_topic_constructs_correctly() {
        let kb_root = Utf8PathBuf::from("/home/user/Docs/KB");
        let path = rvf_path_for_topic("french-cooking", &kb_root);
        assert_eq!(
            path,
            Utf8PathBuf::from("/home/user/Docs/KB/french-cooking.rvf")
        );
    }

    #[test]
    #[cfg(unix)]
    fn rvf_path_for_topic_nested_root() {
        let kb_root = Utf8PathBuf::from("/tmp/test-kb");
        let path = rvf_path_for_topic("rust-programming", &kb_root);
        assert_eq!(path.as_str(), "/tmp/test-kb/rust-programming.rvf");
    }

    #[test]
    fn rvf_path_for_topic_joins_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let path = rvf_path_for_topic("my-topic", &kb_root);
        assert!(path.as_str().ends_with("my-topic.rvf"));
        assert!(path.starts_with(&kb_root));
    }

    #[tokio::test]
    async fn run_push_errors_clearly_when_rvf_missing() {
        let dir = tempfile::tempdir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).unwrap();
        let result = run_push(
            "nonexistent-topic".to_string(),
            Some("127.0.0.1".to_string()),
            None,
            kb_root,
        )
        .await;
        assert!(
            matches!(result, Err(learn_core::LearnError::Acquire(_))),
            "expected Err(LearnError::Acquire) for missing .rvf, got: {result:?}"
        );
        if let Err(learn_core::LearnError::Acquire(msg)) = result {
            assert!(
                msg.contains("nonexistent-topic"),
                "error should name the topic; got: {msg}"
            );
            assert!(
                msg.contains("learn ingest"),
                "error should suggest learn ingest; got: {msg}"
            );
        }
    }
}
