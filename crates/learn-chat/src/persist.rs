//! JSONL persistence for chat sessions.
//!
//! Layout: `<kb_root>/_chat/<topic>/<session-id>.jsonl`
//!
//! Each line is a JSON object of one of two shapes:
//! - Header line: `{"type":"header","session_id":…,"topic":…,"started_at":…,"depth":…}`
//! - Turn line:   `{"type":"turn","role":…,"content":…,"citations":…,"timestamp":…,"hits":…}`
//!
//! Writes use a `.tmp` → rename pattern for atomicity.

use camino::{Utf8Path, Utf8PathBuf};
use learn_core::{LearnError, Result, Topic};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{ChatSession, SessionHeader, Turn};

// ── Line variants ─────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum JournalLine {
    Header(SessionHeader),
    Turn(Turn),
}

// ── Path helpers ──────────────────────────────────────────────────────────────

/// `<kb_root>/_chat/<topic>/`
pub fn chat_dir_for_topic(topic: &Topic, kb_root: &Utf8Path) -> Utf8PathBuf {
    kb_root.join("_chat").join(topic.as_str())
}

/// `<kb_root>/_chat/<topic>/<session_id>.jsonl`
pub fn session_path(session_id: Uuid, topic: &Topic, kb_root: &Utf8Path) -> Utf8PathBuf {
    chat_dir_for_topic(topic, kb_root).join(format!("{session_id}.jsonl"))
}

// ── Write operations ──────────────────────────────────────────────────────────

/// Write the session header as the first line of the JSONL file.
pub fn write_header(session: &ChatSession, kb_root: &Utf8Path) -> Result<()> {
    let dir = chat_dir_for_topic(&session.topic, kb_root);
    std::fs::create_dir_all(dir.as_std_path())?;

    let header = SessionHeader {
        session_id: session.id,
        topic: session.topic.clone(),
        started_at: session.started_at,
        depth: session.depth,
    };
    let line = JournalLine::Header(header);
    atomic_append(session.id, &session.topic, kb_root, &line)
}

/// Append a single turn atomically to the JSONL file.
pub fn append_turn(turn: &Turn, session_id: Uuid, topic: &Topic, kb_root: &Utf8Path) -> Result<()> {
    let line = JournalLine::Turn(turn.clone());
    atomic_append(session_id, topic, kb_root, &line)
}

/// Serialize `line` and append it atomically using a tmp-rename pattern.
///
/// We write to `<path>.tmp`, then `rename` — both in the same directory so
/// the rename is atomic even on macOS HFS+/APFS.
fn atomic_append<T: Serialize>(
    session_id: Uuid,
    topic: &Topic,
    kb_root: &Utf8Path,
    value: &T,
) -> Result<()> {
    use std::io::Write;

    let path = session_path(session_id, topic, kb_root);
    let tmp_path = path.with_extension("jsonl.tmp");

    // Serialize the new line.
    let new_line = serde_json::to_string(value).map_err(LearnError::Serde)? + "\n";

    // Read existing content (if any).
    let existing = if path.exists() {
        std::fs::read(path.as_std_path())?
    } else {
        Vec::new()
    };

    // Write existing + new line to tmp.
    let mut tmp = std::fs::File::create(tmp_path.as_std_path()).map_err(LearnError::Io)?;
    tmp.write_all(&existing).map_err(LearnError::Io)?;
    tmp.write_all(new_line.as_bytes()).map_err(LearnError::Io)?;
    tmp.flush().map_err(LearnError::Io)?;
    drop(tmp);

    // Atomic rename.
    std::fs::rename(tmp_path.as_std_path(), path.as_std_path()).map_err(LearnError::Io)?;

    Ok(())
}

// ── Read operations ───────────────────────────────────────────────────────────

/// Read back a session from its JSONL file and reconstruct a [`ChatSession`].
pub fn read_session(session_id: Uuid, topic: &Topic, kb_root: &Utf8Path) -> Result<ChatSession> {
    let path = session_path(session_id, topic, kb_root);
    let content = std::fs::read_to_string(path.as_std_path()).map_err(LearnError::Io)?;

    let mut header_opt: Option<SessionHeader> = None;
    let mut turns: Vec<Turn> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let entry: JournalLine = serde_json::from_str(trimmed).map_err(LearnError::Serde)?;
        match entry {
            JournalLine::Header(h) => header_opt = Some(h),
            JournalLine::Turn(t) => turns.push(t),
        }
    }

    let header = header_opt.ok_or_else(|| {
        LearnError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("session {session_id} JSONL has no header line"),
        ))
    })?;

    Ok(ChatSession {
        id: header.session_id,
        topic: header.topic,
        history: turns,
        started_at: header.started_at,
        depth: header.depth,
    })
}
