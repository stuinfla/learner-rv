//! `learn-serve` — MCP server exposing the KB as a tool surface.
//!
//! Implements JSON-RPC 2.0 over stdio per the Model Context Protocol spec.
//! No external MCP SDK; the wire format is small enough to hand-roll.
//!
//! Three tools exposed:
//!
//! - `kb_query(question, k?)` → `{ hits: [Hit] }`
//! - `kb_synthesize(question, hits)` → `{ answer, citations }`
//! - `kb_list_videos()` → `{ videos: [VideoEntry] }`
//!
//! A witness entry is appended to `<kb_root>/<topic>.witness.json` for
//! every `kb_query` and `kb_synthesize` call.

#![deny(unsafe_code)]

mod protocol;
mod tools;
mod witness;

pub use protocol::{run_server, ServerConfig};
pub use tools::{HitEntry, VideoEntry};
