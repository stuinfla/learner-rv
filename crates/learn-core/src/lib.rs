//! Core pipeline types for learn-rs.
//!
//! Every stage emits and consumes one of these shapes. The CLI wires them
//! into a pipeline; the crates that follow each take a typed input and
//! produce a typed output. No ad-hoc tuples, no stringly-typed payloads.

use camino::Utf8PathBuf;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use url::Url;

/// Canonical topic identifier — a slug derived from a human label.
///
/// Each topic owns one `.rvf` file at `<kb_root>/<slug>.rvf`. Different topics
/// are fully isolated: separate files, separate HNSW indices, no shared
/// metadata. Re-ingesting against the same topic appends to the existing file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Topic(String);

impl Topic {
    pub const MAX_LEN: usize = 40;

    /// Build a topic from a human label. Lowercases, replaces runs of
    /// non-alphanumerics with `-`, strips leading/trailing `-`, caps at 40
    /// chars without breaking inside a word boundary, errors on empty.
    pub fn new(input: &str) -> Result<Self> {
        let lower = input.trim().to_lowercase();
        let mut out = String::with_capacity(lower.len());
        let mut prev_dash = true; // suppress leading -
        for ch in lower.chars() {
            if ch.is_ascii_alphanumeric() {
                out.push(ch);
                prev_dash = false;
            } else if !prev_dash {
                out.push('-');
                prev_dash = true;
            }
        }
        while out.ends_with('-') {
            out.pop();
        }
        if out.len() > Self::MAX_LEN {
            // trim at last `-` within bounds, else hard cap
            let bound = &out[..Self::MAX_LEN];
            let cut = bound.rfind('-').unwrap_or(Self::MAX_LEN);
            out.truncate(cut);
            while out.ends_with('-') {
                out.pop();
            }
        }
        if out.is_empty() {
            return Err(LearnError::Topic(format!(
                "topic name {input:?} normalizes to empty"
            )));
        }
        Ok(Topic(out))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Path to the topic's primary `.rvf` file under `kb_root`.
    pub fn rvf_path(&self, kb_root: &camino::Utf8Path) -> Utf8PathBuf {
        kb_root.join(format!("{}.rvf", self.0))
    }

    /// Path to the topic's raw-source cache directory.
    pub fn raw_dir(&self, kb_root: &camino::Utf8Path) -> Utf8PathBuf {
        kb_root.join("_raw").join(&self.0)
    }

    /// Path to the topic's manifest (ingestion state).
    pub fn manifest_path(&self, kb_root: &camino::Utf8Path) -> Utf8PathBuf {
        kb_root.join("_meta").join(format!("{}.json", self.0))
    }
}

impl std::fmt::Display for Topic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Error)]
pub enum LearnError {
    #[error("acquire failed: {0}")]
    Acquire(String),
    #[error("transcribe failed: {0}")]
    Transcribe(String),
    #[error("chunk failed: {0}")]
    Chunk(String),
    #[error("embed failed: {0}")]
    Embed(String),
    #[error("index failed: {0}")]
    Index(String),
    #[error("retrieve failed: {0}")]
    Retrieve(String),
    #[error("synth failed: {0}")]
    Synth(String),
    #[error("apply failed: {0}")]
    Apply(String),
    #[error("graph failed: {0}")]
    Graph(String),
    #[error("invalid topic: {0}")]
    Topic(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, LearnError>;

/// A canonical reference to one source video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoRef {
    pub video_id: String,
    pub url: Url,
    pub title: Option<String>,
    pub channel: Option<String>,
    pub channel_id: Option<String>,
    pub duration_seconds: Option<f64>,
    pub published_at: Option<String>,
}

/// Output of `learn-acquire`: paths to the captions and/or audio file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Acquired {
    pub video: VideoRef,
    pub captions_vtt: Option<Utf8PathBuf>,
    pub audio_mp3: Option<Utf8PathBuf>,
    pub raw_dir: Utf8PathBuf,
}

/// Discriminates the origin of a [`Segment`] or [`Chunk`].
///
/// Serializes as PascalCase (`"Caption"`, `"FrameDescription"`, `"Mixed"`).
/// Old data that predates this field deserializes to [`SegmentKind::Caption`]
/// via the `#[serde(default)]` on the containing struct.
/// The snake_case alias `"frame_description"` is also accepted for backward
/// compat with data written by earlier versions of this library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SegmentKind {
    /// Text derived from auto-captions or Whisper ASR.
    #[default]
    Caption,
    /// Text derived from Sonnet-vision frame description.
    #[serde(alias = "frame_description")]
    FrameDescription,
    /// Chunk spans both Caption and FrameDescription source segments.
    Mixed,
}

/// One transcript line: timestamped text from captions, whisper, or frame vision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
    pub confidence: Option<f32>,
    pub speaker: Option<String>,
    /// Origin of this segment. Defaults to [`SegmentKind::Caption`] so that
    /// JSON written before this field was added deserializes correctly.
    #[serde(default)]
    pub kind: SegmentKind,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TranscriptSource {
    Captions,
    WhisperLocal,
}

/// Output of `learn-asr` or caption parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub video_id: String,
    pub language: Option<String>,
    pub source: TranscriptSource,
    pub segments: Vec<Segment>,
}

/// One semantic chunk ready for embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: String,
    pub video_id: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
    pub token_count: usize,
    /// Origin of this chunk. Defaults to [`SegmentKind::Caption`] so that
    /// meta JSON written before this field was added deserializes correctly.
    #[serde(default)]
    pub kind: SegmentKind,
}

/// A chunk with its dense embedding attached.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedded {
    pub chunk: Chunk,
    pub embedding: Vec<f32>,
    pub embedding_model: String,
}

/// A retrieved chunk plus its score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    pub chunk: Chunk,
    pub score: f32,
    pub rank: usize,
}

/// Final answer with citations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    pub text: String,
    pub citations: Vec<Citation>,
    pub abstained: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub video_id: String,
    pub title: Option<String>,
    pub url: Url,
    pub start_seconds: f64,
}

/// Per-topic ingestion manifest persisted under `~/Docs/KB/<topic>/manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    pub topic: String,
    pub videos: std::collections::BTreeMap<String, VideoState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoState {
    pub video_id: String,
    pub status: IngestStatus,
    pub fetched_at: Option<String>,
    pub indexed_at: Option<String>,
    pub chunk_count: usize,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IngestStatus {
    Pending,
    Acquired,
    Transcribed,
    Chunked,
    Embedded,
    Indexed,
    Failed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learn_error_graph_variant_formats_correctly() {
        let e = LearnError::Graph("x".into());
        assert_eq!(format!("{e}"), "graph failed: x");
        // Also verify Debug works.
        assert!(format!("{e:?}").contains("Graph"));
    }

    #[test]
    fn segment_round_trip() {
        let seg = Segment {
            start_seconds: 0.0,
            end_seconds: 1.5,
            text: "hello".into(),
            confidence: Some(0.9),
            speaker: None,
            kind: SegmentKind::Caption,
        };
        let s = serde_json::to_string(&seg).unwrap();
        let back: Segment = serde_json::from_str(&s).unwrap();
        assert_eq!(seg.text, back.text);
        assert_eq!(back.kind, SegmentKind::Caption);
    }

    /// Old JSON that lacks the `kind` field must deserialize to Caption.
    #[test]
    fn segment_kind_defaults_to_caption_for_old_json() {
        let old_json = r#"{"start_seconds":0.0,"end_seconds":5.0,"text":"hello","confidence":null,"speaker":null}"#;
        let seg: Segment = serde_json::from_str(old_json).unwrap();
        assert_eq!(
            seg.kind,
            SegmentKind::Caption,
            "missing 'kind' field must default to Caption"
        );
    }

    /// FrameDescription round-trips correctly (PascalCase in JSON).
    #[test]
    fn segment_frame_description_round_trip() {
        let seg = Segment {
            start_seconds: 10.0,
            end_seconds: 10.1,
            text: "A diagram showing...".into(),
            confidence: None,
            speaker: None,
            kind: SegmentKind::FrameDescription,
        };
        let s = serde_json::to_string(&seg).unwrap();
        let back: Segment = serde_json::from_str(&s).unwrap();
        assert_eq!(back.kind, SegmentKind::FrameDescription);
        // Serializes as PascalCase to match the grep-visible form in meta.json.
        assert!(
            s.contains("FrameDescription"),
            "kind should serialize as 'FrameDescription'; got: {s}"
        );
    }

    /// The snake_case alias is accepted for backward compat with old meta.json files.
    #[test]
    fn segment_frame_description_accepts_snake_case_alias() {
        let json = r#"{"start_seconds":10.0,"end_seconds":10.1,"text":"frame","confidence":null,"speaker":null,"kind":"frame_description"}"#;
        let seg: Segment = serde_json::from_str(json).unwrap();
        assert_eq!(
            seg.kind,
            SegmentKind::FrameDescription,
            "snake_case alias 'frame_description' should deserialize to FrameDescription"
        );
    }

    #[test]
    fn manifest_default_empty() {
        let m = Manifest::default();
        assert!(m.videos.is_empty());
    }

    #[test]
    fn topic_basic_slug() {
        assert_eq!(
            Topic::new("French Cooking").unwrap().as_str(),
            "french-cooking"
        );
        assert_eq!(
            Topic::new("Indexed Arbitrage").unwrap().as_str(),
            "indexed-arbitrage"
        );
    }

    #[test]
    fn topic_collapses_runs_and_strips_edges() {
        assert_eq!(Topic::new("  AI / ML  ").unwrap().as_str(), "ai-ml");
        assert_eq!(Topic::new("---hello---").unwrap().as_str(), "hello");
        assert_eq!(
            Topic::new("How to make CROISSANTS!!!").unwrap().as_str(),
            "how-to-make-croissants"
        );
    }

    #[test]
    fn topic_unicode_drops_to_ascii_runs() {
        // non-ASCII becomes a `-`; only ASCII alnum survives
        let t = Topic::new("Café français").unwrap();
        assert_eq!(t.as_str(), "caf-fran-ais");
    }

    #[test]
    fn topic_rejects_empty_after_normalize() {
        assert!(Topic::new("").is_err());
        assert!(Topic::new("   ").is_err());
        assert!(Topic::new("!!!").is_err());
    }

    #[test]
    fn topic_caps_at_max_len_without_dangling_dash() {
        let long = "a-very-long-topic-name-that-exceeds-the-forty-character-cap-by-far";
        let t = Topic::new(long).unwrap();
        assert!(t.as_str().len() <= Topic::MAX_LEN);
        assert!(!t.as_str().ends_with('-'));
    }

    #[test]
    fn topic_paths_use_slug() {
        let kb = camino::Utf8PathBuf::from("/tmp/kb");
        let t = Topic::new("French Cooking").unwrap();
        assert_eq!(t.rvf_path(&kb), "/tmp/kb/french-cooking.rvf");
        assert_eq!(t.raw_dir(&kb), "/tmp/kb/_raw/french-cooking");
        assert_eq!(t.manifest_path(&kb), "/tmp/kb/_meta/french-cooking.json");
    }

    #[test]
    fn topic_round_trip_serde() {
        let t = Topic::new("Indexed Arbitrage").unwrap();
        let s = serde_json::to_string(&t).unwrap();
        let back: Topic = serde_json::from_str(&s).unwrap();
        assert_eq!(t, back);
    }
}
