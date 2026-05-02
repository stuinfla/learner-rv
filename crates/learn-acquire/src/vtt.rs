//! WebVTT caption parser.
//!
//! Handles YouTube auto-generated VTT files: parses cue timestamps, strips
//! inline tags, and deduplicates the rolling "same line repeated 2-3×" pattern
//! that YouTube uses for live auto-subs.

use camino::Utf8Path;
use learn_core::{LearnError, Result, Segment, Transcript, TranscriptSource};
use regex::Regex;
use std::sync::OnceLock;

// ── helpers ──────────────────────────────────────────────────────────────────

static TIMESTAMP_RE: OnceLock<Regex> = OnceLock::new();
static TAG_RE: OnceLock<Regex> = OnceLock::new();

fn timestamp_re() -> &'static Regex {
    TIMESTAMP_RE.get_or_init(|| {
        Regex::new(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
            .unwrap()
    })
}

fn tag_re() -> &'static Regex {
    TAG_RE.get_or_init(|| Regex::new(r"<[^>]*>").unwrap())
}

fn parse_seconds(h: &str, m: &str, s: &str, ms: &str) -> f64 {
    let h: f64 = h.parse().unwrap_or(0.0);
    let m: f64 = m.parse().unwrap_or(0.0);
    let s: f64 = s.parse().unwrap_or(0.0);
    let ms: f64 = ms.parse().unwrap_or(0.0);
    h * 3600.0 + m * 60.0 + s + ms / 1000.0
}

fn strip_tags(s: &str) -> String {
    let cleaned = tag_re().replace_all(s, "");
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── public API ────────────────────────────────────────────────────────────────

/// Parse a WebVTT file into segments. Returns `Ok(vec![])` on malformed input
/// rather than propagating parse errors — callers can fall back to ASR.
pub fn parse_vtt(path: &Utf8Path) -> Result<Vec<Segment>> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return Err(LearnError::Acquire(format!("cannot read VTT {path}: {e}"))),
    };
    Ok(parse_vtt_str(&raw))
}

/// Parse VTT content from a string slice (extracted for testability).
pub(crate) fn parse_vtt_str(raw: &str) -> Vec<Segment> {
    // Strip UTF-8 BOM if present.
    let raw = raw.strip_prefix('\u{FEFF}').unwrap_or(raw);
    let re = timestamp_re();
    let mut segments: Vec<Segment> = Vec::new();

    let mut start = 0.0_f64;
    let mut end = 0.0_f64;
    let mut in_cue = false;
    let mut cue_lines: Vec<String> = Vec::new();

    for line in raw.lines() {
        let line = line.trim();
        if let Some(caps) = re.captures(line) {
            // flush previous cue if any
            if in_cue && !cue_lines.is_empty() {
                let text = cue_lines.join(" ").trim().to_string();
                if !text.is_empty() {
                    push_or_merge(&mut segments, start, end, text);
                }
                cue_lines.clear();
            }
            start = parse_seconds(&caps[1], &caps[2], &caps[3], &caps[4]);
            end = parse_seconds(&caps[5], &caps[6], &caps[7], &caps[8]);
            in_cue = true;
        } else if in_cue {
            if line.is_empty() {
                // blank line = end of cue block
                if !cue_lines.is_empty() {
                    let text = cue_lines.join(" ").trim().to_string();
                    if !text.is_empty() {
                        push_or_merge(&mut segments, start, end, text);
                    }
                    cue_lines.clear();
                }
                in_cue = false;
            } else if !is_meta_line(line) {
                let stripped = strip_tags(line);
                if !stripped.is_empty() {
                    cue_lines.push(stripped);
                }
            }
        }
    }

    // flush final cue if file ends without trailing blank
    if in_cue && !cue_lines.is_empty() {
        let text = cue_lines.join(" ").trim().to_string();
        if !text.is_empty() {
            push_or_merge(&mut segments, start, end, text);
        }
    }

    dedupe_rolling(&mut segments);
    segments
}

/// Lines that look like metadata inside a cue block (NOTE, WEBVTT header, etc.)
fn is_meta_line(line: &str) -> bool {
    line.starts_with("NOTE")
        || line.starts_with("STYLE")
        || line.starts_with("REGION")
        || line == "WEBVTT"
}

/// Push a new segment, or merge into the last one if the text is identical.
fn push_or_merge(segments: &mut Vec<Segment>, start: f64, end: f64, text: String) {
    if let Some(last) = segments.last_mut() {
        if last.text == text {
            last.end_seconds = end;
            return;
        }
    }
    segments.push(Segment {
        start_seconds: start,
        end_seconds: end,
        text,
        confidence: None,
        speaker: None,
    });
}

/// Second-pass dedup: remove any remaining consecutive identical segments
/// (YouTube sometimes produces them across cue boundaries).
fn dedupe_rolling(segments: &mut Vec<Segment>) {
    let mut i = 1;
    while i < segments.len() {
        if segments[i].text == segments[i - 1].text {
            let new_end = segments[i].end_seconds;
            segments[i - 1].end_seconds = new_end;
            segments.remove(i);
        } else {
            i += 1;
        }
    }
}

/// Wrap parsed segments into a `Transcript`.
pub fn segments_to_transcript(video_id: String, segments: Vec<Segment>) -> Transcript {
    Transcript {
        video_id,
        language: Some("en".into()),
        source: TranscriptSource::Captions,
        segments,
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_VTT: &str = r#"WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.500
Hello world

00:00:02.500 --> 00:00:05.000
This is a test

"#;

    const ROLLING_VTT: &str = r#"WEBVTT

00:00:01.000 --> 00:00:03.000
Rolling caption

00:00:02.000 --> 00:00:04.000
Rolling caption

00:00:03.000 --> 00:00:05.000
Rolling caption

00:00:05.000 --> 00:00:07.000
New text here

"#;

    const TAGS_VTT: &str = r#"WEBVTT

00:00:00.500 --> 00:00:02.000
<c.colorCCCCCC>Hello</c> <b>world</b>

"#;

    const MALFORMED: &str = "this is not a valid vtt file at all";

    #[test]
    fn parse_happy_path() {
        let segs = parse_vtt_str(SIMPLE_VTT);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "Hello world");
        assert!((segs[0].start_seconds - 0.0).abs() < 0.001);
        assert!((segs[0].end_seconds - 2.5).abs() < 0.001);
        assert_eq!(segs[1].text, "This is a test");
    }

    #[test]
    fn parse_dedupes_rolling() {
        let segs = parse_vtt_str(ROLLING_VTT);
        // 3 identical "Rolling caption" cues collapse to 1; "New text here" stays
        assert_eq!(
            segs.len(),
            2,
            "expected 2 segments after dedup, got {segs:?}"
        );
        assert_eq!(segs[0].text, "Rolling caption");
        // merged end should be the last of the three cues
        assert!((segs[0].end_seconds - 5.0).abs() < 0.001);
        assert_eq!(segs[1].text, "New text here");
    }

    #[test]
    fn parse_strips_inline_tags() {
        let segs = parse_vtt_str(TAGS_VTT);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "Hello world");
    }

    #[test]
    fn parse_malformed_returns_empty() {
        let segs = parse_vtt_str(MALFORMED);
        assert!(segs.is_empty());
    }

    #[test]
    fn parses_vtt_without_trailing_newline() {
        // Input: WEBVTT header + one cue, terminated WITHOUT a final \n.
        let raw = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello";
        let segs = parse_vtt_str(raw);
        assert_eq!(segs.len(), 1, "expected 1 segment, got {segs:?}");
        assert_eq!(segs[0].text, "hello");
    }

    #[test]
    fn parses_vtt_with_utf8_bom() {
        let raw = "\u{FEFF}WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n";
        let segs = parse_vtt_str(raw);
        assert_eq!(
            segs.len(),
            1,
            "expected 1 segment after BOM strip, got {segs:?}"
        );
        assert_eq!(segs[0].text, "hi");
    }

    #[test]
    fn segments_json_round_trip() {
        let segs = parse_vtt_str(SIMPLE_VTT);
        let json = serde_json::to_string(&segs).expect("serialize");
        let back: Vec<Segment> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(segs.len(), back.len());
        for (a, b) in segs.iter().zip(back.iter()) {
            assert_eq!(a.text, b.text);
            assert!((a.start_seconds - b.start_seconds).abs() < 0.0001);
        }
    }
}
