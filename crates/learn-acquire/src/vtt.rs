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
        kind: learn_core::SegmentKind::Caption,
    });
}

/// Second-pass dedup: handle YouTube's rolling-caption overlap pattern.
///
/// yt-dlp auto-captions repeat the previous cue's text as the first line of
/// each new cue, then append the new words.  After tag-stripping and joining
/// this produces segments where `segments[i].text` *starts with*
/// `segments[i-1].text`.  We strip that prefix so only the fresh words remain,
/// then drop any segment whose text becomes empty after stripping.
fn dedupe_rolling(segments: &mut Vec<Segment>) {
    let mut i = 1;
    while i < segments.len() {
        let prev_text = segments[i - 1].text.clone();
        let cur_text = &segments[i].text;

        if cur_text == &prev_text {
            // Exact duplicate — extend previous and drop current.
            let new_end = segments[i].end_seconds;
            segments[i - 1].end_seconds = new_end;
            segments.remove(i);
        } else if let Some(tail) = cur_text.strip_prefix(prev_text.as_str()) {
            // Current starts with previous — keep only the new tail.
            let tail = tail.trim().to_string();
            if tail.is_empty() {
                // Nothing new; extend previous and drop current.
                let new_end = segments[i].end_seconds;
                segments[i - 1].end_seconds = new_end;
                segments.remove(i);
            } else {
                segments[i].text = tail;
                i += 1;
            }
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

    /// Real yt-dlp auto-caption pattern: each cue carries the previous cue's
    /// full text as its first line, then adds new words.  After tag-stripping
    /// the segments arrive as:
    ///   seg[0] = "Hi everyone."
    ///   seg[1] = "Hi everyone. Real pleasure to be here"   ← starts with seg[0]
    ///   seg[2] = "Real pleasure to be here and I'm excited" ← starts with tail of seg[1]
    ///
    /// Expected output: three non-overlapping segments whose texts concatenate
    /// cleanly into the full spoken sentence.
    const YTDLP_OVERLAP_VTT: &str = r#"WEBVTT
Kind: captions
Language: en

00:00:02.480 --> 00:00:04.230 align:start position:0%
Hi everyone.

00:00:04.230 --> 00:00:04.240 align:start position:0%
Hi everyone. Real pleasure to be here

00:00:04.240 --> 00:00:07.190 align:start position:0%
Hi everyone. Real pleasure to be here
and I'm very excited

00:00:07.190 --> 00:00:07.200 align:start position:0%
and I'm very excited

00:00:07.200 --> 00:00:10.310 align:start position:0%
and I'm very excited
to showcase the demo

"#;

    #[test]
    fn ytdlp_overlap_deduplicated() {
        let segs = parse_vtt_str(YTDLP_OVERLAP_VTT);
        // After dedup the texts must not repeat words from the previous segment.
        for i in 1..segs.len() {
            assert!(
                !segs[i].text.starts_with(&segs[i - 1].text),
                "seg[{i}] still starts with seg[{}]: {:?} vs {:?}",
                i - 1,
                segs[i - 1].text,
                segs[i].text
            );
        }
        // Joined text should contain each key phrase exactly once.
        let joined = segs
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            joined.contains("Hi everyone."),
            "missing 'Hi everyone.' in: {joined}"
        );
        assert!(
            joined.contains("Real pleasure to be here"),
            "missing carry-forward in: {joined}"
        );
        assert!(
            joined.contains("to showcase the demo"),
            "missing final phrase in: {joined}"
        );
        // Sanity: no empty segments.
        for seg in &segs {
            assert!(!seg.text.is_empty(), "empty segment after dedup: {seg:?}");
        }
    }

    /// Overlap where the new segment is ONLY the prefix (nothing new added).
    /// Should collapse into the previous segment (extend its end_seconds).
    const PURE_REPEAT_VTT: &str = r#"WEBVTT

00:00:00.000 --> 00:00:02.000
Hello world

00:00:02.000 --> 00:00:04.000
Hello world

00:00:04.000 --> 00:00:06.000
Hello world and more

"#;

    #[test]
    fn pure_repeat_collapses_then_extends() {
        let segs = parse_vtt_str(PURE_REPEAT_VTT);
        // "Hello world" repeated twice → collapsed to one; "and more" is new.
        assert_eq!(segs.len(), 2, "expected 2 segments, got {segs:?}");
        assert_eq!(segs[0].text, "Hello world");
        // The second segment carries only the new words.
        assert_eq!(segs[1].text, "and more");
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
