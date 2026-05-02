//! `learn-chunk` — turn a `Transcript` into a stream of `Chunk`s with
//! sentence-aware boundaries, target-token sizing, and overlap.
//!
//! Phase 1 is naive sentence packing. Topic-shift detection is wired in
//! Phase 2 by composing a separate function on top — not by adding a
//! callback to this signature. Karpathy: minimal scope.

#![deny(unsafe_code)]

use learn_core::{Chunk, LearnError, Result, Transcript};
use tracing::debug;

/// Approximate token count for a string slice.
///
/// Formula: `chars / 4`. BGE tokenizer is close to this on English.
/// Phase 2 will swap this for a real tokenizer call.
#[inline]
fn approx_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

/// Configuration for [`chunk_transcript`].
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target size of each chunk in approximate tokens. Default: 300.
    pub target_tokens: usize,
    /// How many tokens of overlap to carry from the end of one chunk into the
    /// start of the next. Default: 50.
    pub overlap_tokens: usize,
    /// Minimum size in tokens for the final (tail) chunk. If the tail would be
    /// smaller than this it is merged into the previous chunk. Default: 80.
    pub min_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_tokens: 300,
            overlap_tokens: 50,
            min_tokens: 80,
        }
    }
}

/// A single sentence fragment extracted from a [`Segment`], carrying its
/// time span and approximate token count.
#[derive(Debug, Clone)]
struct Sentence {
    text: String,
    start_seconds: f64,
    end_seconds: f64,
    tokens: usize,
}

/// Split one segment's text into sentence fragments at `[.!?]` boundaries.
/// If the text has no terminating punctuation the whole segment becomes one
/// fragment. Whitespace-only or empty fragments are discarded.
fn split_segment_into_sentences(text: &str, start: f64, end: f64) -> Vec<Sentence> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    let mut sentences = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];
        current.push(ch);

        // Sentence boundary: [.!?] followed by whitespace OR end of input.
        if matches!(ch, '.' | '!' | '?') {
            let at_end = i + 1 >= len;
            let next_is_space = i + 1 < len && chars[i + 1].is_whitespace();
            if at_end || next_is_space {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    let tokens = approx_tokens(&trimmed);
                    sentences.push(Sentence {
                        text: trimmed,
                        start_seconds: start,
                        end_seconds: end,
                        tokens,
                    });
                }
                current.clear();
                // Skip the whitespace after the punctuation.
                if next_is_space {
                    i += 1;
                }
            }
        }
        i += 1;
    }

    // Any remaining text that had no sentence-terminating punctuation.
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        let tokens = approx_tokens(&trimmed);
        sentences.push(Sentence {
            text: trimmed,
            start_seconds: start,
            end_seconds: end,
            tokens,
        });
    }

    sentences
}

/// Collect all sentences from every segment in order.
fn sentences_from_transcript(transcript: &Transcript) -> Vec<Sentence> {
    transcript
        .segments
        .iter()
        .flat_map(|seg| split_segment_into_sentences(&seg.text, seg.start_seconds, seg.end_seconds))
        .collect()
}

/// Build a [`Chunk`] from a slice of sentences.
fn build_chunk(sentences: &[Sentence], video_id: &str, idx: usize) -> Chunk {
    debug_assert!(!sentences.is_empty());
    let text = sentences
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let token_count = approx_tokens(&text);
    let start_seconds = sentences.first().map(|s| s.start_seconds).unwrap_or(0.0);
    let end_seconds = sentences.last().map(|s| s.end_seconds).unwrap_or(0.0);
    Chunk {
        chunk_id: format!("{}:{}", video_id, idx),
        video_id: video_id.to_string(),
        start_seconds,
        end_seconds,
        text,
        token_count,
    }
}

/// Find the suffix of `sentences` whose cumulative token count is at most
/// `overlap_tokens`. Returns the starting index into `sentences`.
fn overlap_start(sentences: &[Sentence], overlap_tokens: usize) -> usize {
    let mut accum = 0usize;
    let mut start = sentences.len(); // default: carry nothing
    for s in sentences.iter().rev() {
        let new_accum = accum + s.tokens;
        if new_accum > overlap_tokens {
            break;
        }
        accum = new_accum;
        start -= 1;
    }
    start
}

/// Chunk a [`Transcript`] into a `Vec<Chunk>` using [`ChunkConfig`].
pub fn chunk_transcript(transcript: &Transcript, cfg: &ChunkConfig) -> Result<Vec<Chunk>> {
    if cfg.target_tokens == 0 {
        return Err(LearnError::Chunk("target_tokens must be > 0".to_string()));
    }
    if cfg.overlap_tokens >= cfg.target_tokens {
        return Err(LearnError::Chunk(
            "overlap_tokens must be < target_tokens".to_string(),
        ));
    }

    let all_sentences = sentences_from_transcript(transcript);
    if all_sentences.is_empty() {
        debug!(video_id = %transcript.video_id, "no sentences — returning empty chunks");
        return Ok(vec![]);
    }

    let video_id = &transcript.video_id;
    let mut chunks: Vec<Chunk> = Vec::new();
    // The working window: sentences accumulated for the current chunk.
    let mut window: Vec<Sentence> = Vec::new();
    let mut window_tokens: usize = 0;

    for sentence in all_sentences {
        let new_tokens = window_tokens + sentence.tokens;

        if new_tokens >= cfg.target_tokens && !window.is_empty() {
            // Emit the current window.
            window.push(sentence); // include triggering sentence in this chunk
            let idx = chunks.len();
            chunks.push(build_chunk(&window, video_id, idx));
            debug!(
                chunk_idx = idx,
                tokens = chunks.last().unwrap().token_count,
                "emitted chunk"
            );

            // Carry overlap into next window.
            let start = overlap_start(&window, cfg.overlap_tokens);
            window = window[start..].to_vec();
            window_tokens = window.iter().map(|s| s.tokens).sum();
        } else {
            window_tokens += sentence.tokens;
            window.push(sentence);
        }
    }

    // Handle the tail.
    if !window.is_empty() {
        let tail_tokens = approx_tokens(
            &window
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
        );

        if tail_tokens < cfg.min_tokens && !chunks.is_empty() {
            // Merge tail into the previous chunk.
            let mut prev = chunks.pop().unwrap();
            let merged_text = format!(
                "{} {}",
                prev.text,
                window
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            prev.text = merged_text;
            prev.token_count = approx_tokens(&prev.text);
            prev.end_seconds = window
                .last()
                .map(|s| s.end_seconds)
                .unwrap_or(prev.end_seconds);
            debug!(
                chunk_idx = prev.chunk_id,
                "merged runt tail into previous chunk"
            );
            chunks.push(prev);
        } else {
            let idx = chunks.len();
            chunks.push(build_chunk(&window, video_id, idx));
            debug!(chunk_idx = idx, "emitted tail chunk");
        }
    }

    Ok(chunks)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use learn_core::{Segment, TranscriptSource};
    use proptest::prelude::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn seg(start: f64, end: f64, text: &str) -> Segment {
        Segment {
            start_seconds: start,
            end_seconds: end,
            text: text.to_string(),
            confidence: None,
            speaker: None,
        }
    }

    fn transcript(video_id: &str, segments: Vec<Segment>) -> Transcript {
        Transcript {
            video_id: video_id.to_string(),
            language: None,
            source: TranscriptSource::Captions,
            segments,
        }
    }

    fn default_cfg() -> ChunkConfig {
        ChunkConfig::default()
    }

    fn run(t: &Transcript, cfg: &ChunkConfig) -> Vec<Chunk> {
        chunk_transcript(t, cfg).expect("chunk_transcript failed")
    }

    // ── unit tests ────────────────────────────────────────────────────────────

    #[test]
    fn empty_transcript_returns_empty_chunks() {
        let t = transcript("vid1", vec![]);
        let chunks = run(&t, &default_cfg());
        assert!(chunks.is_empty());
    }

    #[test]
    fn single_short_segment_round_trip() {
        let t = transcript("vid1", vec![seg(0.0, 5.0, "Hello world.")]);
        let chunks = run(&t, &default_cfg());
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert_eq!(c.video_id, "vid1");
        assert_eq!(c.text, "Hello world.");
        assert_eq!(c.start_seconds, 0.0);
        assert_eq!(c.end_seconds, 5.0);
    }

    /// Build ~1000 chars of text across several segments so we get 3-4 chunks
    /// with default target=300 tokens (≈1200 chars).
    #[test]
    fn large_transcript_produces_multiple_chunks() {
        // Each sentence is ~80 chars ≈ 20 tokens.
        // 50 sentences × 20 tokens = 1000 tokens → should produce ≥3 chunks.
        let sentence = "The quick brown fox jumps over the lazy dog near the river bank.";
        assert!(sentence.len() < 100);
        let full_text = (0..50).map(|_| sentence).collect::<Vec<_>>().join(" ");

        let t = transcript("vid_large", vec![seg(0.0, 300.0, &full_text)]);
        let cfg = ChunkConfig {
            target_tokens: 300,
            overlap_tokens: 50,
            min_tokens: 80,
        };
        let chunks = run(&t, &cfg);
        assert!(
            chunks.len() >= 3,
            "expected ≥3 chunks, got {}",
            chunks.len()
        );
        for c in &chunks {
            // Each chunk (except the last) should be at or near target.
            assert!(
                c.token_count >= cfg.min_tokens || chunks.last().unwrap().chunk_id == c.chunk_id,
                "chunk {} has {} tokens < min_tokens {}",
                c.chunk_id,
                c.token_count,
                cfg.min_tokens
            );
        }
    }

    #[test]
    fn chunks_are_time_monotonic() {
        let t = transcript(
            "vid_mono",
            vec![
                seg(0.0, 10.0, "First sentence here."),
                seg(10.0, 20.0, "Second sentence here."),
                seg(20.0, 30.0, "Third sentence here."),
            ],
        );
        let chunks = run(&t, &default_cfg());
        let starts: Vec<f64> = chunks.iter().map(|c| c.start_seconds).collect();
        for w in starts.windows(2) {
            assert!(w[1] >= w[0], "time not monotonic: {} < {}", w[1], w[0]);
        }
    }

    #[test]
    fn overlap_text_appears_in_consecutive_chunks() {
        // Build enough tokens so we get at least 2 chunks.
        let sentence = "The quick brown fox jumps over the lazy dog near the river.";
        let sentences_block = (0..30).map(|_| sentence).collect::<Vec<_>>().join(" ");

        let t = transcript("vid_overlap", vec![seg(0.0, 60.0, &sentences_block)]);
        let cfg = ChunkConfig {
            target_tokens: 120,
            overlap_tokens: 30,
            min_tokens: 20,
        };
        let chunks = run(&t, &cfg);
        assert!(
            chunks.len() >= 2,
            "need ≥2 chunks for overlap test, got {}",
            chunks.len()
        );

        // The last sentence(s) of chunk 0 should appear somewhere in chunk 1.
        let c0_words: Vec<&str> = chunks[0].text.split_whitespace().collect();
        let tail_words = &c0_words[c0_words.len().saturating_sub(8)..];
        let c1_text = &chunks[1].text;
        let overlap_found = tail_words.iter().any(|w| c1_text.contains(*w));
        assert!(
            overlap_found,
            "no overlap found between chunk 0 and chunk 1"
        );
    }

    #[test]
    fn runt_tail_is_merged_into_previous_chunk() {
        // Craft a transcript that fills one chunk fully then leaves a tiny tail.
        // sentence ≈ 60 chars ≈ 15 tokens. 25 sentences ≈ 375 tokens → emits 1
        // chunk at 300, then ~75 tokens tail. With min_tokens=80 that tail should
        // merge.
        let sentence = "The quick brown fox jumps over the lazy dog.";
        let sentences_block = (0..25).map(|_| sentence).collect::<Vec<_>>().join(" ");

        let t = transcript("vid_runt", vec![seg(0.0, 25.0, &sentences_block)]);
        let cfg = ChunkConfig {
            target_tokens: 300,
            overlap_tokens: 50,
            min_tokens: 80,
        };
        let chunks = run(&t, &cfg);

        // The final chunk should be >= min_tokens (because any runt was merged).
        if let Some(last) = chunks.last() {
            assert!(
                last.token_count >= cfg.min_tokens,
                "last chunk has {} tokens < min_tokens {}",
                last.token_count,
                cfg.min_tokens
            );
        }
    }

    #[test]
    fn chunk_ids_are_unique_within_video() {
        let sentence = "This is sentence number one here now.";
        let block = (0..60).map(|_| sentence).collect::<Vec<_>>().join(" ");
        let t = transcript("vid_ids", vec![seg(0.0, 60.0, &block)]);
        let cfg = ChunkConfig {
            target_tokens: 150,
            overlap_tokens: 30,
            min_tokens: 40,
        };
        let chunks = run(&t, &cfg);
        let mut ids: Vec<&str> = chunks.iter().map(|c| c.chunk_id.as_str()).collect();
        ids.dedup();
        // dedup leaves unique run, original len matches deduped only if all unique.
        // Use a set instead for a stronger check.
        let id_set: std::collections::HashSet<&str> =
            chunks.iter().map(|c| c.chunk_id.as_str()).collect();
        assert_eq!(id_set.len(), chunks.len(), "duplicate chunk_ids found");
    }

    #[test]
    fn consecutive_identical_sentences_no_infinite_loop() {
        // All identical text — must terminate and produce non-empty chunks.
        let text = "Hello. Hello. Hello. Hello. Hello. Hello. Hello. Hello. Hello. Hello.";
        let t = transcript("vid_dup", vec![seg(0.0, 10.0, text)]);
        let cfg = ChunkConfig {
            target_tokens: 10,
            overlap_tokens: 3,
            min_tokens: 2,
        };
        let chunks = run(&t, &cfg);
        assert!(!chunks.is_empty(), "expected at least one chunk");
        for c in &chunks {
            assert!(!c.text.is_empty(), "chunk text must not be empty");
        }
    }

    #[test]
    fn segment_without_sentence_punctuation_is_not_dropped() {
        // A segment with no [.!?] should be treated as one fragment and included.
        let t = transcript(
            "vid_nopunct",
            vec![seg(0.0, 5.0, "No punctuation at all here")],
        );
        let chunks = run(&t, &default_cfg());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "No punctuation at all here");
    }

    // ── error path tests ──────────────────────────────────────────────────────

    #[test]
    fn errors_when_target_tokens_zero() {
        let cfg = ChunkConfig {
            target_tokens: 0,
            overlap_tokens: 0,
            min_tokens: 0,
        };
        let t = transcript("vid_err", vec![seg(0.0, 1.0, "hi")]);
        assert!(
            matches!(chunk_transcript(&t, &cfg), Err(LearnError::Chunk(_))),
            "expected Err(LearnError::Chunk) for target_tokens=0"
        );
    }

    #[test]
    fn errors_when_overlap_geq_target() {
        let cfg = ChunkConfig {
            target_tokens: 100,
            overlap_tokens: 100,
            min_tokens: 50,
        };
        let t = transcript("vid_err2", vec![seg(0.0, 1.0, "hi")]);
        assert!(
            matches!(chunk_transcript(&t, &cfg), Err(LearnError::Chunk(_))),
            "expected Err(LearnError::Chunk) for overlap_tokens >= target_tokens"
        );
    }

    // ── proptest properties ───────────────────────────────────────────────────

    // Arbitrary transcript generator: 0–20 segments, text drawn from
    // a small vocabulary to exercise punctuation splitting.
    fn arb_segment() -> impl Strategy<Value = Segment> {
        (
            0.0f64..1000.0f64,
            prop::collection::vec(
                prop_oneof![
                    Just("The cat sat on the mat."),
                    Just("Hello world"),
                    Just("One two three four five."),
                    Just("No punctuation here"),
                    Just("Short."),
                    Just("A longer sentence with more words and detail!"),
                    Just("Another sentence? Yes indeed."),
                ],
                1..8usize,
            ),
        )
            .prop_map(|(start, words)| {
                let end = start + 5.0;
                let text = words.join(" ");
                Segment {
                    start_seconds: start,
                    end_seconds: end,
                    text,
                    confidence: None,
                    speaker: None,
                }
            })
    }

    fn arb_transcript() -> impl Strategy<Value = Transcript> {
        prop::collection::vec(arb_segment(), 0..20usize).prop_map(|mut segs| {
            // Sort segments by start_seconds to keep times monotonic.
            segs.sort_by(|a, b| {
                a.start_seconds
                    .partial_cmp(&b.start_seconds)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // Adjust end times to not overlap.
            for i in 1..segs.len() {
                if segs[i].start_seconds <= segs[i - 1].end_seconds {
                    segs[i].start_seconds = segs[i - 1].end_seconds;
                    segs[i].end_seconds = segs[i].start_seconds + 5.0;
                }
            }
            Transcript {
                video_id: "prop-vid".to_string(),
                language: None,
                source: TranscriptSource::Captions,
                segments: segs,
            }
        })
    }

    proptest! {
        /// Property 1: every chunk has start_seconds <= end_seconds.
        #[test]
        fn prop_start_le_end(t in arb_transcript()) {
            let cfg = ChunkConfig { target_tokens: 50, overlap_tokens: 10, min_tokens: 5 };
            let chunks = chunk_transcript(&t, &cfg).unwrap();
            for c in &chunks {
                prop_assert!(
                    c.start_seconds <= c.end_seconds,
                    "chunk {} has start {} > end {}",
                    c.chunk_id,
                    c.start_seconds,
                    c.end_seconds
                );
            }
        }

        /// Property 2: no chunk has empty text.
        #[test]
        fn prop_no_empty_text(t in arb_transcript()) {
            let cfg = ChunkConfig { target_tokens: 50, overlap_tokens: 10, min_tokens: 5 };
            let chunks = chunk_transcript(&t, &cfg).unwrap();
            for c in &chunks {
                prop_assert!(!c.text.trim().is_empty(), "chunk {} has empty text", c.chunk_id);
            }
        }

        /// Property 3: char count of all chunk texts (joined) >= char count of all
        /// source segment texts (joined), accounting for overlap headroom.
        ///
        /// Because overlap copies sentences into multiple chunks, the joined chunk
        /// text will be LARGER than the source text. We verify the lower bound:
        /// joined source text length <= joined chunk text length.
        #[test]
        fn prop_chunks_cover_source_text(t in arb_transcript()) {
            let cfg = ChunkConfig { target_tokens: 50, overlap_tokens: 10, min_tokens: 5 };
            let chunks = chunk_transcript(&t, &cfg).unwrap();

            let source_chars: usize = t.segments.iter()
                .map(|s| s.text.trim().chars().count())
                .sum();
            let chunk_chars: usize = chunks.iter()
                .map(|c| c.text.chars().count())
                .sum();

            // Chunk text >= source text because overlap duplicates some sentences.
            prop_assert!(
                chunk_chars >= source_chars,
                "chunks cover fewer chars ({}) than source ({})",
                chunk_chars,
                source_chars
            );
        }
    }

    // ── Formal invariant proof harnesses (Phase 4B — Option C proptest) ──────
    //
    // These are the correctness-critical invariants for `chunk_transcript`.
    // They are implemented as property-based tests with `proptest` because:
    //   (a) the toolchain is pinned to stable Rust 1.91.1, ruling out `kani`
    //       (which requires nightly + a separate toolchain install);
    //   (b) `ruvector-verified` covers HNSW vector-dimension proofs, not
    //       chunker contracts.
    //
    // Run with: `cargo test -p learn-chunk`
    // Every property exercises 256 random cases by default (proptest default).
    // To raise coverage: `PROPTEST_CASES=10000 cargo test -p learn-chunk`.
    //
    // If kani becomes available in CI, these can be promoted to
    // `#[cfg(kani)] #[kani::proof]` harnesses — the invariant logic is
    // identical, only the annotation changes.

    proptest! {
        /// Invariant: every output chunk has `start_seconds <= end_seconds`.
        ///
        /// Covers arbitrary transcripts with 0–20 segments and varied token
        /// budgets to ensure no path through the chunker violates time ordering.
        #[test]
        fn chunker_chunks_have_valid_time_ranges(
            t in arb_transcript(),
            target in 20usize..400usize,
        ) {
            // overlap must be < target; fix it at 10 % of target (min 1).
            let overlap = (target / 10).max(1);
            let min_t = (target / 5).max(1);
            let cfg = ChunkConfig {
                target_tokens: target,
                overlap_tokens: overlap,
                min_tokens: min_t,
            };
            let chunks = chunk_transcript(&t, &cfg).unwrap();
            for c in &chunks {
                prop_assert!(
                    c.start_seconds <= c.end_seconds,
                    "chunk {} violates start_seconds ({}) <= end_seconds ({})",
                    c.chunk_id,
                    c.start_seconds,
                    c.end_seconds
                );
            }
        }

        /// Invariant: for any non-empty input, no output chunk has empty text.
        ///
        /// Verifies the chunker never emits a zero-length text field regardless
        /// of segment content, punctuation style, or token budget.
        #[test]
        fn chunker_no_empty_chunks(
            t in arb_transcript(),
            target in 20usize..400usize,
        ) {
            let overlap = (target / 10).max(1);
            let min_t = (target / 5).max(1);
            let cfg = ChunkConfig {
                target_tokens: target,
                overlap_tokens: overlap,
                min_tokens: min_t,
            };
            let chunks = chunk_transcript(&t, &cfg).unwrap();
            for c in &chunks {
                prop_assert!(
                    !c.text.trim().is_empty(),
                    "chunk {} has empty text; transcript had {} segments",
                    c.chunk_id,
                    t.segments.len()
                );
            }
        }

        /// Invariant: every emitted chunk has
        ///   `token_count <= target_tokens + max_sentence_overshoot`.
        ///
        /// `max_sentence_overshoot` is the maximum approx-token count of any
        /// single sentence in the transcript, because the chunker always
        /// includes the triggering sentence in the emitted chunk before
        /// splitting (one-sentence look-ahead model).  The bound is therefore:
        ///
        ///   chunk.token_count <= target_tokens + longest_sentence_tokens
        ///
        /// The merged-tail path may exceed this slightly (tail is appended to
        /// a full chunk), so the test excludes tail-merged chunks: a chunk is
        /// a candidate for this check only when the transcript produced more
        /// than one chunk (guaranteeing at least one non-tail emission).
        #[test]
        fn chunker_token_count_within_target_envelope(
            t in arb_transcript(),
            target in 40usize..400usize,
        ) {
            let overlap = (target / 10).max(1);
            let min_t = (target / 5).max(1);
            let cfg = ChunkConfig {
                target_tokens: target,
                overlap_tokens: overlap,
                min_tokens: min_t,
            };
            let chunks = chunk_transcript(&t, &cfg).unwrap();

            if chunks.len() <= 1 {
                // Single or zero chunks: the token-envelope rule only constrains
                // non-tail chunks. Nothing to check here.
                return Ok(());
            }

            // Compute the max token count among all individual sentences across
            // all segments.  This is the per-sentence overshoot bound.
            let max_sentence_tokens: usize = t
                .segments
                .iter()
                .flat_map(|seg| {
                    split_segment_into_sentences(&seg.text, seg.start_seconds, seg.end_seconds)
                })
                .map(|s| s.tokens)
                .max()
                .unwrap_or(0);

            let envelope = target + max_sentence_tokens;

            // Check all chunks except the last (which may be a merged tail).
            for c in &chunks[..chunks.len() - 1] {
                prop_assert!(
                    c.token_count <= envelope,
                    "chunk {} has {} tokens, exceeds envelope {} (target={} + max_sentence={})",
                    c.chunk_id,
                    c.token_count,
                    envelope,
                    target,
                    max_sentence_tokens
                );
            }
        }
    }
}
