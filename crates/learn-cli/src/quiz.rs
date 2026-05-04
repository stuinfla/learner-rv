//! `learn quiz <topic>` — flashcard-style Q&A review loop against the KB.

#![deny(unsafe_code)]

use camino::Utf8PathBuf;
use colored::Colorize;
use learn_core::Topic;
use learn_synth::QuizCard;
use std::io::{BufRead, Write};

/// Number of days to schedule next review per response kind (Ebbinghaus spacing).
const SPACING_KNEW_DAYS: u64 = 14;
const SPACING_UNSURE_DAYS: u64 = 4;
const SPACING_WRONG_DAYS: u64 = 1;

/// A user's self-assessment of a card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Response {
    Knew,
    Unsure,
    Wrong,
}

impl Response {
    /// Parse a single character (or empty input) into a [`Response`].
    ///
    /// - `'k'` / `'K'` / empty (Enter) → [`Response::Knew`]
    /// - `'u'` / `'U'`                 → [`Response::Unsure`]
    /// - `'w'` / `'W'`                 → [`Response::Wrong`]
    ///
    /// Returns `None` for unrecognised input so the REPL can re-prompt.
    pub fn parse(input: &str) -> Option<Self> {
        match input.trim() {
            "" | "k" | "K" => Some(Response::Knew),
            "u" | "U" => Some(Response::Unsure),
            "w" | "W" => Some(Response::Wrong),
            _ => None,
        }
    }

    fn spacing_days(self) -> u64 {
        match self {
            Response::Knew => SPACING_KNEW_DAYS,
            Response::Unsure => SPACING_UNSURE_DAYS,
            Response::Wrong => SPACING_WRONG_DAYS,
        }
    }
}

/// Compute the path for the quiz card cache.
///
/// `<kb_root>/_quiz/<topic>.jsonl`
pub fn quiz_cache_path(kb_root: &Utf8PathBuf, topic: &str) -> Utf8PathBuf {
    kb_root.join("_quiz").join(format!("{topic}.jsonl"))
}

/// Return `true` if `path` exists and was modified within the last 24 hours.
fn cache_is_fresh(path: &Utf8PathBuf) -> bool {
    let meta = match std::fs::metadata(path.as_std_path()) {
        Ok(m) => m,
        Err(_) => return false,
    };
    let modified = match meta.modified() {
        Ok(t) => t,
        Err(_) => return false,
    };
    match modified.elapsed() {
        Ok(age) => age.as_secs() < 86_400,
        Err(_) => false,
    }
}

/// Shuffle `items` in-place using a simple Fisher-Yates algorithm seeded from
/// the current time. No external rand dependency.
fn shuffle<T>(items: &mut [T]) {
    use std::time::{SystemTime, UNIX_EPOCH};
    // Cheap pseudo-random seed from system time nanoseconds.
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(42);
    let n = items.len();
    if n < 2 {
        return;
    }
    // xorshift64 PRNG
    let mut state = if seed == 0 { 1 } else { seed };
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        items.swap(i, j);
    }
}

/// Add seconds to the current Unix timestamp and format as RFC 3339 date string.
fn days_from_now_rfc3339(days: u64) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let target = now_secs + days * 86_400;
    // Reuse the same civil-from-days algorithm as main.rs rfc3339_now.
    let days_since_epoch = (target / 86_400) as i64;
    let z: i64 = days_since_epoch + 719_468;
    let era: i64 = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe: i64 = z - era * 146_097;
    let yoe: i64 = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y: i64 = yoe + era * 400;
    let doy: i64 = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp: i64 = (5 * doy + 2) / 153;
    let d: i64 = doy - (153 * mp + 2) / 5 + 1;
    let mo: i64 = if mp < 10 { mp + 3 } else { mp - 9 };
    let y: i64 = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}")
}

/// Run the interactive quiz loop.
pub async fn run_quiz(
    topic_str: String,
    count: usize,
    spaced: bool,
    kb_root: Utf8PathBuf,
) -> learn_core::Result<()> {
    let topic = Topic::new(&topic_str)?;
    let embedder_path = crate::default_model_dir();

    // 1. Open index; exit early if KB has no data.
    let index = learn_index::LearnIndex::open(&kb_root, topic.clone())?;
    if index.manifest().videos.is_empty() {
        eprintln!(
            "error: topic '{topic_str}' has no data (KB missing or not yet ingested).\n\
             Run: learn ingest \"<url>\" --topic {topic_str}"
        );
        std::process::exit(2);
    }

    // 2. Build retriever + BM25.
    let mut retriever =
        learn_retrieve::Retriever::for_topic(index, &topic, embedder_path.as_ref())?;
    retriever.refresh_bm25()?;

    // 3. Load or generate cards.
    let cache_path = quiz_cache_path(&kb_root, topic.as_str());
    let mut cards: Vec<QuizCard> = if cache_is_fresh(&cache_path) {
        load_cards_from_cache(&cache_path)?
    } else {
        // Broad query for diverse coverage.
        let hits = retriever
            .search("key concepts techniques methods principles", 20)
            .await?;
        if hits.is_empty() {
            eprintln!("(no relevant chunks found — try re-ingesting the topic)");
            std::process::exit(3);
        }
        let synth = learn_synth::select_synthesizer()?;
        let generated = synth
            .generate_quiz_cards(topic.as_str(), &hits, count)
            .await?;
        if generated.is_empty() {
            eprintln!("(model returned no quiz cards — the KB may be too sparse)");
            std::process::exit(3);
        }
        save_cards_to_cache(&cache_path, &generated)?;
        generated
    };

    // 4. Shuffle, then take up to `count`.
    shuffle(&mut cards);
    cards.truncate(count);

    let total = cards.len();
    println!();
    println!(
        "{}  ({} cards)",
        format!("Quiz: {topic_str}").bold().cyan(),
        total
    );
    println!();

    // 5. Interactive REPL.
    let mut knew = 0usize;
    let mut unsure = 0usize;
    let mut wrong = 0usize;
    let mut weak_areas: Vec<(String, Response)> = Vec::new();
    let mut spaced_entries: Vec<(QuizCard, Response)> = Vec::new();

    let separator = "━".repeat(45);

    for (i, card) in cards.iter().enumerate() {
        println!("Card {} of {}", i + 1, total);
        println!("{}", separator.dimmed());
        println!("{} {}", "Q:".bold().yellow(), card.question);
        println!();
        print!("Press Enter to reveal answer… ");
        std::io::stdout()
            .flush()
            .map_err(learn_core::LearnError::Io)?;
        {
            let stdin = std::io::stdin();
            let mut locked = stdin.lock();
            let mut _buf = String::new();
            locked
                .read_line(&mut _buf)
                .map_err(learn_core::LearnError::Io)?;
        }

        println!("{} {}", "A:".bold().green(), card.answer);
        println!();
        println!(
            "Source: youtu.be/{}?t={}",
            card.video_id, card.start_seconds as u64
        );
        println!();

        // Prompt for self-assessment with retry on unrecognised input.
        let response = loop {
            print!(
                "Did you know this? [{}]new it / [{}]nsure / [{}]rong  → ",
                "k".bold(),
                "u".bold(),
                "w".bold()
            );
            std::io::stdout()
                .flush()
                .map_err(learn_core::LearnError::Io)?;
            let line = {
                let stdin = std::io::stdin();
                let mut locked = stdin.lock();
                let mut buf = String::new();
                locked
                    .read_line(&mut buf)
                    .map_err(learn_core::LearnError::Io)?;
                buf
            };
            if let Some(r) = Response::parse(&line) {
                break r;
            }
            println!("  (type k, u, or w — or press Enter for 'knew it')");
        };

        match response {
            Response::Knew => {
                knew += 1;
                println!("{}", "✓ Knew it".green());
            }
            Response::Unsure => {
                unsure += 1;
                println!("{}", "~ Unsure".yellow());
                weak_areas.push((card.question.clone(), Response::Unsure));
            }
            Response::Wrong => {
                wrong += 1;
                println!("{}", "✗ Wrong".red());
                weak_areas.push((card.question.clone(), Response::Wrong));
            }
        }
        println!();

        if spaced {
            spaced_entries.push((card.clone(), response));
        }
    }

    // 6. Summary.
    println!("{}", separator.dimmed());
    println!(
        "{}  {} cards  ·  knew: {}  ·  unsure: {}  ·  wrong: {}",
        "Quiz complete!".bold(),
        total,
        knew.to_string().green(),
        unsure.to_string().yellow(),
        wrong.to_string().red(),
    );

    if !weak_areas.is_empty() {
        println!();
        println!("{}", "Weak areas to revisit:".bold());
        for (q, r) in &weak_areas {
            let label = match r {
                Response::Wrong => "(wrong)".red().to_string(),
                Response::Unsure => "(unsure)".yellow().to_string(),
                Response::Knew => String::new(),
            };
            // Trim long questions for display.
            let display: String = q.chars().take(70).collect();
            let ellipsis = if q.len() > 70 { "…" } else { "" };
            println!("  • {display}{ellipsis}  {label}");
        }
    }
    println!();

    // 7. Spaced-repetition: append next_review to the JSONL cache.
    if spaced && !spaced_entries.is_empty() {
        append_spaced_schedule(&cache_path, &spaced_entries)?;
        println!(
            "{}",
            "Spaced-repetition schedule written to cache.".dimmed()
        );
    }

    Ok(())
}

// ── Cache helpers ─────────────────────────────────────────────────────────────

fn load_cards_from_cache(path: &Utf8PathBuf) -> learn_core::Result<Vec<QuizCard>> {
    let raw = std::fs::read_to_string(path.as_std_path()).map_err(learn_core::LearnError::Io)?;
    let mut cards = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Each line is a JSON object; `next_review` key may be present — ignore it.
        let card: QuizCard = serde_json::from_str(line).map_err(learn_core::LearnError::Serde)?;
        cards.push(card);
    }
    Ok(cards)
}

fn save_cards_to_cache(path: &Utf8PathBuf, cards: &[QuizCard]) -> learn_core::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent.as_std_path()).map_err(learn_core::LearnError::Io)?;
    }
    let mut out = String::new();
    for card in cards {
        out.push_str(&serde_json::to_string(card).map_err(learn_core::LearnError::Serde)?);
        out.push('\n');
    }
    std::fs::write(path.as_std_path(), out.as_bytes()).map_err(learn_core::LearnError::Io)
}

/// Append `{ "chunk_id": "...", "next_review": "YYYY-MM-DD" }` lines to the
/// JSONL cache so a future scheduler can surface due cards.
fn append_spaced_schedule(
    path: &Utf8PathBuf,
    entries: &[(QuizCard, Response)],
) -> learn_core::Result<()> {
    use std::io::Write as _;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path.as_std_path())
        .map_err(learn_core::LearnError::Io)?;
    let mut writer = std::io::BufWriter::new(file);
    for (card, resp) in entries {
        let next = days_from_now_rfc3339(resp.spacing_days());
        let entry = serde_json::json!({
            "chunk_id": card.source_chunk_id,
            "next_review": next,
        });
        writer
            .write_all(format!("{entry}\n").as_bytes())
            .map_err(learn_core::LearnError::Io)?;
    }
    writer.flush().map_err(learn_core::LearnError::Io)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiz_cache_path_constructs_correctly() {
        let kb = Utf8PathBuf::from("/home/user/Docs/KB");
        let path = quiz_cache_path(&kb, "french-cooking");
        assert_eq!(
            path,
            Utf8PathBuf::from("/home/user/Docs/KB/_quiz/french-cooking.jsonl")
        );
    }

    #[test]
    fn parse_response_char_knew() {
        assert_eq!(Response::parse("k"), Some(Response::Knew));
        assert_eq!(Response::parse("K"), Some(Response::Knew));
        assert_eq!(Response::parse(""), Some(Response::Knew));
        assert_eq!(Response::parse("\n"), Some(Response::Knew));
        assert_eq!(Response::parse("  "), Some(Response::Knew));
    }

    #[test]
    fn parse_response_char_unsure() {
        assert_eq!(Response::parse("u"), Some(Response::Unsure));
        assert_eq!(Response::parse("U"), Some(Response::Unsure));
    }

    #[test]
    fn parse_response_char_wrong() {
        assert_eq!(Response::parse("w"), Some(Response::Wrong));
        assert_eq!(Response::parse("W"), Some(Response::Wrong));
    }

    #[test]
    fn parse_response_char_unknown_returns_none() {
        assert_eq!(Response::parse("x"), None);
        assert_eq!(Response::parse("?"), None);
        assert_eq!(Response::parse("yes"), None);
    }

    #[test]
    fn spacing_days_values() {
        assert_eq!(Response::Knew.spacing_days(), SPACING_KNEW_DAYS);
        assert_eq!(Response::Unsure.spacing_days(), SPACING_UNSURE_DAYS);
        assert_eq!(Response::Wrong.spacing_days(), SPACING_WRONG_DAYS);
    }

    #[test]
    fn shuffle_does_not_lose_elements() {
        let mut items: Vec<u32> = (0..20).collect();
        let original = items.clone();
        shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, original);
    }

    #[test]
    fn days_from_now_rfc3339_format() {
        let s = days_from_now_rfc3339(0);
        // Must be YYYY-MM-DD (10 chars).
        assert_eq!(s.len(), 10, "date string must be 10 chars; got: {s}");
        let parts: Vec<&str> = s.split('-').collect();
        assert_eq!(parts.len(), 3, "must have YYYY-MM-DD structure");
        assert_eq!(parts[0].len(), 4);
        assert_eq!(parts[1].len(), 2);
        assert_eq!(parts[2].len(), 2);
    }
}
