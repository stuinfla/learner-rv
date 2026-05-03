//! `learn cloud` — SVG word-cloud visualisation of a KB topic (or all topics).
//!
//! Two modes:
//! - `learn cloud <topic>` — single-topic word cloud (1200×800 SVG)
//! - `learn cloud`          — meta-cloud across ALL topics as a grid of cards

use camino::Utf8PathBuf;
use learn_core::{Chunk, LearnError, Result};
use std::collections::HashMap;

// ── Stop-word list (~100 common English words) ────────────────────────────────

const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any",
    "are", "aren", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "cannot", "could", "couldn", "did", "didn", "do", "does", "doesn",
    "doing", "don", "down", "during", "each", "few", "for", "from", "further", "get", "gets",
    "got", "had", "hadn", "has", "hasn", "have", "haven", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "however", "if", "in", "into", "is", "isn", "it",
    "its", "itself", "just", "know", "let", "like", "ll", "me", "more", "most", "my", "myself",
    "need", "no", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "out",
    "over", "own", "re", "same", "she", "should", "shouldn", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "then", "there", "these", "they", "this", "through", "to",
    "too", "up", "us", "use", "used", "using", "ve", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "will", "with", "would", "you", "your",
];

// ── Sidecar loading ───────────────────────────────────────────────────────────

/// Load chunks from a topic's `.meta.json` sidecar by parsing via serde_json::Value
/// so we don't need `serde` as a direct crate dependency in learn-cli.
fn load_chunks(path: &Utf8PathBuf) -> Result<Vec<Chunk>> {
    let bytes = std::fs::read(path.as_std_path()).map_err(LearnError::Io)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(LearnError::Serde)?;
    let chunks_obj = value
        .get("chunks")
        .and_then(|v| v.as_object())
        .ok_or_else(|| LearnError::Index("meta.json missing 'chunks' field".to_owned()))?;
    let mut out = Vec::with_capacity(chunks_obj.len());
    for (_key, v) in chunks_obj {
        let chunk: Chunk = serde_json::from_value(v.clone()).map_err(LearnError::Serde)?;
        out.push(chunk);
    }
    Ok(out)
}

// ── Public entry-points ───────────────────────────────────────────────────────

/// Single-topic word cloud.
pub fn run_cloud_topic(
    topic_str: &str,
    out: Option<Utf8PathBuf>,
    print_html: bool,
    kb_root: &Utf8PathBuf,
) -> Result<()> {
    let meta_path = kb_root.join(format!("{topic_str}.meta.json"));
    let chunks = load_chunks(&meta_path)?;

    if chunks.is_empty() {
        return Err(LearnError::Index(format!(
            "topic '{topic_str}' has no chunks in meta sidecar"
        )));
    }

    let video_ids: std::collections::HashSet<&str> =
        chunks.iter().map(|c| c.video_id.as_str()).collect();
    let chunk_count = chunks.len();
    let freq_map = frequency_map(chunks.iter().map(|c| c.text.as_str()));
    let top_words = top_n(&freq_map, 100);
    let total_tokens: u32 = top_words.iter().map(|(_, f)| f).sum();

    let title = format!(
        "{} \u{2014} {} video{}, {} chunks, {} tokens",
        topic_str,
        video_ids.len(),
        if video_ids.len() == 1 { "" } else { "s" },
        chunk_count,
        total_tokens,
    );

    let svg = render_word_cloud(&title, &top_words, 1200, 800);

    let out_path = out.unwrap_or_else(|| kb_root.join(format!("{topic_str}.cloud.svg")));
    write_svg(
        &out_path,
        &svg,
        print_html,
        &title,
        chunk_count,
        video_ids.len(),
    )?;
    println!("wrote {out_path}");
    Ok(())
}

/// Meta-cloud across all topics (grid of cards).
pub fn run_cloud_meta(
    out: Option<Utf8PathBuf>,
    print_html: bool,
    kb_root: &Utf8PathBuf,
) -> Result<()> {
    let entries = glob_topic_metas(kb_root)?;
    if entries.is_empty() {
        println!("No topic meta files found in {kb_root}");
        return Ok(());
    }

    let total_chunks: usize = entries.iter().map(|e| e.chunk_count).sum();
    let kb_bytes = kb_size_bytes(kb_root);
    let title = format!(
        "Learn-RV Knowledge Base \u{2014} {} topic{}, {} total chunks, {:.1} MB",
        entries.len(),
        if entries.len() == 1 { "" } else { "s" },
        total_chunks,
        kb_bytes as f64 / 1_048_576.0,
    );

    let svg = render_meta_cloud(&title, &entries, 1600, 1200);

    let meta_dir = kb_root.join("_meta");
    std::fs::create_dir_all(meta_dir.as_std_path()).map_err(LearnError::Io)?;
    let out_path = out.unwrap_or_else(|| meta_dir.join("all-topics-cloud.svg"));
    write_svg(
        &out_path,
        &svg,
        print_html,
        &title,
        total_chunks,
        entries.len(),
    )?;
    println!("wrote {out_path}");
    Ok(())
}

// ── Tokenisation ──────────────────────────────────────────────────────────────

/// Tokenise text: lowercase, split on non-alpha, filter stopwords and tokens < 4 chars.
pub fn tokenize(text: &str) -> Vec<String> {
    let stop: std::collections::HashSet<&str> = STOPWORDS.iter().copied().collect();
    text.to_lowercase()
        .split(|c: char| !c.is_alphabetic())
        .filter(|tok| {
            tok.len() >= 4 && !stop.contains(*tok) && !tok.chars().all(|c| c.is_ascii_digit())
        })
        .map(str::to_owned)
        .collect()
}

/// Build a frequency map from an iterator of text slices.
pub fn frequency_map<'a>(texts: impl Iterator<Item = &'a str>) -> HashMap<String, u32> {
    let mut map: HashMap<String, u32> = HashMap::new();
    for text in texts {
        for tok in tokenize(text) {
            *map.entry(tok).or_insert(0) += 1;
        }
    }
    map
}

/// Return the top `n` (word, freq) pairs sorted by frequency descending.
pub fn top_n(freq: &HashMap<String, u32>, n: usize) -> Vec<(String, u32)> {
    let mut pairs: Vec<(String, u32)> = freq.iter().map(|(k, v)| (k.clone(), *v)).collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    pairs.truncate(n);
    pairs
}

/// Compute font size for a word given its frequency and max frequency.
/// Clamped to [12, 72] px using sqrt scaling.
pub fn font_size(freq: u32, max_freq: u32) -> u32 {
    if max_freq == 0 {
        return 12;
    }
    let scale = (freq as f64 / max_freq as f64).sqrt();
    let size = (72.0 * scale).round() as u32;
    size.clamp(12, 72)
}

// ── Layout helpers ────────────────────────────────────────────────────────────

/// A placed word in the cloud.
struct Placed {
    word: String,
    x: i32,
    y: i32,
    font_sz: u32,
    rotate: bool,
    color: String,
}

/// Greedy bounding-box placement. Returns placed words within (width × height).
fn layout_words(words: &[(String, u32)], width: i32, height: i32) -> Vec<Placed> {
    let max_freq = words.first().map(|(_, f)| *f).unwrap_or(1);
    let total = words.len() as f64;
    let mut placed: Vec<(i32, i32, i32, i32)> = Vec::new(); // (x, y, w, h)
    let mut result = Vec::new();

    let pad_x = 20_i32;
    let pad_y = 60_i32;
    let mut cursor_x = pad_x;
    let mut cursor_y = pad_y + 20;
    let mut row_height = 0_i32;
    let line_gap = 8_i32;

    for (idx, (word, freq)) in words.iter().enumerate() {
        let fs = font_size(*freq, max_freq) as i32;
        let rotate = idx % 4 == 3;

        let char_w = (fs as f32 * 0.6) as i32;
        let (box_w, box_h) = if rotate {
            (fs + 4, word.len() as i32 * char_w + 4)
        } else {
            (word.len() as i32 * char_w + 4, fs + 4)
        };

        if cursor_x + box_w > width - pad_x {
            cursor_x = pad_x;
            cursor_y += row_height + line_gap;
            row_height = 0;
        }

        if cursor_y + box_h > height - 20 {
            break;
        }

        let x = cursor_x;
        let y = cursor_y;
        let overlaps = placed.iter().any(|(px, py, pw, ph)| {
            x < px + pw && x + box_w > *px && y < py + ph && y + box_h > *py
        });
        if overlaps {
            cursor_x += 4;
            continue;
        }

        placed.push((x, y, box_w, box_h));

        let pct = 1.0 - idx as f64 / total;
        let color = lerp_color(pct);

        result.push(Placed {
            word: word.clone(),
            x,
            y: y + fs,
            font_sz: fs as u32,
            rotate,
            color,
        });

        cursor_x += box_w + 6;
        if box_h > row_height {
            row_height = box_h;
        }
    }
    result
}

/// Linearly interpolate between #7CCBFF (low-freq, blue) and #FF8856 (high-freq, orange).
fn lerp_color(t: f64) -> String {
    let t = t.clamp(0.0, 1.0);
    let r = (0x7C as f64 + (0xFF as f64 - 0x7C as f64) * t).round() as u8;
    let g = (0xCB as f64 + (0x88 as f64 - 0xCB as f64) * t).round() as u8;
    let b = (0xFF as f64 + (0x56 as f64 - 0xFF as f64) * t).round() as u8;
    format!("#{r:02X}{g:02X}{b:02X}")
}

// ── SVG renderers ─────────────────────────────────────────────────────────────

fn render_word_cloud(title: &str, words: &[(String, u32)], width: i32, height: i32) -> String {
    let placed = layout_words(words, width, height - 60);

    let bg = "#1a1e2e";
    let title_fill = "#c8d0e0";

    let mut svg = String::with_capacity(32_768);
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\" \
         width=\"{width}\" height=\"{height}\">"
    ));
    svg.push_str(&format!(
        "<rect width=\"{width}\" height=\"{height}\" fill=\"{bg}\"/>"
    ));
    svg.push_str(&format!(
        "<text x=\"20\" y=\"40\" font-family=\"sans-serif\" font-size=\"18\" \
         fill=\"{title_fill}\" font-weight=\"bold\">{}</text>",
        xml_escape(title)
    ));

    for p in &placed {
        let transform = if p.rotate {
            format!(
                " transform=\"rotate(-90,{},{})\"",
                p.x + p.font_sz as i32 / 2,
                p.y
            )
        } else {
            String::new()
        };
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"{}\" \
             fill=\"{}\"{}>{}</text>",
            p.x,
            p.y + 60,
            p.font_sz,
            p.color,
            transform,
            xml_escape(&p.word)
        ));
    }

    svg.push_str("</svg>");
    svg
}

/// One card summary for the meta-cloud.
pub struct TopicCard {
    pub name: String,
    pub chunk_count: usize,
    pub video_count: usize,
    pub top_words: Vec<(String, u32)>,
}

fn render_meta_cloud(title: &str, cards: &[TopicCard], width: i32, height: i32) -> String {
    let cols = 3_i32;
    let rows = ((cards.len() as i32) + cols - 1) / cols;
    let card_w = (width - 40) / cols;
    let card_h = if rows > 0 {
        ((height - 80) / rows).min(220)
    } else {
        220
    };

    let bg = "#1a1e2e";
    let title_fill = "#c8d0e0";
    let card_bg = "#252a3d";
    let card_stroke = "#3a4060";
    let name_fill = "#e0e8ff";
    let subtitle_fill = "#8899bb";

    let mut svg = String::with_capacity(65_536);
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\" \
         width=\"{width}\" height=\"{height}\">"
    ));
    svg.push_str(&format!(
        "<rect width=\"{width}\" height=\"{height}\" fill=\"{bg}\"/>"
    ));
    svg.push_str(&format!(
        "<text x=\"20\" y=\"48\" font-family=\"sans-serif\" font-size=\"22\" \
         fill=\"{title_fill}\" font-weight=\"bold\">{}</text>",
        xml_escape(title)
    ));

    for (i, card) in cards.iter().enumerate() {
        let col = i as i32 % cols;
        let row = i as i32 / cols;
        let cx = 20 + col * (card_w + 10);
        let cy = 70 + row * (card_h + 10);

        svg.push_str(&format!(
            "<rect x=\"{cx}\" y=\"{cy}\" width=\"{card_w}\" height=\"{card_h}\" \
             rx=\"8\" fill=\"{card_bg}\" stroke=\"{card_stroke}\" stroke-width=\"1\"/>"
        ));
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"15\" \
             fill=\"{name_fill}\" font-weight=\"bold\">{}</text>",
            cx + 10,
            cy + 24,
            xml_escape(&card.name)
        ));
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"11\" \
             fill=\"{subtitle_fill}\">{} chunks \u{00b7} {} video{}</text>",
            cx + 10,
            cy + 40,
            card.chunk_count,
            card.video_count,
            if card.video_count == 1 { "" } else { "s" },
        ));

        let max_f = card.top_words.first().map(|(_, f)| *f).unwrap_or(1);
        let mut wx = cx + 10;
        let mut wy = cy + 60;
        for (idx, (word, freq)) in card.top_words.iter().enumerate() {
            let fs = font_size(*freq, max_f).clamp(10, 28) as i32;
            let char_w = (fs as f32 * 0.6) as i32;
            let word_w = word.len() as i32 * char_w + 6;
            if wx + word_w > cx + card_w - 10 {
                wx = cx + 10;
                wy += fs + 6;
            }
            if wy + fs > cy + card_h - 8 {
                break;
            }
            let pct = 1.0 - idx as f64 / card.top_words.len() as f64;
            let color = lerp_color(pct);
            svg.push_str(&format!(
                "<text x=\"{wx}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"{fs}\" \
                 fill=\"{color}\">{}</text>",
                wy + fs,
                xml_escape(word)
            ));
            wx += word_w;
        }
    }

    svg.push_str("</svg>");
    svg
}

// ── File I/O ──────────────────────────────────────────────────────────────────

fn write_svg(
    path: &Utf8PathBuf,
    svg: &str,
    print_html: bool,
    title: &str,
    chunk_count: usize,
    video_count: usize,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent.as_std_path()).map_err(LearnError::Io)?;
    }
    if print_html {
        let now = crate::rfc3339_now();
        let html = build_html_wrapper(svg, title, chunk_count, video_count, &now);
        let html_path = path.with_extension("html");
        std::fs::write(html_path.as_std_path(), html.as_bytes()).map_err(LearnError::Io)?;
    }
    std::fs::write(path.as_std_path(), svg.as_bytes()).map_err(LearnError::Io)?;
    Ok(())
}

fn build_html_wrapper(
    svg: &str,
    title: &str,
    chunk_count: usize,
    video_count: usize,
    timestamp: &str,
) -> String {
    let title_esc = xml_escape(title);
    let plural = if video_count == 1 { "" } else { "s" };
    format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\
         <meta charset=\"UTF-8\"><title>{title_esc}</title>\n\
         <style>body{{margin:0;background:#1a1e2e;font-family:sans-serif;}}\
         .bar{{padding:12px 20px;background:#252a3d;color:#c8d0e0;font-size:14px;\
         border-bottom:1px solid #3a4060;}}\
         .bar span{{margin-right:24px;}}</style></head>\n\
         <body><div class=\"bar\">\n\
         <span><b>{title_esc}</b></span>\n\
         <span>{chunk_count} chunks</span>\n\
         <span>{video_count} video{plural}</span>\n\
         <span>Generated {timestamp}</span>\n\
         </div>\n{svg}\n</body></html>"
    )
}

/// Glob `<kb_root>/*.meta.json`, return one TopicCard per file.
fn glob_topic_metas(kb_root: &Utf8PathBuf) -> Result<Vec<TopicCard>> {
    let mut cards = Vec::new();
    let read_dir = std::fs::read_dir(kb_root.as_std_path()).map_err(LearnError::Io)?;
    for entry in read_dir.flatten() {
        let p = entry.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.ends_with(".meta.json") {
            continue;
        }
        let topic_name = name.trim_end_matches(".meta.json").to_owned();
        // Skip hidden/system or dotted names.
        if topic_name.starts_with('_') || topic_name.contains('.') {
            continue;
        }
        let meta_path = Utf8PathBuf::from_path_buf(p.clone()).unwrap_or_default();
        let chunks = match load_chunks(&meta_path) {
            Ok(v) if !v.is_empty() => v,
            _ => continue,
        };
        let video_ids: std::collections::HashSet<&str> =
            chunks.iter().map(|c| c.video_id.as_str()).collect();
        let freq = frequency_map(chunks.iter().map(|c| c.text.as_str()));
        let top_words = top_n(&freq, 5);
        cards.push(TopicCard {
            name: topic_name,
            chunk_count: chunks.len(),
            video_count: video_ids.len(),
            top_words,
        });
    }
    cards.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(cards)
}

/// Approximate total size of `<kb_root>/*.rvf` files in bytes.
fn kb_size_bytes(kb_root: &Utf8PathBuf) -> u64 {
    let Ok(rd) = std::fs::read_dir(kb_root.as_std_path()) else {
        return 0;
    };
    rd.flatten()
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .map(|x| x == "rvf")
                .unwrap_or(false)
        })
        .filter_map(|e| e.metadata().ok().map(|m| m.len()))
        .sum()
}

/// Escape XML special characters for SVG text nodes.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_filters_stopwords_and_short_tokens() {
        let tokens = tokenize("the quick brown fox jumps over lazy dog and a cat");
        // "the", "and", "a", "over" are stopwords; "fox" (3 chars) and "cat" (3 chars) filtered
        assert!(
            !tokens.contains(&"the".to_owned()),
            "'the' must be filtered (stopword)"
        );
        assert!(
            !tokens.contains(&"and".to_owned()),
            "'and' must be filtered (stopword)"
        );
        assert!(
            !tokens.contains(&"over".to_owned()),
            "'over' must be filtered (stopword)"
        );
        assert!(tokens.contains(&"quick".to_owned()), "'quick' must be kept");
        assert!(tokens.contains(&"brown".to_owned()), "'brown' must be kept");
        assert!(tokens.contains(&"jumps".to_owned()), "'jumps' must be kept");
        assert!(tokens.contains(&"lazy".to_owned()), "'lazy' must be kept");
    }

    #[test]
    fn tokenize_filters_short_tokens_under_4_chars() {
        let tokens = tokenize("fox cat data information");
        // "fox" (3) and "cat" (3) filtered; "data" (4) and "information" kept
        assert!(
            !tokens.contains(&"fox".to_owned()),
            "3-char token must be filtered"
        );
        assert!(
            !tokens.contains(&"cat".to_owned()),
            "3-char token must be filtered"
        );
        assert!(
            tokens.contains(&"data".to_owned()),
            "4-char token must be kept"
        );
        assert!(
            tokens.contains(&"information".to_owned()),
            "'information' must be kept"
        );
    }

    #[test]
    fn frequency_count_aggregates_across_chunks() {
        let chunks = vec!["hello world hello", "hello world extra"];
        let freq = frequency_map(chunks.into_iter());
        assert_eq!(
            *freq.get("hello").unwrap_or(&0),
            3,
            "hello must appear 3 times"
        );
        assert_eq!(
            *freq.get("world").unwrap_or(&0),
            2,
            "world must appear 2 times"
        );
        assert_eq!(
            *freq.get("extra").unwrap_or(&0),
            1,
            "extra must appear 1 time"
        );
    }

    #[test]
    fn font_size_scales_correctly_for_top_word_vs_low_word() {
        // Top word (freq == max) must produce 72px.
        assert_eq!(font_size(100, 100), 72, "top word must produce 72px");
        // Low word must be smaller.
        let low = font_size(1, 100);
        let high = font_size(100, 100);
        assert!(
            high > low,
            "top-word font_size {high} must be > low-word font_size {low}"
        );
        assert!(low >= 12, "font_size must be >= 12, got {low}");
        assert!(high <= 72, "font_size must be <= 72, got {high}");
        // Zero max_freq must not panic.
        assert_eq!(font_size(0, 0), 12, "font_size(0,0) must return 12");
    }

    #[test]
    fn xml_escape_handles_special_chars() {
        assert_eq!(
            xml_escape("a & b < c > d \"e\""),
            "a &amp; b &lt; c &gt; d &quot;e&quot;"
        );
    }

    #[test]
    fn top_n_returns_sorted_by_freq_descending() {
        let mut map = HashMap::new();
        map.insert("apple".to_owned(), 5u32);
        map.insert("banana".to_owned(), 10u32);
        map.insert("cherry".to_owned(), 2u32);
        let top = top_n(&map, 2);
        assert_eq!(top[0].0, "banana", "highest-freq word must be first");
        assert_eq!(top[1].0, "apple", "second-highest must be second");
        assert_eq!(top.len(), 2, "must truncate to n=2");
    }

    #[test]
    fn lerp_color_returns_valid_hex() {
        let low = lerp_color(0.0);
        let high = lerp_color(1.0);
        assert!(
            low.starts_with('#') && low.len() == 7,
            "low color must be a 7-char hex: {low}"
        );
        assert!(
            high.starts_with('#') && high.len() == 7,
            "high color must be a 7-char hex: {high}"
        );
        // Low-freq end is blue-ish (#7CCBFF).
        assert_eq!(
            low.to_uppercase(),
            "#7CCBFF",
            "low color must match blue anchor"
        );
        // High-freq end is orange-ish (#FF8856).
        assert_eq!(
            high.to_uppercase(),
            "#FF8856",
            "high color must match orange anchor"
        );
    }

    /// Integration test: generate cloud for verified-demo and validate SVG.
    /// Requires ~/Docs/KB/verified-demo.meta.json to exist.
    #[test]
    #[ignore = "requires real KB at ~/Docs/KB/verified-demo.meta.json"]
    fn integration_cloud_verified_demo() {
        let home = dirs::home_dir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(home.join("Docs").join("KB")).unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let out = Utf8PathBuf::from_path_buf(out_dir.path().join("test.cloud.svg")).unwrap();

        run_cloud_topic("verified-demo", Some(out.clone()), false, &kb_root).unwrap();

        let bytes = std::fs::read(out.as_std_path()).unwrap();
        assert!(!bytes.is_empty(), "SVG must be non-empty");
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("<svg"), "output must contain <svg");
        assert!(text.contains("</svg>"), "output must close with </svg>");
        assert!(
            text.len() >= 4_096,
            "SVG must be >= 4 KB, got {} bytes",
            text.len()
        );
    }
}
