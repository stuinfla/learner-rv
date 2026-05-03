//! Prompt construction helpers for chat synthesis.
//!
//! Prior turns are condensed to avoid blowing up the context window.
//! Each prior assistant turn is truncated to 200 chars.

use crate::Turn;

/// Maximum characters per prior turn included in the context prefix.
const MAX_PRIOR_TURN_CHARS: usize = 200;

/// Maximum number of prior turn pairs (user+assistant) to include.
const MAX_PRIOR_PAIRS: usize = 5;

/// Build the user content to pass to the synthesizer.
///
/// When there are prior turns, we prepend a condensed conversation log so
/// the synthesizer can refer back to earlier answers. The current question
/// is always appended at the end.
pub fn build_user_content(question: &str, history: &[Turn]) -> String {
    if history.is_empty() {
        return question.to_string();
    }

    let mut parts: Vec<String> = Vec::new();
    parts.push("[Prior conversation (condensed)]:".to_string());

    // Take last MAX_PRIOR_PAIRS pairs (user+assistant).
    let pair_count = history.len() / 2;
    let skip_pairs = pair_count.saturating_sub(MAX_PRIOR_PAIRS);
    let skip_turns = skip_pairs * 2;

    for turn in history.iter().skip(skip_turns) {
        let label = match turn.role {
            crate::Role::User => "User",
            crate::Role::Assistant => "Assistant",
        };
        let text = truncate(&turn.content, MAX_PRIOR_TURN_CHARS);
        parts.push(format!("{label}: {text}"));
    }

    parts.push(String::new());
    parts.push(format!("[Current question]: {question}"));
    parts.join("\n")
}

/// Truncate `s` to at most `max` bytes (UTF-8 safe, appends "…" when cut).
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_user_content_empty_history_returns_question() {
        let result = build_user_content("what is this?", &[]);
        assert_eq!(result, "what is this?");
    }

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_long_string_appends_ellipsis() {
        let s = "a".repeat(300);
        let t = truncate(&s, 200);
        assert!(t.len() <= 204); // 200 + "…" (3 bytes UTF-8)
        assert!(t.ends_with('…'));
    }
}
