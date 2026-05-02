//! Heuristic scoring rubric for curriculum candidates.
//!
//! Five factors — each in [0.0, 1.0] — are combined as a weighted sum.
//!
//! | Factor            | Weight | Notes                                     |
//! |-------------------|--------|-------------------------------------------|
//! | title_alignment   | 0.35   | cosine sim of title embedding vs topic    |
//! | channel_authority | 0.25   | log10(view_count+1)/8.0 clamped to 1.0    |
//! | recency_bonus     | 0.20   | exp decay by age_days * recency_bias/365  |
//! | duration_sanity   | 0.15   | triangle: 0@0s, 1@600–3600s, 0@7200s     |
//! | has_captions      | 0.05   | 1.0=confirmed, 0.5=unknown, 0.0=absent   |
//!
//! The weights sum to exactly 1.0.

use crate::Candidate;
use crate::StudyDepth;

/// Type alias to avoid repeating the complex closure type.
pub type EmbedFn<'a> = &'a dyn Fn(&str) -> Vec<f32>;

/// Scoring configuration — exposes weights for documentation; values match the design memo.
pub struct ScoringConfig {
    pub title_alignment: f32,
    pub channel_authority: f32,
    pub recency_bonus: f32,
    pub duration_sanity: f32,
    pub has_captions: f32,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            title_alignment: 0.35,
            channel_authority: 0.25,
            recency_bonus: 0.20,
            duration_sanity: 0.15,
            has_captions: 0.05,
        }
    }
}

impl ScoringConfig {
    /// Weights must sum to 1.0 — verified in tests.
    pub fn weights_sum(&self) -> f32 {
        self.title_alignment
            + self.channel_authority
            + self.recency_bonus
            + self.duration_sanity
            + self.has_captions
    }
}

/// Score all candidates in place and sort descending by final score.
///
/// `topic_embed_fn`: if `Some`, called with a title string to produce a
/// 384-dim (or any-dim) embedding; cosine similarity against `topic_embed`
/// gives `title_alignment`. If `None`, 0.5 is used as a neutral placeholder.
pub fn score_candidates(
    candidates: &mut [Candidate],
    topic_embed: Option<&[f32]>,
    topic_embed_fn: Option<EmbedFn<'_>>,
    depth: &StudyDepth,
) {
    let cfg = ScoringConfig::default();
    let today = chrono::Utc::now().date_naive();

    for c in candidates.iter_mut() {
        let ta = compute_title_alignment(&c.title, topic_embed, topic_embed_fn);
        let ca = compute_channel_authority(c.view_count);
        let rb = compute_recency_bonus(c.upload_date.as_deref(), today, depth.recency_bias);
        let ds = compute_duration_sanity(c.duration_seconds, depth.allow_long_form);
        let hc = match c.has_captions {
            Some(true) => 1.0_f32,
            Some(false) => 0.0,
            None => 0.5,
        };

        c.score = cfg.title_alignment * ta
            + cfg.channel_authority * ca
            + cfg.recency_bonus * rb
            + cfg.duration_sanity * ds
            + cfg.has_captions * hc;
    }

    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn compute_title_alignment(
    title: &str,
    topic_embed: Option<&[f32]>,
    embed_fn: Option<EmbedFn<'_>>,
) -> f32 {
    match (topic_embed, embed_fn) {
        (Some(topic_vec), Some(f)) => {
            let title_vec = f(title);
            cosine_sim(&title_vec, topic_vec)
        }
        _ => 0.5,
    }
}

fn compute_channel_authority(view_count: Option<u64>) -> f32 {
    let v = view_count.unwrap_or(0) as f64;
    let score = (v + 1.0).log10() / 8.0;
    score.clamp(0.0, 1.0) as f32
}

/// `recency_bias` in [0.0, 1.0]: 0.0 = ignore date, 1.0 = strongly prefer recent.
/// Missing upload_date returns 0.5 (neutral) per memo.
pub fn compute_recency_bonus(
    upload_date: Option<&str>,
    today: chrono::NaiveDate,
    recency_bias: f32,
) -> f32 {
    let Some(date_str) = upload_date else {
        return 0.5;
    };
    // yt-dlp flat playlist returns YYYYMMDD
    let Ok(date) = chrono::NaiveDate::parse_from_str(date_str, "%Y%m%d") else {
        return 0.5;
    };
    let age_days = (today - date).num_days().max(0) as f32;
    let exponent = -age_days * recency_bias / 365.0;
    exponent.exp().clamp(0.0, 1.0)
}

/// Triangle function per design memo.
/// Ideal band: 600–3600 s. Extended upper: 7200 s when `allow_long_form`.
fn compute_duration_sanity(duration_seconds: Option<f64>, allow_long_form: bool) -> f32 {
    let Some(dur) = duration_seconds else {
        return 0.5;
    };
    let upper_ideal = 3600.0_f64;
    let upper_limit = if allow_long_form { 7200.0 } else { upper_ideal };
    let lower = 120.0_f64; // 2 min — penalise below this
    let ideal_start = 600.0_f64; // 10 min

    if dur <= 0.0 {
        return 0.0;
    }
    if dur < lower {
        return (dur / lower).clamp(0.0, 1.0) as f32;
    }
    if dur <= ideal_start {
        let partial = lower / ideal_start;
        return (partial + (1.0 - partial) * (dur - lower) / (ideal_start - lower)).clamp(0.0, 1.0)
            as f32;
    }
    if dur <= upper_ideal {
        return 1.0;
    }
    if dur >= upper_limit {
        return 0.0;
    }
    ((upper_limit - dur) / (upper_limit - upper_ideal)).clamp(0.0, 1.0) as f32
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.5;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.5;
    }
    (dot / (mag_a * mag_b)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_rubric_weights_sum_to_one() {
        let cfg = ScoringConfig::default();
        let sum = cfg.weights_sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights sum {sum} != 1.0");
    }

    #[test]
    fn score_rubric_recency_high_recency_bias_prefers_newer() {
        let today = chrono::NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let recent = "20251201"; // ~31 days ago
        let old = "20200101"; // ~6 years ago
        let bias = 1.0_f32;

        let r_recent = compute_recency_bonus(Some(recent), today, bias);
        let r_old = compute_recency_bonus(Some(old), today, bias);
        assert!(
            r_recent > r_old,
            "recency_bonus({recent})={r_recent} should exceed recency_bonus({old})={r_old}"
        );
    }

    #[test]
    fn recency_missing_date_returns_neutral() {
        let today = chrono::NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let score = compute_recency_bonus(None, today, 1.0);
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn duration_sanity_ideal_band_is_one() {
        let score = compute_duration_sanity(Some(1800.0), false);
        assert!((score - 1.0).abs() < 1e-6, "got {score}");
    }

    #[test]
    fn duration_sanity_penalises_extremes() {
        let short = compute_duration_sanity(Some(10.0), false);
        let ideal = compute_duration_sanity(Some(1200.0), false);
        let long_ = compute_duration_sanity(Some(7200.0), false);
        assert!(short < ideal);
        assert!(long_ < ideal);
    }
}
