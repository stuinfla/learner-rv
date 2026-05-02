//! `learn-eval` — runs per-topic golden Q&A regressions.
//!
//! # Usage
//!
//! ```text
//! learn eval <topic>
//! ```
//!
//! The harness loads `<kb_root>/<topic>/eval/golden.yaml`, runs each item
//! through the provided `Retriever` + `Synthesizer`, and returns an
//! `EvalReport` with per-question pass/fail and an aggregate score.
//!
//! Call `validate_golden` without real model instances for a dry-run that
//! only checks YAML structure.

#![deny(unsafe_code)]

use async_trait::async_trait;
use camino::Utf8Path;
use chrono::Utc;
use learn_core::{Answer, Hit, LearnError, Result};
use serde::{Deserialize, Serialize};

// ─── Golden set types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ItemMode {
    Ask,
    Apply,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenItem {
    pub id: String,
    pub question: String,
    pub mode: ItemMode,
    #[serde(default)]
    pub apply_task: Option<String>,
    #[serde(default)]
    pub apply_format: Option<String>,
    #[serde(default)]
    pub expected_substrings: Vec<String>,
    #[serde(default)]
    pub forbidden_substrings: Vec<String>,
    pub min_citations: usize,
    pub abstain_acceptable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenSet {
    pub topic: String,
    pub version: u32,
    pub items: Vec<GoldenItem>,
}

// ─── Report types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemResult {
    pub id: String,
    pub passed: bool,
    pub abstained: bool,
    pub answer_excerpt: String,
    pub citations_count: usize,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub topic: String,
    pub started_at: String,
    pub finished_at: String,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub abstained: usize,
    pub items: Vec<ItemResult>,
    pub aggregate_score: f32,
}

// ─── Trait abstractions ───────────────────────────────────────────────────────

#[async_trait]
pub trait Retriever {
    async fn search(&mut self, query: &str, k: usize) -> Result<Vec<Hit>>;
}

#[async_trait]
pub trait Synthesizer {
    async fn ask(&self, topic: &str, question: &str, hits: &[Hit]) -> Result<Answer>;
    async fn apply(&self, topic: &str, task: &str, format: &str, hits: &[Hit]) -> Result<Answer>;
}

// ─── Public functions ─────────────────────────────────────────────────────────

/// Load and deserialize a `GoldenSet` from a YAML file.
pub fn load_golden(path: &Utf8Path) -> Result<GoldenSet> {
    let text = std::fs::read_to_string(path.as_std_path())?;
    serde_yaml::from_str(&text).map_err(|e| {
        LearnError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
}

/// Validate golden set schema without executing any model calls (dry-run mode).
///
/// Checks:
/// - `topic` is non-empty
/// - Each item has a non-empty `id` and `question`
/// - `Apply` items have `apply_task`
/// - `min_citations >= 1` unless `abstain_acceptable == true`
pub fn validate_golden(set: &GoldenSet) -> Result<()> {
    if set.topic.trim().is_empty() {
        return Err(LearnError::Topic(
            "golden set topic must not be empty".into(),
        ));
    }
    for item in &set.items {
        if item.id.trim().is_empty() {
            return Err(LearnError::Topic(format!(
                "item with empty id found in topic {:?}",
                set.topic
            )));
        }
        if item.question.trim().is_empty() {
            return Err(LearnError::Topic(format!(
                "item {:?} has empty question",
                item.id
            )));
        }
        if item.mode == ItemMode::Apply && item.apply_task.is_none() {
            return Err(LearnError::Topic(format!(
                "item {:?} is Apply mode but has no apply_task",
                item.id
            )));
        }
        if item.min_citations == 0 && !item.abstain_acceptable {
            return Err(LearnError::Topic(format!(
                "item {:?}: min_citations=0 with abstain_acceptable=false is a misconfiguration",
                item.id
            )));
        }
    }
    Ok(())
}

/// Run the full harness against real (or mock) `Retriever` and `Synthesizer`.
pub async fn run_eval(
    set: &GoldenSet,
    retriever: &mut impl Retriever,
    synth: &dyn Synthesizer,
) -> Result<EvalReport> {
    let started_at = Utc::now().to_rfc3339();
    let mut results: Vec<ItemResult> = Vec::with_capacity(set.items.len());

    for item in &set.items {
        let hits = retriever.search(&item.question, 5).await?;

        let answer = match item.mode {
            ItemMode::Ask => synth.ask(&set.topic, &item.question, &hits).await?,
            ItemMode::Apply => {
                let task = item.apply_task.as_deref().unwrap_or("");
                let fmt = item.apply_format.as_deref().unwrap_or("text");
                synth.apply(&set.topic, task, fmt, &hits).await?
            }
        };

        let result = score_item(item, &answer);
        results.push(result);
    }

    let finished_at = Utc::now().to_rfc3339();
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let abstained = results.iter().filter(|r| r.abstained).count();
    let failed = total - passed;
    let aggregate_score = if total == 0 {
        0.0
    } else {
        passed as f32 / total as f32
    };

    Ok(EvalReport {
        topic: set.topic.clone(),
        started_at,
        finished_at,
        total,
        passed,
        failed,
        abstained,
        items: results,
        aggregate_score,
    })
}

// ─── Internal scoring ─────────────────────────────────────────────────────────

fn score_item(item: &GoldenItem, answer: &Answer) -> ItemResult {
    let excerpt: String = answer.text.chars().take(200).collect();
    let citations_count = answer.citations.len();

    if answer.abstained {
        let passed = item.abstain_acceptable;
        let reasoning = if passed {
            "model abstained — abstain is acceptable for this item".into()
        } else {
            "model abstained — expected an answer".into()
        };
        return ItemResult {
            id: item.id.clone(),
            passed,
            abstained: true,
            answer_excerpt: excerpt,
            citations_count,
            reasoning,
        };
    }

    let lower = answer.text.to_lowercase();

    let expected_ok = item.expected_substrings.is_empty()
        || item
            .expected_substrings
            .iter()
            .any(|s| lower.contains(&s.to_lowercase()));

    let forbidden_hit: Option<&String> = item
        .forbidden_substrings
        .iter()
        .find(|s| lower.contains(&s.to_lowercase()));

    let citations_ok = citations_count >= item.min_citations;

    let passed = expected_ok && forbidden_hit.is_none() && citations_ok;
    let reasoning = build_reasoning(
        expected_ok,
        forbidden_hit,
        citations_ok,
        item,
        citations_count,
    );

    ItemResult {
        id: item.id.clone(),
        passed,
        abstained: false,
        answer_excerpt: excerpt,
        citations_count,
        reasoning,
    }
}

fn build_reasoning(
    expected_ok: bool,
    forbidden_hit: Option<&String>,
    citations_ok: bool,
    item: &GoldenItem,
    citations_count: usize,
) -> String {
    let mut reasons: Vec<String> = Vec::new();

    if !expected_ok {
        reasons.push("missing expected substring".into());
    }
    if let Some(f) = forbidden_hit {
        reasons.push(format!("forbidden substring present: {f:?}"));
    }
    if !citations_ok {
        reasons.push(format!(
            "citations {citations_count} < min_citations {}",
            item.min_citations
        ));
    }

    if reasons.is_empty() {
        "pass: all checks satisfied".into()
    } else {
        format!("fail: {}", reasons.join("; "))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use learn_core::{Answer, Chunk, Citation, Hit};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use url::Url;

    // ── Sample YAML fixture ───────────────────────────────────────────────────

    const SAMPLE_YAML: &str = r#"
topic: french-cooking
version: 1
items:
  - id: lamination_basics
    question: "What is lamination and why does it matter for croissants?"
    mode: ask
    expected_substrings: ["butter", "fold"]
    forbidden_substrings: ["AI invented"]
    min_citations: 1
    abstain_acceptable: false
  - id: roux_technique
    question: "Describe the roux technique."
    mode: ask
    expected_substrings: ["flour", "fat"]
    forbidden_substrings: []
    min_citations: 1
    abstain_acceptable: false
"#;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_hit(text: &str) -> Hit {
        Hit {
            chunk: Chunk {
                chunk_id: "c1".into(),
                video_id: "v1".into(),
                start_seconds: 0.0,
                end_seconds: 10.0,
                text: text.into(),
                token_count: 5,
            },
            score: 0.9,
            rank: 0,
        }
    }

    fn make_citation() -> Citation {
        Citation {
            video_id: "v1".into(),
            title: Some("Test Video".into()),
            url: Url::parse("https://www.youtube.com/watch?v=abc123").unwrap(),
            start_seconds: 0.0,
        }
    }

    fn good_answer() -> Answer {
        Answer {
            text: "Lamination folds butter into the dough repeatedly.".into(),
            citations: vec![make_citation()],
            abstained: false,
        }
    }

    // ── Mock Retriever / Synthesizer ──────────────────────────────────────────

    struct MockRetriever {
        hits: Vec<Hit>,
    }

    #[async_trait]
    impl Retriever for MockRetriever {
        async fn search(&mut self, _query: &str, _k: usize) -> Result<Vec<Hit>> {
            Ok(self.hits.clone())
        }
    }

    struct MockSynth {
        answer: Answer,
    }

    #[async_trait]
    impl Synthesizer for MockSynth {
        async fn ask(&self, _t: &str, _q: &str, _h: &[Hit]) -> Result<Answer> {
            Ok(self.answer.clone())
        }
        async fn apply(&self, _t: &str, _tk: &str, _f: &str, _h: &[Hit]) -> Result<Answer> {
            Ok(self.answer.clone())
        }
    }

    // A synthesizer that cycles through a fixed list of answers.
    struct SequentialSynth {
        answers: Vec<Answer>,
        idx: AtomicUsize,
    }

    #[async_trait]
    impl Synthesizer for SequentialSynth {
        async fn ask(&self, _t: &str, _q: &str, _h: &[Hit]) -> Result<Answer> {
            let i = self.idx.fetch_add(1, Ordering::SeqCst);
            Ok(self.answers[i % self.answers.len()].clone())
        }
        async fn apply(&self, _t: &str, _tk: &str, _f: &str, _h: &[Hit]) -> Result<Answer> {
            Ok(self.answers[0].clone())
        }
    }

    // ── Unit tests ────────────────────────────────────────────────────────────

    #[test]
    fn load_golden_parses_sample_yaml() {
        let set: GoldenSet = serde_yaml::from_str(SAMPLE_YAML).unwrap();
        assert_eq!(set.items.len(), 2);
        assert_eq!(set.topic, "french-cooking");
        assert_eq!(set.version, 1);
        assert_eq!(set.items[0].id, "lamination_basics");
    }

    #[test]
    fn validate_golden_rejects_empty_topic() {
        let set = GoldenSet {
            topic: "  ".into(),
            version: 1,
            items: vec![],
        };
        let err = validate_golden(&set).unwrap_err();
        assert!(err.to_string().contains("topic"));
    }

    #[test]
    fn validate_golden_rejects_zero_min_citations_with_abstain_unacceptable() {
        let item = GoldenItem {
            id: "x".into(),
            question: "What is X?".into(),
            mode: ItemMode::Ask,
            apply_task: None,
            apply_format: None,
            expected_substrings: vec![],
            forbidden_substrings: vec![],
            min_citations: 0,
            abstain_acceptable: false,
        };
        let set = GoldenSet {
            topic: "test-topic".into(),
            version: 1,
            items: vec![item],
        };
        let err = validate_golden(&set).unwrap_err();
        assert!(err.to_string().contains("misconfiguration"));
    }

    #[tokio::test]
    async fn run_eval_with_mock_retriever_and_synthesizer_passes_known_good() {
        let set: GoldenSet = serde_yaml::from_str(SAMPLE_YAML).unwrap();
        // Answer contains both "butter" and "fold" (satisfies item[0])
        // and "flour" and "fat" (satisfies item[1])
        let answer = Answer {
            text: "Lamination folds butter into dough; roux combines flour and fat.".into(),
            citations: vec![make_citation()],
            abstained: false,
        };
        let mut retriever = MockRetriever {
            hits: vec![make_hit("butter fold technique")],
        };
        let synth = MockSynth { answer };
        let report = run_eval(&set, &mut retriever, &synth).await.unwrap();
        assert_eq!(report.total, 2);
        assert_eq!(report.passed, 2);
        assert_eq!(report.failed, 0);
    }

    #[tokio::test]
    async fn run_eval_with_mock_returns_fail_when_forbidden_substring_present() {
        let set: GoldenSet = serde_yaml::from_str(SAMPLE_YAML).unwrap();
        // Contains "AI invented" which is forbidden in item[0]
        let bad_answer = Answer {
            text: "Lamination — AI invented this technique — uses butter and fold.".into(),
            citations: vec![make_citation()],
            abstained: false,
        };
        let mut retriever = MockRetriever {
            hits: vec![make_hit("lamination")],
        };
        let synth = MockSynth { answer: bad_answer };
        let report = run_eval(&set, &mut retriever, &synth).await.unwrap();
        assert!(
            !report.items[0].passed,
            "item[0] should fail due to forbidden substring"
        );
        assert!(
            report.items[0].reasoning.contains("forbidden"),
            "reasoning should mention forbidden"
        );
    }

    #[tokio::test]
    async fn run_eval_aggregates_correctly_with_partial_passes() {
        let set: GoldenSet = serde_yaml::from_str(SAMPLE_YAML).unwrap();

        // item[0] needs "butter" or "fold" — this answer has them
        let pass_answer = good_answer();

        // item[1] needs "flour" or "fat" — this answer has neither
        let fail_answer = Answer {
            text: "A roux is cooked starch mixed with water.".into(),
            citations: vec![make_citation()],
            abstained: false,
        };

        let synth = SequentialSynth {
            answers: vec![pass_answer, fail_answer],
            idx: AtomicUsize::new(0),
        };
        let mut retriever = MockRetriever { hits: vec![] };
        let report = run_eval(&set, &mut retriever, &synth).await.unwrap();

        assert_eq!(report.total, 2);
        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);
        assert!(
            (report.aggregate_score - 0.5).abs() < f32::EPSILON,
            "aggregate_score should be 0.5, got {}",
            report.aggregate_score
        );
    }
}
