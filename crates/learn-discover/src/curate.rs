//! Anthropic curation call — sends scored candidates to Claude Opus for
//! sub-topic clustering and prerequisite ordering.
//!
//! If `ANTHROPIC_API_KEY` is unset or empty, the function fails-soft by
//! returning a heuristic-only list (empty sub_topic, "(heuristic only)" rationale).

use crate::{Candidate, CurriculumPick};
use learn_core::{Result, VideoRef};
use url::Url;

const SYSTEM_PROMPT: &str = "\
You are a curriculum designer. The user wants to learn about a specific topic.
You are given a list of YouTube video candidates with metadata and heuristic scores.
Your job: select the best videos that together form a coherent, progressive curriculum.

Rules:
1. Prioritise breadth: cover distinct sub-topics, not the same ground twice.
2. Order from prerequisites and fundamentals to advanced material.
3. Deduplicate: if two videos cover identical ground, keep the higher-scored one.
4. Prefer videos with confirmed captions (has_captions: true).
5. Output ONLY valid JSON — no prose before or after.";

const USER_TEMPLATE: &str = "\
Topic: {{topic_description}}

Select exactly {{surface_count}} videos from the candidates below.
Return a JSON array with this schema per element:
[
  {
    \"video_id\": \"<string>\",
    \"sub_topic\": \"<one short phrase describing what this video covers>\",
    \"rationale\": \"<one sentence: why this video belongs in the curriculum>\",
    \"rank\": <1-based integer, 1 = watch first>,
    \"prerequisite_ranks\": [<list of rank integers that should be watched before this one>]
  }
]

Candidates (JSON array):
{{candidates_json}}";

/// Attempt to curate via the Anthropic API.
/// Falls back to heuristic ordering if the key is missing.
pub async fn curate(
    candidates: &[Candidate],
    topic: &str,
    surface_count: usize,
) -> Result<Vec<CurriculumPick>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
    if api_key.trim().is_empty() {
        tracing::warn!("ANTHROPIC_API_KEY not set — falling back to heuristic-only curriculum");
        return Ok(heuristic_picks(candidates, surface_count));
    }

    match call_anthropic(candidates, topic, surface_count, &api_key).await {
        Ok(picks) => Ok(picks),
        Err(e) => {
            tracing::warn!("Anthropic curation failed ({e}); falling back to heuristic ranking");
            Ok(heuristic_picks(candidates, surface_count))
        }
    }
}

async fn call_anthropic(
    candidates: &[Candidate],
    topic: &str,
    surface_count: usize,
    api_key: &str,
) -> std::result::Result<Vec<CurriculumPick>, String> {
    let candidates_json = build_candidates_json(candidates);
    let user_msg = USER_TEMPLATE
        .replace("{{topic_description}}", topic)
        .replace("{{surface_count}}", &surface_count.to_string())
        .replace("{{candidates_json}}", &candidates_json);

    let body = serde_json::json!({
        "model": "claude-opus-4-7",
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "messages": [{ "role": "user", "content": user_msg }]
    });

    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Anthropic API {status}: {text}"));
    }

    let resp_json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("response parse: {e}"))?;

    let content = resp_json["content"][0]["text"]
        .as_str()
        .ok_or("missing content[0].text")?;

    parse_curation_response(content, candidates)
}

fn build_candidates_json(candidates: &[Candidate]) -> String {
    let arr: Vec<serde_json::Value> = candidates
        .iter()
        .map(|c| {
            serde_json::json!({
                "video_id": c.video_id,
                "title": c.title,
                "channel": c.channel,
                "duration_seconds": c.duration_seconds,
                "upload_date": c.upload_date,
                "score": c.score,
                "has_captions": c.has_captions.unwrap_or(false),
            })
        })
        .collect();
    serde_json::to_string(&arr).unwrap_or_default()
}

fn parse_curation_response(
    content: &str,
    candidates: &[Candidate],
) -> std::result::Result<Vec<CurriculumPick>, String> {
    // Trim any accidental markdown fences
    let trimmed = content
        .trim()
        .trim_start_matches("```json")
        .trim_end_matches("```")
        .trim();
    let arr: Vec<serde_json::Value> =
        serde_json::from_str(trimmed).map_err(|e| format!("JSON parse: {e}"))?;

    let candidate_map: std::collections::HashMap<&str, &Candidate> = candidates
        .iter()
        .map(|c| (c.video_id.as_str(), c))
        .collect();

    let mut picks = Vec::new();
    for item in &arr {
        let video_id = item["video_id"].as_str().unwrap_or("").to_owned();
        let sub_topic = item["sub_topic"].as_str().unwrap_or("").to_owned();
        let rationale = item["rationale"].as_str().unwrap_or("").to_owned();
        let rank = item["rank"].as_u64().unwrap_or(0) as usize;

        let Some(c) = candidate_map.get(video_id.as_str()) else {
            tracing::warn!("Anthropic returned unknown video_id {video_id}, skipping");
            continue;
        };

        picks.push(CurriculumPick {
            video: candidate_to_videoref(c),
            sub_topic,
            rationale,
            rank,
        });
    }

    picks.sort_by_key(|p| p.rank);
    Ok(picks)
}

fn heuristic_picks(candidates: &[Candidate], surface_count: usize) -> Vec<CurriculumPick> {
    candidates
        .iter()
        .take(surface_count)
        .enumerate()
        .map(|(i, c)| CurriculumPick {
            video: candidate_to_videoref(c),
            sub_topic: String::new(),
            rationale: "(heuristic only)".to_owned(),
            rank: i + 1,
        })
        .collect()
}

fn candidate_to_videoref(c: &Candidate) -> VideoRef {
    let url = format!("https://www.youtube.com/watch?v={}", c.video_id)
        .parse::<Url>()
        .expect("video URL is always valid");
    VideoRef {
        video_id: c.video_id.clone(),
        url,
        title: Some(c.title.clone()),
        channel: c.channel.clone(),
        channel_id: c.channel_id.clone(),
        duration_seconds: c.duration_seconds,
        published_at: c.upload_date.clone(),
    }
}

/// Render the curation prompt as it would be sent (for testing / logging).
pub fn render_curation_prompt(topic: &str, surface_count: usize, candidates_json: &str) -> String {
    USER_TEMPLATE
        .replace("{{topic_description}}", topic)
        .replace("{{surface_count}}", &surface_count.to_string())
        .replace("{{candidates_json}}", candidates_json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curation_prompt_includes_topic_and_candidates_placeholders() {
        let topic = "advanced Rust async programming";
        let surface_count = 5;
        let candidates_json = r#"[{"video_id":"abc","title":"Rust async"}]"#;
        let rendered = render_curation_prompt(topic, surface_count, candidates_json);

        assert!(
            rendered.contains(topic),
            "rendered prompt missing topic: {rendered}"
        );
        assert!(
            rendered.contains(&surface_count.to_string()),
            "rendered prompt missing surface_count: {rendered}"
        );
        assert!(
            rendered.contains(candidates_json),
            "rendered prompt missing candidates_json: {rendered}"
        );
        // Template placeholders must be gone
        assert!(!rendered.contains("{{topic_description}}"));
        assert!(!rendered.contains("{{surface_count}}"));
        assert!(!rendered.contains("{{candidates_json}}"));
    }

    #[test]
    fn system_prompt_is_non_empty_and_contains_key_rules() {
        assert!(SYSTEM_PROMPT.contains("curriculum designer"));
        assert!(SYSTEM_PROMPT.contains("valid JSON"));
    }
}
