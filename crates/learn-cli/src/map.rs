//! `learn map` — UMAP/PCA galaxy of KB chunks projected into 2-D concept space.
//!
//! Two modes:
//! - `learn map`          — all topics, writes `~/Docs/KB/_meta/knowledge-map.svg`
//! - `learn map <topic>`  — single topic, writes `~/Docs/KB/<topic>.map.svg`
//!
//! ## Algorithm
//! 1. Load all `<topic>.emb.bin` + `<topic>.meta.json` files (or one topic).
//! 2. PCA projection to 2-D via power iteration (hand-rolled, no new crate).
//! 3. K-means clustering: k = clamp(sqrt(N), 3, 12).
//! 4. Top-5 concept words per cluster from chunk text (TF, filtered by stopwords).
//! 5. SVG layout: left 60% galaxy, right 40% legend + cluster labels + stats.

use camino::Utf8PathBuf;
use learn_core::{Chunk, LearnError, Result};
use serde::Deserialize;
use std::collections::HashMap;

use crate::cloud::{frequency_map, top_n};

// ── Constants ─────────────────────────────────────────────────────────────────

const MAX_CHUNKS: usize = 2_000;
const SVG_W: f64 = 1600.0;
const SVG_H: f64 = 1000.0;
/// Left panel is 60% of width.
const GALAXY_W: f64 = SVG_W * 0.60;
const PANEL_X: f64 = GALAXY_W + 20.0;
#[allow(dead_code)]
const PANEL_W: f64 = SVG_W - PANEL_X - 20.0;

/// 20 visually distinct topic colours (dark-background friendly).
const TOPIC_PALETTE: &[&str] = &[
    "#4FC3F7", "#FF8A65", "#81C784", "#F06292", "#FFD54F", "#BA68C8", "#4DB6AC", "#FF7043",
    "#AED581", "#E57373", "#4DD0E1", "#FFB74D", "#9575CD", "#A5D6A7", "#F48FB1", "#26C6DA",
    "#FFA726", "#66BB6A", "#EF5350", "#AB47BC",
];

// ── Sidecar deserialization (mirrors learn-index private struct) ───────────────

#[derive(Deserialize, Default)]
struct MetaSidecar {
    #[allow(dead_code)]
    dimension: u16,
    #[serde(default)]
    chunks: HashMap<String, Chunk>,
}

// ── FNV-1a 64-bit (mirrors learn-index private function) ─────────────────────

fn chunk_id_to_u64(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for b in s.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ── Per-chunk point (post-projection) ─────────────────────────────────────────

pub(crate) struct Point {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) topic: String,
    pub(crate) text: String,
    #[allow(dead_code)]
    pub(crate) video_id: String,
}

// ── Public entry-points ────────────────────────────────────────────────────────

/// Generate a knowledge map for all topics.
pub fn run_map_all(out: Option<Utf8PathBuf>, kb_root: &Utf8PathBuf) -> Result<()> {
    let (topics, chunks_per_topic, embeddings_per_topic) = load_all_topics(kb_root)?;

    if topics.is_empty() {
        println!("No topic meta files found in {kb_root}");
        return Ok(());
    }

    let points = project_to_2d(&topics, &chunks_per_topic, &embeddings_per_topic)?;

    if points.is_empty() {
        println!("No embeddings found across topics — ingest some content first.");
        return Ok(());
    }

    let total_chunks: usize = chunks_per_topic.values().map(|v| v.len()).sum();
    let total_videos: usize = unique_video_count(&chunks_per_topic);
    let kb_bytes = kb_size_bytes(kb_root);
    let date_str = crate::rfc3339_now()
        .split('T')
        .next()
        .unwrap_or("unknown")
        .to_owned();

    let svg = render_map_svg(
        &points,
        &topics,
        total_chunks,
        total_videos,
        kb_bytes,
        &date_str,
    );

    let meta_dir = kb_root.join("_meta");
    std::fs::create_dir_all(meta_dir.as_std_path()).map_err(LearnError::Io)?;
    let out_path = out.unwrap_or_else(|| meta_dir.join("knowledge-map.svg"));
    std::fs::write(out_path.as_std_path(), svg.as_bytes()).map_err(LearnError::Io)?;
    println!("wrote {out_path}");
    Ok(())
}

/// Generate a knowledge map for a single topic.
pub fn run_map_topic(
    topic_str: &str,
    out: Option<Utf8PathBuf>,
    kb_root: &Utf8PathBuf,
) -> Result<()> {
    let meta_path = kb_root.join(format!("{topic_str}.meta.json"));
    let emb_path = kb_root.join(format!("{topic_str}.emb.bin"));

    let chunks = load_meta(&meta_path)?;
    let embeddings = load_emb(&emb_path)?;

    if embeddings.is_empty() {
        return Err(LearnError::Index(format!(
            "topic '{topic_str}' has no embeddings — run `learn ingest` first"
        )));
    }

    let topics = vec![topic_str.to_owned()];
    let mut chunks_per_topic = HashMap::new();
    chunks_per_topic.insert(topic_str.to_owned(), chunks);
    let mut embeddings_per_topic = HashMap::new();
    embeddings_per_topic.insert(topic_str.to_owned(), embeddings);

    let points = project_to_2d(&topics, &chunks_per_topic, &embeddings_per_topic)?;

    let total_chunks = chunks_per_topic.values().map(|v| v.len()).sum();
    let total_videos = unique_video_count(&chunks_per_topic);
    let kb_bytes = {
        let rvf_path = kb_root.join(format!("{topic_str}.rvf"));
        std::fs::metadata(rvf_path.as_std_path())
            .map(|m| m.len())
            .unwrap_or(0)
    };
    let date_str = crate::rfc3339_now()
        .split('T')
        .next()
        .unwrap_or("unknown")
        .to_owned();

    let svg = render_map_svg(
        &points,
        &topics,
        total_chunks,
        total_videos,
        kb_bytes,
        &date_str,
    );

    let out_path = out.unwrap_or_else(|| kb_root.join(format!("{topic_str}.map.svg")));
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent.as_std_path()).map_err(LearnError::Io)?;
    }
    std::fs::write(out_path.as_std_path(), svg.as_bytes()).map_err(LearnError::Io)?;
    println!("wrote {out_path}");
    Ok(())
}

// ── Data loading ───────────────────────────────────────────────────────────────

/// Returns (topic_names, chunks_per_topic, embeddings_per_topic).
#[allow(clippy::type_complexity)]
fn load_all_topics(
    kb_root: &Utf8PathBuf,
) -> Result<(
    Vec<String>,
    HashMap<String, Vec<Chunk>>,
    HashMap<String, HashMap<u64, Vec<f32>>>,
)> {
    let mut topics = Vec::new();
    let mut chunks_map: HashMap<String, Vec<Chunk>> = HashMap::new();
    let mut emb_map: HashMap<String, HashMap<u64, Vec<f32>>> = HashMap::new();

    let rd = std::fs::read_dir(kb_root.as_std_path()).map_err(LearnError::Io)?;
    for entry in rd.flatten() {
        let p = entry.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.ends_with(".meta.json") {
            continue;
        }
        let topic_name = name.trim_end_matches(".meta.json").to_owned();
        if topic_name.starts_with('_') || topic_name.contains('.') {
            continue;
        }
        let meta_path = Utf8PathBuf::from_path_buf(p).unwrap_or_default();
        let emb_path = kb_root.join(format!("{topic_name}.emb.bin"));

        let chunks = load_meta(&meta_path).unwrap_or_default();
        let embeddings = load_emb(&emb_path).unwrap_or_default();

        if embeddings.is_empty() {
            continue;
        }

        topics.push(topic_name.clone());
        chunks_map.insert(topic_name.clone(), chunks);
        emb_map.insert(topic_name, embeddings);
    }
    topics.sort();
    Ok((topics, chunks_map, emb_map))
}

fn load_meta(meta_path: &Utf8PathBuf) -> Result<Vec<Chunk>> {
    if !meta_path.exists() {
        return Ok(Vec::new());
    }
    let bytes = std::fs::read(meta_path.as_std_path()).map_err(LearnError::Io)?;
    let sidecar: MetaSidecar = serde_json::from_slice(&bytes).map_err(LearnError::Serde)?;
    Ok(sidecar.chunks.into_values().collect())
}

/// Load `.emb.bin` into id→vector map. Uses the same format as learn-index.
fn load_emb(emb_path: &Utf8PathBuf) -> Result<HashMap<u64, Vec<f32>>> {
    if !emb_path.exists() {
        return Ok(HashMap::new());
    }
    let raw = std::fs::read(emb_path.as_std_path()).map_err(LearnError::Io)?;
    if raw.len() < 12 {
        return Ok(HashMap::new());
    }
    let dim = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(raw[4..12].try_into().unwrap()) as usize;
    if dim == 0 || count == 0 {
        return Ok(HashMap::new());
    }
    let _record_size = 8 + 4 + dim * 4;
    let mut map: HashMap<u64, Vec<f32>> = HashMap::with_capacity(count);
    let mut offset = 12usize;
    while offset + 8 + 4 <= raw.len() {
        let id = u64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap());
        offset += 8;
        let vec_bytes = u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + vec_bytes > raw.len() {
            break;
        }
        let vec: Vec<f32> = raw[offset..offset + vec_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        map.insert(id, vec);
        offset += vec_bytes;
    }
    Ok(map)
}

// ── Projection pipeline ────────────────────────────────────────────────────────

/// Build `Point` vec by pairing chunk metadata with its embedding, then PCA to 2-D.
fn project_to_2d(
    topics: &[String],
    chunks_per_topic: &HashMap<String, Vec<Chunk>>,
    embeddings_per_topic: &HashMap<String, HashMap<u64, Vec<f32>>>,
) -> Result<Vec<Point>> {
    // 1. Gather (chunk, embedding, topic) triples.
    let mut triples: Vec<(Chunk, Vec<f32>, String)> = Vec::new();
    for topic in topics {
        let chunks = chunks_per_topic
            .get(topic)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let embs = match embeddings_per_topic.get(topic) {
            Some(m) => m,
            None => continue,
        };
        for chunk in chunks {
            let id = chunk_id_to_u64(&chunk.chunk_id);
            if let Some(emb) = embs.get(&id) {
                triples.push((chunk.clone(), emb.clone(), topic.clone()));
            }
        }
    }

    if triples.is_empty() {
        return Ok(Vec::new());
    }

    // 2. Subsample to MAX_CHUNKS, preserving per-topic ratio.
    let triples = subsample(triples, topics, MAX_CHUNKS);

    // 3. PCA to 2-D.
    let vecs: Vec<&[f32]> = triples.iter().map(|(_, e, _)| e.as_slice()).collect();
    let coords = pca_2d(&vecs);

    // 4. Rescale to [pad, galaxy_dim - pad].
    let pad = 30.0_f64;
    let galaxy_w = GALAXY_W - pad * 2.0;
    let galaxy_h = SVG_H - pad * 2.0 - 60.0; // leave room for title

    let (min_x, max_x) = min_max(coords.iter().map(|c| c[0]));
    let (min_y, max_y) = min_max(coords.iter().map(|c| c[1]));
    let rx = if max_x > min_x { max_x - min_x } else { 1.0 };
    let ry = if max_y > min_y { max_y - min_y } else { 1.0 };

    let points = triples
        .into_iter()
        .zip(coords)
        .map(|((chunk, _, topic), coord)| {
            let x = (coord[0] - min_x) / rx * galaxy_w + pad;
            let y = (coord[1] - min_y) / ry * galaxy_h + pad + 60.0;
            Point {
                x,
                y,
                topic,
                text: chunk.text,
                video_id: chunk.video_id,
            }
        })
        .collect();

    Ok(points)
}

/// Random-subsample preserving per-topic ratio using a deterministic LCG.
fn subsample(
    mut triples: Vec<(Chunk, Vec<f32>, String)>,
    topics: &[String],
    max: usize,
) -> Vec<(Chunk, Vec<f32>, String)> {
    if triples.len() <= max {
        return triples;
    }
    let total = triples.len();
    // Per-topic budget proportional to its share.
    let mut topic_counts: HashMap<String, usize> = HashMap::new();
    for (_, _, t) in &triples {
        *topic_counts.entry(t.clone()).or_insert(0) += 1;
    }
    let mut budget: HashMap<String, usize> = HashMap::new();
    for topic in topics {
        let cnt = topic_counts.get(topic).copied().unwrap_or(0);
        budget.insert(topic.clone(), (cnt * max / total).max(1));
    }
    // Fill remaining from largest topics.
    let used: usize = budget.values().sum();
    let remaining = max.saturating_sub(used);
    let mut sorted_topics: Vec<String> = topics.to_vec();
    sorted_topics.sort_by(|a, b| topic_counts.get(b).cmp(&topic_counts.get(a)));
    let mut extra = remaining;
    for t in &sorted_topics {
        if extra == 0 {
            break;
        }
        *budget.entry(t.clone()).or_insert(0) += 1;
        extra -= 1;
    }

    // Shuffle within each topic using deterministic LCG, then take budget.
    triples.sort_by(|a, b| a.2.cmp(&b.2));
    let mut result = Vec::with_capacity(max);
    let mut i = 0;
    while i < triples.len() {
        let topic = triples[i].2.clone();
        let j = triples[i..].partition_point(|(_, _, t)| *t == topic) + i;
        let slice = &mut triples[i..j];
        let b = budget.get(&topic).copied().unwrap_or(0);
        lcg_shuffle(slice);
        result.extend_from_slice(&slice[..b.min(slice.len())]);
        i = j;
    }
    result
}

/// Deterministic LCG shuffle (Knuth variant).
fn lcg_shuffle<T>(slice: &mut [T]) {
    let n = slice.len();
    if n < 2 {
        return;
    }
    let mut state: u64 = 0x1234_5678_abcd_ef01;
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let j = (state >> 33) as usize % (i + 1);
        slice.swap(i, j);
    }
}

// ── Hand-rolled PCA (power iteration for top 2 eigenvectors) ─────────────────

/// Project `n` vectors of dimension `d` to 2-D via PCA.
///
/// 1. Centre: subtract per-dimension mean.
/// 2. Power iteration to find first eigenvector of covariance (X^T X).
/// 3. Deflate and repeat for second eigenvector.
/// 4. Project each centred vector onto the two eigenvectors.
pub fn pca_2d(vecs: &[&[f32]]) -> Vec<[f64; 2]> {
    let n = vecs.len();
    if n == 0 {
        return Vec::new();
    }
    let d = vecs[0].len();
    if d < 2 {
        return vecs.iter().map(|_| [0.0, 0.0]).collect();
    }

    // Centre.
    let mean: Vec<f64> = (0..d)
        .map(|j| vecs.iter().map(|v| v[j] as f64).sum::<f64>() / n as f64)
        .collect();
    let centred: Vec<Vec<f64>> = vecs
        .iter()
        .map(|v| {
            v.iter()
                .enumerate()
                .map(|(j, x)| *x as f64 - mean[j])
                .collect()
        })
        .collect();

    // First eigenvector.
    let ev1 = power_iter(&centred, d, 60);
    // Deflate.
    let deflated = deflate(&centred, &ev1);
    // Second eigenvector.
    let ev2 = power_iter(&deflated, d, 60);

    // Project.
    centred
        .iter()
        .map(|row| [dot(row, &ev1), dot(row, &ev2)])
        .collect()
}

/// Power iteration: find the dominant eigenvector of X^T X (covariance).
fn power_iter(data: &[Vec<f64>], d: usize, iters: usize) -> Vec<f64> {
    // Start with all-ones (deterministic seed).
    let mut v: Vec<f64> = vec![1.0 / (d as f64).sqrt(); d];
    for _ in 0..iters {
        // w = X^T (X v)
        let xv: Vec<f64> = data.iter().map(|row| dot(row, &v)).collect();
        let mut w: Vec<f64> = vec![0.0; d];
        for (row, xvi) in data.iter().zip(&xv) {
            for (j, rj) in row.iter().enumerate() {
                w[j] += rj * xvi;
            }
        }
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break;
        }
        v = w.iter().map(|x| x / norm).collect();
    }
    v
}

/// Deflate: remove the component along `ev` from every row.
fn deflate(data: &[Vec<f64>], ev: &[f64]) -> Vec<Vec<f64>> {
    data.iter()
        .map(|row| {
            let proj = dot(row, ev);
            row.iter().zip(ev).map(|(r, e)| r - proj * e).collect()
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn min_max(iter: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for v in iter {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if mn == f64::INFINITY {
        (0.0, 1.0)
    } else if mn == mx {
        (mn - 0.5, mx + 0.5)
    } else {
        (mn, mx)
    }
}

// ── K-means ───────────────────────────────────────────────────────────────────

/// Run k-means on the 2-D points, return cluster assignment per point.
/// k = clamp(floor(sqrt(n)), 3, 12).
pub fn kmeans_2d(points: &[[f64; 2]], k_override: Option<usize>) -> Vec<usize> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k_override
        .unwrap_or_else(|| (n as f64).sqrt().floor() as usize)
        .clamp(3, 12)
        .min(n);

    // Initialise centroids by spacing evenly over sorted x.
    let mut sorted_idx: Vec<usize> = (0..n).collect();
    sorted_idx.sort_by(|&a, &b| points[a][0].partial_cmp(&points[b][0]).unwrap());
    let step = n / k;
    let mut centroids: Vec<[f64; 2]> = (0..k)
        .map(|i| points[sorted_idx[(i * step).min(n - 1)]])
        .collect();

    let mut labels = vec![0usize; n];
    for _ in 0..50 {
        // Assign.
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let new_label = (0..k)
                .min_by(|&a, &b| {
                    dist2(p, &centroids[a])
                        .partial_cmp(&dist2(p, &centroids[b]))
                        .unwrap()
                })
                .unwrap_or(0);
            if labels[i] != new_label {
                labels[i] = new_label;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Update centroids.
        let mut sums = vec![[0.0_f64, 0.0_f64]; k];
        let mut counts = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            let c = labels[i];
            sums[c][0] += p[0];
            sums[c][1] += p[1];
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                centroids[c] = [sums[c][0] / counts[c] as f64, sums[c][1] / counts[c] as f64];
            }
        }
    }
    labels
}

fn dist2(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

// ── Cluster label extraction ───────────────────────────────────────────────────

/// For each cluster, collect chunk texts, compute TF, return top 5 words.
///
/// Filters stopwords and tokens under 4 chars (delegates to `cloud::tokenize`).
fn cluster_top_words(points: &[Point], labels: &[usize], k: usize) -> Vec<Vec<String>> {
    let mut cluster_texts: Vec<Vec<&str>> = vec![Vec::new(); k];
    for (i, p) in points.iter().enumerate() {
        if i < labels.len() {
            cluster_texts[labels[i]].push(p.text.as_str());
        }
    }
    cluster_texts
        .iter()
        .map(|texts| {
            let freq = frequency_map(texts.iter().copied());
            top_n(&freq, 5).into_iter().map(|(w, _)| w).collect()
        })
        .collect()
}

// ── SVG rendering ─────────────────────────────────────────────────────────────

fn render_map_svg(
    points: &[Point],
    topics: &[String],
    total_chunks: usize,
    total_videos: usize,
    kb_bytes: u64,
    date_str: &str,
) -> String {
    let n = points.len();

    // Build topic → colour mapping.
    let topic_colors: HashMap<String, &str> = topics
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), TOPIC_PALETTE[i % TOPIC_PALETTE.len()]))
        .collect();

    // K-means clustering.
    let coords: Vec<[f64; 2]> = points.iter().map(|p| [p.x, p.y]).collect();
    let labels = kmeans_2d(&coords, None);
    let k = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let top_words = cluster_top_words(points, &labels, k);

    // Cluster centroids (for glow + label placement).
    let centroids = compute_centroids(&coords, &labels, k);

    let bg = "#0e1117";
    let panel_bg = "#161b2e";

    let mut svg = String::with_capacity(200_000);
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" \
         viewBox=\"0 0 {SVG_W} {SVG_H}\" \
         width=\"{SVG_W}\" height=\"{SVG_H}\">"
    ));

    // Background.
    svg.push_str(&format!(
        "<rect width=\"{SVG_W}\" height=\"{SVG_H}\" fill=\"{bg}\"/>"
    ));

    // Panel background (right 40%).
    svg.push_str(&format!(
        "<rect x=\"{GALAXY_W}\" y=\"0\" width=\"{}\" height=\"{SVG_H}\" fill=\"{panel_bg}\"/>",
        SVG_W - GALAXY_W
    ));

    // Title.
    let title = format!("Learn-RV Knowledge Map — {date_str}");
    svg.push_str(&format!(
        "<text x=\"20\" y=\"36\" font-family=\"monospace,sans-serif\" font-size=\"20\" \
         fill=\"#e8eaf6\" font-weight=\"bold\">{}</text>",
        xml_escape(&title)
    ));

    // Galaxy divider line.
    svg.push_str(&format!(
        "<line x1=\"{GALAXY_W}\" y1=\"0\" x2=\"{GALAXY_W}\" y2=\"{SVG_H}\" \
         stroke=\"#2a3050\" stroke-width=\"1\"/>"
    ));

    // Cluster glow circles at centroids.
    for (ci, centroid) in centroids.iter().enumerate() {
        let col = TOPIC_PALETTE[ci % TOPIC_PALETTE.len()];
        svg.push_str(&format!(
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"40\" fill=\"{col}\" opacity=\"0.04\"/>",
            centroid[0], centroid[1]
        ));
    }

    // Data points.
    for p in points {
        let col = topic_colors.get(&p.topic).copied().unwrap_or("#aaaaaa");
        let title_text = xml_escape(&format!(
            "{} | {}",
            p.topic,
            &p.text[..p.text.len().min(60)]
        ));
        svg.push_str(&format!(
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"{col}\" opacity=\"0.75\">\
             <title>{title_text}</title></circle>",
            p.x, p.y
        ));
    }

    // ── Right panel ───────────────────────────────────────────────────────────

    let panel_x = PANEL_X;
    let mut panel_y = 30.0_f64;

    // Legend title.
    svg.push_str(&format!(
        "<text x=\"{panel_x}\" y=\"{panel_y}\" font-family=\"monospace,sans-serif\" \
         font-size=\"14\" fill=\"#9fa8da\" font-weight=\"bold\">TOPICS</text>",
    ));
    panel_y += 20.0;

    // Legend entries (sorted by chunk count descending).
    let mut topic_chunk_counts: Vec<(String, usize)> = topics
        .iter()
        .map(|t| {
            let cnt = points.iter().filter(|p| p.topic == *t).count();
            (t.clone(), cnt)
        })
        .collect();
    topic_chunk_counts.sort_by(|a, b| b.1.cmp(&a.1));

    for (t, cnt) in &topic_chunk_counts {
        let col = topic_colors.get(t).copied().unwrap_or("#aaaaaa");
        svg.push_str(&format!(
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"5\" fill=\"{col}\"/>",
            panel_x + 6.0,
            panel_y - 3.0
        ));
        let label = format!("{t} ({cnt})");
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{panel_y}\" font-family=\"monospace,sans-serif\" \
             font-size=\"11\" fill=\"#c5cae9\">{}</text>",
            panel_x + 16.0,
            xml_escape(&label)
        ));
        panel_y += 16.0;
        if panel_y > SVG_H * 0.45 {
            break; // Don't overflow panel
        }
    }

    // Cluster labels section.
    panel_y += 16.0;
    svg.push_str(&format!(
        "<text x=\"{panel_x}\" y=\"{panel_y}\" font-family=\"monospace,sans-serif\" \
         font-size=\"14\" fill=\"#9fa8da\" font-weight=\"bold\">CLUSTERS</text>",
    ));
    panel_y += 18.0;

    for (ci, words) in top_words.iter().enumerate() {
        if words.is_empty() {
            continue;
        }
        let col = TOPIC_PALETTE[ci % TOPIC_PALETTE.len()];
        let label = format!("#{}: {}", ci + 1, words.join(", "));
        let truncated = if label.len() > 38 {
            format!("{}…", &label[..38])
        } else {
            label
        };
        svg.push_str(&format!(
            "<text x=\"{panel_x}\" y=\"{panel_y}\" font-family=\"monospace,sans-serif\" \
             font-size=\"10\" fill=\"{col}\">{}</text>",
            xml_escape(&truncated)
        ));
        panel_y += 14.0;
        if panel_y > SVG_H * 0.85 {
            break;
        }
    }

    // Stats bar at bottom of right panel.
    let stats_y = SVG_H - 80.0;
    svg.push_str(&format!(
        "<line x1=\"{GALAXY_W}\" y1=\"{stats_y}\" x2=\"{SVG_W}\" y2=\"{stats_y}\" \
         stroke=\"#2a3050\" stroke-width=\"1\"/>"
    ));
    let stats = format!(
        "{} topics  •  {} chunks  •  {} videos  •  {:.1} MB",
        topics.len(),
        total_chunks,
        total_videos,
        kb_bytes as f64 / 1_048_576.0
    );
    svg.push_str(&format!(
        "<text x=\"{panel_x}\" y=\"{:.1}\" font-family=\"monospace,sans-serif\" \
         font-size=\"11\" fill=\"#7986cb\">{}</text>",
        stats_y + 18.0,
        xml_escape(&stats)
    ));
    let rendered = format!("{n} points  •  {k} clusters  •  PCA projection");
    svg.push_str(&format!(
        "<text x=\"{panel_x}\" y=\"{:.1}\" font-family=\"monospace,sans-serif\" \
         font-size=\"10\" fill=\"#546e7a\">{}</text>",
        stats_y + 34.0,
        xml_escape(&rendered)
    ));

    // Cluster number labels in the galaxy.
    for (ci, centroid) in centroids.iter().enumerate() {
        let col = TOPIC_PALETTE[ci % TOPIC_PALETTE.len()];
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"monospace,sans-serif\" \
             font-size=\"9\" fill=\"{col}\" opacity=\"0.6\">#{}</text>",
            centroid[0] + 4.0,
            centroid[1] - 4.0,
            ci + 1
        ));
    }

    svg.push_str("</svg>");
    svg
}

fn compute_centroids(coords: &[[f64; 2]], labels: &[usize], k: usize) -> Vec<[f64; 2]> {
    let mut sums = vec![[0.0_f64, 0.0_f64]; k];
    let mut counts = vec![0usize; k];
    for (i, c) in coords.iter().enumerate() {
        if i < labels.len() {
            let cl = labels[i];
            sums[cl][0] += c[0];
            sums[cl][1] += c[1];
            counts[cl] += 1;
        }
    }
    (0..k)
        .map(|ci| {
            if counts[ci] > 0 {
                [
                    sums[ci][0] / counts[ci] as f64,
                    sums[ci][1] / counts[ci] as f64,
                ]
            } else {
                [0.0, 0.0]
            }
        })
        .collect()
}

fn unique_video_count(chunks_per_topic: &HashMap<String, Vec<Chunk>>) -> usize {
    let mut ids = std::collections::HashSet::new();
    for chunks in chunks_per_topic.values() {
        for c in chunks {
            ids.insert(c.video_id.clone());
        }
    }
    ids.len()
}

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

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// PCA should separate synthetic clusters that live at orthogonal corners.
    #[test]
    fn pca_2d_projection_separates_orthogonal_synthetic_clusters() {
        // Three clusters: A near e1, B near e2, C near -e1.
        // We use 1024-dim vectors with most dims near 0.
        let dim = 64usize; // smaller for test speed
        let n_per_cluster = 20usize;

        let mut vecs: Vec<Vec<f32>> = Vec::new();
        // Cluster A: strong component at dim 0.
        for i in 0..n_per_cluster {
            let mut v = vec![0.0f32; dim];
            v[0] = 1.0 + (i as f32) * 0.01;
            vecs.push(v);
        }
        // Cluster B: strong component at dim 1.
        for i in 0..n_per_cluster {
            let mut v = vec![0.0f32; dim];
            v[1] = 1.0 + (i as f32) * 0.01;
            vecs.push(v);
        }
        // Cluster C: strong negative component at dim 0.
        for i in 0..n_per_cluster {
            let mut v = vec![0.0f32; dim];
            v[0] = -1.0 - (i as f32) * 0.01;
            vecs.push(v);
        }

        let refs: Vec<&[f32]> = vecs.iter().map(Vec::as_slice).collect();
        let coords = pca_2d(&refs);
        assert_eq!(coords.len(), n_per_cluster * 3);

        // A and C should have opposite signs on one axis.
        let mean_a_x =
            coords[..n_per_cluster].iter().map(|c| c[0]).sum::<f64>() / n_per_cluster as f64;
        let mean_c_x = coords[2 * n_per_cluster..]
            .iter()
            .map(|c| c[0])
            .sum::<f64>()
            / n_per_cluster as f64;

        // A and C are on opposite sides of the first principal component.
        assert!(
            (mean_a_x - mean_c_x).abs() > 0.1,
            "A and C clusters should be separated on first PC; got mean_a={mean_a_x:.4}, mean_c={mean_c_x:.4}"
        );
    }

    /// K-means should recover three well-separated synthetic clusters.
    #[test]
    fn kmeans_recovers_synthetic_clusters() {
        // Three tight clusters at (-5,0), (5,0), (0,5).
        let mut points: Vec<[f64; 2]> = Vec::new();
        for i in 0..10 {
            let off = i as f64 * 0.05;
            points.push([-5.0 + off, off]);
        }
        for i in 0..10 {
            let off = i as f64 * 0.05;
            points.push([5.0 + off, off]);
        }
        for i in 0..10 {
            let off = i as f64 * 0.05;
            points.push([off, 5.0 + off]);
        }

        let labels = kmeans_2d(&points, Some(3));
        assert_eq!(labels.len(), 30);

        // Each group of 10 should share the same label.
        let g0: std::collections::HashSet<usize> = labels[..10].iter().copied().collect();
        let g1: std::collections::HashSet<usize> = labels[10..20].iter().copied().collect();
        let g2: std::collections::HashSet<usize> = labels[20..].iter().copied().collect();

        assert_eq!(
            g0.len(),
            1,
            "first cluster must be uniform; got labels {:?}",
            &labels[..10]
        );
        assert_eq!(
            g1.len(),
            1,
            "second cluster must be uniform; got labels {:?}",
            &labels[10..20]
        );
        assert_eq!(
            g2.len(),
            1,
            "third cluster must be uniform; got labels {:?}",
            &labels[20..]
        );
        assert_ne!(g0, g1, "cluster 0 and 1 must differ");
        assert_ne!(g1, g2, "cluster 1 and 2 must differ");
    }

    /// top_words should filter stopwords and return highest-frequency terms.
    #[test]
    fn top_words_per_cluster_filters_stopwords() {
        let points = vec![
            Point {
                x: 0.0,
                y: 0.0,
                topic: "t".into(),
                text: "the quick brown fox jumps over the lazy dog and the dog runs fast".into(),
                video_id: "v1".into(),
            },
            Point {
                x: 0.0,
                y: 0.0,
                topic: "t".into(),
                text: "quick brown fox leaps over quick steps forward quickly".into(),
                video_id: "v1".into(),
            },
        ];
        let labels = vec![0, 0];
        let result = cluster_top_words(&points, &labels, 1);
        assert_eq!(result.len(), 1);
        let words = &result[0];
        // "the", "and", "over" are stopwords — must not appear.
        assert!(
            !words.contains(&"the".to_owned()),
            "stopword 'the' must be filtered; got: {words:?}"
        );
        assert!(
            !words.contains(&"and".to_owned()),
            "stopword 'and' must be filtered; got: {words:?}"
        );
        // "quick" is 5 chars, not a stopword — must appear.
        assert!(
            words.contains(&"quick".to_owned()),
            "'quick' should be in top words; got: {words:?}"
        );
    }

    /// Integration: generate map for the live KB (requires real KB).
    #[test]
    #[ignore = "requires real KB at ~/Docs/KB with embeddings"]
    fn integration_map_generates_valid_svg() {
        let home = dirs::home_dir().unwrap();
        let kb_root = Utf8PathBuf::from_path_buf(home.join("Docs").join("KB")).unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let out = Utf8PathBuf::from_path_buf(out_dir.path().join("test-map.svg")).unwrap();

        run_map_all(Some(out.clone()), &kb_root).unwrap();

        let meta = std::fs::metadata(out.as_std_path()).unwrap();
        assert!(
            meta.len() >= 20_480,
            "SVG must be >= 20 KB; got {} bytes",
            meta.len()
        );
        let text = std::fs::read_to_string(out.as_std_path()).unwrap();
        assert!(text.contains("<svg"), "must contain <svg");
        assert!(text.contains("</svg>"), "must close </svg>");
    }
}
