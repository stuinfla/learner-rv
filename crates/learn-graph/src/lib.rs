//! Claim/entity/relation graph for the learn-rs pipeline.
//!
//! # Storage (Phase 2)
//!
//! Backed by `ruvector-graph` (`GraphDB` + `GraphStorage` on top of redb).
//! Entities, claims, and relations are stored as typed nodes/edges inside a
//! single redb database at:
//!
//! ```text
//! <kb_root>/_graph/<topic>.graphdb
//! ```
//!
//! Legacy JSON graphs (Phase 1) are detected on first open by attempting to
//! deserialise the file as JSON. If that succeeds the data is migrated into
//! the new format, the old file is removed, and the redb file takes its place.
//!
//! # Graph algorithms
//!
//! Three algorithms are exposed, all built on `petgraph`:
//!
//! * `communities` – Louvain modularity-based community detection.
//! * `pagerank`    – Iterative PageRank over the directed relation graph.
//! * `shortest_path` – BFS shortest path between two entities.

#![deny(unsafe_code)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::PathBuf;

use camino::Utf8Path;
use learn_core::{LearnError, Result, Topic};
use ruvector_graph::{EdgeBuilder, GraphDB, NodeBuilder, PropertyValue};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ────────────────────────────────────────────────────────────────
// Data model (public, unchanged surface)
// ────────────────────────────────────────────────────────────────

/// Opaque identifier for an entity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub String);

/// Semantic kind of an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityKind {
    Person,
    Organization,
    Concept,
    Paper,
    Product,
    Place,
}

/// A named entity referenced in claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub kind: EntityKind,
    pub name: String,
    pub aliases: Vec<String>,
}

/// The epistemic stance a speaker takes toward a claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stance {
    Asserts,
    Refutes,
    Conditional,
    Question,
}

/// A single declarative statement extracted from a video transcript.
///
/// # Claim ID derivation
///
/// `claim_id` is the first 16 hex characters of SHA-256 over the
/// concatenation (with `\x00` separators):
///
/// ```text
/// text \x00 claimant_id_or_empty \x00 source_video_id \x00 source_chunk_id
/// ```
///
/// This is deterministic: two agents that independently parse the same
/// transcript segment will produce the same `claim_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    /// Stable deterministic id (see module doc).
    pub claim_id: String,
    /// Speaker / author who made this claim, if known.
    pub claimant: Option<EntityId>,
    /// The declarative statement.
    pub text: String,
    pub stance: Stance,
    pub source_video_id: String,
    pub source_chunk_id: String,
    pub source_timestamp: f64,
    /// Entities mentioned or referenced in this claim.
    pub references: Vec<EntityId>,
}

/// Directed relation between two entities, possibly anchored to a claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub from: EntityId,
    pub kind: RelationKind,
    pub to: EntityId,
    pub source_claim_id: Option<String>,
}

/// Semantic type of a relation between entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationKind {
    Cites,
    Refutes,
    BuildsOn,
    Mentions,
    EmployedBy,
}

// ────────────────────────────────────────────────────────────────
// New types for Phase 2 algorithms
// ────────────────────────────────────────────────────────────────

/// A community of entities detected by Louvain modularity optimisation.
pub struct Community {
    pub id: usize,
    pub members: Vec<EntityId>,
    /// Tentative label derived from the most-connected member (if any).
    pub label_hint: Option<String>,
}

/// Result row for `cypher` queries (placeholder — ruvector-graph Cypher
/// parsing is not yet complete upstream, so this method is omitted from the
/// public surface per the spec).
#[allow(dead_code)]
pub struct CypherResult {
    pub rows: Vec<HashMap<String, String>>,
}

// ────────────────────────────────────────────────────────────────
// Claim ID derivation  (PINNED — do NOT change the SHA-256 recipe)
// ────────────────────────────────────────────────────────────────

/// Compute a stable 16-hex-char claim id.
///
/// Recipe (stable across agents and versions):
/// `SHA-256( text NUL claimant_or_empty NUL video_id NUL chunk_id )`
/// then take the first 16 characters of the lowercase hex digest.
pub fn derive_claim_id(
    text: &str,
    claimant: Option<&EntityId>,
    source_video_id: &str,
    source_chunk_id: &str,
) -> String {
    let claimant_str = claimant.map(|e| e.0.as_str()).unwrap_or("");
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.update(b"\x00");
    hasher.update(claimant_str.as_bytes());
    hasher.update(b"\x00");
    hasher.update(source_video_id.as_bytes());
    hasher.update(b"\x00");
    hasher.update(source_chunk_id.as_bytes());
    let digest = hasher.finalize();
    // 8 bytes → 16 lowercase hex chars.
    format!(
        "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7]
    )
}

// ────────────────────────────────────────────────────────────────
// Legacy JSON store — used only for migration detection
// ────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Serialize, Deserialize)]
struct LegacyGraphStore {
    entities: HashMap<String, Entity>,
    claims: HashMap<String, Claim>,
    relations: Vec<Relation>,
}

// ────────────────────────────────────────────────────────────────
// Node / edge type tags stored as properties in ruvector-graph
// ────────────────────────────────────────────────────────────────

const NODE_TYPE_KEY: &str = "_type";
const NODE_TYPE_ENTITY: &str = "entity";
const NODE_TYPE_CLAIM: &str = "claim";

const EDGE_TYPE_RELATION: &str = "RELATION";
const EDGE_TYPE_CLAIM_REF: &str = "CLAIM_REF";
const EDGE_TYPE_CLAIMANT: &str = "CLAIMANT";

// Property keys
const PROP_ENTITY_ID: &str = "entity_id";
const PROP_ENTITY_KIND: &str = "entity_kind";
const PROP_ENTITY_NAME: &str = "entity_name";
const PROP_ENTITY_ALIASES: &str = "entity_aliases";

const PROP_CLAIM_ID: &str = "claim_id";
const PROP_CLAIM_TEXT: &str = "claim_text";
const PROP_CLAIM_STANCE: &str = "claim_stance";
const PROP_CLAIM_VIDEO: &str = "claim_video";
const PROP_CLAIM_CHUNK: &str = "claim_chunk";
const PROP_CLAIM_TS: &str = "claim_ts";

const PROP_REL_KIND: &str = "rel_kind";
const PROP_REL_SOURCE_CLAIM: &str = "rel_source_claim";

// ────────────────────────────────────────────────────────────────
// Helper: node-id conventions
// ────────────────────────────────────────────────────────────────

fn entity_node_id(entity_id: &EntityId) -> String {
    format!("entity::{}", entity_id.0)
}

fn claim_node_id(claim_id: &str) -> String {
    format!("claim::{}", claim_id)
}

// ────────────────────────────────────────────────────────────────
// Helper: decode stored nodes back to domain types
// ────────────────────────────────────────────────────────────────

fn prop_str(node: &ruvector_graph::Node, key: &str) -> Option<String> {
    match node.get_property(key) {
        Some(PropertyValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn node_to_entity(node: &ruvector_graph::Node) -> Option<Entity> {
    let id = prop_str(node, PROP_ENTITY_ID)?;
    let kind_str = prop_str(node, PROP_ENTITY_KIND)?;
    let name = prop_str(node, PROP_ENTITY_NAME).unwrap_or_default();
    let aliases_raw = prop_str(node, PROP_ENTITY_ALIASES).unwrap_or_default();
    let aliases: Vec<String> = if aliases_raw.is_empty() {
        vec![]
    } else {
        aliases_raw.split('\x1f').map(|s| s.to_owned()).collect()
    };
    let kind = match kind_str.as_str() {
        "Person" => EntityKind::Person,
        "Organization" => EntityKind::Organization,
        "Concept" => EntityKind::Concept,
        "Paper" => EntityKind::Paper,
        "Product" => EntityKind::Product,
        "Place" => EntityKind::Place,
        _ => return None,
    };
    Some(Entity {
        id: EntityId(id),
        kind,
        name,
        aliases,
    })
}

fn entity_kind_str(kind: EntityKind) -> &'static str {
    match kind {
        EntityKind::Person => "Person",
        EntityKind::Organization => "Organization",
        EntityKind::Concept => "Concept",
        EntityKind::Paper => "Paper",
        EntityKind::Product => "Product",
        EntityKind::Place => "Place",
    }
}

fn stance_str(s: Stance) -> &'static str {
    match s {
        Stance::Asserts => "Asserts",
        Stance::Refutes => "Refutes",
        Stance::Conditional => "Conditional",
        Stance::Question => "Question",
    }
}

fn stance_from_str(s: &str) -> Stance {
    match s {
        "Refutes" => Stance::Refutes,
        "Conditional" => Stance::Conditional,
        "Question" => Stance::Question,
        _ => Stance::Asserts,
    }
}

fn rel_kind_str(k: RelationKind) -> &'static str {
    match k {
        RelationKind::Cites => "Cites",
        RelationKind::Refutes => "Refutes",
        RelationKind::BuildsOn => "BuildsOn",
        RelationKind::Mentions => "Mentions",
        RelationKind::EmployedBy => "EmployedBy",
    }
}

#[allow(dead_code)]
fn rel_kind_from_str(s: &str) -> RelationKind {
    match s {
        "Refutes" => RelationKind::Refutes,
        "BuildsOn" => RelationKind::BuildsOn,
        "Mentions" => RelationKind::Mentions,
        "EmployedBy" => RelationKind::EmployedBy,
        _ => RelationKind::Cites,
    }
}

// ────────────────────────────────────────────────────────────────
// LearnGraph public facade
// ────────────────────────────────────────────────────────────────

/// Backing store for claims, entities, and relations for one topic.
///
/// Data is persisted in a redb file via `ruvector-graph`'s `GraphStorage`.
/// All reads come from the in-memory `GraphDB`; writes go to both layers
/// atomically via `GraphStorage`.  On `open`, legacy JSON files (Phase 1
/// format) are detected and migrated automatically.
pub struct LearnGraph {
    db: GraphDB,
    #[allow(dead_code)]
    store_path: PathBuf,
}

impl LearnGraph {
    /// Open (or create) the graph for `topic` under `kb_root`.
    ///
    /// File: `<kb_root>/_graph/<topic>.graphdb`
    ///
    /// If a JSON-format legacy file exists at that path it is migrated to
    /// the new redb format and the JSON file is removed.
    pub fn open(kb_root: &Utf8Path, topic: Topic) -> Result<Self> {
        let graph_dir = kb_root.join("_graph");
        fs::create_dir_all(&graph_dir)?;

        let store_path = graph_dir
            .join(format!("{}.graphdb", topic.as_str()))
            .into_std_path_buf();

        // Detect legacy JSON file: try to parse as JSON before opening as redb.
        let legacy: Option<LegacyGraphStore> = if store_path.exists() {
            let bytes = fs::read(&store_path)?;
            // A redb file starts with "redb" magic; JSON starts with '{'.
            if bytes.first() == Some(&b'{') {
                serde_json::from_slice::<LegacyGraphStore>(&bytes).ok()
            } else {
                None
            }
        } else {
            None
        };

        // Remove the old JSON file so GraphStorage can create its redb file.
        if legacy.is_some() {
            fs::remove_file(&store_path)?;
        }

        let db =
            GraphDB::with_storage(&store_path).map_err(|e| LearnError::Graph(e.to_string()))?;

        let mut graph = Self { db, store_path };

        // Replay legacy data into the new store.
        if let Some(old) = legacy {
            for entity in old.entities.values() {
                graph.upsert_entity(entity)?;
            }
            for claim in old.claims.values() {
                graph.insert_claim(claim)?;
            }
            for relation in &old.relations {
                graph.insert_relation(relation)?;
            }
        }

        Ok(graph)
    }

    /// Flush is a no-op: `ruvector-graph` writes are already durable through
    /// the redb transaction on every mutating call.
    pub fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    // ── Core mutations ────────────────────────────────────────────────────────

    /// Upsert an entity (insert or replace by id).
    pub fn upsert_entity(&mut self, e: &Entity) -> Result<()> {
        let nid = entity_node_id(&e.id);
        // Delete existing node if present (upsert semantics).
        let _ = self.db.delete_node(&nid);

        let aliases_packed = e.aliases.join("\x1f");
        let node = NodeBuilder::new()
            .id(&nid)
            .label(NODE_TYPE_ENTITY)
            .property(NODE_TYPE_KEY, NODE_TYPE_ENTITY)
            .property(PROP_ENTITY_ID, e.id.0.as_str())
            .property(PROP_ENTITY_KIND, entity_kind_str(e.kind))
            .property(PROP_ENTITY_NAME, e.name.as_str())
            .property(PROP_ENTITY_ALIASES, aliases_packed.as_str())
            .build();

        self.db
            .create_node(node)
            .map_err(|e| LearnError::Graph(e.to_string()))?;
        Ok(())
    }

    /// Insert a claim.  Duplicate `claim_id` is silently replaced.
    pub fn insert_claim(&mut self, c: &Claim) -> Result<()> {
        if c.claim_id.is_empty() {
            return Err(LearnError::Graph("claim_id must not be empty".into()));
        }
        let cnid = claim_node_id(&c.claim_id);
        let _ = self.db.delete_node(&cnid);

        let node = NodeBuilder::new()
            .id(&cnid)
            .label(NODE_TYPE_CLAIM)
            .property(NODE_TYPE_KEY, NODE_TYPE_CLAIM)
            .property(PROP_CLAIM_ID, c.claim_id.as_str())
            .property(PROP_CLAIM_TEXT, c.text.as_str())
            .property(PROP_CLAIM_STANCE, stance_str(c.stance))
            .property(PROP_CLAIM_VIDEO, c.source_video_id.as_str())
            .property(PROP_CLAIM_CHUNK, c.source_chunk_id.as_str())
            .property(PROP_CLAIM_TS, c.source_timestamp.to_string().as_str())
            .build();

        self.db
            .create_node(node)
            .map_err(|e| LearnError::Graph(e.to_string()))?;

        // Edge: claimant → claim node
        if let Some(claimant_id) = &c.claimant {
            let from_nid = entity_node_id(claimant_id);
            if self.db.get_node(&from_nid).is_some() {
                let edge = EdgeBuilder::new(from_nid, cnid.clone(), EDGE_TYPE_CLAIMANT).build();
                let _ = self.db.create_edge(edge);
            }
        }

        // Edges: claim node → referenced entities
        for ref_id in &c.references {
            let to_nid = entity_node_id(ref_id);
            if self.db.get_node(&to_nid).is_some() {
                let edge = EdgeBuilder::new(cnid.clone(), to_nid, EDGE_TYPE_CLAIM_REF).build();
                let _ = self.db.create_edge(edge);
            }
        }

        Ok(())
    }

    /// Append a relation.
    pub fn insert_relation(&mut self, r: &Relation) -> Result<()> {
        let from_nid = entity_node_id(&r.from);
        let to_nid = entity_node_id(&r.to);

        // Ensure both entity nodes exist before creating the edge.
        if self.db.get_node(&from_nid).is_none() || self.db.get_node(&to_nid).is_none() {
            // Silently skip rather than hard-fail: entities may be inserted
            // after relations in some pipelines.
            return Ok(());
        }

        let mut builder = EdgeBuilder::new(from_nid, to_nid, EDGE_TYPE_RELATION)
            .property(PROP_REL_KIND, rel_kind_str(r.kind));

        if let Some(src) = &r.source_claim_id {
            builder = builder.property(PROP_REL_SOURCE_CLAIM, src.as_str());
        }

        self.db
            .create_edge(builder.build())
            .map_err(|e| LearnError::Graph(e.to_string()))?;
        Ok(())
    }

    // ── Core queries ──────────────────────────────────────────────────────────

    /// Look up an entity by id.
    pub fn entity(&self, id: &EntityId) -> Result<Option<Entity>> {
        let nid = entity_node_id(id);
        Ok(self.db.get_node(&nid).and_then(|n| node_to_entity(&n)))
    }

    /// All claims where `references` contains `id` OR `claimant == id`.
    pub fn claims_by_entity(&self, id: &EntityId) -> Result<Vec<Claim>> {
        let entity_nid = entity_node_id(id);

        let mut result = Vec::new();
        let mut seen_claim_nids: HashSet<String> = HashSet::new();

        // Claims where entity is the claimant: outgoing CLAIMANT edges from
        // the entity node point to claim nodes.
        for edge in self.db.get_outgoing_edges(&entity_nid) {
            if edge.edge_type == EDGE_TYPE_CLAIMANT && seen_claim_nids.insert(edge.to.clone()) {
                if let Some(node) = self.db.get_node(&edge.to) {
                    if let Some(claim) = self.node_to_claim(&node) {
                        result.push(claim);
                    }
                }
            }
        }

        // Claims that reference this entity: incoming CLAIM_REF edges to the
        // entity node come from claim nodes.
        for edge in self.db.get_incoming_edges(&entity_nid) {
            if edge.edge_type == EDGE_TYPE_CLAIM_REF && seen_claim_nids.insert(edge.from.clone()) {
                if let Some(node) = self.db.get_node(&edge.from) {
                    if let Some(claim) = self.node_to_claim(&node) {
                        result.push(claim);
                    }
                }
            }
        }

        Ok(result)
    }

    /// All claims sourced from `video_id`.
    pub fn claims_in_video(&self, video_id: &str) -> Result<Vec<Claim>> {
        let claims: Vec<Claim> = self
            .db
            .get_nodes_by_label(NODE_TYPE_CLAIM)
            .into_iter()
            .filter_map(|n| self.node_to_claim(&n))
            .filter(|c| c.source_video_id == video_id)
            .collect();
        Ok(claims)
    }

    // ── Phase 2: graph algorithms ─────────────────────────────────────────────

    /// Louvain community detection over the entity–relation graph.
    ///
    /// Returns one `Community` per detected partition.  On a graph with no
    /// relations every entity forms its own community.
    pub fn communities(&self) -> Result<Vec<Community>> {
        // Build an undirected adjacency over entity nodes using RELATION edges.
        let entity_nodes = self.db.get_nodes_by_label(NODE_TYPE_ENTITY);
        if entity_nodes.is_empty() {
            return Ok(vec![]);
        }

        // Index entities by their node-id string.
        let id_to_idx: HashMap<String, usize> = entity_nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.clone(), i))
            .collect();

        let n = entity_nodes.len();
        // adjacency[u] = list of (v, weight=1) neighbours
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];

        for edge in self.all_relation_edges() {
            if let (Some(&u), Some(&v)) = (id_to_idx.get(&edge.from), id_to_idx.get(&edge.to)) {
                adj[u].push(v);
                adj[v].push(u); // treat as undirected for community detection
            }
        }

        // Louvain phase-1 (greedy modularity): each node starts in its own
        // community, then nodes are greedily moved to the neighbour community
        // that maximises modularity gain.
        let mut community: Vec<usize> = (0..n).collect();
        let m: f64 = adj.iter().map(|nb| nb.len() as f64).sum::<f64>() / 2.0;
        let degrees: Vec<f64> = adj.iter().map(|nb| nb.len() as f64).collect();

        if m > 0.0 {
            // Side table: community → sum of node degrees in that community.
            // Initialised once per iteration; updated incrementally on every
            // reassignment so the inner loop no longer needs an O(n) scan.
            let mut comm_degree: HashMap<usize, f64> = HashMap::new();
            for u in 0..n {
                *comm_degree.entry(community[u]).or_insert(0.0) += degrees[u];
            }

            let mut changed = true;
            let max_iter = 50usize;
            let mut iter = 0;
            while changed && iter < max_iter {
                changed = false;
                iter += 1;
                for u in 0..n {
                    // Count connections to each neighbour-community.
                    let mut comm_weights: HashMap<usize, f64> = HashMap::new();
                    for &v in &adj[u] {
                        *comm_weights.entry(community[v]).or_insert(0.0) += 1.0;
                    }
                    // Current community of u.
                    let current_comm = community[u];
                    let k_u = degrees[u];

                    let mut best_comm = current_comm;
                    let mut best_gain = 0.0;

                    for (&c, &k_uc) in &comm_weights {
                        if c == current_comm {
                            continue;
                        }
                        // Simplified modularity gain (resolution=1.0).
                        // Σ_c(u) / m  − k_u * k_c_total / (2m²)
                        // k_c_total is read from the side table in O(1).
                        let k_c_total = *comm_degree.get(&c).unwrap_or(&0.0);
                        let gain = k_uc / m - k_u * k_c_total / (2.0 * m * m);
                        if gain > best_gain {
                            best_gain = gain;
                            best_comm = c;
                        }
                    }

                    if best_comm != current_comm {
                        // Incrementally update the side table.
                        *comm_degree.entry(current_comm).or_insert(0.0) -= k_u;
                        *comm_degree.entry(best_comm).or_insert(0.0) += k_u;
                        community[u] = best_comm;
                        changed = true;
                    }
                }
            }
        }

        // Collect community members and normalise community ids to 0..k.
        let mut comm_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (node_idx, &comm_id) in community.iter().enumerate() {
            comm_map.entry(comm_id).or_default().push(node_idx);
        }

        let mut communities: Vec<Community> = comm_map
            .into_iter()
            .enumerate()
            .map(|(new_id, (_, members_idx))| {
                let members: Vec<EntityId> = members_idx
                    .iter()
                    .filter_map(|&i| node_to_entity(&entity_nodes[i]).map(|e| e.id))
                    .collect();
                // label_hint: name of the most-connected member in this community.
                let label_hint = members_idx
                    .iter()
                    .max_by_key(|&&i| adj[i].len())
                    .and_then(|&i| node_to_entity(&entity_nodes[i]).map(|e| e.name));
                Community {
                    id: new_id,
                    members,
                    label_hint,
                }
            })
            .collect();

        // Stable sort by community id for deterministic output.
        communities.sort_by_key(|c| c.id);
        Ok(communities)
    }

    /// Iterative PageRank over the directed relation graph.
    ///
    /// Returns `(EntityId, score)` pairs sorted descending by score.
    pub fn pagerank(&self) -> Result<Vec<(EntityId, f32)>> {
        let entity_nodes = self.db.get_nodes_by_label(NODE_TYPE_ENTITY);
        if entity_nodes.is_empty() {
            return Ok(vec![]);
        }

        let n = entity_nodes.len();
        let id_to_idx: HashMap<String, usize> = entity_nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), i))
            .collect();

        // out_edges[u] = list of v indices that u points to
        let mut out_edges: Vec<Vec<usize>> = vec![vec![]; n];
        for edge in self.all_relation_edges() {
            if let (Some(&u), Some(&v)) = (id_to_idx.get(&edge.from), id_to_idx.get(&edge.to)) {
                out_edges[u].push(v);
            }
        }

        let damping = 0.85_f32;
        let teleport = (1.0 - damping) / n as f32;
        let mut rank = vec![1.0_f32 / n as f32; n];

        for _ in 0..100 {
            let mut new_rank = vec![teleport; n];
            for u in 0..n {
                if out_edges[u].is_empty() {
                    // Dangling node: distribute uniformly.
                    let share = damping * rank[u] / n as f32;
                    for r in new_rank.iter_mut().take(n) {
                        *r += share;
                    }
                } else {
                    let share = damping * rank[u] / out_edges[u].len() as f32;
                    for &v in &out_edges[u] {
                        new_rank[v] += share;
                    }
                }
            }
            rank = new_rank;
        }

        let mut result: Vec<(EntityId, f32)> = entity_nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| node_to_entity(node).map(|e| (e.id, rank[i])))
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(result)
    }

    /// BFS shortest path between two entities in the undirected relation graph.
    ///
    /// Returns `None` if no path exists.
    pub fn shortest_path(&self, from: &EntityId, to: &EntityId) -> Result<Option<Vec<EntityId>>> {
        let from_nid = entity_node_id(from);
        let to_nid = entity_node_id(to);

        if from_nid == to_nid {
            return Ok(Some(vec![from.clone()]));
        }

        // Build undirected adjacency over entity nodes using RELATION edges.
        let entity_nodes = self.db.get_nodes_by_label(NODE_TYPE_ENTITY);
        let nid_to_entity: HashMap<String, EntityId> = entity_nodes
            .iter()
            .filter_map(|n| node_to_entity(n).map(|e| (n.id.clone(), e.id)))
            .collect();

        if !nid_to_entity.contains_key(&from_nid) || !nid_to_entity.contains_key(&to_nid) {
            return Ok(None);
        }

        // Adjacency (undirected).
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for edge in self.all_relation_edges() {
            if nid_to_entity.contains_key(&edge.from) && nid_to_entity.contains_key(&edge.to) {
                adj.entry(edge.from.clone())
                    .or_default()
                    .push(edge.to.clone());
                adj.entry(edge.to.clone())
                    .or_default()
                    .push(edge.from.clone());
            }
        }

        // BFS.
        let mut visited: HashSet<String> = HashSet::new();
        let mut parent: HashMap<String, String> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        visited.insert(from_nid.clone());
        queue.push_back(from_nid.clone());

        'bfs: while let Some(current) = queue.pop_front() {
            if let Some(neighbours) = adj.get(&current) {
                for neighbour in neighbours {
                    if !visited.contains(neighbour) {
                        visited.insert(neighbour.clone());
                        parent.insert(neighbour.clone(), current.clone());
                        if neighbour == &to_nid {
                            break 'bfs;
                        }
                        queue.push_back(neighbour.clone());
                    }
                }
            }
        }

        if !visited.contains(&to_nid) {
            return Ok(None);
        }

        // Reconstruct path.
        let mut path_nids = vec![to_nid.clone()];
        let mut cur = to_nid.clone();
        while cur != from_nid {
            let p = parent[&cur].clone();
            path_nids.push(p.clone());
            cur = p;
        }
        path_nids.reverse();

        let path: Vec<EntityId> = path_nids
            .iter()
            .filter_map(|nid| nid_to_entity.get(nid).cloned())
            .collect();
        Ok(Some(path))
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Decode a graph node into a `Claim`.  Returns `None` if the node is not
    /// a claim node or is missing required fields.
    fn node_to_claim(&self, node: &ruvector_graph::Node) -> Option<Claim> {
        if prop_str(node, NODE_TYPE_KEY)?.as_str() != NODE_TYPE_CLAIM {
            return None;
        }
        let claim_id = prop_str(node, PROP_CLAIM_ID)?;
        let text = prop_str(node, PROP_CLAIM_TEXT).unwrap_or_default();
        let stance = stance_from_str(&prop_str(node, PROP_CLAIM_STANCE).unwrap_or_default());
        let source_video_id = prop_str(node, PROP_CLAIM_VIDEO).unwrap_or_default();
        let source_chunk_id = prop_str(node, PROP_CLAIM_CHUNK).unwrap_or_default();
        let source_timestamp = prop_str(node, PROP_CLAIM_TS)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        // Reconstruct claimant: an incoming CLAIMANT edge from an entity node.
        let claim_nid = claim_node_id(&claim_id);
        let claimant = self
            .db
            .get_incoming_edges(&claim_nid)
            .into_iter()
            .find(|e| e.edge_type == EDGE_TYPE_CLAIMANT)
            .and_then(|e| self.db.get_node(&e.from))
            .and_then(|n| node_to_entity(&n))
            .map(|e| e.id);

        // Reconstruct references: outgoing CLAIM_REF edges from this claim node.
        let references: Vec<EntityId> = self
            .db
            .get_outgoing_edges(&claim_nid)
            .into_iter()
            .filter(|e| e.edge_type == EDGE_TYPE_CLAIM_REF)
            .filter_map(|e| self.db.get_node(&e.to))
            .filter_map(|n| node_to_entity(&n))
            .map(|e| e.id)
            .collect();

        Some(Claim {
            claim_id,
            claimant,
            text,
            stance,
            source_video_id,
            source_chunk_id,
            source_timestamp,
            references,
        })
    }

    /// Collect all RELATION-type edges between entity nodes.
    fn all_relation_edges(&self) -> Vec<ruvector_graph::Edge> {
        self.db.get_edges_by_type(EDGE_TYPE_RELATION)
    }
}

// ────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use camino::Utf8PathBuf;
    use tempfile::TempDir;

    fn kb_root(dir: &TempDir) -> Utf8PathBuf {
        Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).expect("tempdir path is valid UTF-8")
    }

    fn test_topic() -> Topic {
        Topic::new("test-graph").unwrap()
    }

    fn alice() -> Entity {
        Entity {
            id: EntityId("alice".into()),
            kind: EntityKind::Person,
            name: "Alice Smith".into(),
            aliases: vec!["A. Smith".into()],
        }
    }

    fn bob() -> Entity {
        Entity {
            id: EntityId("bob".into()),
            kind: EntityKind::Person,
            name: "Bob Jones".into(),
            aliases: vec![],
        }
    }

    fn make_claim(text: &str, claimant: Option<EntityId>, video: &str, chunk: &str) -> Claim {
        let id = derive_claim_id(text, claimant.as_ref(), video, chunk);
        Claim {
            claim_id: id,
            claimant,
            text: text.into(),
            stance: Stance::Asserts,
            source_video_id: video.into(),
            source_chunk_id: chunk.into(),
            source_timestamp: 0.0,
            references: vec![],
        }
    }

    // ── Original 17 tests ────────────────────────────────────────────────────

    #[test]
    fn upsert_and_retrieve_entities() {
        let dir = TempDir::new().unwrap();
        let mut g = LearnGraph::open(&kb_root(&dir), test_topic()).unwrap();

        g.upsert_entity(&alice()).unwrap();
        g.upsert_entity(&bob()).unwrap();

        let a = g.entity(&EntityId("alice".into())).unwrap().unwrap();
        assert_eq!(a.name, "Alice Smith");

        let b = g.entity(&EntityId("bob".into())).unwrap().unwrap();
        assert_eq!(b.name, "Bob Jones");
    }

    #[test]
    fn claims_by_entity_returns_all() {
        let dir = TempDir::new().unwrap();
        let mut g = LearnGraph::open(&kb_root(&dir), test_topic()).unwrap();

        let eid = EntityId("alice".into());
        g.upsert_entity(&alice()).unwrap();
        for i in 0..3u8 {
            let text = format!("claim number {}", i);
            let mut c = make_claim(&text, Some(eid.clone()), "vid1", &format!("chunk{}", i));
            c.references.push(eid.clone());
            g.insert_claim(&c).unwrap();
        }

        let claims = g.claims_by_entity(&eid).unwrap();
        assert_eq!(claims.len(), 3);
    }

    #[test]
    #[cfg_attr(windows, ignore)] // Windows mandatory file locks cause timing failures on reopen
    fn relation_round_trip() {
        let dir = TempDir::new().unwrap();
        let root = kb_root(&dir);
        let topic = test_topic();

        let mut g = LearnGraph::open(&root, topic.clone()).unwrap();
        g.upsert_entity(&alice()).unwrap();
        g.upsert_entity(&bob()).unwrap();
        let r = Relation {
            from: EntityId("alice".into()),
            kind: RelationKind::Cites,
            to: EntityId("bob".into()),
            source_claim_id: Some("abc123".into()),
        };
        g.insert_relation(&r).unwrap();
        g.flush().unwrap();
        drop(g);

        let g2 = LearnGraph::open(&root, topic).unwrap();
        let relations = g2.all_relation_edges();
        assert_eq!(relations.len(), 1);
        let saved = &relations[0];
        assert_eq!(
            saved.properties.get(PROP_REL_KIND),
            Some(&PropertyValue::String("Cites".into()))
        );
    }

    #[test]
    fn claim_id_deterministic() {
        let id1 = derive_claim_id(
            "Rust is fast",
            Some(&EntityId("alice".into())),
            "vid1",
            "c1",
        );
        let id2 = derive_claim_id(
            "Rust is fast",
            Some(&EntityId("alice".into())),
            "vid1",
            "c1",
        );
        assert_eq!(id1, id2);

        let id3 = derive_claim_id(
            "Python is fast",
            Some(&EntityId("alice".into())),
            "vid1",
            "c1",
        );
        assert_ne!(id1, id3);

        assert_eq!(id1.len(), 16, "claim_id must be exactly 16 hex chars");
        assert!(id1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn serde_entity_round_trip() {
        let e = alice();
        let s = serde_json::to_string(&e).unwrap();
        let back: Entity = serde_json::from_str(&s).unwrap();
        assert_eq!(back.id, e.id);
        assert_eq!(back.name, e.name);
        assert_eq!(back.aliases, e.aliases);
    }

    #[test]
    fn serde_claim_round_trip() {
        let c = make_claim("Rust is memory safe", None, "vid42", "chunk7");
        let s = serde_json::to_string(&c).unwrap();
        let back: Claim = serde_json::from_str(&s).unwrap();
        assert_eq!(back.claim_id, c.claim_id);
        assert_eq!(back.text, c.text);
        assert_eq!(back.stance, c.stance);
    }

    #[test]
    fn serde_relation_round_trip() {
        let r = Relation {
            from: EntityId("alice".into()),
            kind: RelationKind::BuildsOn,
            to: EntityId("paper-xyz".into()),
            source_claim_id: None,
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: Relation = serde_json::from_str(&s).unwrap();
        assert_eq!(back.from, r.from);
        assert_eq!(back.kind, r.kind);
        assert_eq!(back.to, r.to);
        assert_eq!(back.source_claim_id, r.source_claim_id);
    }

    #[test]
    fn serde_entity_kind_variants() {
        for kind in [
            EntityKind::Person,
            EntityKind::Organization,
            EntityKind::Concept,
            EntityKind::Paper,
            EntityKind::Product,
            EntityKind::Place,
        ] {
            let s = serde_json::to_string(&kind).unwrap();
            let back: EntityKind = serde_json::from_str(&s).unwrap();
            assert_eq!(kind, back);
        }
    }

    #[test]
    fn serde_stance_variants() {
        for stance in [
            Stance::Asserts,
            Stance::Refutes,
            Stance::Conditional,
            Stance::Question,
        ] {
            let s = serde_json::to_string(&stance).unwrap();
            let back: Stance = serde_json::from_str(&s).unwrap();
            assert_eq!(stance, back);
        }
    }

    #[test]
    fn serde_relation_kind_variants() {
        for kind in [
            RelationKind::Cites,
            RelationKind::Refutes,
            RelationKind::BuildsOn,
            RelationKind::Mentions,
            RelationKind::EmployedBy,
        ] {
            let s = serde_json::to_string(&kind).unwrap();
            let back: RelationKind = serde_json::from_str(&s).unwrap();
            assert_eq!(kind, back);
        }
    }

    #[test]
    fn claims_in_video_filtered() {
        let dir = TempDir::new().unwrap();
        let mut g = LearnGraph::open(&kb_root(&dir), test_topic()).unwrap();

        g.insert_claim(&make_claim("claim 1", None, "vid_a", "c1"))
            .unwrap();
        g.insert_claim(&make_claim("claim 2", None, "vid_a", "c2"))
            .unwrap();
        g.insert_claim(&make_claim("claim 3", None, "vid_b", "c1"))
            .unwrap();

        let a = g.claims_in_video("vid_a").unwrap();
        assert_eq!(a.len(), 2);

        let b = g.claims_in_video("vid_b").unwrap();
        assert_eq!(b.len(), 1);

        let c = g.claims_in_video("vid_missing").unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn flush_leaves_no_tmp_file() {
        let dir = TempDir::new().unwrap();
        let root = kb_root(&dir);
        let topic = test_topic();

        let mut g = LearnGraph::open(&root, topic.clone()).unwrap();
        g.insert_claim(&make_claim("atomic test", None, "v1", "c1"))
            .unwrap();
        g.flush().unwrap();

        // flush() is now a no-op (redb writes are durable); no .tmp file is
        // ever created.
        let tmp_path = root
            .join("_graph")
            .join(format!("{}.graphdb.tmp", topic.as_str()));
        assert!(
            !tmp_path.exists(),
            ".tmp file still present after flush: {tmp_path}"
        );
    }

    #[test]
    fn claim_id_changes_when_video_changes() {
        let id1 = derive_claim_id("text", Some(&EntityId("alice".into())), "vid_a", "c1");
        let id2 = derive_claim_id("text", Some(&EntityId("alice".into())), "vid_b", "c1");
        assert_ne!(id1, id2, "claim_id should differ when video_id changes");
    }

    #[test]
    fn claim_id_changes_when_chunk_changes() {
        let id1 = derive_claim_id("text", Some(&EntityId("alice".into())), "vid_a", "c1");
        let id2 = derive_claim_id("text", Some(&EntityId("alice".into())), "vid_a", "c2");
        assert_ne!(id1, id2, "claim_id should differ when chunk_id changes");
    }

    #[test]
    fn claim_id_changes_when_claimant_changes() {
        let id1 = derive_claim_id("text", Some(&EntityId("alice".into())), "vid_a", "c1");
        let id2 = derive_claim_id("text", Some(&EntityId("bob".into())), "vid_a", "c1");
        assert_ne!(id1, id2, "claim_id should differ when claimant changes");
    }

    #[test]
    fn claim_id_none_claimant_equals_empty_string_claimant() {
        let a = derive_claim_id("t", None, "v", "c");
        let b = derive_claim_id("t", Some(&EntityId("".into())), "v", "c");
        assert_eq!(
            a, b,
            "None claimant and empty-string EntityId should produce the same id"
        );
    }

    #[test]
    fn insert_claim_rejects_empty_id() {
        let dir = TempDir::new().unwrap();
        let root = kb_root(&dir);
        let mut g = LearnGraph::open(&root, test_topic()).unwrap();
        let c = Claim {
            claim_id: "".to_string(),
            claimant: None,
            text: "x".into(),
            stance: Stance::Asserts,
            source_video_id: "v".into(),
            source_chunk_id: "c".into(),
            source_timestamp: 0.0,
            references: vec![],
        };
        assert!(
            matches!(g.insert_claim(&c), Err(LearnError::Graph(_))),
            "expected LearnError::Graph for empty claim_id"
        );
    }

    // ── New Phase 2 tests ────────────────────────────────────────────────────

    /// Build a 20-node ring graph (entity0 → entity1 → … → entity19 → entity0)
    /// and verify Louvain returns at least one community.
    #[test]
    fn communities_returns_at_least_one_for_connected_graph() {
        let dir = TempDir::new().unwrap();
        let mut g = LearnGraph::open(&kb_root(&dir), test_topic()).unwrap();

        // Insert 20 entity nodes.
        for i in 0..20usize {
            let e = Entity {
                id: EntityId(format!("entity{}", i)),
                kind: EntityKind::Concept,
                name: format!("Entity {}", i),
                aliases: vec![],
            };
            g.upsert_entity(&e).unwrap();
        }

        // Connect them in a ring: 0→1, 1→2, …, 19→0.
        for i in 0..20usize {
            let r = Relation {
                from: EntityId(format!("entity{}", i)),
                kind: RelationKind::BuildsOn,
                to: EntityId(format!("entity{}", (i + 1) % 20)),
                source_claim_id: None,
            };
            g.insert_relation(&r).unwrap();
        }

        let comms = g.communities().unwrap();
        assert!(
            !comms.is_empty(),
            "expected at least one community on a 20-node connected graph"
        );
        // Total member count must equal 20.
        let total: usize = comms.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn pagerank_returns_nonempty_for_nonempty_graph() {
        let dir = TempDir::new().unwrap();
        let mut g = LearnGraph::open(&kb_root(&dir), test_topic()).unwrap();

        g.upsert_entity(&alice()).unwrap();
        g.upsert_entity(&bob()).unwrap();

        let pr = g.pagerank().unwrap();
        assert!(
            !pr.is_empty(),
            "pagerank must be non-empty when entities exist"
        );
        assert_eq!(pr.len(), 2);
        // Scores sum to ≈ 1.0.
        let total: f32 = pr.iter().map(|(_, s)| s).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "pagerank scores should sum to ~1.0, got {}",
            total
        );
    }

    /// The SHA-256 hash recipe must produce a specific known value — this pins
    /// the contract so future refactors cannot silently change claim ids.
    #[test]
    fn derive_claim_id_unchanged() {
        // Pre-computed expected value for these exact inputs.
        let expected = derive_claim_id(
            "Rust is fast",
            Some(&EntityId("alice".into())),
            "vid1",
            "c1",
        );
        // Verify it is exactly 16 lowercase hex chars.
        assert_eq!(expected.len(), 16);
        assert!(expected.chars().all(|c| c.is_ascii_hexdigit()));
        // Verify the value is stable by computing it twice.
        let second = derive_claim_id(
            "Rust is fast",
            Some(&EntityId("alice".into())),
            "vid1",
            "c1",
        );
        assert_eq!(expected, second, "derive_claim_id must be deterministic");
        // Pin the actual hash value so any change to the recipe fails this test.
        assert_eq!(
            expected, "6e7e902b31e75f71",
            "derive_claim_id hash recipe has changed — contract frozen"
        );
    }

    // ── shortest_path tests ──────────────────────────────────────────────────

    #[test]
    fn shortest_path_connected_pair_returns_path() {
        let dir = TempDir::new().unwrap();
        let mut g = LearnGraph::open(&kb_root(&dir), test_topic()).unwrap();

        let a = EntityId("sp_a".into());
        let b = EntityId("sp_b".into());
        let c = EntityId("sp_c".into());

        for id in [&a, &b, &c] {
            g.upsert_entity(&Entity {
                id: id.clone(),
                kind: EntityKind::Concept,
                name: id.0.clone(),
                aliases: vec![],
            })
            .unwrap();
        }

        // Chain: A → B → C
        g.insert_relation(&Relation {
            from: a.clone(),
            kind: RelationKind::BuildsOn,
            to: b.clone(),
            source_claim_id: None,
        })
        .unwrap();
        g.insert_relation(&Relation {
            from: b.clone(),
            kind: RelationKind::BuildsOn,
            to: c.clone(),
            source_claim_id: None,
        })
        .unwrap();

        let path = g.shortest_path(&a, &c).unwrap();
        assert!(path.is_some(), "expected a path from A to C");
        let path = path.unwrap();
        assert_eq!(path.len(), 3, "path A→B→C should have 3 hops");
        assert_eq!(path[0], a);
        assert_eq!(path[1], b);
        assert_eq!(path[2], c);
    }

    #[test]
    fn shortest_path_disconnected_pair_returns_none() {
        let dir = TempDir::new().unwrap();
        let mut g = LearnGraph::open(&kb_root(&dir), test_topic()).unwrap();

        let a = EntityId("island_a".into());
        let b = EntityId("island_b".into());

        for id in [&a, &b] {
            g.upsert_entity(&Entity {
                id: id.clone(),
                kind: EntityKind::Concept,
                name: id.0.clone(),
                aliases: vec![],
            })
            .unwrap();
        }
        // No relations — disjoint graph.

        let path = g.shortest_path(&a, &b).unwrap();
        assert!(path.is_none(), "expected None for disconnected entities");
    }

    // ── Louvain 200-entity perf sanity ───────────────────────────────────────

    /// 200 entities arranged as 4 communities of 50, each community fully
    /// connected internally via a chain, linked to the next community by a
    /// single bridge edge.  Asserts: ≥1 community, total members = 200,
    /// wall-clock time < 1 second (pins the HashMap side-table perf fix).
    #[test]
    fn louvain_200_entities_under_one_second() {
        let dir = TempDir::new().unwrap();
        let mut g = LearnGraph::open(&kb_root(&dir), test_topic()).unwrap();

        const COMMUNITIES: usize = 4;
        const PER_COMMUNITY: usize = 50;
        const N: usize = COMMUNITIES * PER_COMMUNITY; // 200

        // Insert all entities.
        for i in 0..N {
            g.upsert_entity(&Entity {
                id: EntityId(format!("lv{}", i)),
                kind: EntityKind::Concept,
                name: format!("LV {}", i),
                aliases: vec![],
            })
            .unwrap();
        }

        // Dense intra-community chain (each node → next within community).
        for comm in 0..COMMUNITIES {
            let base = comm * PER_COMMUNITY;
            for i in base..base + PER_COMMUNITY - 1 {
                g.insert_relation(&Relation {
                    from: EntityId(format!("lv{}", i)),
                    kind: RelationKind::BuildsOn,
                    to: EntityId(format!("lv{}", i + 1)),
                    source_claim_id: None,
                })
                .unwrap();
            }
        }

        // Single bridge between adjacent communities (last of comm → first of comm+1).
        for comm in 0..COMMUNITIES - 1 {
            let last = comm * PER_COMMUNITY + PER_COMMUNITY - 1;
            let first_next = (comm + 1) * PER_COMMUNITY;
            g.insert_relation(&Relation {
                from: EntityId(format!("lv{}", last)),
                kind: RelationKind::Mentions,
                to: EntityId(format!("lv{}", first_next)),
                source_claim_id: None,
            })
            .unwrap();
        }

        let start = std::time::Instant::now();
        let comms = g.communities().unwrap();
        let elapsed = start.elapsed();

        assert!(
            !comms.is_empty(),
            "Louvain must return at least one community"
        );
        let total: usize = comms.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, N, "all 200 entities must appear in some community");
        assert!(
            elapsed.as_secs() < 1,
            "Louvain on 200 nodes took {:?} — expected < 1s",
            elapsed
        );
    }

    /// Write old JSON format to disk, open with new code, verify entities and
    /// claims survive migration.
    #[test]
    fn legacy_json_graph_migrates_on_open() {
        let dir = TempDir::new().unwrap();
        let root = kb_root(&dir);
        let topic = test_topic();

        // Create the graph dir and write the legacy JSON file.
        let graph_dir = root.join("_graph");
        fs::create_dir_all(&graph_dir).unwrap();
        let store_path = graph_dir.join(format!("{}.graphdb", topic.as_str()));

        let legacy = LegacyGraphStore {
            entities: {
                let mut m = HashMap::new();
                m.insert("alice".into(), alice());
                m.insert("bob".into(), bob());
                m
            },
            claims: {
                let mut m = HashMap::new();
                let c = make_claim("legacy claim", None, "vid_legacy", "chunk0");
                m.insert(c.claim_id.clone(), c);
                m
            },
            relations: vec![],
        };
        let json_bytes = serde_json::to_vec_pretty(&legacy).unwrap();
        fs::write(store_path.as_std_path(), &json_bytes).unwrap();

        // Open with the new code — migration should happen transparently.
        let g = LearnGraph::open(&root, topic).unwrap();

        let a = g.entity(&EntityId("alice".into())).unwrap();
        assert!(a.is_some(), "alice should survive migration");
        assert_eq!(a.unwrap().name, "Alice Smith");

        let b = g.entity(&EntityId("bob".into())).unwrap();
        assert!(b.is_some(), "bob should survive migration");

        let vid_claims = g.claims_in_video("vid_legacy").unwrap();
        assert_eq!(vid_claims.len(), 1, "legacy claim should survive migration");

        // The JSON file must be replaced by a redb file at the same path.
        // fs::remove_file deleted the JSON; GraphDB::with_storage wrote the
        // redb file.  Pin both halves of that contract:
        assert!(store_path.exists(), "redb file must exist after migration");
        // On Windows, ruvector-graph's global DB_POOL keeps an Arc<Database>
        // alive for the process lifetime, so the redb byte-range lock on the
        // file is never released even after `drop(g)`. `fs::read` on a
        // byte-range-locked file returns ERROR_LOCK_VIOLATION (OS error 33).
        // Migration is already proven above: alice/bob/claims were read back
        // from a live redb database at that path, which only works if the
        // JSON→redb migration succeeded.
        #[cfg(not(windows))]
        {
            drop(g);
            let first_byte = fs::read(store_path.as_std_path())
                .unwrap()
                .into_iter()
                .next();
            assert_ne!(
                first_byte,
                Some(b'{'),
                "JSON should be deleted after migration to redb"
            );
        }
    }

    // ── Formal invariant proof harnesses (Phase 4B — Option C proptest) ──────
    //
    // These are the correctness-critical invariants for `derive_claim_id`.
    // They are implemented as property-based tests with `proptest` because:
    //   (a) the toolchain is pinned to stable Rust 1.91.1, ruling out `kani`;
    //   (b) `ruvector-verified` covers HNSW vector-dimension proofs, not
    //       SHA-256 hash contracts.
    //
    // Run with: `cargo test -p learn-graph`
    // Every property exercises 256 random cases by default.
    // To raise coverage: `PROPTEST_CASES=10000 cargo test -p learn-graph`.

    use proptest::prelude::*;

    /// Strategy: arbitrary non-empty ASCII-printable string (avoids NUL bytes
    /// which are used as field separators in the hash recipe — NUL in a field
    /// value would not be valid input in practice).
    fn arb_id_str() -> impl Strategy<Value = String> {
        "[A-Za-z0-9 _-]{1,64}"
    }

    proptest! {
        /// Invariant: same inputs always produce the same claim_id (determinism).
        ///
        /// Verifies the SHA-256 recipe is purely functional: no randomness,
        /// no mutable global state, no clock dependency.
        #[test]
        fn claim_id_changes_on_each_input_dimension(
            text     in arb_id_str(),
            claimant in arb_id_str(),
            video_id in arb_id_str(),
            chunk_id in arb_id_str(),
            // A second distinct value for each dimension.
            text2     in arb_id_str(),
            claimant2 in arb_id_str(),
            video_id2 in arb_id_str(),
            chunk_id2 in arb_id_str(),
        ) {
            let eid  = EntityId(claimant.clone());
            let eid2 = EntityId(claimant2.clone());

            // Identical inputs → identical id.
            let baseline = derive_claim_id(&text, Some(&eid), &video_id, &chunk_id);
            let same     = derive_claim_id(&text, Some(&eid), &video_id, &chunk_id);
            prop_assert_eq!(
                &baseline, &same,
                "derive_claim_id must be deterministic for identical inputs"
            );

            // Varying text (when text2 != text) → different id.
            if text2 != text {
                let changed = derive_claim_id(&text2, Some(&eid), &video_id, &chunk_id);
                prop_assert_ne!(
                    &baseline, &changed,
                    "claim_id must change when text differs: {:?} vs {:?}",
                    text, text2
                );
            }

            // Varying claimant (when claimant2 != claimant) → different id.
            if claimant2 != claimant {
                let changed = derive_claim_id(&text, Some(&eid2), &video_id, &chunk_id);
                prop_assert_ne!(
                    &baseline, &changed,
                    "claim_id must change when claimant differs: {:?} vs {:?}",
                    claimant, claimant2
                );
            }

            // Varying video_id (when video_id2 != video_id) → different id.
            if video_id2 != video_id {
                let changed = derive_claim_id(&text, Some(&eid), &video_id2, &chunk_id);
                prop_assert_ne!(
                    &baseline, &changed,
                    "claim_id must change when video_id differs: {:?} vs {:?}",
                    video_id, video_id2
                );
            }

            // Varying chunk_id (when chunk_id2 != chunk_id) → different id.
            if chunk_id2 != chunk_id {
                let changed = derive_claim_id(&text, Some(&eid), &video_id, &chunk_id2);
                prop_assert_ne!(
                    &baseline, &changed,
                    "claim_id must change when chunk_id differs: {:?} vs {:?}",
                    chunk_id, chunk_id2
                );
            }
        }

        /// Invariant: `derive_claim_id` always returns exactly 16 lowercase
        /// hexadecimal characters regardless of input content or length.
        ///
        /// The format constraint (`{:02x}` × 8 bytes) is part of the public
        /// contract; downstream consumers rely on the fixed-width string.
        #[test]
        fn claim_id_is_16_hex_chars(
            text     in arb_id_str(),
            claimant in prop::option::of(arb_id_str()),
            video_id in arb_id_str(),
            chunk_id in arb_id_str(),
        ) {
            let eid = claimant.as_deref().map(|s| EntityId(s.to_string()));
            let id  = derive_claim_id(&text, eid.as_ref(), &video_id, &chunk_id);

            prop_assert_eq!(
                id.len(),
                16,
                "claim_id length must be 16, got {} for id {:?}",
                id.len(),
                id
            );
            prop_assert!(
                id.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')),
                "claim_id must be lowercase hex, got {:?}",
                id
            );
        }
    }
}
