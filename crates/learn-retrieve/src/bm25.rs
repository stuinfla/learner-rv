//! In-memory tantivy BM25 index, rebuilt from the sidecar chunk store.

use learn_core::Chunk;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexWriter, TantivyDocument};
use tracing::warn;

const FIELD_CHUNK_ID: &str = "chunk_id";
const FIELD_TEXT: &str = "text";

/// Holds a live in-memory tantivy index.  Drop + rebuild on `refresh_bm25`.
pub(crate) struct Bm25State {
    index: Index,
    field_text: Field,
    field_chunk_id: Field,
}

impl Bm25State {
    /// Build a fresh in-memory BM25 index from `chunks`.
    pub(crate) fn build(chunks: &[&Chunk]) -> anyhow::Result<Self> {
        let mut sb = Schema::builder();
        let field_chunk_id = sb.add_text_field(FIELD_CHUNK_ID, STRING | STORED);
        let field_text = sb.add_text_field(FIELD_TEXT, TEXT);
        let schema = sb.build();

        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer(15_000_000)?;
        for chunk in chunks {
            writer.add_document(doc!(
                field_chunk_id => chunk.chunk_id.as_str(),
                field_text     => chunk.text.as_str(),
            ))?;
        }
        writer.commit()?;

        Ok(Self {
            index,
            field_text,
            field_chunk_id,
        })
    }

    /// Query BM25 index; returns `(chunk_id, score)` pairs in score order.
    pub(crate) fn search(&self, query_text: &str, k: usize) -> anyhow::Result<Vec<(String, f32)>> {
        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.field_text]);

        let sanitised = sanitise_query(query_text);
        if sanitised.is_empty() {
            return Ok(Vec::new());
        }

        let query = match parser.parse_query(&sanitised) {
            Ok(q) => q,
            Err(e) => {
                warn!("BM25 parse error: {e}; returning empty");
                return Ok(Vec::new());
            }
        };

        let top = searcher.search(&query, &TopDocs::with_limit(k))?;
        let mut results = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr)?;
            let chunk_id = doc
                .get_first(self.field_chunk_id)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            results.push((chunk_id, score));
        }
        Ok(results)
    }
}

/// Strip tantivy special chars; return plain keyword string.
pub(crate) fn sanitise_query(q: &str) -> String {
    q.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
