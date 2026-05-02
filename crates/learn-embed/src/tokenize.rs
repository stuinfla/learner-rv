//! Tokenization helpers: encode text → padded integer arrays for ORT.

use learn_core::{LearnError, Result};
use ndarray::Array2;
use tokenizers::{Encoding, Tokenizer};
use tracing::trace;

use crate::BatchArrays;

/// Tokenize one string, truncating to `max_seq_len` tokens.
pub(crate) fn encode_single(tok: &Tokenizer, text: &str, max_seq_len: usize) -> Result<Encoding> {
    let enc = tok
        .encode(text, false)
        .map_err(|e| LearnError::Embed(format!("tokenize: {e}")))?;
    trace!(n_tokens = enc.len(), "encoded single text");
    Ok(truncate(enc, max_seq_len))
}

/// Tokenize a batch of strings, truncating each to `max_seq_len`.
pub(crate) fn encode_batch<'a>(
    tok: &Tokenizer,
    texts: impl Iterator<Item = &'a str>,
    max_seq_len: usize,
) -> Result<Vec<Encoding>> {
    texts.map(|t| encode_single(tok, t, max_seq_len)).collect()
}

/// Convert a slice of `Encoding` into the three int64 arrays expected by BGE.
///
/// - `input_ids`:      `[batch, seq_len]`  padded with 0
/// - `attention_mask`: `[batch, seq_len]`  1 for real tokens, 0 for padding
/// - `token_type_ids`: `[batch, seq_len]`  all 0 (single-segment model)
pub(crate) fn encodings_to_arrays(encs: &[Encoding]) -> BatchArrays {
    let batch = encs.len();
    let seq_len = encs.iter().map(|e| e.len()).max().unwrap_or(0);

    let mut input_ids = Array2::<i64>::zeros((batch, seq_len));
    let mut attention_mask = Array2::<i64>::zeros((batch, seq_len));
    let token_type_ids = Array2::<i64>::zeros((batch, seq_len));

    for (i, enc) in encs.iter().enumerate() {
        for (j, &id) in enc.get_ids().iter().enumerate() {
            input_ids[[i, j]] = id as i64;
            attention_mask[[i, j]] = 1;
        }
    }

    BatchArrays {
        input_ids,
        attention_mask,
        token_type_ids,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Truncate an `Encoding` to at most `max_len` tokens.
fn truncate(enc: Encoding, max_len: usize) -> Encoding {
    if enc.len() <= max_len {
        return enc;
    }
    let ids = enc.get_ids()[..max_len].to_vec();
    let type_ids = enc.get_type_ids()[..max_len].to_vec();
    let tokens: Vec<String> = enc.get_tokens()[..max_len]
        .iter()
        .map(|s| s.to_owned())
        .collect();
    let words: Vec<Option<u32>> = enc.get_word_ids()[..max_len].to_vec();
    let offsets = enc.get_offsets()[..max_len].to_vec();
    let special_tokens = enc.get_special_tokens_mask()[..max_len].to_vec();
    let attention_mask = enc.get_attention_mask()[..max_len].to_vec();

    Encoding::new(
        ids,
        type_ids,
        tokens,
        words,
        offsets,
        special_tokens,
        attention_mask,
        vec![],             // overflowing: none after truncation
        Default::default(), // sequence_ranges
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Encoding;

    fn dummy_encodings(lengths: &[usize]) -> Vec<Encoding> {
        lengths
            .iter()
            .map(|&n| {
                Encoding::new(
                    vec![1u32; n],
                    vec![0u32; n],
                    vec!["x".to_owned(); n],
                    vec![None; n],
                    vec![(0usize, 1usize); n],
                    vec![0u32; n],
                    vec![1u32; n],
                    vec![],
                    Default::default(),
                )
            })
            .collect()
    }

    #[test]
    fn arrays_shape_is_batch_x_max_seq() {
        let encs = dummy_encodings(&[3, 5, 2]);
        let arrays = encodings_to_arrays(&encs);
        assert_eq!(arrays.input_ids.shape(), &[3, 5]);
        assert_eq!(arrays.attention_mask.shape(), &[3, 5]);
        assert_eq!(arrays.token_type_ids.shape(), &[3, 5]);
    }

    #[test]
    fn attention_mask_is_1_for_real_tokens() {
        let encs = dummy_encodings(&[3, 2]);
        let arrays = encodings_to_arrays(&encs);
        // row 0: all 3 are real
        assert_eq!(arrays.attention_mask.row(0).to_vec(), vec![1, 1, 1]);
        // row 1: 2 real + 1 padding
        assert_eq!(arrays.attention_mask.row(1).to_vec(), vec![1, 1, 0]);
    }

    #[test]
    fn token_type_ids_all_zero() {
        let encs = dummy_encodings(&[4]);
        let arrays = encodings_to_arrays(&encs);
        let zeros: Vec<i64> = vec![0; 4];
        assert_eq!(arrays.token_type_ids.row(0).to_vec(), zeros);
    }
}
