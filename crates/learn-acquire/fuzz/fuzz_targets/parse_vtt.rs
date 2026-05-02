#![no_main]

// Fuzz target for the VTT parser in learn-acquire.
//
// Strategy:
//   - Take arbitrary bytes from the fuzzer.
//   - Lossily decode to UTF-8 (preserving as many bytes as possible).
//   - Write to a temp file and call the public `parse_vtt` entry point,
//     which exercises the file-IO path as well as the parser.
//   - Discard the result — we're checking for panics and memory safety,
//     not correctness. The parser is documented to return Ok(vec![]) on
//     malformed input rather than propagating errors.
//
// Run:
//   cargo +nightly fuzz run parse_vtt
//
// With a corpus from fixtures:
//   cargo +nightly fuzz run parse_vtt -- fuzz/corpus/parse_vtt/
//
// Generate a corpus seed from the fixture file:
//   mkdir -p fuzz/corpus/parse_vtt
//   cp ../../../../fixtures/short.vtt fuzz/corpus/parse_vtt/short.vtt

use libfuzzer_sys::fuzz_target;
use camino::Utf8PathBuf;
use std::io::Write as _;

fuzz_target!(|data: &[u8]| {
    // Lossily decode bytes to a UTF-8 string (replaces invalid sequences with U+FFFD).
    let content = String::from_utf8_lossy(data);

    // Write to a temp file so we exercise the full public API including I/O.
    let mut tmp = match tempfile_named() {
        Some(f) => f,
        None => return, // If we can't create a temp file, skip this input.
    };

    if tmp.file.write_all(content.as_bytes()).is_err() {
        return;
    }
    // Flush so the file handle is visible to read_to_string inside parse_vtt.
    let _ = tmp.file.flush();

    // Call the public parse_vtt entry point; discard Ok/Err.
    let path = Utf8PathBuf::try_from(tmp.path.clone()).unwrap_or_default();
    let _ = learn_acquire::vtt::parse_vtt(&path);
});

// ── helpers ───────────────────────────────────────────────────────────────────

struct TempVtt {
    file: std::fs::File,
    path: std::path::PathBuf,
}

impl Drop for TempVtt {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn tempfile_named() -> Option<TempVtt> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("fuzz_vtt_{ns}.vtt"));
    let file = std::fs::File::create(&path).ok()?;
    Some(TempVtt { file, path })
}
