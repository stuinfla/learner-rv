# Fuzz Harness — `learn-acquire`

This directory contains a `cargo-fuzz` harness for the WebVTT parser in
`learn-acquire`.

## Prerequisites

- Rust nightly toolchain: `rustup toolchain install nightly`
- cargo-fuzz: `cargo install cargo-fuzz`

## Running the fuzzer

From the workspace root:

```bash
cd crates/learn-acquire
cargo +nightly fuzz run parse_vtt
```

Or with a time limit (e.g., 60 seconds):

```bash
cargo +nightly fuzz run parse_vtt -- -max_total_time=60
```

## Seeding the corpus

Copy the fixture VTT file as a seed before running to give the fuzzer a
realistic starting point:

```bash
mkdir -p fuzz/corpus/parse_vtt
cp ../../../../fixtures/short.vtt fuzz/corpus/parse_vtt/short.vtt
```

Then run with the corpus directory:

```bash
cargo +nightly fuzz run parse_vtt -- fuzz/corpus/parse_vtt/
```

## What is being tested

The fuzz target exercises `learn_acquire::vtt::parse_vtt`, which is the public
entry point that reads a file from disk and delegates to the internal
`parse_vtt_str`. Arbitrary bytes are written to a temp file and fed to the
parser. Panics and address-sanitizer violations are reported as findings.

Expected behaviour: `parse_vtt` must never panic on arbitrary input. It may
return `Ok(vec![])` for malformed content or `Err(LearnError::Acquire(…))` if
the file cannot be read — both are acceptable.

## CI note

The fuzz harness uses `[workspace]` to opt out of the main workspace resolver
so it can use nightly features without affecting `cargo build --workspace` on
stable. The CI matrix does **not** run the fuzzer continuously; add a dedicated
fuzz job with `cargo +nightly fuzz run parse_vtt -- -max_total_time=300` if
you want scheduled fuzz runs.
