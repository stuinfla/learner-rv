// orientation.rs — Verifies that `learn` (no args) prints the friendly
// orientation block and exits with code 0.
//
// Run with:
//   cargo test --workspace orientation
//
// The test locates the `learn` binary that cargo builds into the workspace
// target directory, so it requires a prior `cargo build --workspace`.
// In CI, run `cargo build --workspace` before `cargo test --workspace`.

use std::process::Command;

/// Path to the workspace `target/debug/learn` binary.
fn learn_bin() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR is the `tests/` crate root.
    // The workspace target dir is two levels up.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest)
        .parent()
        .expect("tests/ crate has a parent");
    workspace_root.join("target").join("debug").join("learn")
}

#[test]
fn no_args_prints_orientation_and_exits_zero() {
    let bin = learn_bin();
    assert!(
        bin.exists(),
        "learn binary not found at {}; run `cargo build --workspace` first",
        bin.display()
    );

    let output = Command::new(&bin)
        .output()
        .expect("failed to spawn learn binary");

    assert!(
        output.status.success(),
        "expected exit code 0, got {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Learn-RV"),
        "orientation header missing from stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("30-second quickstart"),
        "quickstart section missing from stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("learn ingest"),
        "ingest example missing from stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("~/Docs/KB/<topic>.rvf"),
        "KB location line missing from stdout:\n{stdout}"
    );
}

#[test]
fn help_flag_still_exits_zero_and_contains_usage() {
    let bin = learn_bin();
    if !bin.exists() {
        return; // skip silently if binary not built yet
    }

    let output = Command::new(&bin)
        .arg("--help")
        .output()
        .expect("failed to spawn learn binary with --help");

    assert!(
        output.status.success(),
        "expected exit code 0 for --help, got {:?}",
        output.status.code()
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Clap puts "Usage:" in the help text
    assert!(
        stdout.contains("Usage") || stdout.contains("usage"),
        "--help output should contain usage info:\n{stdout}"
    );
}
