# Building learn-rs

This workspace depends on RuVector_Clean crates via a `../ruvector` symlink
(a sibling of the `learn-rs` directory, resolved through Cargo relative paths).

**Stuart's machine:** the symlink is already present at
`/Users/stuartkerr/Code/Video watcher skill/ruvector -> ~/RuVector_Clean`.

**CI / other machines:** create the sibling symlink before running `cargo check`:
```
ln -s /path/to/RuVector_Clean /path/to/parent-dir/ruvector
```
The symlink must be a sibling of `learn-rs/`, not inside it.
