# Releasing Savant crates

This document describes how to release the Savant workspace crates to [crates.io](https://crates.io). The workspace uses a single shared version (`[workspace.package].version` in the root `Cargo.toml`), and every crate inherits it via `version.workspace = true`.

## Bumping versions

Internal workspace dependencies use a major-only version requirement (`version = "2"`) declared once in `[workspace.dependencies]` at the workspace root. This minimizes churn for routine releases:

- **Minor or patch bump** (e.g. `2.1.0` → `2.1.1`, `2.2.0`, `2.3.4`): edit exactly one line — `[workspace.package].version` in the root `Cargo.toml`. No `[workspace.dependencies]` line changes because `"2"` already covers the `^2.0.0` range. You can automate this with [`cargo-edit`](https://github.com/killercup/cargo-edit):

  ```sh
  cargo set-version --workspace --bump patch
  # or: cargo set-version --workspace --bump minor
  ```

- **Major bump** (e.g. `2.x` → `3.0.0`): bump `[workspace.package].version` AND do one search-and-replace of `version = "2"` → `version = "3"` in the `[workspace.dependencies]` section of the root `Cargo.toml`. Both changes fit in a single commit.

## Publish order

Crates.io enforces that every dependency is published before any crate that depends on it. The file [`release-order.txt`](release-order.txt) encodes the DAG publish order (leaves first). Use it as the source of truth.

```sh
while IFS= read -r crate; do
    case "$crate" in
        ""|\#*) continue ;;
    esac
    echo ">>> publishing $crate"
    cargo publish -p "$crate"
done < release-order.txt
```

For a dry run that catches missing READMEs, invalid metadata, or broken paths without uploading anything:

```sh
while IFS= read -r crate; do
    case "$crate" in
        ""|\#*) continue ;;
    esac
    cargo publish --dry-run -p "$crate" || { echo "FAIL: $crate"; exit 1; }
done < release-order.txt
```

## Pre-release checklist

1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo build --workspace` (all features off)
4. `cargo build -p savant-rs --features deepstream` (requires DeepStream 7.x + GStreamer + CUDA host)
5. `cargo test --workspace --no-run`
6. Python smoke tests: `SAVANT_FEATURES=deepstream make sp-pytest`
7. Dry-run publish in DAG order (see above).
8. Bump version (see "Bumping versions").
9. Commit & tag: `git commit -m "release: vX.Y.Z"` && `git tag vX.Y.Z && git push --follow-tags`.
10. Publish in DAG order (replace `--dry-run` with nothing).

## Not published

The following workspace members are intentionally never published to crates.io:

- (workspace-local binaries / helpers that are not part of the public Savant API; if any are re-added in future, mark them with `publish = false` in their `Cargo.toml`).
