#!/usr/bin/env bash
set -e

# Rust toolchain is already in the base image (py314t_rust).
# Just install dev-time cargo tools.

# ── cargo-binstall: fast pre-built binary installer ───────────────────────────
curl -L --proto '=https' --tlsv1.2 -sSf \
    https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh \
    | bash

# ── Cargo developer tools (pre-built binaries via binstall) ───────────────────
cargo binstall --no-confirm \
    cargo-nextest \
    cargo-watch \
    cargo-edit \
    cargo-audit \
    sccache

# ── sccache: use as Rust compiler wrapper to speed up rebuilds ─────────────────
echo 'export RUSTC_WRAPPER=sccache' >> "$HOME/.bashrc"
