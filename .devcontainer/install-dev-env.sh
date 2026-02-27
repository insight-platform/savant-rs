#!/bin/sh -e

# ── Rust toolchain ────────────────────────────────────────────────────────────
curl -o rustup.sh --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs
sh rustup.sh -y
. "$HOME/.cargo/env"
rustup update
rustc -V

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
    sccache \
    cbindgen

# cargo-expand has no pre-built binary; compile from source
cargo install --locked cargo-expand

# ── sccache: use as Rust compiler wrapper to speed up rebuilds ─────────────────
echo 'export RUSTC_WRAPPER=sccache' >> "$HOME/.bashrc"

# ── Python packages ───────────────────────────────────────────────────────────
pip install --no-cache-dir \
    auditwheel \
    skia-python \
    maturin \
    virtualenv \
    ruff \
    pytest \
    pytest-asyncio \
    black \
    isort \
    unify \
    mypy \
    ipython
