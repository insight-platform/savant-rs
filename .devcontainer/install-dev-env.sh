#!/bin/sh -e

curl -o rustup.sh --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs
sh rustup.sh -y
source $HOME/.cargo/env
rustup update
rustc -V

pip install --no-cache-dir \
    skia-python \
    maturin
