#!/bin/bash

set -e

curl -o rustup.sh --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs
sh rustup.sh -y
source $HOME/.cargo/env
rustup update
rustc -V

rustup target add aarch64-unknown-linux-gnu

cd savant_capi
cargo build --target x86_64-unknown-linux-gnu --release

export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++

cargo build --target aarch64-unknown-linux-gnu --release

cd ..
mkdir -p savant_capi/artifacts
cp target/aarch64-unknown-linux-gnu/release/libsavant_capi.so savant_capi/artifacts/libsavant_capi_aarch64.so
cp target/x86_64-unknown-linux-gnu/release/libsavant_capi.so savant_capi/artifacts/libsavant_capi_x86_64.so
cp savant_capi/capi/savant_capi.h savant_capi/artifacts/savant_capi.h
if [ -f artifacts.tar.gz ]; then
  rm artifacts.tar.gz
fi
tar -czf artifacts.tar.gz savant_capi/artifacts