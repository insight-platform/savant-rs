#!/bin/bash

ARCH=$1

set -e

rustup target add $ARCH-unknown-linux-gnu

PB_REL="https://github.com/protocolbuffers/protobuf/releases"
curl -LO $PB_REL/download/v3.15.8/protoc-3.15.8-linux-x86_64.zip
unzip protoc-3.15.8-linux-x86_64.zip -d $HOME/.local
export PATH="$PATH:$HOME/.local/bin"

cd savant_capi
cargo build --target $ARCH-unknown-linux-gnu # --release