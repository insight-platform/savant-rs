#!/bin/sh -e

mkdir -p /opt/libs
mkdir -p /opt/bin

RUST_STD_LIB=$(find / -name 'libstd-*.so' | head -n1)
echo "Rust std lib: $RUST_STD_LIB"
cp $RUST_STD_LIB /opt/libs/
cp target/release/deps/libsavant_core-*.so /opt/libs/
cp target/release/replay /opt/bin/
