#!/bin/sh -e

mkdir -p /opt/libs
mkdir -p /opt/bin
mkdir -p /opt/etc

RUST_STD_LIB=$(find / -name 'libstd-*.so' | head -n1)
echo "Rust std lib: $RUST_STD_LIB"
cp $RUST_STD_LIB /opt/libs/
cp /tmp/build/release/deps/*.so /opt/libs/
cp /tmp/build/release/replay /opt/bin/

cp /opt/savant-rs/services/replay/replay/assets/test.json /opt/etc/config.json
