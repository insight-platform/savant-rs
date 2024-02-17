#!/usr/bin/env bash

set -e

ARGS=-f

RUST_STD_DIR=$( find $HOME -name 'libstd-*.so' | grep stable | head -n1 | xargs dirname)
echo "Rust std dir: $RUST_STD_DIR"

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RUST_STD_DIR:$(pwd)/target/release:$(pwd)/target/release/deps
export LD_LIBRARY_PATH

if [[ -z $PYTHON_INTERPRETER ]]; then
    ARGS=-f
else
    ARGS="-i $PYTHON_INTERPRETER"
fi

echo "Additional build args: $ARGS"

cargo build --release
plugins/prepare_native_plugins.sh release
mkdir -p /opt/dist
cp target/release/plugins*.tar.gz /opt/dist
cd savant_python && maturin build $ARGS --release --out /opt/dist
