#!/usr/bin/env bash

set -e

ARGS=-f

if [[ -z $PYTHON_INTERPRETER ]]; then
    ARGS=-f
else
    ARGS="-i $PYTHON_INTERPRETER"
fi

echo "Additional build args: $ARGS"

cd savant_python && maturin build $ARGS --release --out /opt/dist
cd ../savant_etcd_dynamic_state && maturin build $ARGS --release --out /opt/dist
