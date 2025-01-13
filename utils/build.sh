#!/bin/bash -e

MODE=$1

export PROJECT_DIR=$(pwd)
echo "Project dir: $PROJECT_DIR"
export RUST_TOOLCHAIN=$(rustup default | awk '{print $1}')
echo "Rust toolchain: $RUST_TOOLCHAIN"
RUST_STD_DIR=$(find $HOME -name 'libstd-*.so' 2>/dev/null | grep $RUST_TOOLCHAIN | head -n1 | xargs dirname)
echo "Rust std dir: $RUST_STD_DIR"

export LD_LIBRARY_PATH=$RUST_STD_DIR:$(pwd)/target/$MODE:$(pwd)/target/$MODE/deps
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CARGO_TARGET_DIR/$MODE:$CARGO_TARGET_DIR/$MODE/deps
echo $LD_LIBRARY_PATH

MATURIN_PYTHON_SEARCH_ARGS=-f

if [[ -z $PYTHON_INTERPRETER ]]; then
    MATURIN_PYTHON_SEARCH_ARGS=-f
else
    MATURIN_PYTHON_SEARCH_ARGS="-i $PYTHON_INTERPRETER"
fi

cd $PROJECT_DIR/savant_python
if [ "$MODE" = "debug" ]; then
  EXTRA_FLAGS="$MATURIN_PYTHON_SEARCH_ARGS"
else
  EXTRA_FLAGS="--release $MATURIN_PYTHON_SEARCH_ARGS"
fi

CARGO_INCREMENTAL=true maturin build $EXTRA_FLAGS -o $PROJECT_DIR/dist

cd $PROJECT_DIR

for d in $(find savant_plugins/* -maxdepth 0 -type d); do
		cd $d && CARGO_INCREMENTAL=true maturin build $EXTRA_FLAGS -o $PROJECT_DIR/dist
		cd $PROJECT_DIR
done
