#!/bin/bash -e

ARCHITECTURE=$(uname -m)
MODE=$1
# ensure MODE is set and either "debug" or "release"
if [ -z "$MODE" ]; then
    echo "Usage: $0 <debug|release>"
    exit 1
fi

if [ "$MODE" != "debug" ] && [ "$MODE" != "release" ]; then
    echo "Usage: $0 <debug|release>"
    exit 1
fi

# ensure jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq could not be found. Please install jq."
    exit 1
fi

TARGET_DIR=$(cargo metadata --format-version 1 | jq -r '.target_directory')
echo "Cargo target dir: $TARGET_DIR"

BUILD_ARTIFACT_LOCATION=$TARGET_DIR/$MODE
echo "Build artifact location: $BUILD_ARTIFACT_LOCATION"

ARTIFACT_LOCATION=dist/build_artifacts
echo "Artifact location: $ARTIFACT_LOCATION"
mkdir -p $ARTIFACT_LOCATION

export PROJECT_DIR=$(pwd)
echo "Project dir: $PROJECT_DIR"
export RUST_TOOLCHAIN=$(rustup default | awk '{print $1}')
echo "Rust toolchain: $RUST_TOOLCHAIN"
RUST_STD_DIR=$(find $HOME -name 'libstd-*.so' 2>/dev/null | grep $RUST_TOOLCHAIN | head -n1 | xargs dirname)
echo "Rust std dir: $RUST_STD_DIR"

export LD_LIBRARY_PATH=$RUST_STD_DIR:$(pwd)/target/$MODE:$(pwd)/target/$MODE/deps
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CARGO_TARGET_DIR/$MODE:$CARGO_TARGET_DIR/$MODE/deps
echo "$LD_LIBRARY_PATH"

# build python-embedded libraries
CARGO_BUILD_FLAG=""
if [ "$MODE" = "release" ]; then
  CARGO_BUILD_FLAG="--release"
fi

if [ "$BUILD_ENVIRONMENT" != "manylinux" ]; then

  echo "Building python-embedded libraries"
  if [ -n "$PYTHON_INTERPRETER" ]; then
      export PYO3_PYTHON=$PYTHON_INTERPRETER
  fi
  cargo build $CARGO_BUILD_FLAG -p savant_rs -p savant_gstreamer_elements -p savant_launcher
  cp "$BUILD_ARTIFACT_LOCATION"/*.so $ARTIFACT_LOCATION
  cp "$BUILD_ARTIFACT_LOCATION"/savant_launcher $ARTIFACT_LOCATION
  cp $(find "$HOME" -name 'libstd-*.so' 2>/dev/null | grep "$RUST_TOOLCHAIN") $ARTIFACT_LOCATION

  # pack artifacts in a tarball
  echo "Packing artifacts"
  cd $ARTIFACT_LOCATION && tar -czf ../embedded_python-"$ARCHITECTURE".tar.gz *.so

fi

MATURIN_PYTHON_SEARCH_ARGS=-f

if [[ -z $PYTHON_INTERPRETER ]]; then
    MATURIN_PYTHON_SEARCH_ARGS=-f
else
    MATURIN_PYTHON_SEARCH_ARGS="-i $PYTHON_INTERPRETER"
fi

cd "$PROJECT_DIR"/savant_python
if [ "$MODE" = "debug" ]; then
  EXTRA_FLAGS="$MATURIN_PYTHON_SEARCH_ARGS"
else
  EXTRA_FLAGS="--release $MATURIN_PYTHON_SEARCH_ARGS"
fi

CARGO_INCREMENTAL=true maturin build $EXTRA_FLAGS -o "$PROJECT_DIR"/dist

#cd "$PROJECT_DIR"
#
#for d in $(find savant_plugins/* -maxdepth 0 -type d); do
#		cd $d && CARGO_INCREMENTAL=true maturin build $EXTRA_FLAGS -o "$PROJECT_DIR"/dist
#		cd "$PROJECT_DIR"
#done
