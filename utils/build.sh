#!/bin/bash

# Exit on error, undefined variables
set -eu

# Function to clean up temporary files and processes on exit
cleanup() {
    # Add cleanup tasks here if needed
    exit "${1:-0}"
}

# Set up trap for script termination
trap 'cleanup $?' EXIT
trap 'cleanup 1' INT TERM

ARCHITECTURE=$(uname -m)
readonly ARCHITECTURE

MODE="${1:-}"
readonly MODE

# Validate input parameters
if [ -z "$MODE" ] || { [ "$MODE" != "debug" ] && [ "$MODE" != "release" ]; }; then
    echo >&2 "Usage: $0 <debug|release>"
    exit 1
fi

# Check for required commands
for cmd in jq cargo rustup tar find; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo >&2 "Error: Required command '$cmd' not found"
        exit 1
    fi
done

# Use mktemp for secure temporary directory creation if needed
# TEMP_DIR=$(mktemp -d) || exit 1

# Get cargo metadata safely
TARGET_DIR=$(cargo metadata --format-version 1 | jq -r '.target_directory' || exit 1)
readonly TARGET_DIR
echo "Cargo target dir: $TARGET_DIR"

BUILD_ARTIFACT_LOCATION="${TARGET_DIR}/${MODE}"
readonly BUILD_ARTIFACT_LOCATION
echo "Build artifact location: $BUILD_ARTIFACT_LOCATION"

ARTIFACT_LOCATION="dist/build_artifacts"
readonly ARTIFACT_LOCATION

# Export variables with proper quoting
export PROJECT_DIR
PROJECT_DIR=$(pwd)
echo "Project dir: $PROJECT_DIR"

export RUST_TOOLCHAIN
RUST_TOOLCHAIN=$(rustup default | awk '{print $1}')
echo "Rust toolchain: $RUST_TOOLCHAIN"

# debug
find "$HOME" -name 'libstd-*.so' 2>/dev/null | grep -F "$RUST_TOOLCHAIN"

# Find Rust std directory more securely
RUST_STD_DIR=$(find "$HOME" -name 'libstd-*.so' 2>/dev/null | grep -F "$RUST_TOOLCHAIN" | head -n1 | xargs -r dirname)
readonly RUST_STD_DIR
echo "Rust std dir: $RUST_STD_DIR"

# Set LD_LIBRARY_PATH with proper path handling
LD_LIBRARY_PATH="${RUST_STD_DIR}:${PROJECT_DIR}/target/${MODE}:${PROJECT_DIR}/target/${MODE}/deps"
if [ -n "${CARGO_TARGET_DIR:-}" ]; then
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CARGO_TARGET_DIR}/${MODE}:${CARGO_TARGET_DIR}/${MODE}/deps"
fi
export LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Build configuration
CARGO_BUILD_FLAG=""
if [ "$MODE" = "release" ]; then
    CARGO_BUILD_FLAG="--release"
fi

if [ "${BUILD_ENVIRONMENT:-}" != "manylinux" ]; then
    echo "Building python-embedded libraries"
    if [ -n "${PYTHON_INTERPRETER:-}" ]; then
        export PYO3_PYTHON="$PYTHON_INTERPRETER"
    fi
    
    # Build with explicit targets
    cargo build $CARGO_BUILD_FLAG -p savant_rs # -p savant_gstreamer_elements -p savant_launcher

    # Clean previous artifacts
    rm -rf "$ARTIFACT_LOCATION"

    # Copy artifacts with error checking
    for file in "$BUILD_ARTIFACT_LOCATION"/*.so; do # "$BUILD_ARTIFACT_LOCATION/savant_launcher"; do
        if [ -f "$file" ]; then
            install -D "$file" "$ARTIFACT_LOCATION/$(basename "$file")"
        fi
    done

    # Copy libstd safely
    find "$HOME" -name "libstd-*.so" 2>/dev/null | grep -F "$RUST_TOOLCHAIN" | head -n1 | xargs -r install -D -t "$ARTIFACT_LOCATION"

    # Create tarball with explicit paths
    echo "Packing artifacts"
    (cd "$ARTIFACT_LOCATION" && tar --create --gzip --file "../embedded_python-${ARCHITECTURE}.tar.gz" ./*.so)
fi

# Set maturin arguments
MATURIN_PYTHON_SEARCH_ARGS="-f"
if [ -n "${PYTHON_INTERPRETER:-}" ]; then
    echo "Building for $PYTHON_INTERPRETER"
    MATURIN_PYTHON_SEARCH_ARGS="-i $PYTHON_INTERPRETER"
fi

# Build Python package
cd "$PROJECT_DIR/savant_python" || exit 1
EXTRA_FLAGS="$MATURIN_PYTHON_SEARCH_ARGS"
if [ "$MODE" = "release" ]; then
    echo "Building release version"
    EXTRA_FLAGS="--release $MATURIN_PYTHON_SEARCH_ARGS"
fi

export CARGO_INCREMENTAL=true 
maturin build $EXTRA_FLAGS -o "$PROJECT_DIR/dist"
