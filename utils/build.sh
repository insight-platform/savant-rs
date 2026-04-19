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

. .envrc

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

# DeepStream runtime libraries (nvinfer, nvdsgst_meta, nvds_meta, etc.)
DS_LIB_DIR="/opt/nvidia/deepstream/deepstream/lib"
if [ -d "$DS_LIB_DIR" ]; then
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${DS_LIB_DIR}"
    export LD_LIBRARY_PATH
    echo "Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
fi

# Build configuration
CARGO_BUILD_FLAG=""
if [ "$MODE" = "release" ]; then
    CARGO_BUILD_FLAG="--release"
fi

# Resolve Python interpreter: explicit override > project venv > auto-find.
if [ -z "${PYTHON_INTERPRETER:-}" ] && [ -x "$PROJECT_DIR/venv/bin/python3" ]; then
    PYTHON_INTERPRETER="$PROJECT_DIR/venv/bin/python3"
fi

MATURIN_PYTHON_SEARCH_ARGS="-f"
if [ -n "${PYTHON_INTERPRETER:-}" ]; then
    echo "Building for $PYTHON_INTERPRETER"
    MATURIN_PYTHON_SEARCH_ARGS="-i $PYTHON_INTERPRETER"
fi

# Optional Cargo features forwarded to the savant_rs crate (e.g. gst).
FEATURE_ARGS=""
if [ -n "${SAVANT_FEATURES:-}" ]; then
    echo "Enabling Cargo features: $SAVANT_FEATURES"
    FEATURE_ARGS="--features=${SAVANT_FEATURES}"
fi

# Build Python package
cd "$PROJECT_DIR/savant_python" || exit 1
EXTRA_FLAGS="$MATURIN_PYTHON_SEARCH_ARGS"
if [ "$MODE" = "release" ]; then
    echo "Building release version"
    EXTRA_FLAGS="--release $MATURIN_PYTHON_SEARCH_ARGS"
fi

export CARGO_INCREMENTAL=true
maturin build $EXTRA_FLAGS $FEATURE_ARGS -o "$PROJECT_DIR/dist"

# Run auditwheel repair with system libs excluded.  Maturin's built-in
# repair bundles everything; we need our libs (libsavant_core_py.so) but not
# GStreamer, CUDA, or DeepStream (expected on the target system at runtime).
#
# Only repair linux_* tagged wheels (the raw maturin output).  Previously
# repaired manylinux_* wheels must be removed first to avoid auditwheel
# choking on hash-suffixed SONAMEs from a prior run.
if command -v auditwheel >/dev/null 2>&1; then
    rm -f "$PROJECT_DIR"/dist/savant_rs-*manylinux*.whl
    for whl in "$PROJECT_DIR"/dist/savant_rs-*linux_*.whl; do
        [ -f "$whl" ] || continue
        echo "Repairing wheel (excluding system libs): $whl"
        auditwheel repair "$whl" -w "$PROJECT_DIR/dist" \
            --exclude 'libgst*.so*' \
            --exclude 'libgstreamer*.so*' \
            --exclude 'libgobject*.so*' \
            --exclude 'libglib*.so*' \
            --exclude 'libgio*.so*' \
            --exclude 'libcuda.so*' \
            --exclude 'libcudart*.so*' \
            --exclude 'libnvdsbufferpool*.so*' \
            --exclude 'libnvbufsurftransform*.so*' \
            --exclude 'libnvdsgst_meta*.so*' \
            --exclude 'libnvds_meta*.so*' \
            --exclude 'libEGL*.so*' \
            --exclude 'libGL*.so*'
        rm -f "$whl"
    done
fi

# Add features to wheel filename when SAVANT_FEATURES is set (e.g. savant_rs-1.15.1+gst-cp312-...)
if [ -n "${SAVANT_FEATURES:-}" ]; then
    FEATURE_SUFFIX=$(echo "$SAVANT_FEATURES" | tr ',' '.')
    # Append DeepStream version to suffix when building with deepstream feature
    if echo "$SAVANT_FEATURES" | grep -qw deepstream; then
        DS_LINK="/opt/nvidia/deepstream/deepstream"
        if [ -L "$DS_LINK" ]; then
            DS_VER=$(readlink "$DS_LINK" | sed 's/.*deepstream-//')
            if [ -n "$DS_VER" ]; then
                FEATURE_SUFFIX="${FEATURE_SUFFIX}.ds${DS_VER}"
                echo "Detected DeepStream version: $DS_VER"
            fi
        fi
    fi
    for whl in "$PROJECT_DIR"/dist/savant_rs-*.whl; do
        [ -f "$whl" ] || continue
        base=$(basename "$whl")
        new=$(echo "$base" | sed "s|^savant_rs-\([0-9]*\.[0-9]*\.[0-9]*\)-|savant_rs-\1+${FEATURE_SUFFIX}-|")
        if [ "$base" != "$new" ]; then
            mv "$whl" "$PROJECT_DIR/dist/$new"
            echo "Renamed wheel: $base -> $new"
        fi
    done
fi
