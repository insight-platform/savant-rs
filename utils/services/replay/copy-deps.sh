#!/bin/sh
set -euo pipefail  # Strict error handling

# Define target directories
readonly TARGET_LIB_DIR="/opt/libs"
readonly TARGET_BIN_DIR="/opt/bin"
readonly TARGET_ETC_DIR="/opt/etc"

# Create directories with specific permissions
install -d -m 755 "${TARGET_LIB_DIR}"
install -d -m 755 "${TARGET_BIN_DIR}"
install -d -m 755 "${TARGET_ETC_DIR}"

# Find and validate Rust standard library
RUST_STD_LIB=$(find / -name 'libstd-*.so' -type f -print -quit)
if [ -z "${RUST_STD_LIB}" ]; then
    echo "Error: Could not find Rust standard library" >&2
    exit 1
fi
echo "Rust std lib: ${RUST_STD_LIB}"

# Copy files with proper permissions and error checking
if ! install -m 644 "${RUST_STD_LIB}" "${TARGET_LIB_DIR}/"; then
    echo "Error: Failed to copy Rust standard library" >&2
    exit 1
fi

# Copy dependency libraries
if ! install -m 644 /tmp/build/release/deps/*.so "${TARGET_LIB_DIR}/"; then
    echo "Error: Failed to copy dependency libraries" >&2
    exit 1
fi

# Copy binary with executable permissions
if ! install -m 755 /tmp/build/release/replay "${TARGET_BIN_DIR}/"; then
    echo "Error: Failed to copy replay binary" >&2
    exit 1
fi

# Copy binary with executable permissions
if ! install -m 755 /tmp/build/release/savant_info "${TARGET_BIN_DIR}/"; then
    echo "Error: Failed to copy savant_info binary" >&2
    exit 1
fi


# Copy configuration file
if ! install -m 644 /opt/savant-rs/services/replay/replay/assets/test.json "${TARGET_ETC_DIR}/config.json"; then
    echo "Error: Failed to copy configuration file" >&2
    exit 1
fi

echo "All files copied successfully"
