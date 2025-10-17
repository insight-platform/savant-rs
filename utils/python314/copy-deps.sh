#!/bin/sh
set -euo pipefail  # Strict error handling

# Define target directories
readonly TARGET_BIN_DIR="/opt/bin"
install -d -m 755 "${TARGET_BIN_DIR}"
# Copy binary with executable permissions
if ! install -m 755 /tmp/build/release/savant_info "${TARGET_BIN_DIR}/"; then
    echo "Error: Failed to copy savant_info binary" >&2
    exit 1
fi
