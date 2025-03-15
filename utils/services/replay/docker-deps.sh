#!/bin/sh
set -euo pipefail  # Stricter error handling

# Update and install dependencies with minimal attack surface
apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    clang \
    ca-certificates \
    curl \
    unzip \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libpython3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up working directory
WORK_DIR=$(mktemp -d)
cd "${WORK_DIR}"

# Detect architecture in a more robust way
ARCH=$(dpkg --print-architecture)

# Define versions as variables for easier maintenance
PROTOC_VERSION="3.15.8"
PB_REL="https://github.com/protocolbuffers/protobuf/releases"

# Download and verify protoc based on architecture
case "${ARCH}" in
    amd64)
        PROTOC_FILE="protoc-${PROTOC_VERSION}-linux-x86_64.zip"
        ;;
    arm64)
        PROTOC_FILE="protoc-${PROTOC_VERSION}-linux-aarch_64.zip"
        ;;
    *)
        echo "Unsupported architecture: ${ARCH}"
        exit 1
        ;;
esac

# Download with error checking
if ! curl -LO --fail "${PB_REL}/download/v${PROTOC_VERSION}/${PROTOC_FILE}"; then
    echo "Failed to download protoc"
    exit 1
fi

# Verify and install protoc
if ! unzip -q "${PROTOC_FILE}"; then
    echo "Failed to extract protoc"
    exit 1
fi

install -m 755 bin/protoc /usr/local/bin/
rm -rf "${WORK_DIR}"

# Verify installation
if ! command -v protoc >/dev/null 2>&1; then
    echo "protoc installation failed"
    exit 1
fi

