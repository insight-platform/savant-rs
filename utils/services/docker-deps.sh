#!/bin/sh
set -euo pipefail  # Stricter error handling

# Update and install dependencies with minimal attack surface
apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    clang \
    ca-certificates \
    libva-dev \
    curl \
    unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
