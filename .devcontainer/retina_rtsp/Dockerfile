FROM ghcr.io/insight-platform/py314t_rust:v1.6.0 AS devcontainer

ENV DEBIAN_FRONTEND=noninteractive

# Production deps (docker-deps.sh) + dev tooling
RUN --mount=type=bind,source=utils/services/docker-deps.sh,target=/tmp/docker-deps.sh \
    bash /tmp/docker-deps.sh

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Debuggers and tracing
    gdb \
    lldb \
    strace \
    # Build extras not in base image
    cmake \
    pkg-config \
    git \
    # System monitoring
    htop \
    net-tools \
    # CLI utilities
    ripgrep \
    fd-find \
    bat \
    jq \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && ln -sf /usr/bin/batcat /usr/local/bin/bat \
    && rm -rf /var/lib/apt/lists/*

# Install pinned protoc (same version as production build)
RUN --mount=type=bind,source=utils/services/protoc.sh,target=/tmp/protoc.sh \
    bash /tmp/protoc.sh

# GStreamer — dev headers (build) + runtime plugins (test/run)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

# Install libunwind-dev (same as Dockerfile.services builder-main)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libunwind-dev \
    && rm -rf /var/lib/apt/lists/*

# Relocate Rust toolchain from /root to a shared read-only path
ENV RUSTUP_HOME=/opt/rustup
RUN mv /root/.rustup "$RUSTUP_HOME" \
    && chmod -R a+rX "$RUSTUP_HOME"
# Keep root's cargo bin on PATH for toolchain binaries (rustc, cargo, etc.)
ENV PATH="/root/.cargo/bin:$PATH"

# Create a non-root user with UID 1000 (devcontainer.json's
# updateRemoteUserUID will remap this to the host UID at start)
RUN useradd -m -u 1000 -s /bin/bash dev

# Give dev its own CARGO_HOME for installed binaries and caches,
# but keep the shared RUSTUP_HOME for the toolchain
ENV HOME=/home/dev \
    CARGO_HOME=/home/dev/.cargo
ENV PATH="$CARGO_HOME/bin:$PATH"
# Seed dev's cargo home with the registry config from root
RUN cp -r /root/.cargo/. "$CARGO_HOME" \
    && chown -R dev:dev "$CARGO_HOME"
WORKDIR ${HOME}

USER dev

# Cargo developer tools
RUN --mount=type=bind,source=.devcontainer/retina_rtsp,target=/opt/scripts bash /opt/scripts/install-dev-env.sh
