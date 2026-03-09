FROM ghcr.io/insight-platform/savant_rust:latest AS base

FROM base AS devcontainer

RUN apt-get update && apt-get install -y \
    # C/C++ toolchain
    clang \
    build-essential \
    cmake \
    pkg-config \
    # LLVM / bindgen support
    llvm \
    libclang-dev \
    # Debuggers and tracing
    gdb \
    lldb \
    strace \
    # GStreamer
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    # Protobuf compiler (for prost/tonic codegen)
    protobuf-compiler \
    # Graphics / font libs
    libva-dev \
    libgl-dev \
    libegl-dev \
    libfontconfig-dev \
    fonts-dejavu-core \
    # Python C headers (for native extensions)
    python3-dev \
    # Build tools
    ninja-build \
    # System monitoring
    nvtop \
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


ARG REMOTE_USER
ARG REMOTE_UID
ARG REMOTE_GID

RUN userdel -r ubuntu && \
    groupadd -g ${REMOTE_GID} ${REMOTE_USER} && \
    useradd -m -u ${REMOTE_UID} -g ${REMOTE_GID} -s /bin/bash ${REMOTE_USER} && \
    pip uninstall -y savant_rs

ENV HOME=/home/${REMOTE_USER}
# set directory to homedir
WORKDIR ${HOME}

USER ${REMOTE_USER}

RUN --mount=type=bind,source=.devcontainer,target=/opt/scripts bash /opt/scripts/install-dev-env.sh


