FROM ghcr.io/insight-platform/savant_rust:latest AS base

FROM base AS devcontainer

RUN apt-get update && apt-get install -y clang build-essential \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    nvtop \
    htop \
    ninja-build \
    libgl-dev \
    libegl-dev \
    libfontconfig-dev \
    fonts-dejavu-core


ARG REMOTE_USER
ARG REMOTE_UID
ARG REMOTE_GID

RUN userdel -r ubuntu && \
    groupadd -g ${REMOTE_GID} ${REMOTE_USER} && \
    useradd -m -u ${REMOTE_UID} -g ${REMOTE_GID} -s /bin/bash ${REMOTE_USER}

ENV HOME=/home/${REMOTE_USER}
# set directory to homedir
WORKDIR ${HOME}

USER ${REMOTE_USER}

RUN --mount=type=bind,source=.devcontainer,target=/opt/scripts bash /opt/scripts/install-dev-env.sh

