FROM ghcr.io/insight-platform/savant_rust:latest AS base

FROM base AS devcontainer

RUN apt-get update && apt-get install -y clang build-essential \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

ARG REMOTE_USER
ARG REMOTE_UID
ARG REMOTE_GID
RUN <<EOF
    addgroup --gid ${REMOTE_GID} ${REMOTE_USER}
    adduser --disabled-password --uid ${REMOTE_UID} --gid ${REMOTE_GID} ${REMOTE_USER}
EOF

ENV HOME /home/${REMOTE_USER}
# set directory to homedir
WORKDIR ${HOME}

USER ${REMOTE_USER}

RUN --mount=type=bind,source=.,target=/opt/scripts bash /opt/scripts/install-dev-env.sh
