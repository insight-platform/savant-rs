FROM ubuntu:25.10 AS base

FROM base AS devcontainer

ARG REMOTE_USER
ARG REMOTE_UID
ARG REMOTE_GID

RUN userdel -r ubuntu && \
    groupadd -g ${REMOTE_GID} ${REMOTE_USER} && \
    useradd -m -u ${REMOTE_UID} -g ${REMOTE_GID} -s /bin/bash ${REMOTE_USER}

ENV HOME=/home/${REMOTE_USER}
# set directory to homedir
WORKDIR ${HOME}

RUN --mount=type=bind,source=.,target=/opt/scripts bash /opt/scripts/install-deps.sh

USER ${REMOTE_USER}

RUN --mount=type=bind,source=.,target=/opt/scripts bash /opt/scripts/install-rust.sh

