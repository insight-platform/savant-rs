#!/usr/bin/env bash
set -e

if id ubuntu &>/dev/null; then
    userdel -r ubuntu
fi

groupadd -g ${REMOTE_GID} ${REMOTE_USER}
useradd -m -u ${REMOTE_UID} -g ${REMOTE_GID} -s /bin/bash ${REMOTE_USER}
