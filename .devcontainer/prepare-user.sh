#!/usr/bin/env bash
set -e

existing_user=$(getent passwd "${REMOTE_UID}" | cut -d: -f1 || true)
if [ -n "${existing_user}" ]; then
    userdel -r "${existing_user}"
fi

existing_group=$(getent group "${REMOTE_GID}" | cut -d: -f1 || true)
if [ -n "${existing_group}" ]; then
    groupdel "${existing_group}"
fi

groupadd -g ${REMOTE_GID} ${REMOTE_USER}
useradd -m -u ${REMOTE_UID} -g ${REMOTE_GID} -s /bin/bash ${REMOTE_USER}
