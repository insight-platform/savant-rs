#!/bin/sh -e
set -e

# x86 
if [ "$(uname -m)" = "x86_64" ]; then
    userdel -r ubuntu
fi

groupadd -g ${REMOTE_GID} ${REMOTE_USER}
useradd -m -u ${REMOTE_UID} -g ${REMOTE_GID} -s /bin/bash ${REMOTE_USER}
