#!/bin/sh -e

ARCH=$(uname -m)
PB_REL="https://github.com/protocolbuffers/protobuf/releases"

if [ "$ARCH" = "x86_64" ]; then
    echo "Installing x86_64 dependencies"
    curl -LO $PB_REL/download/v23.2/protoc-23.2-linux-x86_64.zip
    unzip protoc-23.2-linux-x86_64.zip
elif [ "$ARCH" = "aarch64" ]; then
    echo "Installing aarch64 dependencies"
    curl -LO $PB_REL/download/v23.2/protoc-23.2-linux-aarch_64.zip
    unzip protoc-23.2-linux-aarch_64.zip
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

export PATH=$PATH:$PWD/bin
