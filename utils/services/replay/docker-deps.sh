#!/bin/sh -e

apt-get update && apt install -y clang

ARCH=$(uname -m)

PB_REL="https://github.com/protocolbuffers/protobuf/releases"

# x86_64
if [ "$ARCH" = "x86_64" ]; then
  curl -LO $PB_REL/download/v3.15.8/protoc-3.15.8-linux-x86_64.zip
elif [ "$ARCH" = "aarch64" ]; then
  curl -LO $PB_REL/download/v3.15.8/protoc-3.15.8-linux-aarch_64.zip
else
    echo "Unsupported architecture $ARCH"
    exit 1
fi

unzip *.zip
cp bin/protoc /usr/bin
chmod 755 /usr/bin/protoc

