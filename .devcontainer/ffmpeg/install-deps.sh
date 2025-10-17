#!/bin/sh -e

apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install \
    liboping0 \
    liboping-dev \
    clang \
    autoconf \
    automake \
    build-essential \
    cmake \
    git-core \
    libass-dev \
    libavutil-dev \
    libavformat-dev \
    libavfilter-dev \
    libavdevice-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    meson \
    ninja-build \
    pkg-config \
    texinfo \
    wget \
    yasm \
    zlib1g-dev \
    openssl \
    libsasl2-dev \
    libsasl2-2 \
    python3-dev \
    python3-pip \
    curl \
    ffmpeg \
    unzip \
    libunwind-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstrtspserver-1.0-dev \
    libges-1.0-dev \
    jq

ARCH=$(uname -m)

PB_REL="https://github.com/protocolbuffers/protobuf/releases"

cd /usr

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
chmod 755 /usr/bin/protoc