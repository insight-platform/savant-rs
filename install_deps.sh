#!/bin/sh -e

apt-get update
apt-get install -y protobuf-compiler
pip install maturin==0.15
