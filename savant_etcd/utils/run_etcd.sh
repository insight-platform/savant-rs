#!/bin/bash

docker run -it --rm \
  -p 2379:2379 \
  -e ALLOW_NONE_AUTHENTICATION=yes \
  -e ETCD_LISTEN_CLIENT_URLS=http://0.0.0.0:2379 \
  -e ETCD_ADVERTISE_CLIENT_URLS=http://0.0.0.0:2379 \
  --name remote-etcd \
  bitnamilegacy/etcd:3.6.4-debian-12-r4
