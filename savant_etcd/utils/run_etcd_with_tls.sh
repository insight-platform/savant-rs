#!/bin/bash

ROOT_PASSWD=$1

if [ -z $ROOT_PASSWD ]; then
  docker run -it --rm \
    -p 2379:2379 \
    -e ALLOW_NONE_AUTHENTICATION=yes \
    -e ETCD_TRUSTED_CA_FILE=/etc/etcd-ssl/ca.crt \
    -e ETCD_CERT_FILE=/etc/etcd-ssl/server.crt \
    -e ETCD_KEY_FILE=/etc/etcd-ssl/server.key \
    -e ETCD_LISTEN_CLIENT_URLS=https://0.0.0.0:2379 \
    -e ETCD_ADVERTISE_CLIENT_URLS=https://0.0.0.0:2379 \
    -v $(pwd)/../assets/certs:/etc/etcd-ssl \
    --name remote-etcd \
    bitnami/etcd:latest
else
  docker run -it --rm \
    -p 2379:2379 \
    -e ALLOW_NONE_AUTHENTICATION=no \
    -e ETCD_ROOT_PASSWORD=$ROOT_PASSWD \
    -e ETCD_TRUSTED_CA_FILE=/etc/etcd-ssl/ca.crt \
    -e ETCD_CERT_FILE=/etc/etcd-ssl/server.crt \
    -e ETCD_KEY_FILE=/etc/etcd-ssl/server.key \
    -e ETCD_LISTEN_CLIENT_URLS=https://0.0.0.0:2379 \
    -e ETCD_ADVERTISE_CLIENT_URLS=https://0.0.0.0:2379 \
    -v $(pwd)/../assets/certs:/etc/etcd-ssl \
    --name remote-etcd \
    bitnami/etcd:latest
fi

