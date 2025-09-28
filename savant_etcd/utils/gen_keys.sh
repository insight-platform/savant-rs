#!/bin/bash -e

DIR=$(dirname $0)/../assets/certs

# CA
openssl genpkey -algorithm RSA -out $DIR/ca.key
openssl req -new -x509 -days 365000 -key $DIR/ca.key -out $DIR/ca.crt -subj "/CN=local-etcd"

# SERVER
openssl genpkey -algorithm RSA -out $DIR/server.key
openssl req -new -key $DIR/server.key -out $DIR/server.csr -subj "/CN=localhost"
openssl x509 -req -days 365000 -in $DIR/server.csr -CA $DIR/ca.crt -CAkey $DIR/ca.key -CAcreateserial -out $DIR/server.crt -extfile <(echo "subjectAltName=IP:127.0.0.1")

# CLIENT
openssl genpkey -algorithm RSA -out $DIR/client.key
openssl req -new -key $DIR/client.key -out $DIR/client.csr -subj "/CN=localhost"
openssl x509 -req -days 365000 -in $DIR/client.csr -CA $DIR/ca.crt -CAkey $DIR/ca.key -CAcreateserial -out $DIR/client.crt -extfile <(echo "subjectAltName=IP:127.0.0.1")

chmod 0666 $DIR/*