#!/bin/bash

curl --header "Content-Type: application/json" -X POST \
     --data '{"source_id": "in-video", "from": null, "to": null, "limit": 1}' \
     http://127.0.0.1:8080/api/v1/keyframes/find | json_pp