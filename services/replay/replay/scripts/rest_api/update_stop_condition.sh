#!/bin/bash

curl --header "Content-Type: application/json" -X PATCH \
     --data '{"frame_count": 10000}' \
     http://127.0.0.1:8080/api/v1/job/$1/stop-condition | json_pp