#!/bin/bash

curl http://127.0.0.1:8080/api/v1/job/$1 | json_pp