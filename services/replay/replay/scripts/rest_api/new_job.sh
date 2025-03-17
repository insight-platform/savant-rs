#!/bin/bash

query() {

ANCHOR_KEYFRAME=$1

cat <<EOF
{
  "sink": {
    "url": "pub+connect:tcp://127.0.0.1:6666",
    "options": null 
  },
  "configuration": {
    "ts_sync": true,
    "skip_intermediary_eos": false,
    "send_eos": true,
    "stop_on_incorrect_ts": false,
    "ts_discrepancy_fix_duration": {
      "secs": 0,
      "nanos": 33333333
    },
    "min_duration": {
      "secs": 0,
      "nanos": 10000000
    },
    "max_duration": {
      "secs": 0,
      "nanos": 103333333
    },
    "stored_stream_id": "in-video",
    "resulting_stream_id": "vod-video-1",
    "routing_labels": "bypass",
    "max_idle_duration": {
      "secs": 10,
      "nanos": 0
    },
    "max_delivery_duration": {
      "secs": 10,
      "nanos": 0
    },
    "send_metadata_only": false,
    "labels": {
        "namespace": "key"
    }
  },
  "stop_condition": "never",
  "anchor_keyframe": "$ANCHOR_KEYFRAME",
  "anchor_wait_duration": {
    "secs": 1,
    "nanos": 0
  },
  "offset": {
    "blocks": 5
  },
  "attributes": [
    {
      "namespace": "key",
      "name": "value",
      "values": [
        {
          "confidence": 0.5,
          "value": {
            "Integer": 1
          }
        },
        {
          "confidence": null,
          "value": {
            "FloatVector": [
              1.0,
              2.0,
              3.0
            ]
          }
        }
      ],
      "hint": null,
      "is_persistent": true,
      "is_hidden": false
    }
  ]
}
EOF

}

Q=$(query $1)
curl -X PUT -H "Content-Type: application/json" -d "$Q" http://127.0.0.1:8080/api/v1/job