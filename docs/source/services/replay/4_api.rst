REST API
============

Status
------

Returns the status of the server.

.. code-block:: bash

    curl -X GET http://127.0.0.1:8080/api/v1/status

.. list-table::
    :header-rows: 1

    * - Parameter
      - Description
      - Extra
      - Description
    * - Method
      - GET
      - /api/v1/status
      -
    * - Response OK (Running)
      - 200 OK
      - ``"running"``
      - service is functional and can be used
    * - Response OK (Finished)
      - 200 OK
      - ``"finished"``
      - service is functional but cannot be used anymore
    * - Response Error
      - 500 Internal Server Error
      - ``{"error" => "Reason"}``
      - service is not functional

Endpoint: GET /api/v1/status

Shutdown
--------

Stops the server.

.. code-block:: bash

    curl -X POST http://127.0.0.1:8080/api/v1/shutdown

.. list-table::
    :header-rows: 1

    * - Parameter
      - Description
      - Extra
      - Description
    * - Method
      - POST
      - /api/v1/shutdown
      -
    * - Response OK
      - 200 OK
      - ``"ok"``
      - service is shutting down

Find Keyframes
--------------

Finds keyframes in a video stream. Returns found keyframes from oldest to newest.

.. code-block:: bash

    curl --header "Content-Type: application/json" -X POST \
         --data '{"source_id": "in-video", "from": null, "to": null, "limit": 1}' \
         http://127.0.0.1:8080/api/v1/keyframes/find | json_pp

.. list-table::
    :header-rows: 1

    * - Parameter
      - Description
      - Extra
      - Description
    * - Method
      - POST
      - /api/v1/keyframes/find
      -
    * - Request
      - JSON
      - ``{...}``
      - see below
    * - ``source_id``
      - string
      - ``"in-video"``
      - source identifier
    * - ``from``
      - int
      - ``null``
      - start time in seconds (Unix time), optional
    * - ``to``
      - int
      - ``null``
      - end time in seconds (Unix time), optional
    * - ``limit``
      - int
      - ``1``
      - maximum number of keyframes UUIDs to return. Must be a positive integer.
    * - Response OK
      - 200 OK
      - ``{"keyframes" : ["in-video", ["018f76e3-a0b9-7f67-8f76-ab0402fda78e", ...]]}``
      - list of keyframes UUIDs
    * - Response Error
      - 500 Internal Server Error
      - ``{"error" => "Reason"}``
      - problem description

Get Keyframe by UUID
--------------------

Retrieves a specific keyframe by UUID and returns a multipart response containing JSON metadata and binary data parts.

.. code-block:: bash

    curl "http://127.0.0.1:8080/api/v1/keyframe/{uuid}?source_id=in-video" \
      -o keyframe.multipart

.. list-table::
    :header-rows: 1

    * - Parameter
      - Description
      - Extra
      - Description
    * - Method
      - GET
      - /api/v1/keyframe/{uuid}
      - returns multipart/mixed response
    * - ``uuid`` (path)
      - string
      - required
      - Keyframe UUID (obtained from ``find_keyframes``)
    * - ``source_id`` (query)
      - string
      - required
      - video source identifier
    * - Response OK
      - 200 OK
      - multipart/mixed
      - metadata + data parts
    * - Not Found
      - 404 Not Found
      - JSON
      - UUID not found for this source
    * - Invalid Request
      - 400 Bad Request
      - JSON
      - invalid UUID format or empty source id
    * - Internal Error
      - 500 Internal Server Error
      - JSON
      - database or decoding issue

**Response Format (200 OK)**

The response uses ``multipart/mixed`` format with the following parts in order:

1. **Metadata part** (``Content-Type: application/json``): JSON-serialized video frame metadata
2. **Data parts** (``Content-Type: application/octet-stream``): Zero or more binary data blobs (encoded frame content, external references, etc.)

Each data part includes an ``index`` attribute in the Content-Disposition header indicating its position (0, 1, 2, ...). The semantic meaning of each index is application-specific and depends on the upstream producer.

Example response structure:

.. code-block:: text

    --savant-frame-<uuid>
    Content-Type: application/json
    Content-Disposition: inline; name="metadata"
    Content-Length: <length>

    {"uuid": "...", "source_id": "...", "pts": ..., ...}
    --savant-frame-<uuid>
    Content-Type: application/octet-stream
    Content-Disposition: inline; name="data"; index="0"
    Content-Length: <length>

    <binary data>
    --savant-frame-<uuid>--

Create New Job
--------------

Creates a new job. Returns the job UUID.

.. note::
   If default sink options are configured in the service configuration, you can omit the ``options`` field in the ``sink`` object. The default options will be applied automatically. This is useful when you have multiple jobs with similar sink configurations.

.. code-block:: bash

    #!/bin/bash

    query() {

    ANCHOR_KEYFRAME=$1

    cat <<EOF
    {
      "sink": {
        "url": "pub+connect:tcp://127.0.0.1:6666",
        "options": {
          "send_timeout": {
            "secs": 1,
            "nanos": 0
          },
          "send_retries": 3,
          "receive_timeout": {
            "secs": 1,
            "nanos": 0
          },
          "receive_retries": 3,
          "send_hwm": 1000,
          "receive_hwm": 1000,
          "inflight_ops": 100
        }
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
      "stop_condition": {
        "frame_count": 10000
      },
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
    curl -X PUT -H "Content-Type: application/json" -d "$Q" http://127.0.0.1:8080/api/v1/job | json_pp

Example with Default Sink Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you've configured default sink options in the service configuration, you can simplify your job request by omitting the options field:

.. code-block:: json

    {
      "sink": {
        "url": "pub+connect:tcp://127.0.0.1:6666"
      },
      "configuration": {
        // configuration fields
      },
      "stop_condition": {
        "frame_count": 10000
      },
      "anchor_keyframe": "018f76e3-a0b9-7f67-8f76-ab0402fda78e",
      "anchor_wait_duration": {
        "secs": 1,
        "nanos": 0
      },
      "offset": {
        "blocks": 5
      },
      "attributes": [
        // attributes
      ]
    }

Augmenting Attributes
^^^^^^^^^^^^^^^^^^^^^

Attributes are defined in JSON format matching savant-rs `Attribute` struct. For details, please take a look at the `Attribute` struct in the `savant-rs <https://insight-platform.github.io/savant-rs/modules/savant_rs/primitives.html#savant_rs.primitives.Attribute>`_ documentation and the relevant `sample <https://github.com/insight-platform/savant-rs/blob/main/python/primitives/attribute.py>`_.

Attributes, passed to the job, automatically ingested in every frame metadata to provide the stream receiver with extra knowledge about the job. For example, you can pass the track ID for the object you want to handle additionally.

Job Labels
^^^^^^^^^^

These labels are used for the user need. When you have a lot of concurrent jobs you may want to associate some metadata with them.

.. code-block:: javascript

    "labels": {
      "key": "value"
    }

When you request the information about the running or stopped jobs, you can effectively distinguish them based on them.

Offset
^^^^^^

Offset defines the starting point of the job. It is required to shift back in time from the anchor keyframe. The offset can be defined in two ways:

- number of fully-decodable blocks;
- number of seconds.

Number of Blocks
~~~~~~~~~~~~~~~~

.. code-block:: javascript

    {
      "blocks": <int>
    }

Rewinds to the specified number of blocks (keyframes) before the anchor keyframe.

Number of Seconds
~~~~~~~~~~~~~~~~~

.. code-block:: javascript

    {
      "seconds": <float>
    }

Rewinds to the specified number of seconds before the anchor keyframe but always starts from the keyframe.

Job Stop Condition JSON Body
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Last Frame
~~~~~~~~~~

.. code-block:: javascript

    {
      "last_frame": {
        "uuid": <UUID>,
      }
    }

When the next frame UUID is "larger" than the specified, the job will stop. Because the system uses strictly increasing UUIDv7 for frame UUIDs, you can construct a UUIDv7 with the desired timestamp to match the timestamp.

Frame Count
~~~~~~~~~~~

.. code-block:: javascript

    {
      "frame_count": <COUNT>
    }

The job will stop when the specified number of frames is processed.

Keyframe Count
~~~~~~~~~~~~~~

.. code-block:: javascript

    {
      "key_frame_count": <COUNT>
    }

The job will stop when the specified number of keyframes is processed.

Timestamp Delta
~~~~~~~~~~~~~~~

.. code-block:: javascript

    {
      "ts_delta_sec": {
        "max_delta_sec": <float, seconds> // 1.0
      }
    }

The job will stop when the encoded timestamp delta between the last frame and the current frame is larger than the specified value.

Realtime Delta
~~~~~~~~~~~~~~

.. code-block:: javascript

    {
      "real_time_delta_ms": {
        "configured_delta_ms": <int, milliseconds> // 1000
      }
    }

The job will stop when the job live time is larger than the specified value.

Now
~~~

.. code-block:: javascript

    "now"

The job will stop immediately.

Never
~~~~~

.. code-block:: javascript

    "never"

The job will never stop.

Time-synchronized And Fast Jobs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With Replay, you can re-stream with different speed and time synchronization. The system can handle the following cases:

- as-fast-as-possible re-streaming (in most cases it is limited by a receiver);
- time-synchronized re-streaming (sends according to encoded PTS/DTS labels and time corrections);

.. note::

    Regardless of the mode, the system never changes encoded PTS and DTS labels, Replay just re-streams regulating frame delivery.

The mode is defined by the following parameter:

.. code-block:: javascript

    "ts_sync": true

When ``ts_sync`` is set to ``true``, the system will re-stream the video in time-synchronized mode. The system will deliver frames according to the encoded timestamps.

When ``ts_sync`` is set to ``false``, the system will re-stream the video as fast as possible.

Egress FPS Control And Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replay allows setting minimum, maximum, and ts discrepancy fix duration for the job. The system will deliver frames according to the specified durations. These parameters only work in time-synchronized mode.

- **min duration**: prevents the system from delivering frames faster than the specified duration; even if frame encoded timestamps (PTS/DTS) are closer, the system will wait for the specified duration before delivering the next frame;

- **max duration**: prevents the system from delivering frames slower than the specified duration; even if frame encoded timestamps (PTS/DTS) are further, the system will deliver the next frame after the specified duration;

- **ts discrepancy fix duration**: when the system detects a non-monotonic discrepancy between the encoded timestamps, it will correct the discrepancy by delivering the frame according to the specified duration.

Routing Labels
^^^^^^^^^^^^^^

Routing labels is a mechanism allowing mark Savant protocol packets with extra tags, helping to route them in the processing graph. Those labels are not supported by modules or sinks but can be used by a custom routing nodes, implemented by users with Savant `ClientSDK <https://docs.savant-ai.io/develop/advanced_topics/10_client_sdk.html>`__.

Replay supports three policies for routing labels:

Bypass
~~~~~~

.. code-block:: javascript

    "routing_labels": "bypass"

The system will not add any routing labels to the packets.

Replace
~~~~~~~

.. code-block:: javascript

    "routing_labels": { "replace": ["label1", "label2"] }

The system will replace all routing labels with the specified ones.

Append
~~~~~~

.. code-block:: javascript

    "routing_labels": { "append": ["label1", "label2"] }

The system will append the specified labels to the existing ones.

All Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Parameters
    :header-rows: 1

    * - Parameter
      - Description
      - Default
      - Example

    * - ``sink``
      - sink configuration
      - ``{...}``
      - ``{...}``
    * - ``sink.url``
      - sink URL
      - ``pub+connect:tcp://127.0.0.1:6666``
      - ``"dealer+connect:tcp://1.1.1.1:6666"``
    * - ``sink.options``
      - sink options
      - ``null``
      - ``{...}``
    * - ``sink.options.send_timeout``
      - send timeout
      - ``{"secs": 1, "nanos": 0}``
      - ``{"secs": 5, "nanos": 0}``
    * - ``sink.options.send_retries``
      - send retries
      - ``3``
      - ``5``
    * - ``sink.options.receive_timeout``
      - receive timeout, used by ``req/rep`` for every message delivery, ``dealer/router`` for EOS delivery
      - ``{"secs": 1, "nanos": 0}``
      - ``{"secs": 5, "nanos": 0}``
    * - ``sink.options.receive_retries``
      - receive retries, used by ``req/rep`` for every message delivery, ``dealer/router`` for EOS delivery
      - ``3``
      - ``5``
    * - ``sink.options.send_hwm``
      - The high-water mark for the egress stream. This parameter is used to control backpressure. Please consult with 0MQ documentation for more details.
      - ``1000``
      - ``500``
    * - ``sink.options.receive_hwm``
      - The high-water mark for the egress stream. This parameter is used to control backpressure. Please consult with 0MQ documentation for more details. Change only if you are using ``req/rep`` communication.
      - ``100``
      - ``50``
    * - ``sink.options.inflight_ops``
      - The maximum number of inflight operations for the egress stream. This parameter is used to allow the service to endure a high load. Default value is OK for most cases.
      - ``100``
      - ``50``
    * - ``configuration``
      - job configuration
      - ``{...}``
      - ``{...}``
    * - ``configuration.ts_sync``
      - time synchronization mode, when ``true`` the system will deliver frames according to the encoded timestamps
      - ``true``
      - ``false``
    * - ``configuration.skip_intermediary_eos``
      - when ``true`` the system will not deliver EOS encountered in the stream
      - ``false``
      - ``true``
    * - ``configuration.send_eos``
      - when ``true`` the system will deliver EOS at the end of the job
      - ``true``
      - ``false``
    * - ``configuration.stop_on_incorrect_ts``
      - when ``true`` the system will stop the job when it detects incorrect timestamps (next is less than the previous), only valid when ``ts_sync`` is ``true``
      - ``false``
      - ``true``
    * - ``configuration.ts_discrepancy_fix_duration``
      - when the system detects a non-monotonic discrepancy between the encoded timestamps, it will correct the discrepancy by delivering the frame according to the specified duration, only valid when ``ts_sync`` is ``true``.
      - ``{"secs": 0, "nanos": 33333333}``
      - ``{"secs": 0, "nanos": 100000000}``
    * - ``configuration.min_duration``
      - prevents the system from delivering frames faster than the specified duration, only valid when ``ts_sync`` is ``true``.
      - ``{"secs": 0, "nanos": 10000000}``
      - ``{"secs": 0, "nanos": 5000000}``
    * - ``configuration.max_duration``
      - prevents the system from delivering frames slower than the specified duration, only valid when ``ts_sync`` is ``true``.
      - ``{"secs": 0, "nanos": 103333333}``
      - ``{"secs": 0, "nanos": 5000000}``
    * - ``configuration.stored_stream_id``
      - stream which is used to re-stream from
      - ``"in-video"``
      - ``"in-video"``
    * - ``configuration.resulting_stream_id``
      - re-streamed stream identifier
      - ``"vod-video-1"``
      - ``"vod-video-2"``
    * - ``configuration.routing_labels``
      - routing labels, used to mark Savant protocol packets with extra tags; see the `Routing Labels`_ section for more details
      - ``"bypass"``
      - ``"bypass"``
    * - ``configuration.max_idle_duration``
      - the job will stop when it does not receive frames from the storage for the specified duration
      - ``{"secs": 10, "nanos": 0}``
      - ``{"secs": 5, "nanos": 0}``
    * - ``configuration.max_delivery_duration``
      - the job will stop when it cannot deliver a frame to the sink for the specified duration
      - ``{"secs": 10, "nanos": 0}``
      - ``{"secs": 5, "nanos": 0}``
    * - ``configuration.send_metadata_only``
      - when ``true`` the system will deliver only metadata frames, without the actual video data
      - ``false``
      - ``true``
    * - ``configuration.labels``
      - job labels, used to mark the job with extra tags; see the `Job Labels`_ section for more details
      - ``{"namespace": "key"}``
      - ``{"namespace": "key"}``
    * - ``stop_condition``
      - job stop condition; see the `Job Stop Condition JSON Body`_ section for more details
      - ``{...}``
      - ``{...}``
    * - ``anchor_keyframe``
      - anchor keyframe UUID
      - ``"018f76e3-a0b9-7f67-8f76-ab0402fda78e"``
      - ``"018f76e3-a0b9-7f67-8f76-ab0402fda78e"``
    * - ``anchor_wait_duration``
      - defines how long the job waits for late anchor keyframes that are not yet arrived to the system
      - ``null``
      - ``{"secs": 1, "nanos": 0}``
    * - ``offset``
      - job offset; see the `Offset`_ section for more details
      - ``{...}``
      - ``{...}``
    * - ``attributes``
      - job attributes; see the `Augmenting Attributes`_ section for more details
      - ``{...}``
      - ``{...}``


List Job
--------

List the running job matching the given UUID.

.. code-block:: bash

    JOB_UUID=<JOB_UUID> curl http://127.0.0.1:8080/api/v1/job/$JOB_UUID | json_pp

.. list-table::
    :header-rows: 1

    * - Parameter
      - Description
      - Extra
      - Description
    * - Method
      - GET
      - ``/api/v1/job/<jobid>``
      -
    * - Response OK
      - 200 OK
      - ``{...}``
      - see JSON response below
    * - Response Error
      - 500 Internal Server Error
      - ``{"error" => "Reason"}``
      - problem description

JSON Body:

.. code-block:: javascript

    {
       "jobs" : [
          [
             <jobid>,
             { /* job configuration */ },
             { /* job stop condition */}
          ], ...
       ]
    }


List Jobs
---------

List all running jobs.

.. code-block:: bash

    curl http://127.0.0.1:8080/api/v1/job | json_pp

.. list-table::
    :header-rows: 1

    * - Parameter
      - Description
      - Extra
      - Description
    * - Method
      - GET
      - ``/api/v1/job``
      -
    * - Response OK
      - 200 OK
      - ``{...}``
      - see JSON response below
    * - Response Error
      - 500 Internal Server Error
      - ``{"error" => "Reason"}``
      - problem description

JSON Body:

.. code-block:: javascript

    {
       "jobs" : [
          [
             <jobid>,
             { /* job configuration */ },
             { /* job stop condition */}
          ], ...
       ]
    }


List Stopped Jobs
-----------------

List all stopped but not yet evicted jobs.

.. code-block:: bash

    curl http://127.0.0.1:8080/api/v1/job/stopped | json_pp

.. list-table::
    :header-rows: 1

    * - Parameter
      - Description
      - Extra
      - Description
    * - Method
      - GET
      - ``/api/v1/job/stopped``
      -
    * - Response OK
      - 200 OK
      - ``{...}``
      - see JSON response below
    * - Response Error
      - 500 Internal Server Error
      - ``{"error" => "Reason"}``
      - problem description


200 OK JSON Body:

.. code-block:: javascript

    {
       "stopped_jobs" : [
          [
             <jobid>,
             { /* job configuration */ },
             null | "When error, termination reason"
          ], ...
       ]
    }


Delete Job
----------

Forcefully deletes the running job matching the given UUID.

.. code-block:: bash

    JOB_UUID=<JOB_UUID> curl -X DELETE http://127.0.0.1:8080/api/v1/job/$JOB_UUID | json_pp

.. list-table::
    :header-rows: 1

    * - Parameter
      - Description
      - Extra
      - Description
    * - Method
      - DELETE
      - ``/api/v1/job/<jobid>``
      -
    * - Response OK
      - 200 OK
      - ``"ok"``
      - job was deleted
    * - Response Error
      - 500 Internal Server Error
      - ``{"error" => "Reason"}``
      - problem description


Update Job Stop Condition
-------------------------

Updates the stop condition of the running job matching the given UUID.

.. code-block:: bash

    JOB_UUID=<JOB_UUID> curl \
         --header "Content-Type: application/json" -X PATCH \
         --data '{"frame_count": 10000}' \
         http://127.0.0.1:8080/api/v1/job/$JOB_UUID/stop-condition | json_pp

.. list-table::
    :header-rows: 1

    * - Parameter
      - Description
      - Extra
      - Description
    * - Method
      - PATCH
      - ``/api/v1/job/<jobid>/stop-condition``
      -
    * - Request
      - JSON
      - ``{...}``
      - see the `Job Stop Condition JSON Body`_ section in the `Create New Job`_ section
    * - Response OK
      - 200 OK
      - ``"ok"``
      - stop condition was updated
    * - Response Error
      - 500 Internal Server Error
      - ``{"error" => "Reason"}``
      - problem description
