Configuration
=============

The Retina RTSP service is configured using a JSON file that specifies RTSP sources, synchronization options, and sink details. This section describes the configuration structure and options.

Configuration File
------------------

The configuration file is a JSON document with the following structure:

.. code-block:: json

    {
        "sink": {
            "url": "pub+connect:tcp://127.0.0.1:3333",
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
        "rtsp_sources": {
            "auth_group": {
                "sources": [
                    {
                        "source_id": "auth-source",
                        "url": "rtsp://hello.savant.video:8554/stream/auth-source",
                        "options": {
                            "username": "${MY_USERNAME:-admin}",
                            "password": "${MY_PASSWORD:-123456}"
                        }
                    }
                ]
            },
            "rtcp_sr_sync_group": {
                "sources": [
                    {
                        "source_id": "city-traffic",
                        "url": "rtsp://hello.savant.video:8554/stream/city-traffic"
                    },
                    {
                        "source_id": "town-centre",
                        "url": "rtsp://hello.savant.video:8554/stream/town-centre"
                    }
                ],
                "rtcp_sr_sync": {
                    "group_window_duration": {
                        "secs": 5,
                        "nanos": 0
                    },
                    "batch_duration": {
                        "secs": 0,
                        "nanos": 100000000
                    },
                    "network_skew_correction": false,
                    "rtcp_once": false
                }
            }
        },
        "reconnect_interval": {
            "secs": 5,
            "nanos": 0
        },
        "eos_on_restart": true
    }


Environment Variable Substitution
---------------------------------

The configuration file supports variable substitution using environment variables.  Use the following syntax to substitute a variable:

.. code-block:: bash

    ${MY_ENV_VAR} or ${MY_ENV_VAR:-the_default_value}       


Sink Configuration
------------------

The sink configuration is used to specify the sink to which the video data will be sent. All Savant sinks are supported.

RTSP Sources Configuration
--------------------------

The RTSP sources configuration is used to specify the RTSP sources to be used by the service. The sources are organized into groups. Each group can use RTPC SR synchronization or do not use it. If you want serve independent streams, you do not need to specify the synchronization configuration.

RTSP Source Stream Position
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``stream_position`` option is used to specify the stream position of the RTSP source. The default value is ``0``. In most cases, you do not need to specify this option.

RTSP Source Options
^^^^^^^^^^^^^^^^^^^

The RTSP source options are used to specify the options for the RTSP sources. The following options are supported:

* ``username`` - the username for the RTSP source
* ``password`` - the password for the RTSP source

.. code-block:: json

    {
        "source_id": "city-traffic",
        "url": "rtsp://hello.savant.video:8554/stream/city-traffic",
        "options": {
            "username": "${MY_USERNAME:-admin}",
            "password": "${MY_PASSWORD:-123456}"
        }
    }


RTCP SR Sync Configuration
--------------------------

When the synchronization is enabled, the service will use the RTCP SR packets to synchronize the streams. The synchronization is done on the group level, streams across different groups are not synchronized.

When the synchronization is enabled, every Savant VideoFrame contains extra attributes:

* ("retina-rtsp", "batch-id") - the batch id of the frame
* ("retina-rtsp", "batch-group-name") - the group name of the frame
* ("retina-rtsp", "batch-sources") - the list of sources in the batch
* ("retina-rtsp", "ntp-timestamp-ns") - the NTP timestamp of the frame in nanoseconds

RTCP SR Sync Configuration introduces delay in frame delivery due to the need to synchronize the streams. The ``group_window_duration`` is the time window in seconds that the service will try to synchronize the streams. The ``batch_duration`` is the time window in seconds that the service will collect the frames before sending them to the sink.

Every batch contains only one frame from each source.

.. note::

    Even when streams are synchronized, it is not guaranteed that frames will be delivered in order. Every strams delivers frames independently. You must use the ``batch-id`` attribute to track the order of the frames and the ``batch-sources`` attribute to track the completeness of the batch.


Network Skew Correction
^^^^^^^^^^^^^^^^^^^^^^^

The ``network_skew_correction`` flag enables the network skew correction. The default value is ``false``. This flag can be used to correct the synchronization when NTP is not configured properly, but the network is stable and the delay is constant. In this case, the service will use the RTCP SR packets to estimate the delay and correct the synchronization.

.. note::

    We have not tested this feature properly, so use it with caution.

RTCP Once Option
^^^^^^^^^^^^^^^^

The ``rtcp_once`` flag enables the RTCP SR synchronization only once. The default value is ``false``. If the flag is set to ``true``, the service will use the RTCP SR packets to synchronize the streams only once and then it will use the NTP timestamp of the first frame to synchronize the streams. This will be helpful if cameras demonstrate RTCP SR drifts due to incorrect or buggy NTP configuration.

Reconnect Interval
------------------

The ``reconnect_interval`` option is used to specify the interval in seconds between the attempts to reconnect to the RTSP sources.

EOS on Restart
--------------

The ``eos_on_restart`` flag enables the EOS on restart. The default value is ``true``. If the flag is set to ``false``, the service will not send the EOS message to the sink when the stream is restarted.

This parameter is beneficial to reset the remote stream decoder state when the stream is restarted. We found that sometimes NVDEC stucks in the weird state and requires the EOS message to be sent to reset the state. We recommend you to set this parameter to ``true`` in most cases.
