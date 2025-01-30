Re-Streaming Jobs
=================

The re-streaming jobs implement dynamic activities (sources in the Savant terminology) transferring data from the Replay storage to pipelines and sinks. Jobs are spawned by user on demand with Restful API.

Every job consists of:

- **sink**: the destination of the stream;
- **anchor frame**: the frame that starts the job;
- **offset**: the shift back in the history from the anchor frame;
- **configuration**: the specification used by job manager to deliver the data according to the user needs;
- **stop condition**: the condition that stops the job, stop conditions can be updated through API during the job execution, allowing job initiators to terminate the job earlier or make it run longer;
- **attributes**: extra Savant attributes injected into every metadata record.

Sink
----

A sink is a ZeroMQ socket, which is always a `connect`-type socket and can be ``pub``, ``req`` or ``dealer``. Other socket types are not supported. The socket is ``connect`` because it is expected to communicate with pre-existing services.

Jobs can communicate with any properly-defined Savant nodes, accepting data from Savant upstream nodes.

Anchor Frame
------------

In Savant, every frame has strictly increasing UUID, constructed as UUIDv7. UUIDv7 is based on millisecond-precision timestamp allowing to know the exact time of the frame creation and the order of the frames. The UUID is generated when the frame is first encountered, and most processing nodes reuse its UUID without modification.

Replay uses frame UUIDs to navigate the video stream. UUIDs are used to find keyframes, create jobs and update job stop conditions. Every frame has encoded information about the `previous keyframe <https://insight-platform.github.io/savant-rs/modules/savant_rs/primitives.html#savant_rs.primitives.VideoFrame.previous_keyframe_uuid>`__ UUID, thus allowing to navigate the stream to a frame which guarantees the correct decoding of the frame sequence.

**Anchor Frame** is a keyframe UUID which is used as a job starting point. It can be corrected with the offset parameter to look backward, which is often required to start processing a little earlier than the moment of interest.

.. warning::

    MJPEG and image streams have every frame marked as a keyframe. Thus streams create significantly more indexing information in Replay and thus can require more resources to process.

In certain situations users may encounter cases when frames are not yet delivered to Replay at the moment of the job creation. In this case, the job may wait for the frame to be delivered and then start processing the stream. There is a special optional parameter in the job specification for this case.

Job Offset
----------

The offset helps rewind the stream backward from the anchor frame, allowing to start the job earlier than the anchor frame. The offset can be defined in two ways:

- number of fully-decodable blocks;
- number of seconds (float).

The offset guarantees that the job will start from the keyframe matching the criteria.

Configuration
-------------

Configuration defines how job sends the data to the sink. It includes several parameters like:

- real-time-synchronization;
- processing intermediary EOS service messages;
- processing final EOS message;
- behavior on incorrect encoded frame timestamp (PTS or DTS);
- how to fix the incorrect frame timestamp;
- minimal and maximal frame duration;
- mapping of stored source ID to the consumer expected source ID;
- setting routing labels;
- max idle duration (when the job does not accept frames from the storage);
- max delivery duration (when the job cannot deliver frames to the sink);
- send metadata only;
- user labels.

Stop Condition
--------------

The stop condition is a condition that stops the job. Jobs support several different stop conditions, but only one can be specified for a job. The stop condition can be updated during the job execution, allowing job initiators to terminate the job earlier or make it run longer.

The following stop conditions are supported:

- immediate stop;
- never stop;
- particular frame UUID (timestamp);
- frame count;
- keyframe count;
- encoded timestamp delta between the first and the last frame;
- real-time duration (when jobs works longer than the specified number of milliseconds).

When the job is stopped by reaching the specified stop condition, it is considered as successfully completed, otherwise it is considered as failed.

Users can update job stop conditions for running jobs with REST API.

Attributes
----------

Attributes are extra Savant attributes injected into every metadata record. They can be used to add extra information to the metadata, which can be used by the sink or other services.

Job Concurrency
---------------

As many jobs can run concurrently and send data to the same sinks, you need to carefully map stored source IDs to the consumer expected source IDs. This is required to avoid data corruption and to ensure that the data is delivered to the sink in the correct order.

.. warning::

    There is no way to limit jobs concurrency: developers must implement it separately. For Replay, every job is completely independent.

.. note::

    To avoid concurrent jobs you can poll the current job status with the REST API. If the job is stopped you can launch a new job.

Jobs are very lightweight, thus you can have dozens or even hundreds of jobs running concurrently. However, you need to ensure that the sinks can handle the load. Also, every job reads data from the storage, so the storage must be able to handle the load.

Job Persistence
---------------

Currently, jobs are not persistent. When the service is reloaded, all the running jobs are lost. Users must implement job persistence separately.
