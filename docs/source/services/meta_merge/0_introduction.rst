Introduction
============

When developing complex computer vision and video analytics pipelines, developers often need to split a video stream across multiple parallel processing pipelines (e.g. detection on one GPU, classification on another) and then merge the results back into a single enriched stream. Each parallel pipeline adds its own metadata, attributes, and objects to the video frame. The challenge is to collect all these partial results, merge them correctly, and forward the consolidated frame downstream in the correct order.

Meta Merge is a high-performance solution for such fan-in merging problems. It is a Python-extendable metadata merging service that receives the same video frame from multiple ingress streams, each carrying different metadata, and merges them into a single frame using developer-defined Python callbacks. The service maintains a per-source ordered merge queue that ensures frames are sent downstream in the correct temporal order.

Developers can use Meta Merge for various tasks, such as:

- merging detection and classification results from parallel pipelines;
- combining metadata from multiple inference engines running on separate GPUs;
- re-assembling a fan-out/fan-in topology after parallel processing stages;
- implementing custom merge logic that decides which attributes to keep, combine, or discard;
- providing deadline-based expiration so that frames are forwarded even when some pipelines are slow;
- handling late-arriving metadata copies with custom recovery or logging logic.

Let us discuss the architecture and use cases in more detail.

Architecture
------------

Meta Merge operates with the following components:

1. **Multiple Ingress Streams**: Each ingress socket receives messages from a separate upstream pipeline. Every pipeline sends video frames with its own metadata attached.

2. **Merge Queue**: For each source (identified by the source ID), the service maintains an ordered queue of frames keyed by UUID v7 (which encodes a timestamp). New frames are inserted into the queue; if a frame with the same UUID already exists, the merge callback is invoked.

3. **Python Callbacks**: Developers implement callback handlers that control the merge logic, expiration behavior, readiness decisions, and late-arrival handling. The key callbacks are:

   - ``on_merge``: Called when a new copy of a frame arrives. The handler merges metadata from the incoming frame into the current state and returns ``True`` when merging is complete.
   - ``on_head_ready``: Called when the head frame in the queue is marked as ready. Returns a ``Message`` to send or ``None`` to drop.
   - ``on_head_expire``: Called when the head frame expires (exceeds ``max_duration``). Returns a ``Message`` to send or ``None`` to drop.
   - ``on_late_arrival``: Called when a frame arrives too late (its UUID is older than the current queue head).
   - ``on_unsupported_message``: Optional callback for non-video, non-EOS messages.
   - ``on_send``: Optional callback that can override the egress topic before sending.

4. **Single Egress Stream**: Ready frames are sent to a single egress socket in temporal order.

Use Cases
---------

Fan-Out / Fan-In Pipeline Topology
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a common Savant deployment, a single video stream is duplicated and sent to multiple inference pipelines. Each pipeline (e.g. person detection, vehicle detection, license plate recognition) produces different objects and attributes on the same video frame. Meta Merge collects these partial results, invokes the ``on_merge`` callback for each arriving copy, and forwards the consolidated frame once all expected metadata has been merged.

Deadline-Based Merging
^^^^^^^^^^^^^^^^^^^^^^^

Sometimes not all parallel pipelines complete within the expected time. Meta Merge supports a configurable ``max_duration`` for each frame in the queue. If a frame is not marked as ready within this period, it is forcefully expired and the ``on_head_expire`` callback is invoked. This ensures the pipeline does not stall indefinitely, while the developer can decide whether to forward the partially merged frame or drop it.

Conditional Readiness
^^^^^^^^^^^^^^^^^^^^^^

The ``on_merge`` callback returns a boolean indicating whether the merge is complete. This allows developers to implement custom readiness logic: for example, waiting until metadata from exactly N ingress streams has been merged, or checking that specific attributes are present before forwarding the frame.

Late Arrival Handling
^^^^^^^^^^^^^^^^^^^^^^

When a frame copy arrives after the corresponding frame has already been forwarded, it is considered a late arrival. The ``on_late_arrival`` callback is invoked, giving the developer an opportunity to log the event, update external counters, or implement recovery strategies.

Python Extensibility
---------------------

Meta Merge provides six callback hooks for Python handlers:

- **on_merge(ingress_name, topic, current_state, incoming_state) → bool**: Merge incoming metadata into the current state. ``incoming_state`` is ``None`` for the first arrival (which automatically becomes the current state). Return ``True`` to mark the frame as ready for sending.

- **on_head_ready(state) → Optional[Message]**: Called when the head frame is marked ready. Return a ``Message`` to send downstream, or ``None`` to drop the frame.

- **on_head_expire(state) → Optional[Message]**: Called when the head frame has expired. Return a ``Message`` to send (possibly with partial metadata), or ``None`` to drop it.

- **on_late_arrival(state)**: Called when a frame arrives after the queue head has moved past it. Use this for logging or recovery.

- **on_unsupported_message(ingress_name, topic, message, data)**: Optional. Called for non-video, non-EOS messages (e.g. ``UserData``).

- **on_send(message, state, data, labels) → Optional[str]**: Optional. Called before sending a message to the egress. Return a topic string to override the default (source ID), or ``None`` to use the default.

These callbacks receive an ``EgressItem`` object that provides access to the video frame, a state dictionary (for accumulating merge state), data payloads, and labels.

Performance and Scalability
----------------------------

Meta Merge is built in Rust for high performance and includes several optimization features:

- Efficient per-source merge queues using BTreeMap for ordered UUID-based indexing.
- Configurable ``max_duration`` for deadline-based frame expiration.
- Non-blocking ZeroMQ I/O for high-throughput message handling.
- Python callback overhead is minimized by operating on references rather than copies where possible.
- The service can process thousands of frames per second while maintaining correct ordering and merge semantics.

The service is suitable for production deployments where multiple inference pipelines process the same stream in parallel and their results need to be combined before delivery to downstream consumers.
