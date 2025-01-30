Introduction
============

When developing computer vision and video analytics pipelines developers often meet challenging problems requiring non-linear stream processing. Such operations may include video rewinds, on-demand re-streaming, footage retrieval, archiving, etc. These problems are hard to solve without specialized video collection systems optimized for such tasks.

Replay is a solution for such complex problems. It is a high-performance video storage system that allows developers to store, lookup and re-stream video in a non-linear API-driven fashion. It is optimized for high-throughput video storage and retrieval, and it is capable of handling multiple video streams simultaneously. It stores and re-streams not only video data but also metadata associated with the video stream.

Developers can use Replay for various tasks, such as:

- long-term video archive;
- selective video processing;
- particular video fragment processing;
- video playback in real time, with increased or decreased speed;
- video footage saving, and many others.

Let us discuss a couple of such use cases in more detail.

Use Cases
---------

Video Archive
^^^^^^^^^^^^^

With Replay, you can simultaneously process and save video streams to long-term storage. All the discovered metadata are stored together with associated video packets, and in the future, you can re-stream the required parts for additional processing or playback. The system allows configuring TTL, which automatically clears old data.

Selective Video Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes, you do not want to process all the data because it is inefficient, but when the particular event is triggered (e.g., a person crosses the line), you want to process the video around the event for deeper analytics.

Replay allows you to store the video stream and process it later when needed. In this particular use case, there are two pipelines: one for basic analytics and another for advanced processing. The first lightweight pipeline processes the video stream in real-time and looks for key events. When such an event is detected, the pipeline instructs Replay to re-stream the video surrounding the event to the advanced pipeline for complex processing.

Video Footage Saving
^^^^^^^^^^^^^^^^^^^^

Savant has adapters that allow video footage to be saved to the disk. This is useful when you need to store the video stream for further analysis or legal purposes. With Replay, when the pipeline detects an event, it can instruct Replay to send the video surrounding it to the video file sink.

