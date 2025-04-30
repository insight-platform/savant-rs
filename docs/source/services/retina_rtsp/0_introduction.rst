Introduction
============

When developing computer vision and video analytics pipelines, synchronized handling of multiple RTSP video streams is often a challenge. Ensuring proper timing alignment between sources and providing reliable connections to various RTSP endpoints can be complex and error-prone without specialized tools.

Retina RTSP is a solution for handling RTSP streams in a reliable and synchronized manner. It provides a robust way to connect to multiple RTSP sources, synchronize their timelines, and stream the video data to Savant pipelines. This service is essential for applications that require precise timing coordination between multiple video sources.

Developers can use Retina RTSP for various tasks, such as:

- multi-camera synchronized video analytics;
- connecting to secured RTSP sources with authentication;
- ensuring reliable connections with automatic reconnection;
- accurate time synchronization between multiple sources;
- streaming synchronized video feeds to Savant pipelines.

Let us discuss a couple of use cases in more detail.

Use Cases
---------

Multi-Camera Synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With Retina RTSP, you can connect to multiple RTSP sources and synchronize their video streams based on NTP or RTCP SR timestamps. This is crucial for applications that need to analyze events across multiple cameras with precise timing correlations, such as multi-camera tracking or cross-camera event detection.

Secured RTSP Sources
^^^^^^^^^^^^^^^^^^^^

Many RTSP sources require authentication. Retina RTSP supports username and password authentication for protected sources, making it easier to integrate with secure video systems that implement access controls.

Reliable Streaming
^^^^^^^^^^^^^^^^^^

RTSP connections can sometimes be unstable. Retina RTSP implements automatic reconnection mechanisms, ensuring that your pipeline continues to receive video data even if the connection to an RTSP source temporarily fails. This is essential for systems that need high availability and resilience.

Time-Accurate Video Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For applications that require precise timing information, such as motion tracking or event correlation, Retina RTSP provides options for time synchronization. It can correct for network skew and ensure that video frames are delivered with accurate timing information. 