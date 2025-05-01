Introduction
============

Retina RTSP is a solution for handling multiple RTSP streams within a single adapter in a reliable and optionally synchronized manner. It provides a robust way to connect to multiple RTSP sources, synchronize their timelines, and stream the video data to Savant pipelines. This service is essential for applications that require precise timing coordination between multiple video sources.

The service restarts failed RTSP connections automatically, so you do not need to worry about the health of the RTSP streams or rely on external tools to restart the streams. Retina RTSP is a pure Rust adapter that does not use GStreamer or FFmpeg libraries. It is based on `Retina <https://github.com/scottlamb/retina>`_ library by Scott Lamb.

We develop this adapter mostly to work with precise RTSP stream synchronization for multi-camera video analytics. Another options were to use GStreamer or patched FFmpeg. GStreamer has a very fragile RTSP implmentaion and using of patched FFmpeg is also looks difficult because it is a custom build of FFmpeg which is difficult to maintain and extend/fix.

Nevertheless, the FFmpeg-based RTSP `adapter <https://docs.savant-ai.io/develop/savant_101/10_adapters.html#rtsp-source-adapter>`_ is a first-class citizen and will be maintained and recommended for use in cases where precise synchronization is not required and the cams are diverse and include different brands and models, so you cannot guarantee that all the cams are supported by Retina RTSP.

Also, if RTSP streams contain B-frames, Retina RTSP is not an option since the underlying library does not support them. So, before using this adapter, please test that your cameras are normally processed. Nevertheless, we think that this adapter is a future replacement for RTSP processing in Savant pipelines when it comes to working with cameras.

Core Features
-------------

* RTSP Streams Synchronization with RTCP SR protocol messages;
* Multiple RTSP streams handling, which decreases the number of moving parts in the solution;
* Automatic RTSP reconnection;
* Pure-Rust implementation without GStreamer or FFmpeg dependencies;
* Convenient JSON-based configuration.






