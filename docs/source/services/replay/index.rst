Replay Service Documentation
============================

Replay is an advanced storage providing features required for non-linear computer vision and video analytics:

- collects video from multiple streams (archiving with TTL eviction);
- provides a REST API for video re-streaming to Savant sinks or modules;
- supports time-synchronized and fast video re-streaming;
- supports configurable video re-streaming stop conditions;
- supports setting minimum and maximum frame duration to increase or decrease the video playback speed;
- can fix incorrect TS in re-streaming video streams;
- can look backward when video stream re-streamed;
- can set additional attributes to retrieved video streams;
- can work as a sidecar or intermediary service in Savant pipelines.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   0_introduction
   1_platforms
   2_installation
   3_jobs
   4_api