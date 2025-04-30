Retina RTSP Service Documentation
=================================

Retina RTSP is a specialized service for handling RTSP video streams with advanced synchronization capabilities:

- connects to multiple RTSP sources and streams them to Savant sinks or modules;
- supports time-synchronized streaming from multiple sources;
- provides NTP and RTCP SR synchronization mechanisms;
- handles authentication for protected RTSP sources;
- automatically reconnects to sources in case of disconnection;
- can work as a sidecar or intermediary service in Savant pipelines.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   0_introduction
   1_platforms
   2_installation
   3_configuration
