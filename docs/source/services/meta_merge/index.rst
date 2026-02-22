Meta Merge Service Documentation
==================================

Meta Merge is a Python-extendable service that merges metadata from multiple ingress streams into a single egress stream. When the same video frame is processed by several parallel pipelines (e.g. detection, classification, pose estimation), each pipeline attaches its own metadata and attributes to the frame. Meta Merge collects these partial results, merges them into a single consolidated frame, and forwards the result downstream.

This service is crucial for fan-in topologies in distributed Savant pipeline architectures:

- receives the same video frame from multiple ingress streams, each carrying different metadata;
- merges metadata from all ingress copies of a frame into a single frame using Python callbacks;
- maintains a per-source ordered queue so frames are forwarded in the correct order;
- supports configurable expiration so that frames are sent even if not all ingress copies arrive in time;
- handles late-arriving frames via a dedicated callback for custom recovery logic;
- provides EOS (end-of-stream) management with configurable allow/deny policies per ingress;
- offers a send callback that allows overriding the egress topic before the frame is written;
- handles unsupported message types through an optional Python callback.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   0_introduction
   1_platforms
   2_installation
   3_configuration
