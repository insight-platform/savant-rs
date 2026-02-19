Router Service Documentation
=============================

Router is a Python-extendable service that processes, modifies, and routes Savant messages based on their labels and conditions. With the router, you can process multiple ingress streams coming from multiple sockets and route them to multiple egress streams based on configurable matching criteria.

This service is crucial for complex streaming applications requiring conditional processing and routing of streams with many circuits:

- processes multiple ingress streams simultaneously from different sources;
- routes messages to multiple egress destinations based on label matching conditions;
- provides Python-based extensibility for custom message processing logic;
- supports configurable source and topic mapping for each egress endpoint;
- handles high-throughput message routing with efficient caching mechanisms;
- supports complex boolean logic for message routing decisions;
- can modify message labels, attributes, and metadata in real-time;
- provides backpressure control with configurable high watermarks;
- can work as a central routing hub in distributed Savant pipeline architectures.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   0_introduction
   1_platforms
   2_installation
   3_configuration
