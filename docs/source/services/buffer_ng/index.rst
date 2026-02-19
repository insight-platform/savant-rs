Buffer NG Service Documentation
===============================

Buffer NG is a Python-extendable service that can process/modify and buffer messages on disk to prevent their loss when downstream cannot keep up or experience outage. This service is crucial for complex streaming applications working under high load.

This service provides reliable message buffering with the following capabilities:

- persistent disk-based message storage using RocksDB for durability and performance;
- Python-based extensibility for custom message processing logic before buffering and after retrieval;
- configurable buffer capacity with high watermark monitoring for backpressure control;
- efficient message serialization and deserialization for optimal storage and retrieval;
- comprehensive metrics and monitoring for buffer utilization and message flow;
- automatic buffer recovery and persistence across service restarts;
- supports high-throughput message processing with configurable timeouts and retries;
- can handle thousands of messages per second while maintaining low latency;
- provides web-based telemetry interface for monitoring buffer status and performance;
- supports both ingress and egress Python handlers for message transformation;
- can work as a reliable message queue in distributed Savant pipeline architectures.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   0_introduction
   1_platforms
   2_installation
   3_configuration
