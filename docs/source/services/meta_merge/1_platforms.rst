Hardware Requirements
=====================

CPU
---

Currently, we support two platforms:

- ARM64 (Nvidia Jetson, Raspberry Pi 4/5, AWS Graviton, etc);
- X86_64 (Intel/AMD CPUs).

The Meta Merge service is single-threaded for its core processing loop and benefits from fast single-core performance rather than many cores.

RAM
---

The Meta Merge service has modest memory requirements due to its efficient Rust implementation. We recommend having at least 512MB of RAM for basic operations. However, memory usage scales with:

- Number of concurrent ingress connections
- Number of unique sources being merged simultaneously
- Depth of the merge queue (frames waiting for metadata from all pipelines)
- Complexity of Python handlers and the state dictionaries they maintain
- Data payloads attached to frames

For high-throughput deployments processing many sources with deep merge queues, we recommend 2-4GB of RAM to ensure optimal performance.

Storage
-------

Meta Merge has minimal storage requirements as it operates as an in-memory message merging service without persistent data storage. Any standard storage medium (HDD, SSD, or even SD cards) is sufficient. The choice of storage medium does not impact Meta Merge's performance since it operates entirely in-memory.

Network
-------

Meta Merge is designed for high-throughput network operations and benefits from:

- Low-latency network connections between ingress sources and egress destinations
- Adequate network bandwidth to handle aggregate message throughput from all ingress streams
- Stable network connections to prevent message loss during merging

The service supports various ZeroMQ transport protocols (TCP and IPC) and can be configured to optimize for different network topologies and requirements.
