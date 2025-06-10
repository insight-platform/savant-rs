Hardware Requirements
=====================

CPU
---

Currently, we support two platforms:

- ARM64 (Nvidia Jetson, Raspberry Pi 4/5, AWS Graviton, etc);
- X86_64 (Intel/AMD CPUs).

The Router service is optimized for multi-core processors and can efficiently utilize available CPU cores for message processing and routing operations.

RAM
---

The Router service has modest memory requirements due to its efficient Rust implementation. We recommend having at least 512MB of RAM for basic operations. However, memory usage scales with:

- Number of concurrent ingress and egress connections
- Complexity of Python handlers
- Message throughput and processing volume
- Number of in-flight messages configured for ingress and egress connections

For high-throughput deployments processing thousands of messages per second, we recommend 2-4GB of RAM to ensure optimal performance with adequate caching.

Storage
-------

Router has minimal storage requirements as it operates as a message routing service without persistent data storage. Any standard storage medium (HDD, SSD, or even SD cards) is sufficient for Router's storage needs. The choice of storage medium does not impact Router's performance since it operates in-memory.

Network
-------

Router is designed for high-throughput network operations and benefits from:

- Low-latency network connections between ingress sources and egress destinations
- Adequate network bandwidth to handle aggregate message throughput
- Stable network connections to prevent message loss during routing

The service supports various ZeroMQ transport protocols (TCP and IPC) and can be configured to optimize for different network topologies and requirements. 