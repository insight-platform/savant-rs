Hardware Requirements
=====================

CPU
---

Currently, we support two platforms:

- ARM64 (Nvidia Jetson, Raspberry Pi 4/5, AWS Graviton, etc);
- X86_64 (Intel/AMD CPUs).

The Buffer NG service is optimized for multi-core processors and can efficiently utilize available CPU cores for message processing, buffering operations, and RocksDB storage management.

RAM
---

The Buffer NG service has moderate memory requirements due to its efficient Rust implementation and RocksDB storage engine. We recommend having at least 1GB of RAM for basic operations. However, memory usage scales with:

- Buffer size and message volume.
- RocksDB cache configuration and compaction strategies.
- Complexity of Python handlers.
- Message throughput and processing volume.
- Number of in-flight messages configured for ingress and egress connections.

For high-throughput deployments processing thousands of messages per second with large buffer capacities, we recommend 4-8GB of RAM to ensure optimal performance with adequate caching and buffer management.

Storage
-------

Buffer NG requires persistent storage for the RocksDB database that stores buffered messages. The storage requirements depend on:

- Buffer capacity configuration (max_length parameter).
- Message size and complexity.
- Retention policies and buffer cleanup strategies.
- RocksDB compaction and compression settings.

We recommend using SSD storage for optimal performance, especially for high-throughput deployments. The storage medium directly impacts Buffer NG's performance since it performs frequent read/write operations for message persistence.

For production deployments, ensure adequate disk space for:
- The configured buffer capacity.
- RocksDB overhead and compaction temporary files.
- Log files and telemetry data.
- Python handler modules and dependencies.

Network
-------

Buffer NG is designed for high-throughput network operations and benefits from:

- Low-latency network connections between ingress sources and egress destinations.
- Adequate network bandwidth to handle aggregate message throughput.
- Stable network connections to prevent message loss during buffering operations.

The service supports various ZeroMQ transport protocols (TCP and IPC) and can be configured to optimize for different network topologies and requirements. Network performance directly impacts the service's ability to maintain high throughput while providing reliable message buffering.

Disk I/O Performance
--------------------

Since Buffer NG relies heavily on RocksDB for persistent storage, disk I/O performance is crucial for optimal operation:

- Use SSD storage for better random read/write performance.
- Consider NVMe SSDs for high-throughput deployments.
- Monitor disk I/O metrics to identify potential bottlenecks.
- Configure appropriate RocksDB options for your storage characteristics.
- Ensure adequate disk space for buffer growth and RocksDB compaction operations.

The service includes built-in monitoring for disk usage and buffer capacity to help identify storage-related issues before they impact performance.
