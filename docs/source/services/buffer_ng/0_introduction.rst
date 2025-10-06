Introduction
============

When developing complex computer vision and video analytics pipelines, developers often need to ensure message reliability and prevent data loss during downstream service outages, network interruptions, or when processing capacity temporarily cannot keep up with incoming data rates. Such scenarios require robust buffering mechanisms that can persist messages to disk and provide reliable delivery guarantees.

Buffer NG is a high-performance solution for such reliability challenges. It is a Python-extendable message buffering service that allows developers to process, modify, and persistently store Savant messages on disk using RocksDB for optimal performance and durability. The service can handle high-throughput message streams while providing configurable capacity limits and comprehensive monitoring.

Developers can use Buffer NG for various tasks, such as:

- preventing message loss during downstream service outages or maintenance;
- handling burst traffic when downstream processing cannot keep up with incoming rates;
- implementing reliable message queuing with custom processing logic;
- providing message persistence for critical video analytics pipelines;
- enabling message replay capabilities for debugging and analysis;
- implementing custom message filtering and transformation with guaranteed delivery;
- buffering video streams during network interruptions or temporary capacity issues.

Let us discuss a couple of such use cases in more detail.

Use Cases
---------

Reliable Message Queuing
^^^^^^^^^^^^^^^^^^^^^^^^^

Buffer NG provides a persistent message queue that can store millions of messages on disk, ensuring no data loss even during unexpected service shutdowns. Messages are automatically serialized and stored using RocksDB, providing both durability and high-performance retrieval. The service can handle configurable buffer sizes and provides high watermark monitoring to alert when buffer utilization approaches capacity limits.

Downstream Outage Protection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When downstream services experience outages, maintenance windows, or temporary capacity issues, Buffer NG continues to accept and store incoming messages. Once downstream services recover, Buffer NG automatically resumes message delivery, ensuring continuous operation of the entire pipeline without data loss. This is particularly critical for real-time video analytics where missing frames or metadata could compromise analysis results.

Burst Traffic Handling
^^^^^^^^^^^^^^^^^^^^^^^

Buffer NG can handle traffic bursts by temporarily storing excess messages when downstream processing cannot keep up with incoming rates. The service monitors buffer utilization and can be configured to drop messages or apply backpressure when capacity limits are approached, allowing for graceful degradation under extreme load conditions.

Custom Message Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^

Buffer NG supports Python-based extensibility for custom message processing both before messages are stored (ingress handlers) and after they are retrieved from the buffer (egress handlers). This allows developers to implement custom filtering, transformation, enrichment, or validation logic while maintaining message reliability and persistence.


Python Extensibility
---------------------

One of Buffer NG's key strengths is its Python-based extensibility system. You can implement custom handlers for:

- **Ingress Handlers**: Process and modify incoming messages before they are stored in the buffer, add or remove labels, enrich metadata, or filter messages.
- **Egress Handlers**: Transform messages after they are retrieved from the buffer, modify routing information, or apply additional processing before delivery to downstream services.

These handlers are called at optimal points in the message processing pipeline to minimize performance impact while providing maximum flexibility for custom logic implementation.

Performance and Scalability
----------------------------

Buffer NG is built in Rust for high performance and includes several optimization features:

- Efficient RocksDB-based storage with configurable compaction and caching strategies.
- High watermark monitoring for buffer capacity management and backpressure control.
- Optimized message serialization with bitcode for minimal storage overhead.
- Support for high-throughput streaming with configurable timeouts and retry mechanisms.
- Comprehensive metrics collection for monitoring buffer utilization and message flow.
- Automatic buffer recovery and persistence across service restarts.

The service can handle thousands of messages per second while maintaining low latency and efficient disk usage, making it suitable for production deployments with demanding reliability and performance requirements.
