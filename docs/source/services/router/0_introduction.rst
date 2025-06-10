Introduction
============

When developing complex computer vision and video analytics pipelines, developers often need to route video streams and their associated metadata to different processing endpoints based on dynamic conditions. Such operations may include conditional stream routing, load balancing, selective processing, multi-destination broadcasting, and intelligent stream splitting based on content analysis results.

Router is a high-performance solution for such complex routing problems. It is a Python-extendable message routing service that allows developers to process, modify, and route Savant messages based on configurable label matching conditions and custom Python logic. It can handle multiple ingress streams simultaneously and route them to multiple egress destinations with sophisticated matching criteria.

Developers can use Router for various tasks, such as:

- conditional stream routing based on detected objects or events;
- load balancing between multiple processing pipelines;
- selective stream processing and filtering;
- multi-destination message broadcasting;
- dynamic stream switching and circuit management;
- real-time message transformation and enrichment;
- intelligent stream splitting based on content analysis, and many others.

Let us discuss a couple of such use cases in more detail.

Use Cases
---------

Conditional Stream Routing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

With Router, you can route video streams to different processing pipelines based on real-time analysis results. For example, when a person detection pipeline identifies specific types of objects (vehicles, people, etc.), the Router can route those streams to specialized analysis pipelines while sending other streams to general processing endpoints. The system uses configurable boolean logic expressions to match message labels and route accordingly.

Load Balancing and Failover
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Router can distribute incoming streams across multiple processing instances to balance computational load across several GPUs. This ensures optimal resource utilization and prevents pipeline bottlenecks.

Multi-Destination Broadcasting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes you need to send the same stream to multiple processing endpoints simultaneously. Router supports broadcasting messages to multiple egress destinations based on matching criteria. For instance, a security monitoring stream might need to go to both real-time alert systems and long-term archival storage, with different processing requirements for each destination.

Intelligent Stream Switching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Router can implement complex switching logic based on message content, metadata, or external conditions. Using its Python extensibility, you can implement custom handlers that modify message routing based on time of day, detected content types, system load, or any other criteria. This enables dynamic pipeline reconfiguration without service restarts.

Python Extensibility
---------------------

One of Router's key strengths is its Python-based extensibility system. You can implement custom handlers for:

- **Ingress Handlers**: Process and modify incoming messages, add or remove labels, enrich metadata
- **Egress Source Mappers**: Dynamically modify source identifiers based on routing logic
- **Egress Topic Mappers**: Transform message topics for destination-specific requirements

These handlers are called at optimal points in the message processing pipeline to minimize performance impact while providing maximum flexibility for custom logic implementation.

Performance and Scalability
----------------------------

Router is built in Rust for high performance and includes several optimization features:

- Efficient label-based caching with configurable TTL and size limits
- High watermark monitoring for backpressure control
- Optimized message processing with minimal memory allocations
- Support for high-throughput streaming with configurable timeouts
- Efficient boolean expression evaluation for complex routing conditions

The service can handle thousands of messages per second while maintaining low latency and memory usage, making it suitable for production deployments with demanding performance requirements. 