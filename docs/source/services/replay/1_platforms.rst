Hardware Requirements
==========================

CPU
---

Currently, we support two platforms:

- ARM64 (Nvidia Jetson, Raspberry Pi 4/5, AWS Graviton, etc);
- X86_64 (Intel/AMD CPUs).

RAM
---

The system uses RocksDB as a storage engine, which benefit from having more RAM. We recommend having at least 4GB of RAM. However, you should be able to run Replay even with 512MB of RAM without problems.

Storage
-------

Replay uses RocksDB as a storage engine, which is designed for best operation on SSDs. However, it can work on HDDs as well. We recommend using SSD or HDD, but not SD cards due to a low IOPS and fast wear-out.

If you are using Replay to collect data and run long-running jobs, HDDs are fine. If you are using Replay for quick living, fast video retrievals and other low-latency real-time tasks, SSDs are recommended.
