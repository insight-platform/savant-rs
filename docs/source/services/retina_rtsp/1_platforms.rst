Hardware Requirements
=====================

CPU
---

Currently, we support:

- ARM64 (Nvidia Jetson, Raspberry Pi);
- X86-64 (Intel/AMD CPUs).

We mostly build and test ARM64 support on Jetson Orin Nano and NX platforms. If you find any problems with Raspberry Pi or another ARM64 platform, please let us know.

RAM
---

The system needs a very small amount of RAM. We recommend having at least 1GB of RAM. However, you should be able to run Replay even with 512MB of RAM without problems. Depending on the number of streams, bitrates, and synchronization buffer size, you may need more RAM.

Storage
-------

The service does not require storage aside of the space required to store docker images and container layers. 2 GB of storage is enough.
