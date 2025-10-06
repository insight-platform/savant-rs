Installation and Building
=========================

The Buffer NG service is distributed as a Docker container, which simplifies its installation and usage. You can pull the image from the GitHub Container Registry.

Savant-RS Version
-----------------

The Buffer NG service is part of the Savant-RS project. The Savant-rs version used in Savant must match the version of the Buffer NG service.

How to check the savant-rs version used in Savant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can check the Savant-rs version used in Savant by running the following command:

.. code-block:: bash

   SAVANT_VERSION=0.5.15
   docker run -it ghcr.io/insight-platform/savant-adapters-gstreamer:$SAVANT_VERSION


   ============
   == Savant ==
   ============

   Savant Version 0.5.15
   Savant RS Version 1.10.1
   DeepStream Version 7.0


So, you can see that the version of savant-rs is 1.10.1. Thus, you need to use the same version of the Buffer NG service.

We support the following tags for the Buffer NG service:

* ``latest`` - the latest version (do not use it, we use it for development purposes);
* ``${SAVANT_RS}-rolling`` - the latest version of the Buffer NG service for the specific version of Savant-rs, e.g. ``1.10.1-rolling``;
* ``$SAVANT_VERSION`` - the specific version matching the version of Savant, e.g. ``0.5.15``;
* ``savant-latest`` - the latest version of the Buffer NG service for the `latest` version of Savant.

So, if you want to use the latest Savant, you probably need to use the ``${SAVANT_RS}-rolling`` tag for the Buffer NG service. If you want to use the release version of Savant, you can use the ``$SAVANT_VERSION`` tag.

Image naming
^^^^^^^^^^^^

* X86-64: `savant-buffer-ng-x86 <https://github.com/insight-platform/savant-rs/pkgs/container/savant-buffer-ng-x86>`_;
* ARM64: `savant-buffer-ng-arm64 <https://github.com/insight-platform/savant-rs/pkgs/container/savant-buffer-ng-arm64>`_.

Using The Docker Container
--------------------------

Pull the Docker image:

.. code-block:: bash

   docker pull ghcr.io/insight-platform/savant-buffer-ng-x86:savant-latest

   # or for arm64

   docker pull ghcr.io/insight-platform/savant-buffer-ng-arm64:savant-latest

Run the container with your configuration file:

.. code-block:: bash

   docker run \
      -v /path/to/your/config.json:/opt/etc/configuration.json \
      -v /path/to/your/python/handlers:/opt/python \
      -v /path/to/buffer/storage:/tmp/buffer \
      ghcr.io/insight-platform/savant-buffer-ng-x86:savant-latest

   # or for arm64

   docker run \
      -v /path/to/your/config.json:/opt/etc/configuration.json \
      -v /path/to/your/python/handlers:/opt/python \
      -v /path/to/buffer/storage:/tmp/buffer \
      ghcr.io/insight-platform/savant-buffer-ng-arm64:savant-latest

Here, you need to:

1. Mount your configuration file into the container at ``/opt/etc/configuration.json``
2. Mount your Python handler modules directory to make them accessible to the service
3. Mount a persistent volume for the buffer storage to ensure data persistence across container restarts
4. Ensure your configuration file references the correct paths for Python modules and buffer storage

The configuration file defines the ingress sources, egress destinations, buffer settings, and Python handler specifications.

.. note::

    Sometimes, you may need extra Python modules to be installed in the container. In this situation, you use the image as a base image and install the modules in the container in the Dockerfile.

Environment Variables
^^^^^^^^^^^^^^^^^^^^^^

The Buffer NG service supports environment variable substitution in configuration files. You can use the following pattern:

.. code-block:: bash

   docker run \
      -e ZMQ_SRC_ENDPOINT="tcp://source:5555" \
      -e ZMQ_SINK_ENDPOINT="tcp://sink:5556" \
      -e BUFFER_PATH="/tmp/buffer" \
      -e BUFFER_LEN=1000000 \
      -e BUFFER_THRESHOLD_PERCENTAGE=90 \
      -e STATS_LOG_INTERVAL=60 \
      -e PYTHON_MODULE_ROOT=/opt/python \
      -v /path/to/your/config.json:/opt/etc/configuration.json \
      -v /path/to/your/python/handlers:/opt/python \
      -v /path/to/buffer/storage:/tmp/buffer \
      ghcr.io/insight-platform/savant-buffer-ng-x86:savant-latest

Building From Source
--------------------

If you prefer to build the service from source, you can do so using Docker or Cargo:

1. Clone the Savant RS repository:

   .. code-block:: bash

      git clone https://github.com/insight-platform/savant-rs.git

2. Build the service with Docker:

   .. code-block:: bash

      docker build -t buffer-ng -f docker/services/Dockerfile.buffer_ng .

3. Build the service with Cargo:

   .. code-block:: bash

      cargo build --release -p buffer_ng

   .. note::

      Because of various reasons we use dynamic linking in savant-rs. Thus, you need to copy the dependencies to the distribution. Consult with our Docker-based build to find out how to do it.

Development Setup
-----------------

For development purposes, you can run the Buffer NG service directly from the source:

.. code-block:: bash

   LOGLEVEL=info cargo run -p buffer_ng services/buffer_ng/assets/configuration.json

This approach is useful when developing custom Python handlers or modifying the service configuration during development.

.. note::

   Make sure you have Python development dependencies installed and the correct Python version (3.8+) available in your development environment. Also ensure you have adequate disk space for the buffer storage and that the buffer directory is writable.

Dependencies
------------

The Buffer NG service has the following key dependencies:

- **RocksDB**: For persistent message storage and high-performance key-value operations
- **ZeroMQ**: For message transport and communication with upstream and downstream services
- **Python 3.10+**: For Python handler support and extensibility
- **Savant-RS Core**: For message handling and transport abstractions

These dependencies are automatically handled when using the Docker container or following the build instructions.
