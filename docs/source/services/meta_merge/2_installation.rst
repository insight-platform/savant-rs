Installation and Building
=========================

The Meta Merge service is distributed as a Docker container, which simplifies its installation and usage. You can pull the image from the GitHub Container Registry.

Savant-RS Version
-----------------

The Meta Merge service is part of the Savant-RS project. The Savant-rs version used in Savant must match the version of the Meta Merge service.

How to check the savant-rs version used in Savant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can check the Savant-rs version used in Savant by running the following command:

.. code-block:: bash

   SAVANT_VERSION=0.5.10
   docker run -it ghcr.io/insight-platform/savant-adapters-gstreamer:$SAVANT_VERSION


   ============
   == Savant ==
   ============

   Savant Version 0.5.10
   Savant RS Version 1.4.0
   DeepStream Version 7.0


So, you can see that the version of savant-rs is 1.4.0. Thus, you need to use the same version of the Meta Merge service.

We support the following tags for the Meta Merge service:

* ``latest`` - the latest version (do not use it, we use it for development purposes);
* ``${SAVANT_RS}-rolling`` - the latest version of the Meta Merge service for the specific version of Savant-rs, e.g. ``1.4.0-rolling``;
* ``$SAVANT_VERSION`` - the specific version matching the version of Savant, e.g. ``0.5.10``;
* ``savant-latest`` - the latest version of the Meta Merge service for the `latest` version of Savant.

So, if you want to use the latest Savant, you probably need to use the ``${SAVANT_RS}-rolling`` tag for the Meta Merge service. If you want to use the release version of Savant, you can use the ``$SAVANT_VERSION`` tag.

Image naming
^^^^^^^^^^^^

* X86-64: `savant-meta-merge-x86 <https://github.com/insight-platform/savant-rs/pkgs/container/savant-meta-merge-x86>`_;
* ARM64: `savant-meta-merge-arm64 <https://github.com/insight-platform/savant-rs/pkgs/container/savant-meta-merge-arm64>`_.

Using The Docker Container
--------------------------

Pull the Docker image:

.. code-block:: bash

   docker pull ghcr.io/insight-platform/savant-meta-merge-x86:savant-latest

   # or for arm64

   docker pull ghcr.io/insight-platform/savant-meta-merge-arm64:savant-latest

Run the container with your configuration file:

.. code-block:: bash

   docker run \
      -v /path/to/your/config.json:/opt/etc/configuration.json \
      -v /path/to/your/python/handlers:/opt/python \
      ghcr.io/insight-platform/savant-meta-merge-x86:savant-latest

   # or for arm64

   docker run \
      -v /path/to/your/config.json:/opt/etc/configuration.json \
      -v /path/to/your/python/handlers:/opt/python \
      ghcr.io/insight-platform/savant-meta-merge-arm64:savant-latest

Here, you need to:

1. Mount your configuration file into the container at ``/opt/etc/configuration.json``
2. Mount your Python handler modules directory to make them accessible to the service at ``/opt/python``
3. Ensure your configuration file references the correct paths for Python modules

The configuration file defines the ingress sources, egress destination, merge callbacks, and queue settings.

.. note::

    Sometimes, you may need extra Python modules to be installed in the container. In this situation, you use the image as a base image and install the modules in the container in the Dockerfile.


Environment Variables
^^^^^^^^^^^^^^^^^^^^^^

The Meta Merge service supports environment variable substitution in configuration files. You can use the following pattern:

.. code-block:: bash

   docker run \
      -e PYTHON_MODULE_ROOT=/opt/python \
      -e HOME=/root \
      -e USER=root \
      -v /path/to/your/config.json:/opt/etc/configuration.json \
      -v /path/to/your/python/handlers:/opt/python \
      ghcr.io/insight-platform/savant-meta-merge-x86:savant-latest

Building From Source
--------------------

If you prefer to build the service from source, you can do so using Docker or Cargo:

1. Clone the Savant RS repository:

   .. code-block:: bash

      git clone https://github.com/insight-platform/savant-rs.git

2. Build the service with Docker:

   .. code-block:: bash

      docker build -t meta_merge -f docker/services/Dockerfile.meta_merge .

3. Build the service with Cargo:

   .. code-block:: bash

      cargo build --release -p meta_merge

   .. note::

      Because of various reasons we use dynamic linking in savant-rs. Thus, you need to copy the dependencies to the distribution. Consult with our Docker-based build to find out how to do it.

Development Setup
-----------------

For development purposes, you can run the Meta Merge service directly from the source:

.. code-block:: bash

   LOGLEVEL=info cargo run -p meta_merge services/meta_merge/assets/configuration.json

This approach is useful when developing custom Python handlers or modifying the service configuration during development.

.. note::

   Make sure you have Python development dependencies installed and the correct Python version (3.8+) available in your development environment.
