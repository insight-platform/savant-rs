Installation and Building
=========================

The Retina RTSP service is distributed as a Docker container, which simplifies its installation and usage. You can pull the image from the GitHub Container Registry.

Savant-RS Version
-----------------

The Retina RTSP service is part of the Savant-RS project. The Savant-rs version used in Savant must match the version of the Retina RTSP service.

How to check the savant-rs version used in Savant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can check the Savant-rs version used in Savant by running the following command:

.. code-block:: bash

   SAVANT_VERSION=0.5.9
   docker run -it ghcr.io/insight-platform/savant-adapters-gstreamer:$SAVANT_VERSION


   ============
   == Savant ==
   ============

   Savant Version 0.5.10
   Savant RS Version 1.0.5
   DeepStream Version 7.0


So, you can see that the version of savant-rs is 1.0.5. Thus, you need to use the same version of the Retina RTSP service.

We support the following tags for the Retina RTSP service:

* ``latest`` - the latest version (do not use it, we use it for the development purposes);
* ``${SAVANT_RS}-rolling`` - the latest version of the Retina RTSP service for the specific version of Savant-rs, e.g. ``1.0.5-rolling``;
* ``$SAVANT_VERSION`` - the specific version matching the version of Savant, e.g. ``0.5.9``;
* ``savant-latest`` - the latest version of the Retina RTSP service for the `latest` version of Savant.

So, if you want to use the latest Savant, you probably need to use the ``${SAVANT_RS}-rolling`` tag for the Retina RTSP service. If you want to use the release version of Savant, you can use the ``$SAVANT_VERSION`` tag.


Image naming
^^^^^^^^^^^^

* X86-64: `savant-retina-rtsp-x86 <https://github.com/insight-platform/savant-rs/pkgs/container/savant-retina-rtsp-x86>`_;
* ARM64: `savant-retina-rtsp-arm64 <https://github.com/insight-platform/savant-rs/pkgs/container/savant-retina-rtsp-arm64>`_.


Using The Docker Container
--------------------------

Pull the Docker image:

.. code-block:: bash

   docker pull ghcr.io/insight-platform/savant-retina-rtsp-x86:savant-latest

   # or for arm64

   docker pull ghcr.io/insight-platform/savant-retina-rtsp-arm64:savant-latest

Run the container with your configuration file:

.. code-block:: bash

   docker run \
      -v /path/to/your/config.json:/opt/etc/configuration.json \
      ghcr.io/insight-platform/savant-retina-rtsp-x86:savant-latest

   # or for arm64

   docker run \
      -v /path/to/your/config.json:/opt/etc/configuration.json \
      ghcr.io/insight-platform/savant-retina-rtsp-arm64:savant-latest

Here, you need to mount your configuration file into the container and provide its path as an argument to the service. The configuration file defines the RTSP sources, synchronization options, and sink details.

Building From Source
--------------------

If you prefer to build the service from source, you can do so using Docker or Cargo (without Docker):

1. Clone the Savant RS repository:

   .. code-block:: bash

      git clone https://github.com/insight-platform/savant-rs.git

2. Build the service with Docker:

   .. code-block:: bash

      docker build -t retina-rtsp -f docker/services/Dockerfile.retina_rtsp .

3. Build the service with Cargo:

   .. code-block:: bash

      cargo build --release -p retina_rtsp


   .. note::

      Because of various reasons we use dynamic linking in savant-rs. Thus, you need to copy the dependencies to the distribution. Consult with out Docker-based build to find out how to do it.


