Installation
============

The Retina RTSP service is distributed as a Docker container, which simplifies its installation and usage. You can pull the image from the GitHub Container Registry.

Savant-RS version
-----------------

The Retina RTSP service is part of the Savant-RS project. The Savant-RS version used in Savant must match the version of the Retina RTSP service.

Using the Docker Container
--------------------------

Pull the Docker image:

.. code-block:: bash

   docker pull ghcr.io/insight-platform/savant-retina-rtsp-x86:latest

   # or for arm64

   docker pull ghcr.io/insight-platform/savant-retina-rtsp-arm64:latest

Run the container with your configuration file:

.. code-block:: bash

   docker run \
      -v /path/to/your/config.json:/opt/etc/configuration.json \
      ghcr.io/insight-platform/savant-retina-rtsp-x86:latest

   # or for arm64

   docker run \
      -v /path/to/your/config.json:/opt/etc/configuration.json \
      ghcr.io/insight-platform/savant-retina-rtsp-arm64:latest

Here, you need to mount your configuration file into the container and provide its path as an argument to the service. The configuration file defines the RTSP sources, synchronization options, and sink details.

Building from Source
--------------------

If you prefer to build the service from source, you can do so using Cargo:

1. Clone the Savant RS repository:

   .. code-block:: bash

      git clone https://github.com/insight-platform/savant-rs.git

2. Build the service:

   .. code-block:: bash

      docker build -t retina-rtsp -f docker/services/Dockerfile.retina_rtsp .


