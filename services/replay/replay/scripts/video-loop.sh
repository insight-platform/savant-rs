docker run -it --rm --network="host" \
  -v /tmp/video-loop-source-downloads:/tmp/video-loop-source-downloads \
  -e LOCATION=https://eu-central-1.linodeobjects.com/savant-data/demo/shuffle_dance.mp4 \
  -e DOWNLOAD_PATH=/tmp/video-loop-source-downloads \
  -e ZMQ_ENDPOINT=dealer+connect:tcp://127.0.0.1:5555 \
  -e SOURCE_ID=video \
  -e SYNC_OUTPUT=True \
  --entrypoint /opt/savant/adapters/gst/sources/video_loop.sh \
  ghcr.io/insight-platform/savant-adapters-gstreamer:latest
