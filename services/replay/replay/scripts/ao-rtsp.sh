docker run --rm -it --name sink-always-on-rtsp \
    --gpus=all \
    --network="host" \
    -v ./assets/stub_imgs:/stub_imgs \
    -e ZMQ_ENDPOINT=sub+bind:tcp://127.0.0.1:6666 \
    -e SOURCE_ID=vod-video-1 \
    -e FRAMERATE=25/1 \
    -e STUB_FILE_LOCATION=/stub_imgs/smpte100_640x360.jpeg \
    -e DEV_MODE=True \
    ghcr.io/insight-platform/savant-adapters-deepstream:latest \
    python -m adapters.ds.sinks.always_on_rtsp

