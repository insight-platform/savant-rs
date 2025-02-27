# File Re-Streaming Sample

This sample shows how to ingest video file to Replay and then re-stream it to AO-RTSP with REST API.

## Run Replay

Run Replay with the following command (X86 version):

```bash
docker pull ghcr.io/insight-platform/replay-x86:main

docker run -it --rm \
  --network host \
  -v $(pwd)/replay_config.json:/opt/etc/config.json \
  -v $(pwd)/data:/opt/rocksdb \
  ghcr.io/insight-platform/replay-x86:main
```

## Launch AO-RTSP to View

First, we just launch to have it displaying stub because no actual re-streaming happens.

```bash
docker pull ghcr.io/insight-platform/savant-adapters-deepstream:latest

docker run --rm -it --name sink-always-on-rtsp \
    --gpus=all \
    --network="host" \
    -v ./assets/stub_imgs:/stub_imgs \
    -e ZMQ_ENDPOINT=sub+bind:tcp://127.0.0.1:6666 \
    -e SOURCE_ID=vod-video-1 \
    -e FRAMERATE=25/1 \
    -e STUB_FILE_LOCATION=/stub_imgs/smpte100_1280x720.jpeg \
    -e DEV_MODE=True \
    ghcr.io/insight-platform/savant-adapters-deepstream:latest \
    python -m adapters.ds.sinks.always_on_rtsp
```

Open the following URL in your browser to view the stub image: http://127.0.0.1:888/stream/vod-video-1/

or with FFplay:

```bash
ffplay rtsp://127.0.0.1:554/stream/vod-video-1
```

## Download Sample Video

```bash
wget https://eu-central-1.linodeobjects.com/savant-data/demo/shuffle_dance.mp4
```

## Ingest Video File to Replay

We are going to use the file source adapter to ingest the video file to Replay.

```bash
docker run --rm -it --name source-video-files-test \
    --network host \
    -e FILE_TYPE=video \
    -e SYNC_OUTPUT=False \
    -e ZMQ_ENDPOINT=dealer+connect:tcp://127.0.0.1:5555 \
    -e SOURCE_ID=in-video \
    -e LOCATION=/data/shuffle_dance.mp4 \
    -v $(pwd)/shuffle_dance.mp4:/data/shuffle_dance.mp4:ro \
    --entrypoint /opt/savant/adapters/gst/sources/media_files.sh \
    ghcr.io/insight-platform/savant-adapters-gstreamer:latest
```

## Lookup For The First Key Frame

Let us find the first key frame of the video stored in Replay.

```bash
bash ../../replay/scripts/rest_api/find_keyframes.sh
```

In my case the reported UUID for the first keyframe is:

```bash
{
   "keyframes" : [
      "in-video",
      [
         "018f76e3-a0b9-7f67-8f76-ab0402fda78e"
      ]
   ]
}
```

### Initialize Re-Streaming Job

The job will be started from the first key frame and continue until the stop condition or timeout is met.

```bash
bash ../../replay/scripts/rest_api/new_job.sh 018f76e3-a0b9-7f67-8f76-ab0402fda78e
```

The command will return JobID:

```bash
{
   "new_job" : "018f76f0-b351-7aac-b67f-5c526b54da22"
}
```

### Visit AO-RTSP to View Video Playback

Open the following URL in your browser to view the stub image: http://127.0.0.1:888/stream/vod-video-1/

or with FFplay:

```bash
ffplay rtsp://127.0.0.1:554/stream/vod-video-1
```
