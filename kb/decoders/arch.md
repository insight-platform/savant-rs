# DeepStream Decoders Architecture

## Module tree

```
savant_deepstream/decoders/src/
├── lib.rs
├── config.rs
├── error.rs
├── pipeline.rs
├── prelude.rs
└── stream_detect.rs
```

## High-level design

`NvDecoder` is built on top of the `savant_gstreamer::pipeline` framework and exposes a pull API (`recv`/`recv_timeout`/`try_recv`) rather than callback-based delivery.

Three backend strategies are selected from `DecoderConfig`:

- `Pipeline` for H264/HEVC/VP8/VP9/AV1 and GPU JPEG
- `ImageDecode` for PNG and CPU JPEG (`image` crate decode + upload)
- `RawUpload` for `RawRgb`/`RawRgba` packets

All paths produce `DecodedFrame` whose `buffer` field is `Option<SharedBuffer>`,
allowing ownership handoff (`take`) in downstream processing without moving the
entire frame struct.

## Pipeline backend topology

Typical decode chain:

- `appsrc -> parser -> nvv4l2decoder -> appsink` (video codecs)
- `appsrc -> jpegparse -> nvjpegdec -> appsink` (GPU JPEG)

Framework settings used by decoder pipeline:

- strict ordering policy (`PtsPolicy::StrictDecodeOrder`)
- bounded input/output channels
- operation timeout watchdog
- drain polling interval

## Metadata and correlation model

Pipeline backends use monotonic PTS plus an internal map to recover user metadata (`frame_id`, DTS, duration) on output.

- Primary correlation: PTS map (`pts -> FrameMetadata`)
- Fallback for intra-only cases: FIFO keyed by submit order

`ImageDecode` and `RawUpload` bypass this and emit metadata directly from submit arguments.

## Output buffer strategy

- Output buffers are always allocated from the decoder's target pool (`BufferGenerator`)
- If decoded dimensions differ, decoder uses an auxiliary pool and `NvBufSurfTransform` into destination pool
- Final caller-visible format is RGBA

## Stream packaging detection

`stream_detect.rs` provides:

- `detect_stream_config(codec, data)` for H264/HEVC Annex-B vs AVCC/HVCC detection
- `is_random_access_point(codec, data)` for strict one-AU entrypoint checks

For H264/HEVC, RAP requires in-band parameter sets plus IDR/IRAP in the same access unit.
