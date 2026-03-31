# DeepStream Decoders Architecture

## Module Tree

```
deepstream_decoders/src/
├── lib.rs
├── config.rs
├── decoder.rs
├── error.rs
└── prelude.rs
```

## Backends

`NvDecoder` chooses one of three internal backends:

- `Pipeline`: GStreamer decode + appsink drain thread
- `RawUpload`: direct CPU pixel upload into GPU buffers
- `ImageDecode`: CPU decode via Rust `image` crate, then GPU upload/transform

## GStreamer Pipeline Variants

- V4L2 codecs:
  `appsrc -> parser -> nvv4l2decoder -> appsink`
- JPEG GPU backend:
  `appsrc -> jpegparse -> nvjpegdec -> appsink`

## Non-Pipeline Decode Paths

- JPEG CPU backend: decoded with Rust `image` crate (`load_from_memory`)
- PNG: decoded with Rust `image` crate (`load_from_memory`)
- Raw RGB/RGBA: uploaded directly from packet bytes

Decoded pixels are uploaded to NVMM and transformed into the caller pool
using `TransformConfig` when resolution differs.

## Frame-ID Propagation

Two mechanisms are used together:

1. `bridge_savant_id_meta(&decoder_element)` preserves `SavantIdMeta` across
   decoder elements that allocate new output buffers.
2. Internal PTS mapping:
   `HashMap<pts_ns, (frame_id, dts_ns, duration_ns)>` plus FIFO fallback for
   intra-only codecs.

For `ImageDecode` and `RawUpload`, frame metadata comes directly from
`submit_packet` inputs (no PTS-map lookup path involved).

## Output Buffer Model

- Decoder output buffers are always acquired from caller-provided
  `BufferGenerator`.
- If source and pool sizes match, upload can be direct.
- If sizes differ, an auxiliary temporary pool is used, then
  `NvBufSurfTransform` applies scaling/padding/conversion into final output.

## Error Recovery

- `submit_packet` checks bus errors before push for pipeline backend.
- On error, decoder pipeline is torn down and rebuilt.
- State reset clears PTS map, FIFO, and monotonic PTS tracking.
- Caller receives:
  - `DecoderEvent::PipelineRestarted { reason, lost_frame_count }` on restart
  - `DecoderEvent::Error(DecoderError::PipelineError(...))` on spawn/restart failures
