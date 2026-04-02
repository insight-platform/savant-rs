# DeepStream Decoders Architecture

## Module Tree

```
deepstream_decoders/src/
├── lib.rs
├── config.rs
├── decoder.rs
├── error.rs
├── prelude.rs
└── stream_detect.rs
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

## Stream Format Detection (`stream_detect.rs`)

`detect_stream_config(codec, data)` inspects the prefix of one access unit
(packet) to determine H264/HEVC stream packaging:

- **Annex-B**: detected when the buffer starts with `00 00 01` (3-byte) or
  `00 00 00 01` (4-byte start code). Returns `ByteStream` with no `codec_data`.
- **Length-prefixed (AVCC/HVCC)**: buffer is treated as 4-byte big-endian
  length-prefixed NAL units. SPS/PPS (H264) or VPS/SPS/PPS (HEVC) are
  collected from the NALUs, and `codec_data` is built:
  - H264 → `AVCDecoderConfigurationRecord` (ISO 14496-15), format `Avc`
  - HEVC → `HEVCDecoderConfigurationRecord` (ISO 14496-15), format `Hvc1`
- Returns `None` for unsupported codecs, empty data, malformed prefixes, or
  missing parameter sets.

NAL unit parsing uses `cros-codecs` (`cros_codecs::codec::h264::parser::Nalu`,
`cros_codecs::codec::h265::parser::Nalu`) for typed NALU header inspection.
HEVC SPS parsing for profile/tier/level extraction also uses
`cros_codecs::codec::h265::parser::Parser`.

### Random access points (`is_random_access_point`)

Single-AU predicate: whether the buffer contains everything needed to start
decoding **without** relying on prior packets (in-band parameter sets + IDR/IRAP).

- H264: SPS + PPS + IDR in the same AU.
- HEVC: VPS + SPS + PPS + IRAP VCL (`is_irap()` on the NAL type).

This is stricter than “keyframe only”: a lone IDR without SPS/PPS returns
`false`. For MP4, many packets are keyframes in the demuxer sense but only
those that still carry in-band VPS/SPS/PPS + IRAP satisfy this helper.

### `config-interval` Property

`h264parse` and `h265parse` GStreamer elements are configured with
`config-interval=1` in both:
- `NvDecoder::build_pipeline()` in `decoder.rs`
- `Mp4Demuxer::build_parser_chain()` in `savant_gstreamer/mp4_demuxer.rs`

This forces SPS/PPS/VPS re-injection into every IDR frame, ensuring
downstream elements always have up-to-date parameter sets.

## Error Recovery

- `submit_packet` checks bus errors before push for pipeline backend.
- On error, decoder pipeline is torn down and rebuilt.
- State reset clears PTS map, FIFO, and monotonic PTS tracking.
- Caller receives:
  - `DecoderEvent::PipelineRestarted { reason, lost_frame_count }` on restart
  - `DecoderEvent::Error(DecoderError::PipelineError(...))` on spawn/restart failures
