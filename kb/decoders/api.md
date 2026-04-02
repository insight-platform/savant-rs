# DeepStream Decoders Public API

Crate: `deepstream_decoders`  
Prelude: `use deepstream_decoders::prelude::*;`

## Core Types

- `DecoderConfig` (per-codec enum)
- `NvDecoder` (event/callback-driven decoder)
- `DecodedFrame` (decoded buffer + metadata)
- `DecoderEvent` (Frame/Eos/Error/PipelineRestarted)
- `DecoderError` (typed error enum)

## Stream Format Enums

- `H264StreamFormat`: `ByteStream`, `Avc`, `Avc3`
- `HevcStreamFormat`: `ByteStream`, `Hvc1`, `Hev1`
- `JpegBackend`: `Gpu`, `Cpu`

## DecoderConfig Variants

- `H264(H264DecoderConfig)`
- `Hevc(HevcDecoderConfig)`
- `Vp8(Vp8DecoderConfig)`
- `Vp9(Vp9DecoderConfig)`
- `Av1(Av1DecoderConfig)`
- `Jpeg(JpegDecoderConfig)` (`Gpu` default, or `Cpu`)
- `Png(PngDecoderConfig)`
- `RawRgba(RawRgbaDecoderConfig)`
- `RawRgb(RawRgbDecoderConfig)`

SIG: `DecoderConfig::codec(&self) -> Codec`

## NvDecoder

SIG:
`new<F>(gpu_id: u32, config: &DecoderConfig, pool: BufferGenerator, transform_config: TransformConfig, on_event: F) -> Result<Self, DecoderError> where F: FnMut(DecoderEvent) + Send + 'static`

SIG:
`codec(&self) -> Codec`

SIG:
`submit_packet(&mut self, data: &[u8], frame_id: u128, pts_ns: u64, dts_ns: Option<u64>, duration_ns: Option<u64>) -> Result<(), DecoderError>`

SIG:
`send_eos(&mut self) -> Result<(), DecoderError>`

### Input Contract

- `pts_ns` must be strictly monotonic across `submit_packet` calls.
- Non-monotonic or equal `pts_ns` returns `DecoderError::PtsReordered`.
- After `send_eos`, subsequent `submit_packet` returns `DecoderError::AlreadyFinalized`.

## DecoderEvent

```rust
pub enum DecoderEvent {
    Frame(DecodedFrame),
    Eos,
    Error(DecoderError),
    PipelineRestarted {
        reason: String,
        lost_frame_count: usize,
    },
}
```

## DecodedFrame

```rust
pub struct DecodedFrame {
    pub frame_id: Option<u128>,
    pub pts_ns: u64,
    pub dts_ns: Option<u64>,
    pub duration_ns: Option<u64>,
    pub buffer: SharedBuffer,
    pub codec: Codec,
    pub format: VideoFormat, // always RGBA for decoder output
}
```

## Stream Format Detection

SIG:
`detect_stream_config(codec: Codec, data: &[u8]) -> Option<DecoderConfig>`

Inspects one access unit (packet) to determine H264/HEVC stream packaging.
Returns `Some(DecoderConfig)` with stream format and optional `codec_data`,
or `None` when:
- `codec` is not `Codec::H264` or `Codec::Hevc`
- `data` is empty or prefix is neither Annex-B nor valid length-prefixed
- length-prefixed H264 without both SPS and PPS
- length-prefixed HEVC without VPS, SPS, and PPS, or SPS parsing fails

### Annex-B Detection

When `data` starts with `00 00 01` or `00 00 00 01`:
- H264 → `DecoderConfig::H264(H264DecoderConfig::new(ByteStream))`
- HEVC → `DecoderConfig::Hevc(HevcDecoderConfig::new(ByteStream))`
- No `codec_data` is set.

### Length-Prefixed (AVCC/HVCC) Detection

When `data` does not start with an Annex-B start code, it is parsed as
4-byte big-endian length-prefixed NAL units:
- H264: collects SPS + PPS → builds `AVCDecoderConfigurationRecord` →
  `DecoderConfig::H264(H264DecoderConfig::new(Avc).codec_data(record))`
- HEVC: collects VPS + SPS + PPS → builds `HEVCDecoderConfigurationRecord` →
  `DecoderConfig::Hevc(HevcDecoderConfig::new(Hvc1).codec_data(record))`

### Usage

```rust
use deepstream_decoders::{detect_stream_config, DecoderConfig};
use savant_gstreamer::Codec;

let config = detect_stream_config(Codec::H264, &packet_data);
match config {
    Some(DecoderConfig::H264(c)) => { /* use c.stream_format, c.codec_data */ }
    Some(DecoderConfig::Hevc(c)) => { /* use c.stream_format, c.codec_data */ }
    None => { /* unsupported or malformed */ }
    _ => unreachable!(),
}
```

Re-exported via `deepstream_decoders::prelude::*` and
`deepstream_decoders::detect_stream_config`.

## Random Access Point Detection

SIG:
`is_random_access_point(codec: Codec, data: &[u8]) -> bool`

Returns `true` when a **single access unit** is a valid decode entry point
(same Annex-B vs length-prefixed framing rules as `detect_stream_config`).

- **H264**: at least one SPS, one PPS, and one IDR slice (NAL type 5) in the AU.
- **HEVC**: at least one VPS, SPS, PPS, and one IRAP VCL NAL (IDR, CRA, BLA, etc.;
  uses `cros_codecs` `NaluType::is_irap()`).
- **JPEG, PNG, RawRgba, RawRgb, RawNv12**: `true` if `data` is non-empty.
- **VP8, VP9, AV1**: always `false` (not implemented).
- **Empty `data`**: always `false`.

Re-exported via `deepstream_decoders::prelude::*` and
`deepstream_decoders::is_random_access_point`.

## Output Semantics

- Decoder output delivered to callbacks is RGBA for all supported codecs.
- The caller-provided `pool` controls output dimensions and memory behavior.
- `transform_config` controls GPU scaling/conversion behavior when needed.
