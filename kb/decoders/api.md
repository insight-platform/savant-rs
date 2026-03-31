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

## Output Semantics

- Decoder output delivered to callbacks is RGBA for all supported codecs.
- The caller-provided `pool` controls output dimensions and memory behavior.
- `transform_config` controls GPU scaling/conversion behavior when needed.
