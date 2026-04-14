# DeepStream Decoders Public API

Crate: `deepstream_decoders`  
Prelude: `use deepstream_decoders::prelude::*;`

## Core types

- `DecoderConfig` and per-codec config structs
- `NvDecoderConfig` (framework/pipeline runtime settings)
- `NvDecoder` (channel-based decoder)
- `NvDecoderOutput` (pull-based outputs)
- `DecodedFrame` (decoded RGBA frame + metadata)
- `DecoderError`

## Decoder configuration

`DecoderConfig` variants:

- `H264(H264DecoderConfig)`
- `Hevc(HevcDecoderConfig)`
- `Vp8(Vp8DecoderConfig)`
- `Vp9(Vp9DecoderConfig)`
- `Av1(Av1DecoderConfig)`
- `Jpeg(JpegDecoderConfig)`
- `Png(PngDecoderConfig)`
- `RawRgba(RawRgbaDecoderConfig)`
- `RawRgb(RawRgbDecoderConfig)`

Helpers:

- `DecoderConfig::codec(&self) -> Codec`
- `NvDecoderConfig::new(gpu_id: u32, decoder: DecoderConfig) -> Self`
- `NvDecoderConfig` builder methods: `name`, `input_channel_capacity`, `output_channel_capacity`, `operation_timeout`, `drain_poll_interval`

## NvDecoder

Primary constructor:

`NvDecoder::new(config: NvDecoderConfig, pool: BufferGenerator, transform_config: TransformConfig) -> Result<Self, DecoderError>`

Primary methods:

- `codec(&self) -> Codec`
- `is_failed(&self) -> bool`
- `submit_packet(&self, data: &[u8], frame_id: u128, pts_ns: u64, dts_ns: Option<u64>, duration_ns: Option<u64>) -> Result<(), DecoderError>`
- `send_event(&self, event: gst::Event) -> Result<(), DecoderError>`
- `send_source_eos(&self, source_id: &str) -> Result<(), DecoderError>`
- `recv(&self) -> Result<NvDecoderOutput, DecoderError>`
- `recv_timeout(&self, timeout: Duration) -> Result<Option<NvDecoderOutput>, DecoderError>`
- `try_recv(&self) -> Result<Option<NvDecoderOutput>, DecoderError>`
- `graceful_shutdown(&self, idle_timeout: Option<Duration>, on_output: F) -> Result<(), DecoderError>`
- `shutdown(&self) -> Result<(), DecoderError>`

## NvDecoderOutput

```rust
pub enum NvDecoderOutput {
    Frame(DecodedFrame),
    Event(gst::Event),
    SourceEos { source_id: String },
    Eos,
    Error(DecoderError),
}
```

## DecodedFrame

```rust
pub struct DecodedFrame {
    pub frame_id: Option<u128>,
    pub pts_ns: u64,
    pub dts_ns: Option<u64>,
    pub duration_ns: Option<u64>,
    pub buffer: Option<SharedBuffer>,
    pub codec: Codec,
    pub format: VideoFormat,
}
```

`format` is RGBA for decoder outputs delivered to callers.  
`buffer` is optional so downstream stages can take ownership without consuming
the whole `DecodedFrame` container.

## Stream detection helpers

- `detect_stream_config(codec: Codec, data: &[u8]) -> Option<DecoderConfig>`
- `is_random_access_point(codec: Codec, data: &[u8]) -> bool`

`detect_stream_config` supports H264/HEVC access-unit inspection (Annex-B vs length-prefixed AVCC/HVCC).  
`is_random_access_point` is stricter than keyframe-only checks (requires in-band parameter sets for H264/HEVC in the same AU).
