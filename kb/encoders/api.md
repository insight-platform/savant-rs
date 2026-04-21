# DeepStream Encoders Public API

Crate: `deepstream_encoders`
Prelude: `use deepstream_encoders::prelude::*;`

The encoder public API mirrors `deepstream_decoders` one-to-one: a
codec-specific configuration enum (`EncoderConfig`), a runtime wrapper
that carries pipeline-level knobs (`NvEncoderConfig`), a thread-safe
pipeline handle (`NvEncoder`), and a pull-based output channel
(`NvEncoderOutput`). See `kb/decoders/api.md` for the symmetric decoder
API.

---

## Codec Enum (from savant_gstreamer, re-exported)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codec {
    H264,     // nvv4l2h264enc       -- NVENC required
    Hevc,     // nvv4l2h265enc       -- NVENC required
    Jpeg,     // nvjpegenc           -- nvjpegenc element required
    Av1,      // nvv4l2av1enc        -- NVENC required, dGPU only
    Png,      // pngenc (CPU-based)  -- always available
    RawRgba,  // pseudoencoder       -- always available
    RawRgb,   // pseudoencoder       -- always available
    RawNv12,  // pseudoencoder       -- always available
}
```

Codec → element / caps mappings are identical to the decoder KB; see
`kb/decoders/api.md` for the full table.

---

## Core types

- Per-codec configs: `H264EncoderConfig`, `HevcEncoderConfig`,
  `Av1EncoderConfig`, `JpegEncoderConfig`, `PngEncoderConfig`,
  `RawEncoderConfig`.
- Top-level codec enum: `EncoderConfig`.
- Framework/pipeline settings: `NvEncoderConfig`.
- Encoder handle: `NvEncoder` (channel-based, interior mutable).
- Outputs: `NvEncoderOutput` (pull), `EncodedFrame` (encoded payload).
- Errors: `EncoderError`.

## EncoderConfig (codec-specific enum)

```rust
pub enum EncoderConfig {
    H264(H264EncoderConfig),
    Hevc(HevcEncoderConfig),
    Av1(Av1EncoderConfig),
    Jpeg(JpegEncoderConfig),
    Png(PngEncoderConfig),
    RawRgba(RawEncoderConfig),
    RawRgb(RawEncoderConfig),
    RawNv12(RawEncoderConfig),
}
```

Helpers on `EncoderConfig`:

- `codec(&self) -> Codec`
- `width(&self) -> u32`
- `height(&self) -> u32`
- `format(&self) -> VideoFormat`
- `fps(&self) -> (i32, i32)`
- `to_gst_pairs(&self) -> Vec<(&'static str, String)>`

Per-codec structs expose a builder chain:

- `<Codec>EncoderConfig::new(width, height)` (raw: `RawEncoderConfig::new(width, height, format)`)
- `.format(VideoFormat)` (not available on raw; format is fixed at
  construction to the codec's pixel layout)
- `.fps(num, den)`
- `.props(<Codec>Props)` for platform-aware codec properties

Raw codec configs never take props on their builder; the `RawProps`
unit struct is used only by the top-level `EncoderProperties` enum.

## NvEncoderConfig (framework/pipeline settings)

```rust
pub struct NvEncoderConfig {
    pub name: String,
    pub gpu_id: u32,
    pub mem_type: NvBufSurfaceMemType,
    pub encoder: EncoderConfig,
    pub input_channel_capacity: usize,
    pub output_channel_capacity: usize,
    pub operation_timeout: Duration,
    pub drain_poll_interval: Duration,
}
```

Primary constructor and builder (all return `Self`):

- `NvEncoderConfig::new(gpu_id: u32, encoder: EncoderConfig) -> Self`
- `.name(impl Into<String>)`
- `.mem_type(NvBufSurfaceMemType)`
- `.input_channel_capacity(usize)`
- `.output_channel_capacity(usize)`
- `.operation_timeout(Duration)`
- `.drain_poll_interval(Duration)`

Accessors (`width`, `height`, `codec`, `format`, `fps`) delegate to the
wrapped `EncoderConfig`. This mirrors `NvDecoderConfig` for decoders.

## NvEncoder

Primary constructor:

`NvEncoder::new(config: NvEncoderConfig) -> Result<Self, EncoderError>`

Primary methods (all take `&self`; state is interior-mutable behind
`Arc<Mutex<_>>` so the handle can be cheaply shared with drain
threads):

- `generator(&self) -> Arc<Mutex<BufferGenerator>>` -- acquire NVMM
  buffers sized for the encoder input (pool depth 1).
- `codec(&self) -> Codec`
- `width(&self) -> u32`, `height(&self) -> u32`
- `submit_frame(&self, buffer: gst::Buffer, frame_id: u128, pts_ns: u64, duration_ns: Option<u64>) -> Result<(), EncoderError>`
- `send_event(&self, event: gst::Event) -> Result<(), EncoderError>`
- `send_source_eos(&self, source_id: &str) -> Result<(), EncoderError>`
- `recv(&self) -> Result<NvEncoderOutput, EncoderError>`
- `recv_timeout(&self, timeout: Duration) -> Result<Option<NvEncoderOutput>, EncoderError>`
- `try_recv(&self) -> Result<Option<NvEncoderOutput>, EncoderError>`
- `graceful_shutdown<F>(&self, idle_timeout: Option<Duration>, on_output: F) -> Result<(), EncoderError>`
  where `F: FnMut(NvEncoderOutput)` -- sends a real EOS, drains every
  pending frame/event through the callback, then tears the pipeline
  down.
- `shutdown(&self) -> Result<(), EncoderError>` -- hard stop without
  draining.

`NvEncoder` implements `Drop`; `Drop` sends EOS and stops the
pipeline. Prefer `graceful_shutdown` so callers observe all in-flight
frames.

## NvEncoderOutput

```rust
pub enum NvEncoderOutput {
    Frame(EncodedFrame),
    Event(gst::Event),
    SourceEos { source_id: String },
    Eos,
    Error(EncoderError),
}
```

This is the symmetric dual of `NvDecoderOutput`; the variants have
identical semantics, including in-band delivery of `savant.*`
custom-downstream events via the generic rescue probe in
`GstPipeline`.

## EncodedFrame

```rust
pub struct EncodedFrame {
    pub frame_id: Option<u128>,
    pub pts_ns: u64,
    pub dts_ns: Option<u64>,
    pub duration_ns: Option<u64>,
    pub data: Vec<u8>,
    pub codec: Codec,
    pub keyframe: bool,
    pub time_base: (i32, i32),  // always (1, 1_000_000_000)
}
```

Raw payload sizes:

- `VideoCodec::RawRgba`: `data.len() == width * height * 4`
- `VideoCodec::RawRgb`:  `data.len() == width * height * 3`
- `VideoCodec::RawNv12`: `data.len() == width * height * 3 / 2`

Codec-header-only buffers (e.g. AV1 sequence headers emitted as
standalone GStreamer buffers) are stashed internally and prepended to
the next real encoded frame so that callers only ever see complete
`EncodedFrame`s. See `caveats.md` for the rationale and scope.

---

## EncoderProperties (top-level enum)

```rust
pub enum EncoderProperties {
    H264Dgpu(H264DgpuProps),
    H264Jetson(H264JetsonProps),
    HevcDgpu(HevcDgpuProps),
    HevcJetson(HevcJetsonProps),
    Jpeg(JpegProps),
    Av1Dgpu(Av1DgpuProps),
    Av1Jetson(Av1JetsonProps),
    Png(PngProps),
    RawRgba(RawProps),
    RawRgb(RawProps),
    RawNv12(RawProps),
}
```

| Method | Signature |
|---|---|
| `codec` | `(&self) -> Codec` |
| `platform` | `(&self) -> Option<Platform>` -- None for JPEG/PNG/Raw |
| `to_gst_pairs` | `(&self) -> Vec<(&'static str, String)>` |
| `from_pairs` | `(codec: Codec, platform: Platform, pairs: &HashMap<String,String>) -> Result<Self, EncoderError>` |

### Platform-Specific Property Structs

All fields are `Option<T>` and default to `None` (encoder default
applies). All implement `Debug, Clone, Default`.

- `H264DgpuProps` -- `bitrate`, `control_rate`, `profile`,
  `iframeinterval`, `idrinterval`, `preset` (DgpuPreset),
  `tuning_info`, `qp_range`, `const_qp`, `init_qp`, `max_bitrate`,
  `vbv_buf_size`, `vbv_init`, `cq`, `aq`, `temporal_aq`,
  `extended_colorformat`
- `H264JetsonProps` -- `bitrate`, `control_rate`, `profile`,
  `iframeinterval`, `idrinterval`, `preset_level`, `peak_bitrate`,
  `vbv_size`, `qp_range`, `quant_i_frames`, `quant_p_frames`,
  `ratecontrol_enable`, `maxperf_enable`, `two_pass_cbr`,
  `num_ref_frames`, `insert_sps_pps`, `insert_aud`, `insert_vui`,
  `disable_cabac`
- `HevcDgpuProps` / `HevcJetsonProps` -- same shape as H264 variants
  but with `HevcProfile`
- `Av1DgpuProps` / `Av1JetsonProps` -- Av1-specific subset
- `JpegProps` -- `quality` (0-100)
- `PngProps` -- `compression_level` (0-9, CPU-based `pngenc`)
- `RawProps` -- unit struct, no tunables

### Helper Enums

- `Platform` -- `Dgpu`, `Jetson` (`from_name`: "dgpu"/"gpu" -> Dgpu;
  "jetson"/"tegra" -> Jetson)
- `RateControl` -- `VariableBitrate(0)`, `ConstantBitrate(1)`,
  `ConstantQP(2)`
- `H264Profile` -- `Baseline(0)`, `Main(2)`, `High(4)`, `High444(7)`
- `HevcProfile` -- `Main(0)`, `Main10(1)`, `Frext(3)`
- `DgpuPreset` -- `P1..P7` (lower = faster, higher = better quality)
- `TuningPreset` -- `HighQuality`, `LowLatency`, `UltraLowLatency`,
  `Lossless`
- `JetsonPresetLevel` -- `Disabled`, `UltraFast`, `Fast`, `Medium`,
  `Slow`

---

## Python Bindings

Encoder PyO3 bindings live in `savant_rs.deepstream`, symmetric with
the decoder bindings (NOT in `savant_rs.picasso`). Typical usage:

```python
from savant_rs.deepstream import (
    EncoderConfig, EncoderProperties,
    H264DgpuProps, DgpuPreset, TuningPreset, RateControl,
    VideoFormat,
)
from savant_rs.gstreamer import Codec

props = H264DgpuProps(
    bitrate=4_000_000,
    control_rate=RateControl.VARIABLE_BITRATE,
    preset=DgpuPreset.P3,
    tuning_info=TuningPreset.LOW_LATENCY,
)
cfg = EncoderConfig(Codec.H264, 1920, 1080)
cfg.format(VideoFormat.NV12)
cfg.fps(30, 1)
cfg.gpu_id(0)
cfg.properties(EncoderProperties.h264_dgpu(props))
```

The Python `EncoderConfig` is a thin configuration DSL; its
`to_rust()` path materialises a full `NvEncoderConfig` (wrapping the
codec-specific `EncoderConfig` enum variant with platform-aware props)
for use by Picasso and other consumers.

---

## Prelude Re-exports

```rust
pub use crate::config::{
    Av1EncoderConfig, EncoderConfig, H264EncoderConfig, HevcEncoderConfig,
    JpegEncoderConfig, NvEncoderConfig, PngEncoderConfig, RawEncoderConfig,
};
pub use crate::error::EncoderError;
pub use crate::pipeline::{NvEncoder, NvEncoderOutput};
pub use crate::{EncodedFrame, EncoderProperties};
pub use savant_core::primitives::video_codec::VideoCodec;
pub use deepstream_buffers::{
    cuda_init, BufferGenerator, NvBufSurfaceMemType, SharedBuffer, SurfaceView,
    UniformBatchGenerator, VideoFormat,
};
pub use crate::properties::{
    Av1DgpuProps, Av1JetsonProps, DgpuPreset, H264DgpuProps, H264JetsonProps,
    H264Profile, HevcDgpuProps, HevcJetsonProps, HevcProfile, JetsonPresetLevel,
    JpegProps, Platform, PngProps, RateControl, RawProps, TuningPreset,
};
```
