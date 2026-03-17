# DeepStream Encoders Public API

Crate: `deepstream_encoders`
Prelude: `use deepstream_encoders::prelude::*;`

---

## Codec Enum (from savant_gstreamer, re-exported)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codec {
    H264,     // nvv4l2h264enc       — NVENC required
    Hevc,     // nvv4l2h265enc       — NVENC required
    Jpeg,     // nvjpegenc           — nvjpegenc element required
    Av1,      // nvv4l2av1enc        — NVENC required, dGPU only
    Png,      // pngenc (CPU-based)  — always available
    RawRgba,  // pseudoencoder       — always available
    RawRgb,   // pseudoencoder       — always available
}
```

| Method | Signature | Notes |
|---|---|---|
| `encoder_element` | `(&self) → &'static str` | GStreamer element name. `"identity"` for Raw. |
| `parser_element` | `(&self) → &'static str` | GStreamer parser. `"identity"` for PNG/Raw. |
| `caps_str` | `(&self) → &'static str` | GStreamer caps string |
| `from_name` | `(name: &str) → Option<Self>` | Case-insensitive parse |
| `name` | `(&self) → &'static str` | Canonical name |
| `Display` | format!("{}", codec) | Same as `name()` |

### Codec → element mapping
| Codec | encoder_element | parser_element | caps_str |
|---|---|---|---|
| H264 | nvv4l2h264enc | h264parse | video/x-h264, stream-format=byte-stream |
| Hevc | nvv4l2h265enc | h265parse | video/x-h265, stream-format=byte-stream |
| Jpeg | nvjpegenc | jpegparse | image/jpeg |
| Av1 | nvv4l2av1enc | av1parse | video/x-av1 |
| Png | pngenc | identity | image/png |
| RawRgba | identity | identity | video/x-raw,format=RGBA |
| RawRgb | identity | identity | video/x-raw,format=RGB |

### Codec name ↔ from_name mapping
| Codec | name() | from_name() accepts |
|---|---|---|
| H264 | h264 | h264 |
| Hevc | hevc | hevc, h265 |
| Jpeg | jpeg | jpeg |
| Av1 | av1 | av1 |
| Png | png | png |
| RawRgba | raw_rgba | raw_rgba |
| RawRgb | raw_rgb | raw_rgb |

---

## EncoderConfig

```rust
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    pub format: VideoFormat,        // DEF: NV12
    pub fps_num: i32,               // DEF: 30
    pub fps_den: i32,               // DEF: 1
    pub gpu_id: u32,                // DEF: 0
    pub mem_type: NvBufSurfaceMemType, // DEF: Default
    pub encoder_params: Option<EncoderProperties>,  // DEF: None
}
```

| Method | Signature | Notes |
|---|---|---|
| `new` | `(codec: Codec, width: u32, height: u32) → Self` | Builder entry point |
| `format` | `(self, format: VideoFormat) → Self` | Move semantics — chain in one expression |
| `fps` | `(self, num: i32, den: i32) → Self` | |
| `gpu_id` | `(self, gpu_id: u32) → Self` | |
| `mem_type` | `(self, mem_type: NvBufSurfaceMemType) → Self` | |
| `properties` | `(self, props: EncoderProperties) → Self` | Variant codec must match config codec |

⚠ Builder returns `Self` by value. Chain in one expression.
⚠ `encoder_params` codec must match `config.codec` — validated at `NvEncoder::new` time.

---

## NvEncoder

```rust
pub struct NvEncoder { /* private */ }
```

| Method | Signature | Notes |
|---|---|---|
| `new` | `(config: &EncoderConfig) → Result<Self, EncoderError>` | GPU — builds and starts GStreamer pipeline |
| `generator` | `(&self) → &BufferGenerator` | For acquiring NVMM buffers (wraps `UniformBatchGenerator` with `max_batch_size=1`) |
| `codec` | `(&self) → Codec` | |
| `submit_frame` | `(&mut self, buffer: gst::Buffer, frame_id: u128, pts_ns: u64, duration_ns: Option<u64>) → Result<(), EncoderError>` | PTS must be strictly monotonic. Takes `gst::Buffer` with NvBufSurface memory (e.g. from `generator().acquire().into_buffer()`). |
| `pull_encoded` | `(&mut self) → Result<Option<EncodedFrame>, EncoderError>` | Non-blocking |
| `pull_encoded_timeout` | `(&mut self, timeout_ms: u64) → Result<Option<EncodedFrame>, EncoderError>` | Blocking with timeout |
| `finish` | `(&mut self, drain_timeout_ms: Option<u64>) → Result<Vec<EncodedFrame>, EncoderError>` | Send EOS, drain remaining |
| `check_error` | `(&self) → Result<(), EncoderError>` | Check pipeline bus |

- Implements `Drop` (sends EOS + stops pipeline).
- ⚠ `submit_frame` after `finish()` → `AlreadyFinalized`.
- ⚠ Non-monotonic PTS → `PtsReordered`.
- ⚠ Pool size is always 1 to prevent NVENC DMA read-after-reclaim.

---

## EncodedFrame

```rust
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    pub frame_id: u128,
    pub pts_ns: u64,
    pub dts_ns: Option<u64>,
    pub duration_ns: Option<u64>,
    pub data: Vec<u8>,          // bitstream or tightly-packed pixels (for Raw)
    pub codec: Codec,
    pub keyframe: bool,         // JPEG/PNG/Raw: always true
    pub time_base: (i32, i32),  // always (1, 1_000_000_000)
}
```

⚠ For `Codec::RawRgba`: `data.len() == width * height * 4`
⚠ For `Codec::RawRgb`: `data.len() == width * height * 3`
⚠ Raw data is tightly-packed (stride padding stripped).

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
    Png(PngProps),
    RawRgba(RawProps),
    RawRgb(RawProps),
}
```

| Method | Signature |
|---|---|
| `codec` | `(&self) → Codec` |
| `platform` | `(&self) → Option<Platform>` — None for JPEG/PNG/Raw |
| `to_gst_pairs` | `(&self) → Vec<(&'static str, String)>` |
| `from_pairs` | `(codec: Codec, platform: Platform, pairs: &HashMap<String,String>) → Result<Self, EncoderError>` |

---

## Platform-Specific Property Structs

All fields are `Option<T>` — `None` means use encoder default.
All implement `Debug, Clone, Default`.

### H264DgpuProps (dGPU only)
Key fields: `bitrate`, `control_rate`, `profile` (H264Profile), `iframeinterval`, `idrinterval`, `preset` (DgpuPreset), `tuning_info` (TuningPreset), `qp_range`, `const_qp`, `init_qp`, `max_bitrate`, `vbv_buf_size`, `vbv_init`, `cq`, `aq`, `temporal_aq`, `extended_colorformat`

### H264JetsonProps (Jetson only)
Key fields: `bitrate`, `control_rate`, `profile` (H264Profile), `iframeinterval`, `idrinterval`, `preset_level` (JetsonPresetLevel), `peak_bitrate`, `vbv_size`, `qp_range`, `quant_i_frames`, `quant_p_frames`, `ratecontrol_enable`, `maxperf_enable`, `two_pass_cbr`, `num_ref_frames`, `insert_sps_pps`, `insert_aud`, `insert_vui`, `disable_cabac`

### HevcDgpuProps (dGPU only)
Same structure as H264DgpuProps but with `profile: Option<HevcProfile>`.

### HevcJetsonProps (Jetson only)
Same as H264JetsonProps but with `profile: Option<HevcProfile>`, `enable_lossless` instead of insert_* fields.

### Av1DgpuProps (dGPU only, NVENC)
Key fields: `bitrate`, `control_rate`, `iframeinterval`, `idrinterval`, `preset`, `tuning_info`, `qp_range`, `max_bitrate`, `vbv_buf_size`, `vbv_init`, `cq`, `aq`, `temporal_aq`
⚠ AV1 is NOT supported on Jetson: `from_pairs(Codec::Av1, Platform::Jetson, _)` → `UnsupportedCodec`.

### JpegProps (both platforms)
Key fields: `quality` (0–100, default 85)

### PngProps (both platforms, CPU-based)
Key fields: `compression_level` (0–9, default 6)

### RawProps (both platforms, pseudoencoder)
No configurable properties. `from_pairs` rejects any input.

---

## Helper Enums

### Platform
`Dgpu`, `Jetson`
- `from_name`: "dgpu"/"gpu"/"discrete" → Dgpu; "jetson"/"tegra" → Jetson

### RateControl
`VariableBitrate(0)`, `ConstantBitrate(1)`, `ConstantQP(2)`
- GStreamer values: "0", "1", "2"

### H264Profile
`Baseline(0)`, `Main(2)`, `High(4)`, `High444(7)`

### HevcProfile
`Main(0)`, `Main10(1)`, `Frext(3)`

### DgpuPreset
`P1(1)` through `P7(7)` — lower = faster, higher = better quality

### TuningPreset
`HighQuality(1)`, `LowLatency(2)`, `UltraLowLatency(3)`, `Lossless(4)`

### JetsonPresetLevel
`Disabled(0)`, `UltraFast(1)`, `Fast(2)`, `Medium(3)`, `Slow(4)`

---

## Prelude Re-exports

```rust
pub use crate::encoder::NvEncoder;
pub use crate::error::EncoderError;
pub use crate::{EncodedFrame, EncoderConfig};
pub use savant_gstreamer::Codec;
pub use deepstream_buffers::{
    cuda_init, BufferGenerator, NvBufSurfaceMemType, VideoFormat,
};
pub use crate::properties::{
    Av1DgpuProps, DgpuPreset, EncoderProperties, H264DgpuProps, H264JetsonProps,
    H264Profile, HevcDgpuProps, HevcJetsonProps, HevcProfile, JetsonPresetLevel,
    JpegProps, Platform, PngProps, RateControl, RawProps, TuningPreset,
};
```
