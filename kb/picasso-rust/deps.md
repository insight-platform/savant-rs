# External Crate Dependencies

## Import Map for Tests

### picasso (the crate itself)
```rust
use picasso::prelude::*;
// gives: Callbacks, On*Frame/Render/Eviction/GpuMat/ObjectDrawSpec/StreamReset traits,
//        StreamResetReason,
//        PicassoEngine, PicassoError,
//        OutputMessage,
//        CallbackInvocationOrder, CodecSpec, ConditionalSpec, EvictionDecision,
//        GeneralSpec, ObjectDrawSpec, PtsResetPolicy, SourceSpec

// For low-level worker tests:
use picasso::worker::SourceWorker;
use picasso::message::WorkerMessage;

// For geometry tests:
use picasso::rewrite_frame_transformations;
```

### deepstream_encoders::prelude (GPU encode tests)
```rust
use deepstream_encoders::prelude::*;
// gives: NvEncoder, EncoderError, EncodedFrame, EncoderConfig,
//        Codec (H264, Hevc, Jpeg, Av1, Png, RawRgba, RawRgb),
//        cuda_init, BufferGenerator, NvBufSurfaceMemType, VideoFormat,
//        EncoderProperties, H264DgpuProps, HevcDgpuProps, H264JetsonProps, HevcJetsonProps,
//        JpegProps, PngProps, Av1DgpuProps, RawProps,
//        DgpuPreset, TuningPreset, H264Profile, HevcProfile,
//        JetsonPresetLevel, Platform, RateControl
```

### deepstream_buffers (transform config, GPU utilities)
```rust
use deepstream_buffers::{Padding, Rect, SurfaceView, TransformConfig, buffer_gpu_id};
// Padding: None, Symmetric, RightBottom
// Rect: { top, left, width, height } — optional per-call crop for transform/send_frame
// TransformConfig fields: padding, dst_padding, interpolation, compute_mode, cuda_stream (no src_rect)
// TransformConfig implements Default (Symmetric, Bilinear, Default compute)
// CudaStream: safe RAII wrapper for CUDA stream handles
// CudaStream::default() — legacy null stream; CudaStream::new_non_blocking() — owned non-blocking
// SurfaceView.cuda_stream() — read CUDA stream; SurfaceView.with_cuda_stream(stream) — set stream
// SurfaceView::transform_into(&self, dest: &SurfaceView, config: &TransformConfig, src_rect: Option<&Rect>) — GPU transform per call
// buffer_gpu_id(&gst::BufferRef) → Result<u32, TransformError>  — extract GPU ID from NvBufSurface buffer
// SurfaceView::wrap(buf) — NOGPU stub, surface params zeroed (test-only: requires `testing` feature)
// SurfaceView::from_gst_buffer(buf, slot_index) — extract from NvBufSurface-backed buffer (consumes buf by value)
// SurfaceView::from_buffer(&shared, slot_index) — create view from SharedBuffer (primary for batched/single)
// view.into_gst_buffer() — recover gst::Buffer from view (consumes view); or shared.into_buffer() after dropping view
// SurfaceView::from_cuda_ptr(...) — wrap arbitrary CUDA device memory
// SurfaceView accessors: gst_buffer(), shared_buffer(), data_ptr(), pitch(), width(), height(), gpu_id(), channels()
// SurfaceView CUDA stream: cuda_stream() → &CudaStream, with_cuda_stream(stream) → Self
// surface_ops: memset_surface(&SurfaceView, value), upload_to_surface(&SurfaceView, data, w, h, channels) — take &SurfaceView, not &gst::Buffer
```

### savant_core (frames, objects, geometry)
```rust
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod, VideoFrameTransformation,
};
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::object::{
    BorrowedVideoObject, IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
};
use savant_core::primitives::bbox::RBBox;
use savant_core::primitives::WithAttributes;
use savant_core::draw::ObjectDraw;
```

### gstreamer
```rust
use gstreamer;      // gstreamer::init(), gstreamer::Buffer::new()
use gstreamer::ClockTime;  // for set_pts/set_duration on buffers
```

---

## Key Type Constructors

### VideoFrameProxy
```rust
VideoFrameProxy::new(
    source_id,     // &str
    (30, 1),       // fps (numerator, denominator)
    width,         // i64
    height,        // i64
    content,       // VideoFrameContent::None
    transcoding,   // VideoFrameTranscodingMethod::Copy
    codec,         // Option<VideoCodec> → None
    keyframe,      // Option<bool> → None
    time_base,     // (i64, i64), e.g. (1, 1_000_000_000)
    pts,           // i64
    dts,           // Option<i64> → None
    duration,      // Option<i64> → None
).unwrap()
```

### gstreamer::Buffer (NOGPU stub)
```rust
gstreamer::init().unwrap();
let buf = gstreamer::Buffer::new();
// Wrap as SurfaceView for Picasso APIs:
let view = SurfaceView::wrap(buf);
```

### BufferGenerator (GPU)
```rust
let gen = BufferGenerator::new(
    VideoFormat::RGBA, W, H, 30, 1, 0, NvBufSurfaceMemType::Default,
).unwrap();
assert_eq!(gen.gpu_id(), 0);  // stored GPU ID accessible via getter
let buf = gen.acquire(Some(frame_idx as i64)).unwrap();
let view = SurfaceView::from_buffer(buf, 0).unwrap();
// ⚠ pts/dts/duration are taken from the VideoFrame; do not assume they are in the buffer.
// Set them on the frame before send_frame:
frame.set_pts((idx * dur_ns) as i64).unwrap();
frame.set_duration(Some(dur_ns as i64)).unwrap();
// The pipeline applies frame timestamps to view.buffer_mut().make_mut() at entry.
```

### EncoderConfig
```rust
// dGPU:
EncoderConfig::new(Codec::H264, W, H)
    .format(VideoFormat::RGBA)
    .fps(30, 1)
    .gpu_id(0)  // default: 0 — must match incoming buffer GPU
    .properties(EncoderProperties::H264Dgpu(H264DgpuProps {
        bitrate: Some(2_000_000),
        preset: Some(DgpuPreset::P1),
        tuning_info: Some(TuningPreset::LowLatency),
        iframeinterval: Some(30),
        ..Default::default()
    }))

// Jetson (aarch64):
EncoderConfig::new(Codec::H264, W, H)
    .format(VideoFormat::RGBA)
    .fps(30, 1)
    .properties(EncoderProperties::H264Jetson(H264JetsonProps {
        preset_level: Some(JetsonPresetLevel::UltraFast),
        maxperf_enable: Some(true),
        ..Default::default()
    }))

// Raw pseudoencoder (GPU→CPU, both platforms):
EncoderConfig::new(Codec::RawRgba, W, H)
    .format(VideoFormat::RGBA)
```
⚠ Builder returns `Self` by value (move semantics). Chain in one expression.
⚠ `gpu_id` must match the GPU where incoming NvBufSurface buffers are allocated; `process_encode` validates this at entry.
⚠ **Platform-specific properties:** Use `H264DgpuProps`/`HevcDgpuProps` on dGPU, `H264JetsonProps`/`HevcJetsonProps` on Jetson. Using the wrong variant causes GStreamer property errors. Detect with `cfg!(target_arch = "aarch64")`.

### PNG encoder (CPU-based, GStreamer pngenc)
```rust
EncoderConfig::new(Codec::Png, W, H)
    .format(VideoFormat::RGBA)  // required for PNG
    .fps(30, 1)
    .properties(EncoderProperties::Png(PngProps {
        compression_level: Some(6),  // 0–9, default 6
    }))
```
PNG uses the nvvideoconvert → pngenc GStreamer chain (gst-plugins-good). Format must be RGBA.

### TransformConfig
```rust
TransformConfig::default()  // Padding::Symmetric, Bilinear, Default compute
// src_rect removed — pass Option<Rect> to send_frame or generator.transform() per call
```

### ObjectDraw (from savant_core::draw)
```rust
ObjectDraw::new(
    bounding_box,  // Option<BoundingBoxDraw>
    central_dot,   // Option<DotDraw>
    label,         // Option<LabelDraw>
    blur,          // bool
)
```
Minimal stub for tests: `ObjectDraw::new(None, None, None, false)`

### VideoObjectBuilder
```rust
VideoObjectBuilder::default()
    .id(0)
    .namespace("det".to_string())
    .label("car".to_string())
    .detection_box(RBBox::new(cx, cy, w, h, None))
    .build()
    .unwrap()
```

### Frame attributes (for conditional tests)
```rust
let mut fm = frame.clone();
fm.set_persistent_attribute("namespace", "name", &None, false, vec![]);
// Check: frame.get_attribute("namespace", "name").is_some()
```
