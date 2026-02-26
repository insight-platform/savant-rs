# External Crate Dependencies

## Import Map for Tests

### picasso (the crate itself)
```rust
use picasso::prelude::*;
// gives: Callbacks, On*Frame/Render/Eviction/GpuMat/ObjectDrawSpec traits,
//        PicassoEngine, PicassoError,
//        BypassOutput, EncodedOutput,
//        CodecSpec, ConditionalSpec, EvictionDecision, GeneralSpec, ObjectDrawSpec, SourceSpec

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
//        Codec (H264, HEVC, JPEG, AV1),
//        cuda_init, NvBufSurfaceGenerator, NvBufSurfaceMemType, VideoFormat,
//        EncoderProperties, H264DgpuProps, HevcDgpuProps, H264JetsonProps, HevcJetsonProps,
//        JpegProps, Av1DgpuProps, DgpuPreset, TuningPreset, H264Profile, HevcProfile,
//        JetsonPresetLevel, Platform, RateControl
```

### deepstream_nvbufsurface (transform config, GPU utilities)
```rust
use deepstream_nvbufsurface::{Padding, Rect, TransformConfig, buffer_gpu_id};
// Padding: None, Symmetric, RightBottom
// TransformConfig fields: padding, interpolation, src_rect: Option<Rect>, compute_mode, cuda_stream
// TransformConfig implements Default (Symmetric, Bilinear, no crop, Default compute)
// buffer_gpu_id(&gst::BufferRef) → Result<u32, TransformError>  — extract GPU ID from NvBufSurface buffer
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
    framerate,     // &str, e.g. "30/1"
    width,         // i64
    height,        // i64
    content,       // VideoFrameContent::None
    transcoding,   // VideoFrameTranscodingMethod::Copy
    codec,         // &Option<String> → &None
    keyframe,      // Option<bool> → None
    time_base,     // (i32, i32), e.g. (1, 1_000_000_000)
    pts,           // i64
    dts,           // Option<i64> → None
    duration,      // Option<i64> → None
).unwrap()
```

### gstreamer::Buffer (NOGPU stub)
```rust
gstreamer::init().unwrap();
let buf = gstreamer::Buffer::new();
```

### NvBufSurfaceGenerator (GPU)
```rust
let gen = NvBufSurfaceGenerator::new(
    VideoFormat::RGBA, W, H, 30, 1, 0, NvBufSurfaceMemType::Default,
).unwrap();
assert_eq!(gen.gpu_id(), 0);  // stored GPU ID accessible via getter
let mut buf = gen.acquire_surface(Some(frame_idx as i64)).unwrap();
// ⚠ pts/dts/duration are taken from the VideoFrame; do not assume they are in the buffer.
// Set them on the frame before send_frame:
frame.set_pts((idx * dur_ns) as i64).unwrap();
frame.set_duration(Some(dur_ns as i64)).unwrap();
// The pipeline applies frame timestamps to the buffer at entry.
```

### EncoderConfig
```rust
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
```
⚠ Builder returns `Self` by value (move semantics). Chain in one expression.
⚠ `gpu_id` must match the GPU where incoming NvBufSurface buffers are allocated; `process_encode` validates this at entry.

### TransformConfig
```rust
TransformConfig::default()  // Padding::Symmetric, Bilinear, no crop, Default compute

TransformConfig {
    padding: Padding::Symmetric,
    src_rect: Some(Rect { left: 0, top: 0, width: 960, height: 540 }),
    ..Default::default()
}
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
