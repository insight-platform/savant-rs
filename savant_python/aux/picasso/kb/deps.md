# External Dependencies for Picasso Tests

## Import Map

### savant_rs.picasso (all picasso types)
```python
from savant_rs.picasso import (
    # Encoder enums
    Platform, RateControl, H264Profile, HevcProfile,
    DgpuPreset, TuningPreset, JetsonPresetLevel,
    # Encoder property structs
    H264DgpuProps, HevcDgpuProps, H264JetsonProps, HevcJetsonProps,
    JpegProps, Av1DgpuProps, EncoderProperties, EncoderConfig,
    # Spec types
    GeneralSpec, EvictionDecision, ConditionalSpec, ObjectDrawSpec,
    CodecSpec, SourceSpec,
    # Output messages
    EncodedOutput, BypassOutput,
    # Callbacks & engine
    Callbacks, PicassoEngine,
)
```

### savant_rs.deepstream (GPU/transform, required for encode/bypass)
```python
from savant_rs.deepstream import (
    MemType,              # memory type enum
    NvBufSurfaceGenerator,# GPU buffer pool
    TransformConfig,      # transform config for CodecSpec.encode()
    VideoFormat,          # pixel format enum
    init_cuda,            # CUDA context init
    Padding,              # NONE, RIGHT_BOTTOM, SYMMETRIC
    Interpolation,        # NEAREST, BILINEAR, ALGO1-4, DEFAULT
    ComputeMode,          # DEFAULT, GPU, VIC
)
```

### savant_rs.draw_spec (for ObjectDrawSpec entries)
```python
from savant_rs.draw_spec import (
    BoundingBoxDraw,  # (border_color, background_color, thickness, padding)
    ColorDraw,        # (red, green, blue, alpha) all int 0-255
    DotDraw,          # (color: ColorDraw, radius: int)
    LabelDraw,        # (font_color, background_color, border_color, font_scale, thickness, position, padding, format)
    LabelPosition,    # (position: LabelPositionKind, margin_x, margin_y)
    ObjectDraw,       # (bounding_box?, central_dot?, label?, blur=False, bbox_source=DetectionBox)
    PaddingDraw,      # (left, top, right, bottom) all int
)
```

### savant_rs.gstreamer
```python
from savant_rs.gstreamer import Codec  # H264, HEVC, JPEG, AV1
```

### savant_rs.primitives (frames, objects, geometry)
```python
from savant_rs.primitives import (
    VideoFrame,        # frame container
    VideoFrameContent, # .none(), .external(url, method?)
    VideoObject,       # detection object
    IdCollisionResolutionPolicy,  # .GenerateNewId
    EndOfStream,       # EOS signal
)
from savant_rs.primitives.geometry import RBBox  # (xc, yc, width, height, [angle])
from savant_rs.match_query import MatchQuery     # .idle() for querying objects
```

---

## Key Type Constructors (quick ref)

### VideoFrame
```python
VideoFrame(
    source_id="src-0",
    framerate="30/1",
    width=1280, height=720,
    content=VideoFrameContent.none(),
    time_base=(1, 1_000_000_000),
    pts=0,
)
```

### VideoObject
```python
VideoObject(
    id=0, namespace="detector", label="person",
    detection_box=RBBox(cx, cy, w, h),
    attributes=[], confidence=None,
    track_id=None, track_box=None,
)
# Add to frame:
frame.add_object(obj, IdCollisionResolutionPolicy.GenerateNewId)
```

### NvBufSurfaceGenerator
```python
gen = NvBufSurfaceGenerator(VideoFormat.RGBA, width, height, fps_num, fps_den, gpu_id)
buf_ptr = gen.acquire_surface(id=frame_idx)
# pts/duration are taken from the VideoFrame; set frame.pts and frame.duration before send_frame.
```

### TransformConfig
```python
TransformConfig()  # all defaults: Padding.SYMMETRIC, Interpolation.BILINEAR, ComputeMode.DEFAULT, no crop
TransformConfig(padding=Padding.SYMMETRIC, interpolation=Interpolation.BILINEAR, src_rect=None, compute_mode=ComputeMode.DEFAULT)
# src_rect: Optional[Tuple[top, left, width, height]] for crop region
```

### ColorDraw
```python
ColorDraw(red=255, green=0, blue=0, alpha=255)
ColorDraw.transparent()  # (0,0,0,0)
```

### PaddingDraw
```python
PaddingDraw(left=0, top=0, right=0, bottom=0)
PaddingDraw.default_padding()  # all zeros
```

### LabelPosition
```python
LabelPosition.default_position()  # TopLeftOutside, margin_x=0, margin_y=-10
```
