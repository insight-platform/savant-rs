# savant_python — Python Module API Reference

## Module: `savant_rs` (root)

```python
from savant_rs import version, register_handler, unregister_handler
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `version()` | `-> str` | Returns crate version |
| `register_handler(name, handler)` | `-> None` | Register a named handler |
| `unregister_handler(name)` | `-> None` | Remove a handler |

---

## Module: `savant_rs.primitives`

Core data types for video frames and objects.

| Class | Description |
|-------|-------------|
| `VideoFrame` | Proxy to a video frame (Arc-backed, interior mutability) |
| `VideoFrameBatch` | Batch of frames (HashMap<i64, VideoFrameProxy>) |
| `VideoFrameContent` | Frame content: External, Internal, or None |
| `VideoFrameTranscodingMethod` | Copy or Encoded |
| `VideoFrameUpdate` | Delta to merge into a frame |
| `VideoFrameTransformation` | Coordinate transform (InitialSize, LetterBox, etc.) |
| `Attribute` | Named typed attribute with namespace |
| `AttributeValue` | Typed value (int, float, string, bbox, etc.) |
| `AttributeValueType` | Type discriminant enum |
| `VideoObject` | Standalone video object |
| `BorrowedVideoObject` | Reference to object inside a frame |
| `VideoObjectTree` | Recursive object hierarchy |
| `VideoObjectsView` | View into a collection of objects |
| `EndOfStream` | End-of-stream signal |
| `Shutdown` | Shutdown signal |
| `UserData` | Opaque user data |
| `IdCollisionResolutionPolicy` | GenerateNewId, Overwrite, Error |

---

## Module: `savant_rs.primitives.geometry`

```python
from savant_rs.primitives.geometry import RBBox, Point, Segment, PolygonalArea, Intersection, IntersectionKind
```

### RBBox
```python
class RBBox:
    def __init__(self, xc: float, yc: float, width: float, height: float, angle: Optional[float] = None): ...
    @staticmethod
    def ltwh(left: float, top: float, width: float, height: float) -> RBBox: ...
    @staticmethod
    def ltrb(left: float, top: float, right: float, bottom: float) -> RBBox: ...

    # Properties (get/set)
    xc, yc, width, height: float
    angle: Optional[float]
    area: float  # read-only
    vertices: List[Tuple[float, float]]  # read-only

    # Conversions
    def as_ltwh(self) -> Tuple[float, float, float, float]: ...
    def as_ltrb(self) -> Tuple[float, float, float, float]: ...
    def as_xcycwh(self) -> Tuple[float, float, float, float]: ...

    # Geometry
    def get_wrapping_bbox(self) -> RBBox: ...
    def get_visual_box(self, padding, border_width, max_x, max_y) -> RBBox: ...
    def iou(self, other: RBBox) -> float: ...
    def ios(self, other: RBBox) -> float: ...
    def shift(self, dx: float, dy: float) -> None: ...
    def scale(self, sx: float, sy: float) -> None: ...
    def copy(self) -> RBBox: ...
```

---

## Module: `savant_rs.draw_spec`

```python
from savant_rs.draw_spec import ColorDraw, PaddingDraw, BoundingBoxDraw, DotDraw, LabelDraw, ObjectDraw, LabelPosition, LabelPositionKind, BBoxSource, SetDrawLabelKind
```

---

## Module: `savant_rs.utils`

```python
from savant_rs.utils import eval_expr, gen_frame, gen_empty_frame, ByteBuffer, PropagatedContext, ...
```

### Sub-modules
- `savant_rs.utils.symbol_mapper` — model/object ID ↔ name registry
- `savant_rs.utils.serialization` — `save_message`, `load_message` (+ ByteBuffer variants)

---

## Module: `savant_rs.pipeline`

```python
from savant_rs.pipeline import Pipeline, PipelineConfiguration, VideoPipelineStagePayloadType, StageFunction
```

---

## Module: `savant_rs.match_query`

```python
from savant_rs.match_query import MatchQuery, FloatExpression, IntExpression, StringExpression, QueryFunctions, ...
```

---

## Module: `savant_rs.zmq`

```python
from savant_rs.zmq import (
    WriterConfig, WriterConfigBuilder, WriterSocketType,
    ReaderConfig, ReaderConfigBuilder, ReaderSocketType,
    BlockingWriter, NonBlockingWriter,
    BlockingReader, NonBlockingReader,
    TopicPrefixSpec,
)
```

---

## Module: `savant_rs.telemetry`

```python
from savant_rs.telemetry import init, init_from_file, shutdown, TracerConfiguration, TelemetryConfiguration, ...
```

---

## Module: `savant_rs.logging`

```python
from savant_rs.logging import LogLevel, set_log_level, get_log_level, log_level_enabled, log_message
```

---

## Module: `savant_rs.webserver`

```python
from savant_rs.webserver import init_webserver, stop_webserver, set_shutdown_token, is_shutdown_set, ...
```

### Sub-module: `savant_rs.webserver.kvs`
```python
from savant_rs.webserver.kvs import set_attributes, get_attribute, search_attributes, del_attributes, ...
```

---

## Module: `savant_rs.metrics`

```python
from savant_rs.metrics import CounterFamily, GaugeFamily, delete_metric_family, set_extra_labels
```

---

## Module: `savant_rs.gstreamer` [feature=gst]

```python
from savant_rs.gstreamer import FlowResult, InvocationReason
# With gst feature:
from savant_rs.gstreamer import Codec, Mp4Muxer
# Codec variants: H264, HEVC, JPEG, AV1, PNG, RAW_RGBA, RAW_RGB
```

---

## Module: `savant_rs.deepstream` [feature=deepstream]

```python
from savant_rs.deepstream import (
    SharedBuffer, SurfaceView,
    BufferGenerator, UniformBatchGenerator,
    SurfaceBatch, NonUniformBatch,
    Rect, Padding, Interpolation, ComputeMode, VideoFormat, MemType, SavantIdMetaKind,
    TransformConfig, DstPadding,
    init_cuda, gpu_mem_used_mib, jetson_model, is_jetson_kernel, has_nvenc,
    SkiaContext,                # GPU Skia rendering context
    set_num_filled,             # low-level batch filling
    get_savant_id_meta, get_buffers_info,  # meta functions
    GpuMatCudaArray, make_gpu_mat, nvgstbuf_as_gpu_mat, nvbuf_as_gpu_mat, from_gpumat,  # OpenCV CUDA helpers
    SkiaCanvas,                 # convenience Skia wrapper
)
```

---

## Module: `savant_rs.nvinfer` [feature=deepstream]

```python
from savant_rs.nvinfer import (
    NvInfer, NvInferConfig, Roi,
    MetaClearPolicy, DataType,
    BatchInferenceOutput, ElementOutput, TensorView, InferDims,
)
```

See `savant_python/aux/nvinfer/kb/` for detailed API reference.

---

## Module: `savant_rs.picasso` [feature=deepstream]

```python
from savant_rs.picasso import (
    PicassoEngine, Callbacks, OutputMessage,
    GeneralSpec, EvictionDecision, ConditionalSpec,
    ObjectDrawSpec, CodecSpec, SourceSpec,
    CallbackInvocationOrder,
    PtsResetPolicy, StreamResetReason,
    # Encoder types
    EncoderConfig, EncoderProperties, H264DgpuProps, HevcDgpuProps, ...
)
```

See `savant_python/aux/picasso/kb/` for detailed API reference.
