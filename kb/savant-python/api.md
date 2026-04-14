# savant_python — Python Module API Reference

## Module: `savant_rs` (root)

```python
from savant_rs import (
    version,
    is_release_build,
    register_handler,
    unregister_handler,
    clear_all_handlers,
)
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `version()` | `-> str` | Returns crate version |
| `is_release_build()` | `-> bool` | Returns whether this is a release build |
| `register_handler(name, handler)` | `-> None` | Register a named handler |
| `unregister_handler(name)` | `-> None` | Remove a handler |
| `clear_all_handlers()` | `-> None` | Remove all handlers (also registered on `atexit` at import) |

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
from savant_rs.primitives.geometry import (
    RBBox, BBox, Point, Segment, PolygonalArea, Intersection, IntersectionKind,
    solely_owned_areas, associate_bboxes,
)
```

### Free Functions

| Function | Description |
|----------|-------------|
| `solely_owned_areas(polys, bboxes)` | Compute areas solely owned by each polygon for given bboxes |
| `associate_bboxes(polys, bboxes)` | Associate bounding boxes with polygonal areas |

### BBox
Axis-aligned bounding box (no rotation). See `geometry.pyi` for full API.

```python
class BBox:
    def __init__(self, xc: float, yc: float, width: float, height: float): ...
    @classmethod
    def ltwh(cls, left, top, width, height) -> BBox: ...
    @classmethod
    def ltrb(cls, left, top, right, bottom) -> BBox: ...

    # Properties (get/set)
    xc, yc, width, height, top, left: float
    right, bottom: float  # read-only
    vertices: List[Tuple[float, float]]  # read-only

    # Geometry
    wrapping_box: BBox  # property, returns BBox
    def get_visual_box(self, padding: PaddingDraw, border_width: int) -> BBox: ...
    def iou(self, other: BBox) -> float: ...
    def ios(self, other: BBox) -> float: ...
    def ioo(self, other: BBox) -> float: ...
    def shift(self, dx: float, dy: float) -> BBox: ...
    def scale(self, scale_x: float, scale_y: float) -> BBox: ...
    def as_rbbox(self) -> RBBox: ...
    def copy(self) -> BBox: ...
```

### RBBox
```python
class RBBox:
    def __init__(self, xc: float, yc: float, width: float, height: float, angle: Optional[float] = None): ...
    @classmethod
    def ltwh(cls, left, top, width, height) -> RBBox: ...
    @classmethod
    def ltrb(cls, left, top, right, bottom) -> RBBox: ...

    # Properties (get/set)
    xc, yc, width, height: float
    angle: Optional[float]
    top, left: float
    area: float  # read-only
    right, bottom: float  # read-only
    vertices: List[Tuple[float, float]]  # read-only

    # Conversions
    def as_ltwh(self) -> Tuple[float, float, float, float]: ...
    def as_ltrb(self) -> Tuple[float, float, float, float]: ...
    def as_xcycwh(self) -> Tuple[float, float, float, float]: ...
    def into_bbox(self) -> BBox: ...

    # Geometry
    wrapping_box: BBox  # property, returns BBox (not RBBox)
    def get_visual_box(self, padding: PaddingDraw, border_width: int) -> RBBox: ...
    def iou(self, other: RBBox) -> float: ...
    def ios(self, other: RBBox) -> float: ...
    def ioo(self, other: RBBox) -> float: ...
    def shift(self, dx: float, dy: float) -> RBBox: ...
    def scale(self, scale_x: float, scale_y: float) -> RBBox: ...
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
from savant_rs.utils import (
    eval_expr, gen_frame, gen_empty_frame,
    round_2_digits, estimate_gil_contention, enable_dl_detection,
    incremental_uuid_v7, relative_time_uuid_v7,
    TelemetrySpan, MaybeTelemetrySpan, PropagatedContext, ByteBuffer,
    VideoObjectBBoxType, VideoObjectBBoxTransformation, BBoxMetricType,
    AtomicCounter,
)
```

| Function/Class | Description |
|----------------|-------------|
| `eval_expr(expr, ttl, no_gil)` | Evaluate a Savant expression |
| `gen_frame()` / `gen_empty_frame()` | Generate sample / empty `VideoFrame` for testing |
| `round_2_digits(num)` | Round float to 2 decimal places |
| `estimate_gil_contention()` | Estimate current GIL contention |
| `enable_dl_detection()` | Enable deadlock detection |
| `incremental_uuid_v7()` | Generate an incremental UUID v7 |
| `relative_time_uuid_v7(uuid, offset_millis)` | UUID v7 relative to another UUID |
| `TelemetrySpan` | OpenTelemetry span wrapper (context manager) |
| `MaybeTelemetrySpan` | Optional span wrapper (no-op when None) |
| `PropagatedContext` | Propagated tracing context |
| `ByteBuffer` | Byte buffer with optional checksum |
| `VideoObjectBBoxType` | Enum: Detection, TrackingInfo |
| `VideoObjectBBoxTransformation` | BBox transform: scale/shift |
| `BBoxMetricType` | Enum: IoU, IoSelf, IoOther |
| `AtomicCounter` | Thread-safe atomic counter |

### Sub-modules
- `savant_rs.utils.symbol_mapper` — model/object ID ↔ name registry
- `savant_rs.utils.serialization` — `save_message`, `load_message`, `save_message_to_bytes`, `load_message_from_bytes` (+ ByteBuffer variants), `Message` class

---

## Module: `savant_rs.pipeline`

Also available as `savant_rs.pipeline2` (alias registered in `sys.modules`).

```python
from savant_rs.pipeline import (
    VideoPipeline, VideoPipelineConfiguration,
    VideoPipelineStagePayloadType, StageFunction,
    StageLatencyMeasurements, StageLatencyStat,
    StageProcessingStat, FrameProcessingStatRecord,
    FrameProcessingStatRecordType,
)
```

> **Note**: The Python-visible class names are `VideoPipeline` and `VideoPipelineConfiguration`
> (via `#[pyo3(name = ...)]`), even though the Rust structs are named `Pipeline` and
> `PipelineConfiguration`.

---

## Module: `savant_rs.match_query`

```python
from savant_rs.match_query import (
    MatchQuery, FloatExpression, IntExpression, StringExpression,
    QueryFunctions,  # has filter() and partition() classmethods
    EtcdCredentials, TlsConfig,
    # Resolver functions
    utility_resolver_name, etcd_resolver_name, env_resolver_name, config_resolver_name,
    register_utility_resolver, register_env_resolver, register_etcd_resolver,
    register_config_resolver, update_config_resolver,
    unregister_resolver,
)
```

| Item | Description |
|------|-------------|
| `EtcdCredentials(username, password)` | Credentials for etcd resolver |
| `TlsConfig(ca, cert, key)` | TLS config for etcd resolver |
| `register_etcd_resolver(hosts, ...)` | Register etcd-backed config resolver |
| `register_config_resolver(params)` | Register static config resolver |
| `update_config_resolver(params)` | Update existing config resolver params |
| `unregister_resolver(name)` | Remove a named resolver |
| `QueryFunctions.filter(v, q)` | Filter objects matching query |
| `QueryFunctions.partition(v, q)` | Split objects into matching/non-matching |

---

## Module: `savant_rs.zmq`

```python
from savant_rs.zmq import (
    WriterConfig, WriterConfigBuilder, WriterSocketType,
    ReaderConfig, ReaderConfigBuilder, ReaderSocketType,
    BlockingWriter, NonBlockingWriter,
    BlockingReader, NonBlockingReader,
    TopicPrefixSpec, WriteOperationResult,
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
from savant_rs.webserver import (
    init_webserver, stop_webserver,
    set_shutdown_token, is_shutdown_set,
    set_status_running, set_shutdown_signal,
)
```

### Sub-module: `savant_rs.webserver.kvs`
```python
from savant_rs.webserver.kvs import (
    set_attributes, get_attribute,
    search_attributes, search_keys,
    del_attributes, del_attribute,
    serialize_attributes, deserialize_attributes,
    KvsSubscription, KvsSetOperation, KvsDeleteOperation,
)
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

## Module: `savant_rs.retina_rtsp` [feature=gst]

Embeddable RTSP ingestion service with dynamic group management.

```python
from savant_rs.retina_rtsp import (
    RetinaRtspService, RtspSourceGroup, RtspSource,
    RtspBackend, RtspSourceOptions, SyncConfiguration,
)
```

| Class | Description |
|-------|-------------|
| `RtspBackend` | Enum: `Retina`, `Gstreamer` |
| `RtspSourceOptions` | SIG: `(username: str, password: str)` — RTSP auth credentials |
| `RtspSource` | SIG: `(source_id, url, stream_position=None, options=None)` — single RTSP source |
| `SyncConfiguration` | SIG: `(group_window_duration_ms, batch_duration_ms, network_skew_correction=False, rtcp_once=False)` |
| `RtspSourceGroup` | SIG: `(sources: List[RtspSource], backend=RtspBackend.Retina, rtcp_sr_sync=None)` |
| `RetinaRtspService` | SIG: `(config_path: str)` — loads JSON config, opens shared sink socket |

| Method | Signature | Description |
|--------|-----------|-------------|
| `run_group(group, name)` | `-> None` | Block (GIL released) until stopped. Call from a thread |
| `stop_group(name)` | `-> None` | Stop a group, block until finished |
| `shutdown()` | `-> None` | Stop all groups |
| `running_groups` | `-> List[str]` | Property: names of running groups |

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
    gpu_architecture, gpu_platform_tag,  # GPU detection helpers
    SkiaContext,                # GPU Skia rendering context
    set_num_filled,             # low-level batch filling
    get_savant_id_meta, get_nvbufsurface_info,  # meta functions (was: get_buffers_info)
    GpuMatCudaArray, make_gpu_mat, nvgstbuf_as_gpu_mat, nvbuf_as_gpu_mat, from_gpumat,  # OpenCV CUDA helpers
    SkiaCanvas,                 # convenience Skia wrapper
)
```

| Function | Description |
|----------|-------------|
| `gpu_architecture(gpu_id)` | GPU arch family name (dGPU only, e.g. "ampere", "ada") |
| `gpu_platform_tag(gpu_id)` | Directory-safe platform tag for TRT engine caching |
| `get_nvbufsurface_info(buf)` | Get `(data_ptr, pitch, width, height)` from buffer |

---

## Module: `savant_rs.nvinfer` [feature=deepstream]

```python
from savant_rs.nvinfer import (
    NvInfer, NvInferConfig, Roi,
    MetaClearPolicy, DataType,
    BatchInferenceOutput, ElementOutput, TensorView, InferDims,
)
```

See `kb/nvinfer-python/` for detailed API reference.

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

See `kb/picasso-python/` for detailed API reference.

---

## Module: `savant_rs.nvtracker` [feature=deepstream]

```python
from savant_rs.nvtracker import (
    NvTracker, NvTrackerConfig, TrackingIdResetMode, TrackState,
    TrackedFrame, TrackedObject, MiscTrackFrame, MiscTrackData,
    TrackerOutput, NvTrackerBatchingOperatorConfig,
    TrackerBatchFormationResult, TrackerOperatorFrameOutput,
    SealedDeliveries, TrackerOperatorOutput, NvTrackerBatchingOperator,
)
```

See `kb/nvtracker-python/` for detailed API reference.
