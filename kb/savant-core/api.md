# savant_core — Public API Reference

## Crate Root (`lib.rs`)

| Symbol | Kind | Description |
|--------|------|-------------|
| `EPS` | `const f32` | `0.00001` — crate-wide epsilon |
| `get_or_init_async_runtime()` | fn → `&'static Runtime` | Shared Tokio multi-thread runtime (lazy) |
| `round_2_digits(v: f32)` | fn → `f32` | Round to 2 decimal places |
| `version()` | fn → `String` | `CARGO_PKG_VERSION` |
| `fast_hash(bytes: &[u8])` | fn → `u32` | CRC32 hash |
| `get_tracer()` | fn → `BoxedTracer` | OpenTelemetry tracer (`"video_pipeline"`) |
| `trace!` | macro | Wraps an expression with `log::trace!` before/after for lock-acquisition tracing |
| `function!` | macro | Resolves the enclosing function name at compile time |

### `pub mod rust` re-exports

```rust
pub use otlp::PropagatedContext;
pub use pipeline::stats::{FrameProcessingStatRecord, FrameProcessingStatRecordType,
    StageLatencyMeasurements, StageLatencyStat, StageProcessingStat};
pub use pipeline::{Pipeline, PipelineConfiguration, PipelineConfigurationBuilder,
    PipelineStagePayloadType};
pub use symbol_mapper::{RegistrationPolicy, SymbolMapper};
```

---

## `primitives::bbox` — Bounding Box

### `RBBox` (Arc-backed, thin-clone)

Constructor variants:
- `RBBox::new(xc, yc, width, height, angle: Option<f32>)` — center-based
- `RBBox::ltwh(left, top, width, height) -> Result<Self>` — left-top-width-height
- `RBBox::ltrb(left, top, right, bottom) -> Result<Self>` — left-top-right-bottom

Coordinate accessors (all `f32`):
- `get_xc/yc/width/height()`, `set_xc/yc/width/height()`
- `get_angle() -> Option<f32>`, `set_angle(Option<f32>)`
- `get_area()`, `get_width_to_height_ratio()`

Conversion (fail if rotated, except noted):
- `as_ltwh() -> Result<(f32,f32,f32,f32)>`
- `as_ltrb() -> Result<(f32,f32,f32,f32)>`
- `as_ltwh_int() / as_ltrb_int() -> Result<(i64,i64,i64,i64)>`
- `as_xcycwh() -> (f32,f32,f32,f32)` — always succeeds
- `get_top/left/right/bottom() -> Result<f32>`

Geometry:
- `get_vertices() -> Vec<(f32,f32)>` — 4 corners (rotation-aware)
- `get_wrapping_bbox() -> RBBox` — axis-aligned envelope
- `get_visual_box(padding, border_width, max_x, max_y) -> Result<RBBox>` — clamped to frame
- `new_padded(&self, padding: &PaddingDraw) -> Self`
- `shift(dx, dy)`, `scale(sx, sy)` — in-place mutation
- `iou/ios/ioo(&self, other) -> Result<f32>` — intersection metrics
- `inside(&self, other) -> Result<bool>`
- `inside_viewport(width, height) -> Result<bool>`

Identity:
- `copy() -> Self` — deep copy (new Arc)
- `is_modified() -> bool`, `set_modifications(bool)`
- `geometric_eq(other) -> bool`, `almost_eq(other, eps) -> bool`

### `BBoxMetricType`
`IoU | IoSelf | IoOther`

---

## `primitives::frame` — Video Frame

### `VideoFrameProxy` (Arc<RwLock<Box<VideoFrame>>>)

Construction:
- `new(source_id, fps: (i64, i64), width, height, content, transcoding_method, codec, keyframe, time_base: (i64, i64), pts, dts, duration) -> Result<Self>`
- `smart_copy() -> Self` — deep independent copy

Properties (get/set):
- `source_id`, `uuid`, `pts`, `dts`, `duration`, `width`, `height`
- `fps: (i64, i64)`, `codec`, `keyframe`, `time_base: (i64, i64)`
- `transcoding_method`, `content`, `creation_timestamp_ns`

Transformations:
- `get/add/clear_transformations()`
- `transform_geometry(ops)` — apply bbox operations to all objects
- `transform_backward()` / `transform_forward()` — inverse/forward coordinate remap

Object management:
- `get_all_objects() -> Vec<BorrowedVideoObject>`
- `access_objects(query) -> Vec<BorrowedVideoObject>`
- `create_object(namespace, label, parent_id, detection_box, confidence, track_id, track_box, attributes) -> Result<BorrowedVideoObject>`
- `add_object(object, policy) -> Result<BorrowedVideoObject>`
- `delete_objects(query) / delete_objects_with_ids(ids)`
- `set_parent(query, parent)`, `set_parent_by_id(obj_id, parent_id)`
- `get_children(id)`, `get_parent_chain(obj)`
- `export_complete_object_trees(query, delete) -> Result<Vec<VideoObjectTree>>`
- `import_object_trees(trees)`

Serialization:
- `to_message() -> Message`
- `get_json() / get_json_pretty() -> String`

### `BelongingVideoFrame`
Weak reference back to a `VideoFrameProxy`. Used internally by `VideoObject` to reference its owning frame. Public struct (fields are `pub(crate)`).

### `VideoFrameContent`
`External(ExternalFrame) | Internal(Vec<u8>) | None`

### `VideoFrameTransformation`
`InitialSize(w,h) | LetterBox(...) | Padding(...) | Crop(...)`

### `VideoFrameTranscodingMethod`
`Copy | Encoded`

---

## `primitives::object` — Video Object

### `VideoObject` (builder-derived via `VideoObjectBuilder`)
Fields: `id`, `namespace`, `label`, `draw_label`, `detection_box: RBBox`,
`confidence`, `parent_id`, `track_box: Option<RBBox>`, `track_id`,
`namespace_id: Option<i64>`, `label_id: Option<i64>`, `attributes`,
`frame: Option<BelongingVideoFrame>`

### `BorrowedVideoObject`
Handle to an object living inside a `VideoFrameProxy`. Access via `ObjectOperations` trait.

### `ObjectAccess` trait
Low-level access: `with_object_ref(f)`, `with_object_mut(f)`.
Requires `Sized + Debug + Clone`.

### `WithId` trait
`get_id() -> i64`, `set_id(&mut self, i64)`. Implemented for `VideoObject`.

### `VideoObjectBBoxType`
`Detection | TrackingInfo`

### `VideoObjectBBoxTransformation`
`Scale(f32, f32) | Shift(f32, f32)`

### `ObjectOperations` trait
Getters: `get_id`, `get_namespace`, `get_label`, `get_confidence`,
`get_detection_box`, `get_track_box`, `get_track_id`, `get_draw_label`,
`get_parent_id`, `get_namespace_id`, `get_label_id`

Setters: `set_detection_box`, `set_track_info`, `set_track_box`,
`set_namespace`, `set_label`, `set_confidence`

Operations: `transform_geometry(ops)`, `detached_copy() -> VideoObject`

### `IdCollisionResolutionPolicy`
`GenerateNewId | Overwrite | Error`

### `VideoObjectTree`
Type alias: `ObjectTree<VideoObject>`. A recursive tree where each node holds a `VideoObject` and a `Vec<VideoObjectTree>` of children. Used by `export_complete_object_trees()` / `import_object_trees()`.

---

## `primitives::attribute` — Attributes

### `Attribute`
`namespace`, `name`, `values: Arc<Vec<AttributeValue>>`, `hint`, `is_persistent`, `is_hidden`

### `AttributeValueVariant`
`None | Bytes(Vec<i64>, Vec<u8>) | String | StringVector | Integer | IntegerVector |
Float | FloatVector | Boolean | BooleanVector | BBox(RBBoxData) |
BBoxVector(Vec<RBBoxData>) | Point | PointVector(Vec<Point>) |
Polygon(PolygonalArea) | PolygonVector(Vec<PolygonalArea>) |
Intersection | TemporaryValue(AnyObject)`

### `AttributeSet`
Ordered collection of `Attribute` with O(1) lookup by `(namespace, name)`.

### `WithAttributes` trait
Provides uniform access to the attributes collection on any type that carries attributes (e.g. `VideoObject`, `VideoFrame`). Requires `Send`.

Methods:
- `with_attributes_ref<F, R>(&self, f: F) -> R` where `F: FnOnce(&Vec<Attribute>) -> R` — read-only access
- `with_attributes_mut<F, R>(&mut self, f: F) -> R` where `F: FnOnce(&mut Vec<Attribute>) -> R` — mutable access
- `take_attributes(&mut self) -> Vec<Attribute>` — drain and return all attributes (default impl swaps with empty `Vec`)

---

## `message` — Message Envelope

### `Message`
- Factory: `video_frame(&proxy)`, `end_of_stream(eos)`, `shutdown(s)`,
  `video_frame_batch(&batch)`, `video_frame_update(update)`, `user_data(ud)`,
  `unknown(s)`
- Accessors: `payload()`, `meta()`, type checks (`is_video_frame()`, etc.),
  extractors (`as_video_frame()`, etc.)

### `MessageMeta`
`protocol_version`, `routing_labels`, `span_context: PropagatedContext`, `seq_id`, `system_id`

### `MessageEnvelope`
`VideoFrame | VideoFrameBatch | VideoFrameUpdate | EndOfStream | Shutdown | UserData | Unknown(String)`

### `load_message(bytes) -> Message`, `save_message(m) -> Result<Vec<u8>>`

---

## `match_query` — Object Query DSL

### `MatchQuery` enum (~40 variants)
- Object filters: `Id`, `Namespace`, `Label`, `Confidence`, ...
- BBox filters: `BoxWidth`, `BoxHeight`, `BoxArea`, `BoxAngle`, `BoxMetric`, ...
- Parent/children: `ParentDefined`, `ParentId`, `WithChildren(Box<MatchQuery>, IntExpr)`
- Attribute: `AttributeExists`, `AttributesJMESQuery`
- Frame: `FrameSourceId`, `FrameWidth`, `FrameIsKeyFrame`, ...
- Combinators: `And(Vec)`, `Or(Vec)`, `Not(Box)`, `StopIfFalse(Box)`, `StopIfTrue(Box)`
- `EvalExpr(String)` — evalexpr expression

### Helper fns
- `filter(objs, query) -> Vec<BorrowedVideoObject>`
- `partition(objs, query) -> (matching, non_matching)`
- Builder fns: `eq`, `ne`, `one_of`, `gt`, `ge`, `lt`, `le`, `between`, `contains`, etc.

---

## `geometry` — Affine Transforms

### `Affine2D` — axis-aligned 2D affine `{ sx, sy, tx, ty }`
- `Affine2D::IDENTITY` — const identity transform
- `Affine2D::new(sx, sy, tx, ty) -> Self`
- `from_transformations(chain) -> TransformationChainResult`
- `inverse()`, `then(other)`, `then_scale_to(...)`, `to_bbox_ops()`

### `TransformationChainResult`
`{ affine: Affine2D, initial_size: Option<(u64,u64)>, current_size: Option<(u64,u64)> }`

### `ScaleSpec` — source/dest dimensions + letterbox + crop + inset
Fields: `source_width`, `source_height`, `dest_width`, `dest_height`,
`letterbox: LetterBoxKind`, `crop: Option<CropRect>`, `dst_inset: Option<DstInset>`
- `to_transformations() -> Result<Vec<VideoFrameTransformation>>`

### `LetterBoxKind`
`Stretch | Symmetric | RightBottom`

### `CropRect`
`{ left: u64, top: u64, width: u64, height: u64 }`

### `DstInset`
`{ left: u64, top: u64, right: u64, bottom: u64 }`

### `MIN_EFFECTIVE_DIM: u64 = 16`
Minimum effective dimension after applying `DstInset`.

---

## `draw` — Draw Specifications

| Type | Fields |
|------|--------|
| `PaddingDraw` | `left, top, right, bottom: i64` |
| `ColorDraw` | `red, green, blue, alpha: i64` (0-255) |
| `BoundingBoxDraw` | `border_color, background_color, thickness, padding` |
| `DotDraw` | `color, radius` |
| `LabelPosition` | `position: LabelPositionKind, margin_x: i64, margin_y: i64` |
| `LabelDraw` | `font_color, background_color, border_color, font_scale, thickness, position, padding, format` |
| `ObjectDraw` | `bounding_box, central_dot, label, blur, bbox_source` |
| `LabelPositionKind` | `TopLeftInside, TopLeftOutside, Center` |
| `BBoxSource` | `DetectionBox, TrackingBox` |
| `DrawLabelKind` | `OwnLabel(String), ParentLabel(String)` |

Constructors returning `Result`:
- `PaddingDraw::new(left, top, right, bottom) -> Result<Self>`
- `ColorDraw::new(red, green, blue, alpha) -> Result<Self>`
- `BoundingBoxDraw::new(border_color, background_color, thickness, padding) -> Result<Self>`
- `DotDraw::new(color, radius) -> Result<Self>`

Additional constructors:
- `ObjectDraw::new(bounding_box, central_dot, label, blur) -> Self`
- `ObjectDraw::with_bbox_source(bounding_box, central_dot, label, blur, bbox_source) -> Self`

---

## `pipeline` — Processing Pipeline

### `Pipeline`
- `new(name, stages, config)` — stages: `Vec<(name, payload_type, ingress_fn, egress_fn)>`
- Frame ops: `add_frame`, `add_frame_with_telemetry`, `get_independent_frame`, `delete`
- Batch ops: `move_and_pack_frames(dest, frame_ids: Vec<i64>)`, `move_and_unpack_batch`, `get_batch`, `get_batched_frame`
- Stage moves: `move_as_is`
- Query: `access_objects(frame_id, query)`
- Stats: `get_stat_records`, `get_stat_records_newer_than(id)`, `log_final_fps`
- Updates: `add_frame_update`, `add_batched_frame_update`, `apply_updates`, `clear_updates`
- Config: `set_root_span_name`, `get_root_span_name`, `set_sampling_period`, `get_sampling_period`
- Misc: `memory_handle()`, `clear_source_ordering(source_id)`, `get_stage_type(name)`, `get_stage_queue_len(stage)`, `get_id_locations_len()`, `get_keyframe_history(frame)`

### `PipelineStagePayloadType`
`Frame | Batch`

### `PipelineStageFunctionOrder`
`Ingress | Egress`

### `PipelinePayload`
`Frame(VideoFrameProxy, Vec<VideoFrameUpdate>, Context, Option<String>, SystemTime)` |
`Batch(VideoFrameBatch, Vec<(i64, VideoFrameUpdate)>, HashMap<i64, Context>, Option<String>, Vec<SystemTime>)`

### `PipelineStageFunction` trait
`set_pipeline`, `get_pipeline`, `call(id, stage, order: PipelineStageFunctionOrder, payload: &mut PipelinePayload)`

Note: the `Pipeline` struct wraps `Arc<implementation::Pipeline>`. The `implementation` module is `pub(super)` and not part of the public API.

---

## `transport::zeromq` — ZeroMQ Transport

### Blocking
- `Reader` / `Writer` — not `Send`, single-threaded
- `SyncReader` / `SyncWriter` — sync wrappers

### Non-blocking
- `NonBlockingReader` / `NonBlockingWriter` — `Send + Sync`, use internal Tokio tasks

### Config
- `ReaderConfig` / `WriterConfig` (builder pattern)
- `ReaderSocketType`: `Sub | Router | Rep`
- `WriterSocketType`: `Pub | Dealer | Req`

### `SocketType`
`Reader(ReaderSocketType) | Writer(WriterSocketType)`

### `TopicPrefixSpec`
`SourceId(String) | Prefix(String) | None`

### `WriteOperationResult`
Re-exported from `nonblocking_writer`.

### `parse_zmq_socket_uri(uri: String) -> Result<ZmqSocketUri>`

---

## `symbol_mapper` — Model/Object ID Registry

### `SymbolMapper`
- `register_model_objects(model, objects, policy)` — bulk register
- `get_model_id(model)`, `get_object_id(model, object)`, etc.
- `RegistrationPolicy`: `ErrorIfNonUnique | Override`

### `symbol_mapper::Errors` (thiserror)
`DuplicateName | UnexpectedModelIdObjectId | FullyQualifiedObjectNameParseError | BaseNameParseError | DuplicateId`

---

## `telemetry` — OpenTelemetry

- `init(TelemetryConfiguration)`, `init_from_file(path)`, `shutdown()`
- `TracerConfiguration { service_name, protocol, endpoint, tls, timeout }`
- `ContextPropagationFormat`: `Jaeger | W3C`
- `Protocol`: `Grpc | HttpBinary | HttpJson`
- `ClientTlsConfig { ca: Option<String>, identity: Option<Identity> }`
- `Identity { key: String, certificate: String }`
- `Configurator::new(service_namespace, config) -> Self`, `Configurator::shutdown()`
- `TelemetryConfiguration::no_op() -> Self`, `TelemetryConfiguration::from_file(path) -> Result<Self>`

---

## `protobuf` — Serialization

- `serialize(m: &Message) -> Result<Vec<u8>, Error>`
- `deserialize(bytes: &[u8]) -> Result<Message, Error>`
- `ToProtobuf<'a, T>` trait (generic over target protobuf type): `to_pb() -> Result<Vec<u8>, Error>`
- `from_pb<T, U>(bytes: &[u8]) -> Result<U, Error>` — generic deserialization

### `protobuf::Error` (thiserror)
`ProstDecode | ProstEncode | UuidParse | InvalidVideoFrameParentObject | EnumConversionError | SerializationError`

---

## `webserver` — Embedded HTTP Server

- `init_webserver(port)`, `stop_webserver()`
- `set_status(PipelineStatus)`, `async get_status() -> PipelineStatus`
- `set_shutdown_token(token)`, `is_shutdown_set()`
- `set_shutdown_signal(signal: i32) -> Result<()>`
- `subscribe(subscriber: &str, max_ops: usize) -> Result<KvsSubscription>`
- `get_pipeline(name: &str) -> Option<Arc<implementation::Pipeline>>`
- KVS: `set_attributes`, `get_attribute`, `search_attributes`, `del_attributes`
  - Synchronous handlers: `kvs::synchronous::*` (module path `webserver::kvs`)
  - Asynchronous handlers: `kvs::asynchronous::*` (module path `webserver::kvs`)

---

## `metrics` — Prometheus

- `set_extra_labels(HashMap<String, String>)`
- `new_counter(name, description: Option<&str>, label_names: &[&str], unit: Option<Unit>) -> SharedCounterFamily`
- `new_gauge(name, description: Option<&str>, label_names: &[&str], unit: Option<Unit>) -> SharedGaugeFamily`
- `get_or_create_counter_family(name, description, label_names, unit) -> SharedCounterFamily`
- `get_or_create_gauge_family(name, description, label_names, unit) -> SharedGaugeFamily`
- `get_counter_family(name) -> Option<SharedCounterFamily>`
- `get_gauge_family(name) -> Option<SharedGaugeFamily>`
- `delete_metric_family(name)`
- `export_metrics() -> Vec<MetricExport>`

### Type aliases
- `SharedCounterFamily = Arc<Mutex<Counter>>`
- `SharedGaugeFamily = Arc<Mutex<Gauge>>`

---

## `converters` — Detection Model Output Converters

Postprocessing utilities for converting raw inference tensor output into structured detection results.

### `NmsKind`
How NMS is applied after confidence filtering.

| Variant | Fields | Description |
|---------|--------|-------------|
| `None` | `top_k: usize` | No suppression — keep top-k detections by confidence |
| `ClassAgnostic` | `iou_threshold: f32, top_k: usize` | All classes compete — a high-confidence box can suppress a nearby lower-confidence box of any class |
| `ClassAware` | `iou_threshold: f32, top_k: usize` | NMS runs independently per class — boxes of different classes never suppress each other |

### `YoloFormat`
YOLO output tensor layout.

| Variant | Fields | Description |
|---------|--------|-------------|
| `V8` | `num_classes: usize` | Single tensor `(num_classes+4) x N` — YOLOv8/v11. Rows: `[xc, yc, w, h, cls_0_score, ..., cls_K_score]`. Requires logical transpose. |
| `V5` | `num_classes: usize` | Single tensor `N x (num_classes+5)` — YOLOv5/v7. Columns: `[xc, yc, w, h, obj_conf, cls_0, ..., cls_K]`. Confidence = `obj_conf * max(cls_scores)`. |
| `V4` | `model_width: f32, model_height: f32` | Two tensors — YOLOv4. Tensor 0: boxes `N x 4` (LTRB in 0..1). Tensor 1: scores `N x K`. |
| `V3Raw` | — | Three tensors. Tensor 0: boxes `N x 4` (xc, yc, w, h). Tensor 1: scores `N x K`. Tensor 2: class_ids `N`. |
| `V4PostNms` | `model_width: f32, model_height: f32` | Four post-NMS tensors. Tensor 0: `[1]` num_dets. Tensor 1: `N x 4` boxes (LTRB 0..1). Tensor 2: `N` scores. Tensor 3: `N` class IDs. |

### `ConverterError`

| Variant | Fields | Description |
|---------|--------|-------------|
| `UnexpectedTensorCount` | `expected: &'static str, got: usize` | Wrong number of input tensors for the configured format |
| `ShapeMismatch` | `tensor_index: usize, expected: String, got: Vec<usize>` | Tensor shape does not match expected layout |
| `EmptyTensor` | — | Input tensor has zero elements |
| `NmsInputError` | `String` | NMS input validation failed (length mismatch between boxes/confidences/class_ids) |
| `InvalidClassIndex` | `tensor_index, detection_index, raw: f32, num_score_columns: Option<usize>` | Class id from the model is not usable |
| `InvalidDetectionCount` | `raw: f32` | Scalar detection count tensor is not a finite non-negative value |

Implements `Display` and `std::error::Error`.

### Key Functions

- `nms_class_agnostic(boxes: &[[f32; 4]], confidences: &[f32], iou_threshold: f32, top_k: usize) -> Result<Vec<usize>>` — greedy NMS over all classes; boxes in `[xc, yc, w, h]` form
- `nms_class_aware(boxes: &[[f32; 4]], confidences: &[f32], class_ids: &[usize], iou_threshold: f32, top_k: usize) -> Result<Vec<usize>>` — greedy NMS applied independently per class, results merged and truncated to `top_k`
- `iou_xcycwh(a: &[f32; 4], b: &[f32; 4]) -> f32` — IoU for two boxes given as `[xc, yc, w, h]`

### `YoloDetectionConverter`

Decodes raw YOLO inference tensors into per-class bounding boxes.

Fields:
- `format: YoloFormat` — tensor layout
- `confidence_threshold: f32` — minimum confidence when no per-class override applies
- `per_class_conf_threshold: HashMap<usize, f32>` — per-class confidence overrides
- `class_filter: Option<HashSet<usize>>` — when set, only keep detections whose class id is in this set
- `nms: NmsKind` — NMS strategy applied after filtering

Methods:
- `new(format, confidence_threshold, per_class_conf_threshold, class_filter, nms) -> Self`
- `decode(&self, tensors: &[(&[f32], &[usize])]) -> Result<HashMap<usize, Vec<(f32, RBBox)>>, ConverterError>` — decode tensors into a map from class_id to `(confidence, RBBox)` pairs

---

## `primitives::gstreamer_frame_time` — GStreamer Clock Conversion

Convert `VideoFrameProxy` timestamps to GStreamer clock scale (unsigned nanoseconds).

### Constants

- `GST_TIME_BASE: (i64, i64) = (1, 1_000_000_000)` — GStreamer clock: one nanosecond per unit

### `FrameClockNs`
Timestamps in GStreamer nanosecond clock scale for one compressed packet.

| Field | Type | Description |
|-------|------|-------------|
| `dts_ns` | `Option<u64>` | Decode timestamp in nanoseconds |
| `duration_ns` | `Option<u64>` | Duration in nanoseconds |
| `submission_order_ns` | `u64` | Ordering key: DTS when set, else PTS. Must be strictly increasing for ordered hardware decode pipelines. |

### Functions

- `time_base_to_ns(time_base: (i64, i64), value: i64) -> Option<u64>` — map `value` in `time_base = num/den` seconds per unit to nanoseconds. Returns `None` when `den == 0` or `num == 0`. Negative results clamp to `0`; overflow clamps to `u64::MAX`.
- `frame_clock_ns(frame: &VideoFrameProxy) -> FrameClockNs` — build GStreamer-scale timestamps from a frame proxy
- `normalize_frame_to_gst_ns(frame: &mut VideoFrameProxy)` — rewrite `pts`/`dts`/`duration`/`time_base` to GStreamer nanosecond scale. No-op if already `GST_TIME_BASE`. Logs warning and leaves frame unchanged on invalid time_base.

---

## `primitives::video_codec` — Video Codec Identifier

### `VideoCodec`
Encoded or raw pixel format carried on a `VideoFrame`.

Derives: `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Hash`, `Serialize`, `Deserialize`

| Variant | Canonical name | Notes |
|---------|---------------|-------|
| `H264` | `h264` | Also matches `avc` in `from_name` |
| `Hevc` | `hevc` | Also matches `h265` in `from_name` |
| `Jpeg` | `jpeg` | Hardware JPEG (default in manifests) |
| `SwJpeg` | `swjpeg` | Software JPEG decode (not `sw_jpeg`) |
| `Av1` | `av1` | |
| `Png` | `png` | |
| `Vp8` | `vp8` | |
| `Vp9` | `vp9` | |
| `RawRgba` | `raw_rgba` | |
| `RawRgb` | `raw_rgb` | |
| `RawNv12` | `raw_nv12` | |

Methods:
- `from_name(name: &str) -> Option<Self>` — case-insensitive parse; accepts canonical names and aliases (`avc`, `h265`)
- `name(self) -> &'static str` — canonical wire/JSON name
- `Display` impl — delegates to `name()`
- `FromStr` impl — delegates to `from_name()`; error type is `String`

---

## `utils::release_seal` — One-Shot Release Primitive

### `ReleaseSeal`
Condvar-gated flag flipped once by `release()`. Thread-safe; share via `Arc`.

Backed by `parking_lot::Condvar` + `parking_lot::Mutex<bool>`.

Methods:
- `new() -> Self` — creates a seal in the non-released state
- `release(&self)` — marks the seal as released and wakes all waiters
- `wait(&self)` — blocks until `release()` has been called
- `wait_timeout(&self, timeout: Duration) -> bool` — blocks until released or timeout elapses; returns `true` if released, `false` on timeout
- `is_released(&self) -> bool` — non-blocking check

Implements `Default` (delegates to `new()`).
