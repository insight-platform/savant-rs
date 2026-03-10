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
- `new(source_id, framerate, width, height, content, transcoding_method, codec, keyframe, time_base, pts, dts, duration) -> Result<Self>`
- `smart_copy() -> Self` — deep independent copy

Properties (get/set):
- `source_id`, `uuid`, `pts`, `dts`, `duration`, `width`, `height`
- `framerate`, `codec`, `keyframe`, `time_base`
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

### `VideoFrameContent`
`External(ExternalFrame) | Internal(Vec<u8>) | None`

### `VideoFrameTransformation`
`InitialSize(w,h) | LetterBox(...) | Padding(...) | Crop(...)`

### `VideoFrameTranscodingMethod`
`Copy | Encoded`

---

## `primitives::object` — Video Object

### `VideoObject` (builder-derived)
Fields: `id`, `namespace`, `label`, `draw_label`, `detection_box: RBBox`,
`confidence`, `parent_id`, `track_box: Option<RBBox>`, `track_id`, `attributes`

### `BorrowedVideoObject`
Handle to an object living inside a `VideoFrameProxy`. Access via `ObjectOperations` trait.

### `ObjectOperations` trait
Getters: `get_id`, `get_namespace`, `get_label`, `get_confidence`,
`get_detection_box`, `get_track_box`, `get_track_id`, `get_draw_label`

Setters: `set_detection_box`, `set_track_info`, `set_track_box`,
`set_namespace`, `set_label`, `set_confidence`

Operations: `transform_geometry(ops)`, `detached_copy() -> VideoObject`

### `IdCollisionResolutionPolicy`
`GenerateNewId | Overwrite | Error`

---

## `primitives::attribute` — Attributes

### `Attribute`
`namespace`, `name`, `values: Vec<AttributeValue>`, `hint`, `is_persistent`, `is_hidden`

### `AttributeValue`
Typed value (`None | Bytes | Float | Integer | String | Boolean | BBox(RBBox) |
BBoxList | Point | PointList | Polygon | PolygonList | Intersection | TemporaryValue`)

### `AttributeSet`
Ordered collection of `Attribute` with O(1) lookup by `(namespace, name)`.

---

## `message` — Message Envelope

### `Message`
- Factory: `video_frame(&proxy)`, `end_of_stream(eos)`, `shutdown(s)`,
  `video_frame_batch(&batch)`, `video_frame_update(update)`, `user_data(ud)`,
  `unknown(s)`
- Accessors: `payload()`, `meta()`, type checks (`is_video_frame()`, etc.),
  extractors (`as_video_frame()`, etc.)

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
- `from_transformations(chain) -> TransformationChainResult`
- `inverse()`, `then(other)`, `then_scale_to(...)`, `to_bbox_ops()`

### `ScaleSpec` — source/dest dimensions + letterbox + crop + inset
- `to_transformations() -> Result<Vec<VideoFrameTransformation>>`

---

## `draw` — Draw Specifications

| Type | Fields |
|------|--------|
| `PaddingDraw` | `left, top, right, bottom: i64` |
| `ColorDraw` | `red, green, blue, alpha: i64` (0-255) |
| `BoundingBoxDraw` | `border_color, background_color, thickness, padding` |
| `DotDraw` | `color, radius` |
| `LabelDraw` | `font_color, background_color, border_color, font_scale, thickness, position, padding, format` |
| `ObjectDraw` | `bounding_box, central_dot, label, blur, bbox_source` |
| `LabelPositionKind` | `TopLeftInside, TopLeftOutside, Center` |
| `BBoxSource` | `DetectionBox, TrackingBox` |

---

## `pipeline` — Processing Pipeline

### `Pipeline`
- `new(name, stages, config)` — stages: `Vec<(name, payload_type, ingress_fn, egress_fn)>`
- Frame ops: `add_frame`, `get_independent_frame`, `delete`
- Batch ops: `move_and_pack_frames`, `move_and_unpack_batch`, `get_batch`
- Stage moves: `move_as_is`
- Query: `access_objects(frame_id, query)`
- Stats: `get_stat_records`, `log_final_fps`

### `PipelineStagePayloadType`
`Frame | Batch`

### `PipelineStageFunction` trait
`set_pipeline`, `get_pipeline`, `call(id, stage, order, payload)`

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

### `parse_zmq_socket_uri(uri) -> Result<ZmqSocketUri>`

---

## `symbol_mapper` — Model/Object ID Registry

### `SymbolMapper`
- `register_model_objects(model, objects, policy)` — bulk register
- `get_model_id(model)`, `get_object_id(model, object)`, etc.
- `RegistrationPolicy`: `ErrorIfNonUnique | Override`

---

## `telemetry` — OpenTelemetry

- `init(TelemetryConfiguration)`, `init_from_file(path)`, `shutdown()`
- `TracerConfiguration { service_name, protocol, endpoint, tls, timeout }`
- `ContextPropagationFormat`: `Jaeger | W3C`

---

## `protobuf` — Serialization

- `serialize(m: &Message) -> Result<Vec<u8>>`
- `deserialize(bytes: &[u8]) -> Result<Message>`
- `ToProtobuf` trait: `to_pb() -> Result<Vec<u8>>`

---

## `webserver` — Embedded HTTP Server

- `init_webserver(port)`, `stop_webserver()`
- `set_status(PipelineStatus)`, `get_status()`
- `set_shutdown_token(token)`, `is_shutdown_set()`
- KVS: `set_attributes`, `get_attribute`, `search_attributes`, `del_attributes`
- `subscribe(name, max_ops) -> KvsSubscription`

---

## `metrics` — Prometheus

- `new_counter(name, help, labels)`, `new_gauge(name, help, labels)`
- `export_metrics() -> Vec<MetricExport>`
