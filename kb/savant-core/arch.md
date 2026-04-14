# savant_core Architecture

## Module Tree
```
savant_core/src/
├── lib.rs              # crate root: constants, runtime, version, fast_hash, tracer
├── atomic_f32.rs       # AtomicF32 wrapper (lock-free f32 mutations)
├── converters.rs       # detection model output converters (NmsKind, YoloFormat, ConverterError)
│   ├── nms.rs          # greedy NMS: nms_class_agnostic, nms_class_aware, iou_xcycwh
│   └── yolo.rs         # YoloDetectionConverter: decode YOLO tensors → (class_id → Vec<(conf, RBBox)>)
├── deadlock_detection.rs # parking_lot deadlock detection
├── draw.rs             # draw specifications (PaddingDraw, ColorDraw, BoundingBoxDraw, etc.)
├── eval_cache.rs       # evalexpr LRU cache
├── eval_context.rs     # evaluation context for MatchQuery EvalExpr
├── eval_resolvers.rs   # pluggable resolvers (etcd, env, config, utility)
├── geometry.rs         # Affine2D, TransformationChainResult, ScaleSpec, CropRect, DstInset, LetterBoxKind, MIN_EFFECTIVE_DIM
├── json_api.rs         # ToSerdeJsonValue trait
├── label_template.rs   # label format string parser (e.g. "{namespace}/{label}")
├── macros.rs           # utility macros
├── match_query.rs      # object query DSL (MatchQuery enum, filter/partition)
├── message.rs          # Message, MessageEnvelope, MessageMeta, SeqStore, load/save
│   ├── label_filter.rs
│   └── label_filter_parser.rs
├── otlp.rs             # PropagatedContext, push/pop/current_context
├── pipeline.rs         # Pipeline, PipelineStageFunction, PipelinePayload
│   ├── implementation   # pub(super) mod — inner Pipeline, PipelineConfiguration
│   ├── stage.rs
│   └── stats.rs
├── primitives.rs       # aggregator: Attribute, RBBox, Point, VideoFrameProxy, etc.
│   ├── any_object.rs
│   ├── attribute.rs     # Attribute, with JSON/YAML serde
│   ├── attribute_set.rs # AttributeSet (ordered collection)
│   ├── attribute_value.rs # AttributeValue (typed scalar/list/compound)
│   ├── bbox.rs          # RBBox, RBBoxData, BBoxMetricType
│   │   └── utils.rs     # IoU/IoS/IoO geometry helpers (uses `geo` crate)
│   ├── eos.rs           # EndOfStream
│   ├── frame.rs         # VideoFrame, VideoFrameProxy, ExternalFrame, content types, VideoObjectTree
│   ├── frame_batch.rs   # VideoFrameBatch (HashMap<i64, VideoFrameProxy>)
│   ├── gstreamer_frame_time.rs # GST_TIME_BASE, FrameClockNs, frame_clock_ns, normalize_frame_to_gst_ns, time_base_to_ns
│   ├── frame_update.rs  # VideoFrameUpdate (delta to merge into frame)
│   ├── object.rs        # VideoObject, BorrowedVideoObject, ObjectOperations trait
│   │   └── object_tree.rs # VideoObjectTree (recursive object hierarchy)
│   ├── point.rs         # Point (f32, f32)
│   ├── polygonal_area.rs # PolygonalArea + Intersection
│   ├── segment.rs       # Segment (two Points)
│   ├── shutdown.rs      # Shutdown signal
│   ├── userdata.rs      # UserData (opaque bytes)
│   └── video_codec.rs   # VideoCodec enum (H264, Hevc, Jpeg, SwJpeg, Av1, Png, Vp8, Vp9, RawRgba, RawRgb, RawNv12)
├── protobuf.rs         # serialization to/from protobuf (Message ↔ bytes)
│   └── serialize/       # per-type ToProtobuf/TryFrom impls
│        ├── attribute.rs
│        ├── attribute_set.rs
│        ├── bounding_box.rs
│        ├── intersection_kind.rs
│        ├── message_envelope.rs
│        ├── polygonal_area.rs
│        ├── user_data.rs
│        ├── video_frame.rs
│        ├── video_frame_batch.rs
│        ├── video_frame_content.rs
│        ├── video_frame_transcoding_method.rs
│        ├── video_frame_transformation.rs
│        ├── video_frame_update.rs
│        └── video_object.rs
├── rwlock.rs           # SavantRwLock (parking_lot wrapper)
├── symbol_mapper.rs    # SymbolMapper: model/object ID ↔ name registry
├── telemetry.rs        # OpenTelemetry init, TracerConfiguration, Configurator
├── test.rs             # test utilities (gen_frame, gen_empty_frame)
├── transport.rs        # transport layer
│   └── zeromq/         # ZeroMQ Reader/Writer/SyncReader/SyncWriter/NonBlocking*
├── utils.rs            # clock, DefaultOnce, iterators, RTP PTS mapper, UUID v7, release_seal
│   ├── clock.rs
│   ├── default_once.rs
│   ├── iter.rs
│   ├── release_seal.rs  # ReleaseSeal: one-shot condvar-gated release primitive (parking_lot)
│   ├── rtp_pts_mapper.rs
│   └── uuid_v7.rs
├── metrics.rs          # Prometheus metrics (Counter, Gauge, export)
│   ├── metric_collector.rs
│   └── pipeline_metric_builder.rs
└── webserver.rs        # HTTP server (actix-web): status, metrics, KVS, shutdown
    ├── kvs.rs
    ├── kvs_handlers.rs
    └── kvs_subscription.rs
```

## High-Level Architecture

savant_core is the **foundational Rust library** for the Savant video-analytics
framework. It provides:

1. **Primitives** — core data types (frames, objects, bounding boxes, attributes)
   used throughout the pipeline.
2. **Pipeline** — a multi-stage processing pipeline with frame/batch payloads,
   ingress/egress hooks, and OpenTelemetry integration.
3. **Transport** — ZeroMQ-based messaging (blocking and non-blocking readers/writers).
4. **Serialization** — protobuf-based message encoding/decoding.
5. **Query DSL** — a composable `MatchQuery` enum for filtering/selecting objects.
6. **Geometry** — affine transforms, scale specs, letterboxing.
7. **Telemetry** — OpenTelemetry tracer initialization and context propagation.
8. **Webserver** — embedded HTTP server for status, metrics, and key-value store.
9. **Metrics** — Prometheus metric families.

## Key Data Types and Relationships

```
Message
 ├── MessageMeta (routing labels, seq_id, system_id, span_context: PropagatedContext)
 └── MessageEnvelope
      ├── VideoFrame → VideoFrameProxy (Arc<RwLock<Box<VideoFrame>>>)
      │    ├── source_id, uuid, pts, dts, duration, codec, ...
      │    ├── content: VideoFrameContent (External | Internal | None)
      │    ├── transformations: Vec<VideoFrameTransformation>
      │    ├── attributes: AttributeSet
      │    └── objects: VideoObjectTree
      │         └── VideoObject
      │              ├── id, namespace, label, confidence
      │              ├── namespace_id: Option<i64>, label_id: Option<i64>
      │              ├── detection_box: RBBox
      │              ├── track_box: Option<RBBox>
      │              ├── attributes: AttributeSet
      │              └── parent_id: Option<i64>
      ├── VideoFrameBatch → HashMap<i64, VideoFrameProxy>
      ├── VideoFrameUpdate → delta (add/remove objects, set attributes)
      ├── EndOfStream
      ├── Shutdown
      ├── UserData
      └── Unknown(String)
```

## Pipeline Architecture

```
Pipeline(name, stages: Vec<Stage>)
 ├── Stage 0: "input" (Frame | Batch)
 │    ├── ingress_fn: Option<PipelineStageFunction>
 │    └── egress_fn: Option<PipelineStageFunction>
 ├── Stage 1: "process" (Frame | Batch)
 │    └── ...
 └── Stage N: "output" (Frame | Batch)

Operations:
 add_frame(stage, frame) → id
 move_as_is(dest_stage, ids) — same payload type
 move_and_pack_frames(dest_stage, frame_ids) — Frame → Batch
 move_and_unpack_batch(dest_stage, batch_id) — Batch → Frame
 delete(id) → contexts
```

## Threading Model

- `Pipeline` is `Send + Sync` (behind `Arc`).
- `VideoFrameProxy` is `Send + Sync` (behind `Arc<RwLock<...>>`).
- `RBBox` is `Send + Sync` (behind `Arc` with `AtomicF32`).
- ZeroMQ `Reader`/`Writer` are **not** `Send`; use `NonBlockingReader`/
  `NonBlockingWriter` for async multi-threaded access.
- `get_or_init_async_runtime()` provides a shared Tokio multi-thread runtime.

## Crate Re-export Strategy

Two export layers:
1. **`pub mod rust`** in `lib.rs` and `primitives.rs` — flat re-exports for
   Rust consumers.
2. **`pub mod` + `pub use *`** — module-level re-exports for the full public API.

The Python bindings (`savant_core_py`) import from `savant_core` and wrap types
in `#[pyclass]` structs.
