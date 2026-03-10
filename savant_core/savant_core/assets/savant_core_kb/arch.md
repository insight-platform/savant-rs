# savant_core Architecture

## Module Tree
```
savant_core/src/
├── lib.rs              # crate root: constants, runtime, version, fast_hash, tracer
├── atomic_f32.rs       # AtomicF32 wrapper (lock-free f32 mutations)
├── deadlock_detection.rs # parking_lot deadlock detection
├── draw.rs             # draw specifications (PaddingDraw, ColorDraw, BoundingBoxDraw, etc.)
├── eval_cache.rs       # evalexpr LRU cache
├── eval_context.rs     # evaluation context for MatchQuery EvalExpr
├── eval_resolvers.rs   # pluggable resolvers (etcd, env, config, utility)
├── geometry.rs         # Affine2D, ScaleSpec, CropRect, DstInset, LetterBoxKind
├── json_api.rs         # ToSerdeJsonValue trait
├── label_template.rs   # label format string parser (e.g. "{namespace}/{label}")
├── macros.rs           # utility macros
├── match_query.rs      # object query DSL (MatchQuery enum, filter/partition)
├── message.rs          # Message, MessageEnvelope, MessageMeta, SeqStore, load/save
│   ├── label_filter.rs
│   └── label_filter_parser.rs
├── otlp.rs             # PropagatedContext, push/pop/current_context
├── pipeline.rs         # Pipeline, PipelineStageFunction, PipelinePayload
│   ├── stage.rs
│   ├── stage_function_loader.rs
│   ├── stage_plugin_sample.rs
│   └── stats.rs
├── primitives.rs       # aggregator: Attribute, RBBox, Point, VideoFrameProxy, etc.
│   ├── any_object.rs
│   ├── attribute.rs     # Attribute, with JSON/YAML serde
│   ├── attribute_set.rs # AttributeSet (ordered collection)
│   ├── attribute_value.rs # AttributeValue (typed scalar/list/compound)
│   ├── bbox.rs          # RBBox, RBBoxData, BBoxMetricType
│   │   └── utils.rs     # IoU/IoS/IoO geometry helpers (uses `geo` crate)
│   ├── eos.rs           # EndOfStream
│   ├── frame.rs         # VideoFrame, VideoFrameProxy, ExternalFrame, content types
│   ├── frame_batch.rs   # VideoFrameBatch (HashMap<i64, VideoFrameProxy>)
│   ├── frame_update.rs  # VideoFrameUpdate (delta to merge into frame)
│   ├── object.rs        # VideoObject, BorrowedVideoObject, ObjectOperations trait
│   │   └── object_tree.rs # VideoObjectTree (recursive object hierarchy)
│   ├── point.rs         # Point (f32, f32)
│   ├── polygonal_area.rs # PolygonalArea + Intersection
│   ├── segment.rs       # Segment (two Points)
│   ├── shutdown.rs      # Shutdown signal
│   └── userdata.rs      # UserData (opaque bytes)
├── protobuf.rs         # serialization to/from protobuf (Message ↔ bytes)
│   └── serialize/       # per-type ToProtobuf/TryFrom impls
├── rwlock.rs           # SavantRwLock (parking_lot wrapper)
├── symbol_mapper.rs    # SymbolMapper: model/object ID ↔ name registry
├── telemetry.rs        # OpenTelemetry init, TracerConfiguration, Configurator
├── test.rs             # test utilities (gen_frame, gen_empty_frame)
├── transport.rs        # transport layer
│   └── zeromq/         # ZeroMQ Reader/Writer/SyncReader/SyncWriter/NonBlocking*
├── utils.rs            # clock, DefaultOnce, iterators, RTP PTS mapper, UUID v7
│   ├── clock.rs
│   ├── default_once.rs
│   ├── iter.rs
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
 ├── MessageMeta (routing labels, seq_id, system_id, span context)
 └── MessageEnvelope
      ├── VideoFrame → VideoFrameProxy (Arc<RwLock<Box<VideoFrame>>>)
      │    ├── source_id, uuid, pts, dts, duration, codec, ...
      │    ├── content: VideoFrameContent (External | Internal | None)
      │    ├── transformations: Vec<VideoFrameTransformation>
      │    ├── attributes: AttributeSet
      │    └── objects: VideoObjectTree
      │         └── VideoObject
      │              ├── id, namespace, label, confidence
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
