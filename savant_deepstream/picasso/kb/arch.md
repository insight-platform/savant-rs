# Picasso Architecture

## Module Tree
```
picasso/src/
├── lib.rs            # pub re-exports
├── prelude.rs        # convenience re-exports
├── engine.rs         # PicassoEngine: worker map, watchdog, dispatch
├── worker.rs         # SourceWorker: per-source thread, WorkerState, worker_loop
├── pipeline.rs       # FrameInput struct, sub-module declarations
├── pipeline/
│   ├── encode.rs     # GPU transform → Skia render → encode; DrainHandle for async drain
│   └── bypass.rs     # Bypass mode: transform_backward + callback
├── callbacks.rs      # Callback traits + Callbacks aggregate
├── message.rs        # WorkerMessage, EncodedOutput, BypassOutput
├── error.rs          # PicassoError enum
├── spec.rs           # Sub-module declarations
├── spec/
│   ├── general.rs    # GeneralSpec, EvictionDecision
│   ├── codec.rs      # CodecSpec (Drop/Bypass/Encode)
│   ├── source.rs     # SourceSpec (combines all facets)
│   ├── conditional.rs# ConditionalSpec (attribute gates)
│   └── draw.rs       # ObjectDrawSpec (HashMap<(ns,label), ObjectDraw>)
├── transform.rs      # compute_letterbox_params
├── watchdog.rs       # WatchdogSignal, spawn_watchdog (reaps dead workers)
├── skia.rs           # Skia rendering sub-modules
└── skia/
    ├── context.rs    # DrawContext: font/template cache; resolve_templates (cached) vs resolve_templates_ephemeral (callback-only)
    ├── common.rs     # ResolvedBBox
    ├── object.rs     # draw_object (dispatches to bbox/label/dot/blur)
    ├── bbox.rs       # draw_bounding_box
    ├── label.rs      # draw_label
    ├── dot.rs        # draw_dot
    └── blur.rs       # draw_blur
```

## Threading Model
```
Main thread
  └─ PicassoEngine
       ├─ watchdog thread (picasso-watchdog)
       │    periodically scans worker map, reaps dead workers
       └─ per-source worker threads (picasso-{source_id})
       │    each runs worker_loop:
       │      recv_timeout(idle_timeout)
       │      ├─ Frame → process_frame (Drop/Bypass/Encode)
       │      ├─ Eos → handle_eos (stop drain, flush encoder, fire EOS sentinel)
       │      ├─ UpdateSpec → hot-swap (stop drain + flush if codec changed)
       │      ├─ Shutdown → stop drain + flush + break
       │      └─ Timeout → on_eviction callback → KeepFor/Terminate/TerminateImmediately
       └─ per-source drain threads (picasso-drain-{source_id})
            spawned when encoder is created, stopped on EOS/shutdown/hot-swap
            continuously polls encoder.pull_encoded() in a loop (1ms sleep when idle)
            fires on_encoded_frame callbacks independent of frame submission
```

## Timestamp Source
pts, dts, time_base, and duration are taken from the [`VideoFrameProxy`], not from the
[`gst::Buffer`]. At pipeline entry, `apply_frame_timestamps_to_buffer` is called on
`view.buffer_mut().make_mut()` to copy these values from the frame onto the buffer so
downstream consumers see correct metadata.

## Data Flow (Encode Path)
```
send_frame(source_id, VideoFrameProxy, SurfaceView, src_rect: Option<Rect>)
  → WorkerMessage::Frame(proxy, view, src_rect) via crossbeam channel
  → worker_loop receives
  → apply_frame_timestamps_to_buffer(frame, view.buffer_mut().make_mut())
  → WorkerState::process_frame
    → check encode_attribute gate (skip if missing)
    → match CodecSpec::Encode
      → ensure_encoder (lazy NvEncoder + DrainHandle creation)
      → process_encode:
         1. Lock encoder, get generator
         2. GPU affinity check (view.gpu_id() vs generator.gpu_id → GpuMismatch)
         3. GPU transform via view.buffer() (generator.transform / transform_with_ptr)
         4. Unlock encoder
         5. rewrite_frame_transformations (coordinate mapping)
         6. (if render) resolve draw specs per object
         7. (if render) SkiaRenderer.load_from_nvbuf / from_nvbuf
         8. (if render) draw objects on Skia canvas
         9. (if render + use_on_render) fire on_render callback
        10. (if render) render_to_nvbuf (Skia → GPU surface)
        11. (if use_on_gpumat) fire on_gpumat callback
        12. Lock encoder, submit_frame
        13. Insert into pending_frames (only after successful submit)
        14. Drain thread pulls output independently
```

### Render Omission (Fast Path)
When `use_on_render=false` AND `draw` spec is empty for a source, `should_render`
is false, `render_opts` is `None`, and process_encode skips the entire Skia path:
- `need_ptr=false` → plain `generator.transform()` (no CUDA pointer overhead)
- Skia block (steps 6–10) skipped entirely: no EGL lock, no SkiaRenderer, no canvas
- Frame goes straight from GPU transform to encoder submit

## Data Flow (Bypass Path)
```
Frame → process_bypass:
  1. set_transcoding_method(Copy)
  2. transform_backward (revert bboxes to initial coordinates)
  3. fire on_bypass_frame(BypassOutput { source_id, frame, view })
```

## Data Flow (Drop Path)
```
Frame → CodecSpec::Drop → log debug, return (buffer dropped)
```

## EOS Handling
- Drop/Bypass: fire EOS sentinel via on_encoded_frame(EncodedOutput::EndOfStream)
- Encode: stop drain thread → drain_remaining → encoder.finish(5s timeout) → fire callbacks for remaining frames → EOS sentinel
- After EOS, encoder + drain handle are set to None (re-created on next frame)

## Shared Encoder State
- `SharedEncoder = Arc<parking_lot::Mutex<NvEncoder>>` — shared between worker and drain threads
- `SharedPendingFrames = Arc<parking_lot::Mutex<HashMap<u128, VideoFrameProxy>>>` — frame map shared with drain
- Worker locks encoder for: GPU transform, submit_frame (brief)
- Drain thread locks encoder for: pull_encoded (brief, non-blocking)
- `pending_frames` insert happens AFTER `submit_frame` succeeds (avoids leaking entries on encoder error)

## GPU Affinity
- `EncoderConfig.gpu_id` (default: 0) is the single source of truth per source
- Propagates to: NvEncoder generators, SkiaRenderer, buffer pools
- `NvBufSurfaceGenerator.gpu_id()` exposes the stored GPU ID
- `deepstream_nvbufsurface::buffer_gpu_id(buf)` extracts GPU ID from an NvBufSurface-backed buffer
- `process_encode` checks buffer GPU vs encoder GPU at entry; returns `PicassoError::GpuMismatch` on mismatch
- Check is fail-open: if `buffer_gpu_id` can't extract (e.g., non-NVMM stub buffer in tests), proceeds silently
- Transform (`NvBufSurfTransform`) reads GPU from the source buffer's `gpuId` field independently

## PNG Encoding (CPU-based)
- Uses GStreamer pipeline: appsrc (NVMM RGBA) → nvvideoconvert → pngenc → appsink
- `pngenc` (gst-plugins-good) runs on CPU; nvvideoconvert converts NVMM to system memory
- Requires `VideoFormat::RGBA`; `PngProps` supports `compression_level` (0–9)

## Spec Hot-Swap
- `set_source_spec` on existing worker → `WorkerMessage::UpdateSpec`
- If codec changed (Drop↔Bypass↔Encode, or encode dimensions/codec differ): stop drain thread, flush encoder, drop renderer
- Draw-only change (same codec): encoder + drain thread continue, only spec updated
- Font family change: rebuild DrawContext
- Always rebuild template cache

## OnObjectDrawSpec and Template Cache
When `OnObjectDrawSpec` returns a custom draw spec for an object, its `labelDraw.format`
must not pollute the persistent template cache. Callback-overridden formats are resolved
via `resolve_templates_ephemeral`, which parses on-the-fly and never writes to the cache.
Static-spec formats use `resolve_templates`, which may update the cache as a fallback.
This ensures per-object callback overrides do not affect other objects or future frames.

## Watchdog
- Runs in separate thread, wakes every `idle_timeout/2` seconds
- Scans all workers, removes dead ones from the map
- Shutdown: notified via condvar, exits immediately

## Skia EGL Lock
- `SKIA_EGL_LOCK`: process-global Mutex
- Serializes all SkiaRenderer operations (EGL contexts on same GPU corrupt each other)
- Held during: load_from_nvbuf, canvas draws, render_to_nvbuf

## Key Invariants
- Frame's transformation chain must start with exactly `[InitialSize(w, h)]` before `rewrite_frame_transformations`
- VideoFrameProxy uses interior mutability (clone shares state via Arc)
- `pending_frames` insert is always AFTER successful `submit_frame` (prevents orphaned entries on encoder error)
- Drain thread callback must not block indefinitely (use buffered channels / `try_send` in benchmarks/consumers)
