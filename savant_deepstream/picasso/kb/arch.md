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
    ├── context.rs    # DrawContext: font/template cache
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
[`gst::Buffer`]. At pipeline entry, `apply_frame_timestamps_to_buffer` copies these
values from the frame onto the buffer so downstream consumers see correct metadata.

## Data Flow (Encode Path)
```
send_frame(source_id, VideoFrameProxy, gst::Buffer)
  → WorkerMessage::Frame(proxy, buf) via crossbeam channel
  → worker_loop receives
  → apply_frame_timestamps_to_buffer(frame, buf)
  → WorkerState::process_frame
    → check encode_attribute gate (skip if missing)
    → match CodecSpec::Encode
      → ensure_encoder (lazy NvEncoder creation)
      → process_encode:
         1. GPU transform (NvBufSurfaceGenerator.transform)
         2. rewrite_frame_transformations (coordinate mapping)
         3. (if render) resolve draw specs per object
         4. (if render) SkiaRenderer.load_from_nvbuf / from_nvbuf
         5. (if render) draw objects on Skia canvas
         6. (if render + use_on_render) fire on_render callback
         7. (if render) render_to_nvbuf (Skia → GPU surface)
         8. (if use_on_gpumat) fire on_gpumat callback
         9. encoder.submit_frame (drain thread pulls output independently)
```

## Data Flow (Bypass Path)
```
Frame → process_bypass:
  1. set_transcoding_method(Copy)
  2. transform_backward (revert bboxes to initial coordinates)
  3. fire on_bypass_frame(BypassOutput { source_id, frame, buffer })
```

## Data Flow (Drop Path)
```
Frame → CodecSpec::Drop → log debug, return (buffer dropped)
```

## EOS Handling
- Drop/Bypass: fire EOS sentinel via on_encoded_frame(EncodedOutput::EndOfStream)
- Encode: stop drain thread → drain_remaining → encoder.finish(5s timeout) → fire callbacks for remaining frames → EOS sentinel
- After EOS, encoder + drain handle are set to None (re-created on next frame)

## Spec Hot-Swap
- `set_source_spec` on existing worker → `WorkerMessage::UpdateSpec`
- If codec changed (Drop↔Bypass↔Encode, or encode dimensions/codec differ): drain encoder, drop renderer
- Font family change: rebuild DrawContext
- Always rebuild template cache

## Watchdog
- Runs in separate thread, wakes every `idle_timeout/2` seconds
- Scans all workers, removes dead ones from the map
- Shutdown: notified via condvar, exits immediately

## Skia EGL Lock
- `SKIA_EGL_LOCK`: process-global Mutex
- Serializes all SkiaRenderer operations (EGL contexts on same GPU corrupt each other)
- Held during: load_from_nvbuf, canvas draws, render_to_nvbuf

## Key Invariant
- Frame's transformation chain must start with exactly `[InitialSize(w, h)]` before `rewrite_frame_transformations`
- VideoFrameProxy uses interior mutability (clone shares state via Arc)
