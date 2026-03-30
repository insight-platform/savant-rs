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
├── message.rs        # WorkerMessage, OutputMessage
├── error.rs          # PicassoError enum
├── spec.rs           # Sub-module declarations
├── spec/
│   ├── general.rs    # GeneralSpec, EvictionDecision, PtsResetPolicy
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
`view.gst_buffer().make_mut()` to copy these values from the frame onto the buffer so
downstream consumers see correct metadata.

## Data Flow (Encode Path)
```
send_frame(source_id, VideoFrameProxy, SurfaceView, src_rect: Option<Rect>)
  → WorkerMessage::Frame(proxy, view, src_rect) via crossbeam bounded channel (capacity = GeneralSpec.inflight_queue_size, default 8; GeneralSpec.pts_reset_policy propagated to worker)
  → worker_loop receives
  → apply_frame_timestamps_to_buffer(frame, view.gst_buffer().make_mut())
  → WorkerState::process_frame
    → check encode_attribute gate (skip if missing)
    → match CodecSpec::Encode
      → ensure_encoder (lazy NvEncoder + DrainHandle creation)
      → process_encode:
         1. Lock encoder, get generator
         2. GPU affinity check (view.gpu_id() vs generator.gpu_id → GpuMismatch)
         3. GPU transform via `input.view.transform_into(&dst_view, config, src_rect)`
         4. Unlock encoder
         5. rewrite_frame_transformations (coordinate mapping)
         6. If on_gpumat active OR Skia rendering needed: wrap `dst_buf` in `SharedBuffer::from(dst_buf)`, create `SurfaceView::from_buffer(shared.clone(), 0)` for the entire encode scope
         7. Skia + on_gpumat order per CallbackInvocationOrder:
            - Skia receives `(data_ptr, pitch)` from the SurfaceView; `do_skia_render` uses `view.gst_buffer()` (returns `MutexGuard<gst::Buffer>`) and passes them to `load_from_nvbuf(data_ptr, pitch)` and `render_to_nvbuf_with_ptr(buf_ref, data_ptr, pitch, None)`
            - SkiaGpuMat: Skia (draw specs, load_from_nvbuf, draw, on_render, render_to_nvbuf_with_ptr) then on_gpumat(&SurfaceView)
            - GpuMatSkia: on_gpumat(&SurfaceView) then Skia
            - GpuMatSkiaGpuMat: on_gpumat(&SurfaceView) → Skia → on_gpumat(&SurfaceView)
            (each on_gpumat receives &SurfaceView + worker's cuda_stream; cudaStreamSynchronize after each)
        11. Drop view; pass `SharedBuffer` directly to submit
        12. Lock encoder, submit_frame(buffer, ...) — buffer from shared.into_buffer()
        13. Insert into pending_frames (only after successful submit)
        14. Drain thread pulls output independently
```

### Render Omission (Fast Path)
When `use_on_render=false` AND `draw` spec is empty for a source, `should_render`
is false, `render_opts` is `None`, and process_encode skips the entire Skia path:
- Always uses `generator.transform()` (no separate `transform_with_ptr` path)
- `SurfaceView` is created from `SharedBuffer` via `from_buffer(&shared, 0)` when either `on_gpumat` OR Skia rendering is needed; view is dropped before submit, then `shared.into_buffer()` extracts `gst::Buffer` for `submit_frame`
- Skia block skipped entirely: no EGL lock, no SkiaRenderer, no canvas
- Frame goes straight from GPU transform to encoder submit

## Data Flow (Bypass Path)
```
Frame → process_bypass:
  1. set_transcoding_method(Copy)
  2. transform_backward (revert bboxes to initial coordinates)
  3. fire on_bypass_frame(OutputMessage::VideoFrame(frame))
```

## Data Flow (Drop Path)
```
Frame → CodecSpec::Drop → log debug, return (buffer dropped)
```

## EOS Handling
- Drop: fire EOS sentinel via on_encoded_frame(OutputMessage::EndOfStream)
- Bypass: fire EOS sentinel via on_bypass_frame(OutputMessage::EndOfStream)
- Encode: stop drain thread → drain_remaining → encoder.finish(5s timeout) → fire callbacks for remaining frames → EOS sentinel
- After EOS, encoder + drain handle are set to None (re-created on next frame)

## CallbackInvocationOrder
Controls when the `on_gpumat` callback fires relative to Skia rendering:
- `SkiaGpuMat` (default): Skia render → `on_gpumat`
- `GpuMatSkia`: `on_gpumat` → Skia render
- `GpuMatSkiaGpuMat`: `on_gpumat` → Skia render → `on_gpumat`

After each `on_gpumat` invocation the worker's CUDA stream is synchronised before the next pipeline stage proceeds. Set via `SourceSpec::callback_order`.

## PTS Reset Handling
When a frame's PTS is not strictly greater than the previous frame's PTS (non-monotonic / backward jump), the `PtsResetPolicy` (from `GeneralSpec`) determines the recovery strategy:
- `EosOnDecreasingPts` (default): emit synthetic EOS (drain + flush encoder, fire EOS sentinel), then recreate the encoder. Downstream sees a clean EOS boundary.
- `RecreateOnDecreasingPts`: silently destroy and recreate the encoder without emitting EOS.

In both cases the `on_stream_reset` callback (if set) is fired with `StreamResetReason::PtsDecreased { last_pts_ns, new_pts_ns }` before the encoder is reset. The offending frame is then processed normally on the new encoder. `last_pts_ns` is cleared after reset so the next frame always succeeds.

## Shared Encoder State
- `SharedEncoder = Arc<parking_lot::Mutex<NvEncoder>>` — shared between worker and drain threads
- `SharedPendingFrames = Arc<parking_lot::Mutex<HashMap<u128, VideoFrameProxy>>>` — frame map shared with drain
- Worker locks encoder for: GPU transform, submit_frame (brief)
- Drain thread locks encoder for: pull_encoded (brief, non-blocking)
- `pending_frames` insert happens AFTER `submit_frame` succeeds (avoids leaking entries on encoder error)

## GPU Affinity
- `EncoderConfig.gpu_id` (default: 0) is the single source of truth per source
- Propagates to: NvEncoder generators, SkiaRenderer, buffer pools
- `BufferGenerator.gpu_id()` exposes the stored GPU ID
- `deepstream_buffers::buffer_gpu_id(buf)` extracts GPU ID from an NvBufSurface-backed buffer
- `process_encode` checks buffer GPU vs encoder GPU at entry; returns `PicassoError::GpuMismatch` on mismatch
- Check is fail-open: if `buffer_gpu_id` can't extract (e.g., non-NVMM stub buffer in tests), proceeds silently
- Transform (`NvBufSurfTransform`) reads GPU from the source buffer's `gpuId` field independently

## CUDA Stream Management
Picasso rejects external CUDA streams passed via `TransformConfig.cuda_stream` — `set_source_spec` returns `Err(PicassoError::ExternalCudaStream)` if the stream is not default. Each worker creates its own per-worker non-blocking CUDA stream to avoid global GPU serialization.

## PNG Encoding (CPU-based)
- Uses GStreamer pipeline: appsrc (NVMM RGBA) → nvvideoconvert → pngenc → appsink
- `pngenc` (gst-plugins-good) runs on CPU; nvvideoconvert converts NVMM to system memory
- Requires `VideoFormat::RGBA`; `PngProps` supports `compression_level` (0–9)

## Raw Pseudoencoders (RawRgba, RawRgb)
- Uses GStreamer pipeline: appsrc (NVMM) → nvvideoconvert → capsfilter(video/x-raw) → appsink
- Downloads GPU frames to CPU memory as tightly-packed RGBA or RGB pixel data
- On Jetson (aarch64): `nvvideoconvert` gets `compute-hw=1` to bypass VIC limitations for NV12→RGB/RGBA conversion
- Output `EncodedFrame.data` contains `width * height * bpp` bytes (4 for RGBA, 3 for RGB)
- Every frame is marked as keyframe; stride padding is stripped automatically

## Jetson / Platform-Specific Notes

### Encoder Properties are Platform-Dependent
The `nvv4l2h264enc` / `nvv4l2h265enc` elements have different property APIs on dGPU vs Jetson:
- **dGPU:** `preset-id` (DgpuPreset P1–P7), `tuning-info-id` (TuningPreset)
- **Jetson:** `preset-level` (JetsonPresetLevel), `maxperf-enable`

Tests and benchmarks MUST use the correct property variant. Use `cfg!(target_arch = "aarch64")` to select.

### NVENC Not Available on All Jetsons
Orin Nano does NOT have NVENC. Tests must guard with `nvidia_gpu_utils::has_nvenc(0)` and fall back to JPEG or PNG.

### nvjpegenc Jetson Surface Registration
On Jetson, `nvjpegenc` needs `nvvideoconvert` with `disable-passthrough=true` before it to avoid "Surface not registered" hangs.

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
- Held during: load_from_nvbuf, canvas draws, render_to_nvbuf_with_ptr
- Skia operations receive pre-resolved CUDA pointers `(data_ptr, pitch)` from the caller

## Skia Renderer Jetson Compatibility
On Jetson, `NvBufSurfaceMemType::Default` is VIC-managed (`SurfaceArray`) and
NOT directly CUDA-addressable from the runtime API. The Skia renderer's CUDA-GL
interop paths use **EGL-CUDA interop** (via `EglCudaMeta`) on aarch64 to obtain
zero-copy CUDA device pointers from VIC-managed surfaces. On dGPU, `dataPtr`
from `NvBufSurfaceParams` is used directly.

Both platforms perform `cudaMemcpy2DToArray` / `cudaMemcpy2DFromArray` with
`CUDA_MEMCPY_DEVICE_TO_DEVICE` — no CPU staging is involved.

API: The caller passes `(data_ptr, pitch)` from a `SurfaceView` (created via
`SurfaceView::from_buffer` or `SurfaceView::from_buffer`). On Jetson, the view triggers EGL-CUDA registration
when the pointer is first resolved. The Skia renderer's `load_from_nvbuf` takes
`(data_ptr, pitch)` directly; `from_nvbuf` takes `(width, height, gpu_id, data_ptr, pitch)`;
`render_to_nvbuf_with_ptr` is the primary API (takes `dst_buf`, `dst_ptr`, `dst_pitch`, `config`).
The `render_to_nvbuf` method keeps the `SurfaceView` alive during rendering to ensure the CUDA pointer remains valid for the full render lifetime.

## Key Invariants
- Frame's transformation chain must start with exactly `[InitialSize(w, h)]` before `rewrite_frame_transformations`
- VideoFrameProxy uses interior mutability (clone shares state via Arc)
- `pending_frames` insert is always AFTER successful `submit_frame` (prevents orphaned entries on encoder error)
- Drain thread callback must not block indefinitely (use buffered channels / `try_send` in benchmarks/consumers)
