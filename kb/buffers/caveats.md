# Critical Caveats & Design Decisions

## 1. Buffer Access: `shared_buffer()` and `into_buffer()` — NOT Direct Pool Buffer

### Uniform batches (`SurfaceBatch`)
`as_gst_buffer()` has been **removed**. Use `shared_buffer()` to get a
`SharedBuffer` (cheap Arc clone) for shared access, or use
`into_buffer()` on the `SharedBuffer` to extract a `gst::Buffer`
when sole ownership is available.

Pool-allocated buffers have `surfaceList` pointing to memory managed by
the NvDs allocator. The `SharedBuffer` wraps the pool buffer
directly (no synthetic wrapper buffer needed). The pool buffer stays alive
as long as the `SharedBuffer` (and any `SurfaceView`s derived
from it) are alive.

### Non-uniform batches (`NonUniformBatch`)
`finalize(ids)` is **consuming** (`self`) and returns a `SharedBuffer`
directly. The buffer uses system memory with inlined `surfaceList` and
`GstParentBufferMeta` entries keeping all source GPU buffers alive.

### Extracting `gst::Buffer` for NvInfer / encoder
```rust
let shared = batch.shared_buffer();
drop(batch); // drop batch to release its Arc ref
let buf = shared.into_buffer().expect("sole owner");
// now pass `buf` to NvInfer::infer_sync() or NvEncoder::submit_frame()
```

⚠ `into_buffer()` fails if other references exist. Drop all `SurfaceView`s,
batch structs, and clones first.

---

## 2. Per-Slot Access: `SurfaceView::from_buffer(&shared, slot_index)`

`extract_slot_view()` has been **removed**. Use `SurfaceView::from_buffer`
instead, which takes a `&SharedBuffer` and a slot index, cloning
the Arc internally to keep the source buffer alive. Multiple views for
different slots can coexist — they all share the same underlying buffer.

```rust
let shared = batch.shared_buffer();
let view0 = SurfaceView::from_buffer(&shared, 0).unwrap();
let view1 = SurfaceView::from_buffer(&shared, 1).unwrap();
// view0 and view1 have distinct data_ptr for their respective slots
```

---

## 3. `finalize()` Signatures — Uniform vs Non-Uniform

### Uniform: `finalize(&mut self) -> Result<(), NvBufSurfaceError>`
Non-consuming. Sets `numFilled` from slots written via `transform_slot`.
`SavantIdMeta` is attached at `acquire_batch` time (ids passed there).
After finalize, `transform_slot` is blocked (`AlreadyFinalized`).
Double-finalize also returns `AlreadyFinalized`.

| State | Allowed | Blocked |
|---|---|---|
| Before `finalize()` | `transform_slot`, `view`, `shared_buffer` | — |
| After `finalize()` | `shared_buffer`, `view`, `is_finalized` | `transform_slot` |

### Non-uniform: `finalize(self, ids: Vec<SavantIdMetaKind>) -> Result<SharedBuffer>`
**Consuming** (`self`). Allocates the synthetic buffer, attaches
`GstParentBufferMeta` per source, and returns `SharedBuffer`.

---

## 4. Pool Buffer Lifecycle (Uniform)

```
pool.acquire()
  → gst::Buffer (refcount=1, pool-owned)
  → wrapped in SharedBuffer inside SurfaceBatch
  → transform_slot() calls NvBufSurfTransform to write GPU data
  → finalize(num_filled, ids) sets numFilled + SavantIdMeta
  → shared_buffer() → Arc clone of SharedBuffer
  → SurfaceView::from_buffer(&shared, i) → per-slot GPU view (Arc clone)
  → struct dropped → Arc ref decremented
  → when all SharedBuffer refs dropped → gst::Buffer refcount=0 → returns to pool
```

⚠ The pool has a finite number of buffers (`pool_size`). If
`SharedBuffer` refs leak (outstanding `SurfaceView`s, clones,
or batch structs not dropped), the pool exhausts and
`acquire_batch` blocks/fails. Smoke tests use `pool_size=2` with
50 iterations to catch this.

⚠ To extract `gst::Buffer` via `into_buffer()`, you must be the **sole**
Arc owner. `drop(batch)` before calling `shared.into_buffer()` if `shared`
was obtained via `batch.shared_buffer()`.

---

## 5. `with_mut_buffer_ref` — Mutating Buffers from Python

The PyO3 layer uses `with_mut_buffer_ref()` for functions that need
`&mut gst::BufferRef` (`set_num_filled`, `set_buffer_pts`, etc.).

- **`SharedBuffer` (PySharedBuffer) path:** uses `gst::Buffer::make_mut()` (COW)
  so the caller always gets a writable reference regardless of refcount.
- **Raw `usize` pointer path:** checks `gst_mini_object_is_writable()` via
  FFI before calling `from_mut_ptr`. Returns a clear Python error if
  refcount > 1, advising the user to pass a `SharedBuffer` (PySharedBuffer)
  instead.

⚠ **Never** call `BufferRef::from_mut_ptr()` on a raw pointer without
first checking `gst_mini_object_is_writable`. The gstreamer-rs
`debug_assert_ne!` produces the cryptic `left: 0, right: 0` message.

---

## 6. `NvBufSurface.surfaceList` Is an Absolute Pointer

The C struct `NvBufSurface` contains `NvBufSurfaceParams *surfaceList` —
a raw pointer. For pool-allocated buffers, this points to memory managed by
the NvDs allocator (outside the GstMemory). For self-contained buffers
(our wrappers, non-uniform batches, slot views), this points **within** the
buffer's GstMemory.

After `map_readable()` / `map_writable()` drops, the pointer remains valid
for NVMM memory (persistent mapping) and system memory (stable backing).
But `surfaceList` itself is a snapshot — if the underlying allocation is
reclaimed by the pool, the pointer dangles.

---

## 7. `GstParentBufferMeta` Propagation

`gst_buffer_copy` (triggered by `make_mut()` COW) propagates
`GstParentBufferMeta` to the copy via the meta's transform function.
This means COW copies also keep parent buffers alive. The ref chain
unwinds correctly when all copies are dropped.

---

## 8. `extract_nvbufsurface()` Returns a Pointer Valid Only During Map

```rust
pub unsafe fn extract_nvbufsurface(buf: &BufferRef) -> Result<*mut NvBufSurface, TransformError>
```

This maps the buffer, reads the pointer, then **drops the map**. The
returned pointer is technically valid only while the underlying GstMemory
exists (system memory is stable, NVMM has persistent mappings). But avoid
storing this pointer across operations that might free the buffer.

---

## 9. Build & Test Commands

```bash
# Rust tests (GPU required)
cargo test -p savant-deepstream-buffers

# Python tests (GPU required, full build)
SAVANT_FEATURES=deepstream make release install && make sp-pytest

# Clippy
cargo clippy --features deepstream --all-targets
```

---

## 10. Jetson vs dGPU Memory Access

- **Jetson (aarch64):** `NvBufSurfaceMemType::Default` maps to `SurfaceArray`
  (VIC-managed), which is **not** directly CUDA-addressable via `dataPtr`.
  CUDA access is obtained via **EGL-CUDA interop** (`EglCudaMeta`), which
  provides zero-copy CUDA device pointers from VIC-managed surfaces.

- **Permanent implicit mapping on Jetson:** `cuGraphicsEGLRegisterImage` creates
  a **permanent implicit mapping** — `cuGraphicsUnmapResources` returns error
  999. There is **no RAII map/unmap cycle**. The pointer is valid from
  registration until `cuGraphicsUnregisterResource`.

- **dGPU:** `Default` maps to `CudaDevice`, and `NvBufSurfaceParams::dataPtr`
  is directly CUDA-addressable.

- **Unified access:** `SurfaceView::from_buffer(buf, slot_index)` or
  `SurfaceView::from_buffer(&shared, slot_index)` are the canonical entry points.
  On Jetson they transparently attach `EglCudaMeta` per slot for permanent mapping.
  The POOLED flag ensures the meta survives GstBufferPool recycles — no re-registration
  on each acquisition.

- **Exception:** `clear_surface_black` in `transform.rs` still uses CPU-staging
  (`NvBufSurfaceMap` + memset + `NvBufSurfaceSyncForDevice` + `NvBufSurfaceUnMap`)
  on Jetson, because it receives raw `*mut NvBufSurface` pointers without
  `gst::Buffer` context.

- **nvinfer on Jetson:** `scaling-compute-hw=1` is needed in nvinfer config to
  avoid VIC limitations with small surfaces (< 16×16 pixels).

---

## 12. nvjpegenc Requires nvvideoconvert on Jetson

On Jetson, the NVJPG hardware engine requires surfaces to be "pinned" (registered)
through its own mechanism. Surfaces allocated by the NvDS buffer pool
(`gst_nvds_buffer_pool_new`) are **not** automatically registered, causing
`NVJPGGetSurfPinHandle: Surface not registered` errors at runtime. The encoder
silently fails and stops consuming buffers, which causes upstream `appsrc`
to block on back-pressure — **hanging the pipeline indefinitely**.

**Fix:** Insert `nvvideoconvert` with `disable-passthrough=true` before
`nvjpegenc`. This forces surface re-allocation through nvvideoconvert's own
pool, which creates surfaces compatible with the NVJPG engine. Without
`disable-passthrough=true`, nvvideoconvert operates in passthrough mode when
input/output caps match (same format, same resolution) and simply forwards
the original buffer unmodified.

⚠ This does NOT affect `nvv4l2h264enc` / `nvv4l2h265enc` — those V4L2-based
encoders handle NvDS buffer pool surfaces directly.

---

## 13. NVENC Availability and Orin Nano

- **Orin Nano (8GB and 4GB):** Does NOT have NVENC hardware encoding.
  `nvidia_gpu_utils::has_nvenc(0)` returns `false`.
- **Other Jetson models** (AGX Orin, Orin NX, Xavier, etc.): Have NVENC.
- **Datacenter dGPUs** (H100, A100, A30, B200, etc.): No NVENC.
  `has_nvenc()` uses NVML `encoder_capacity(H264)` to detect.

Tests that use `nvv4l2h264enc` / `nvv4l2h265enc` must guard with
`nvidia_gpu_utils::has_nvenc(0)` and skip (early return) when unavailable.
Tests that use `nvjpegenc` must guard with
`gst::ElementFactory::find("nvjpegenc")` (Jetson-only element).

---

## 14. SurfaceView::upload Takes 5 Arguments

```rust
pub fn upload(&self, data: &[u8], width: u32, height: u32, channels: u32)
    → Result<(), NvBufSurfaceError>
```

Method on `SurfaceView` (not a free function). The `channels` parameter is
commonly forgotten. Use `4` for RGBA, `3` for RGB. Uses CUDA driver API
for GPU upload on both dGPU and Jetson.

---

## 15. nvvideoconvert compute-hw Property on Jetson

On Jetson, the `nvvideoconvert` GStreamer element defaults to using the
Video Image Compositor (VIC) for format conversion. VIC does **not**
support certain conversions (e.g., NV12 → RGBA/RGB), causing:
`RGB/BGR Format transformation is not supported by VIC`

**Fix:** Set `compute-hw` to `"1"` on `nvvideoconvert` to force GPU-based
processing instead of VIC:
```rust
#[cfg(target_arch = "aarch64")]
nvconv.set_property_from_str("compute-hw", "1");
```

⚠ This property only exists on Jetson's `nvvideoconvert`. On dGPU, the
element does not have this property (it always uses GPU).
⚠ Only needed for raw pseudoencoders where `nvvideoconvert` handles
NV12/I420 → RGBA/RGB. NVENC codecs bypass `nvvideoconvert` by using
direct NvBufSurfTransform for format conversion.

---

## 16. Skia Renderer API

The Skia renderer (`skia_renderer.rs`) uses CUDA-GL interop to copy pixels
between NvBufSurface and an OpenGL texture (via `cudaArray`). It **no longer**
has `resolve_nvbuf_cuda_ptr` — pointer resolution is handled internally
by `render_to_nvbuf` or externally by the caller for the `*_with_ptr` / `*_raw` methods.

**API:**
- `load_from_nvbuf(data_ptr, pitch)` — takes `(data_ptr: *const c_void, pitch: usize)` directly (marked `unsafe`). Caller must obtain a valid CUDA pointer (e.g. from `SurfaceView::data_ptr()`).
- `from_nvbuf(width, height, gpu_id, data_ptr, pitch)` — convenience constructor (marked `unsafe`).
- `render_to_nvbuf(dst_buf, config)` — **convenience method** that creates a `SurfaceView` internally to resolve the CUDA pointer. **Keeps the view alive** during rendering to prevent use-after-free on Jetson (see caveat 20). Delegates to `render_to_nvbuf_with_ptr`.
- `render_to_nvbuf_with_ptr(dst_buf, dst_ptr, dst_pitch, config)` — **primary API** (marked `unsafe`). Caller supplies `(dst_ptr, dst_pitch)`. For the scaled path, creates `SurfaceView` internally for the temp buffer.
- `render_to_nvbuf_raw(data_ptr, pitch)` — direct 1:1 GPU copy, **available on ALL platforms** (no longer `#[cfg(not(aarch64))]`).

Both platforms use `cudaMemcpy2DToArray` / `cudaMemcpy2DFromArray` with
`CUDA_MEMCPY_DEVICE_TO_DEVICE` — no CPU staging. For `*_with_ptr` and
`*_raw`, the caller resolves the NvBufSurface CUDA pointer via `SurfaceView`
before calling these methods.

---

## 11. `TransformConfig` Is Move-Only in Loops

`TransformConfig` implements `Clone` but NOT `Copy`. In loops, create a
fresh `TransformConfig::default()` per iteration or call `.clone()`:
```rust
for _ in 0..N {
    let mut batch = gen.acquire_batch(TransformConfig::default(), vec![]).unwrap();
    // ...
}
```

---

## 17. SurfaceView: from_buffer, from_buffer

- **`from_buffer(buf, slot_index)`** — wraps buffer in `SharedBuffer`,
  resolves CUDA ptr for slot. Use for input buffers from generators or pipelines.

- **`from_buffer(&shared, slot_index)`** — primary constructor for batched buffers.
  Takes `&SharedBuffer`, clones the Arc internally. Create one
  `SurfaceView` per slot.

- **`into_gst_buffer()`** — extracts the underlying `gst::Buffer`. **Fails** (returns
  `Err(self)`) when other references exist (sibling views, `shared_buffer()` clones).
  Drop outstanding refs before retrying. Same semantics as `SharedBuffer::into_buffer`.

---

## 18. MAX_BATCH_SLOTS Limit (EglCudaMeta, aarch64)

`EglCudaMeta` supports up to `MAX_BATCH_SLOTS` (**64**) slots per buffer. If
`batchSize` exceeds this, `ensure_meta` returns an error.

This limit is **imposed by our meta layout**, not by Jetson hardware or the
NvBufSurface API: `batchSize` in `nvbufsurface.h` is a `uint32_t` with no documented
maximum. We embed `slots: [SlotRegistration; MAX_BATCH_SLOTS]` in `GstMeta` for a
fixed `gst_meta_register` size and to avoid heap-allocating the slot table per buffer.
Raising the constant only increases per-buffer meta overhead.

---

## 19. into_gst_buffer / SharedBuffer::into_buffer Fail When Other Refs Exist

Both `SurfaceView::into_gst_buffer()` and `SharedBuffer::into_buffer()`
return `Err(self)` when the Arc strong count is > 1. This happens when:
- Multiple `SurfaceView`s share the same buffer (e.g. via `from_buffer`)
- `shared_buffer()` was called and the clone is still alive
- The shared handle was cloned and passed elsewhere

**Fix:** Drop all sibling views and clones before calling `into_gst_buffer()` /
`into_buffer()`. Use `strong_count()` on `SharedBuffer` for diagnostics (1 = sole owner).

---

## 21. `bridge_savant_id_meta` — Per-PTS Overflow Is Not E2E-Testable

The bridge's per-PTS overflow guard (`bridge_savant_id_meta_across: per-PTS
limit (N) exceeded …`) **cannot be asserted from an e2e pipeline test** that
uses `appsrc → queue → appsink` with a BLOCK probe seal.

Why: `appsrc::push_buffer` is **asynchronous** — it just enqueues the buffer
into appsrc's internal queue. Capture probes fire on appsrc's streaming
thread, not the test thread. Once `seal.release()` is called, queue's
downstream restore probe starts popping entries concurrently with captures
still landing, so the per-PTS deque can never be forced to
`MAX_ENTRIES_PER_PTS + 1` deterministically.

The observed symptom is a non-monotonic `existing entries=` log sequence
— the counter grows, then dips, then grows again — and the overflow ERROR
log never fires even when pushing `MAX_ENTRIES_PER_PTS + 1` same-PTS buffers.

**Correct test location:** use the deterministic unit test
`bridge_map_per_pts_eviction` in
`savant_gstreamer/src/pipeline/bridge_meta.rs`, which drives
`capture_into_map()` (a `pub(crate)` helper extracted from the capture
probe) without any GStreamer threading. The e2e test in
`savant_deepstream/buffers/tests/bridge_meta_e2e.rs::bridge_meta_many_same_pts`
only asserts graceful handling (frame count, meta integrity).

---

## 20. Jetson EGL-CUDA Pointer Lifetime in render_to_nvbuf

On Jetson, `render_to_nvbuf` uses `from_glib_none` to temporarily increment
the GstBuffer refcount so it can create a `SurfaceView`. Because `from_glib_none`
creates a new owned `gst::Buffer` that shares the same underlying `GstMiniObject`,
the buffer now has refcount > 1. When `SurfaceView::from_buffer` calls
`make_mut()` on the inner `SharedBuffer`, GStreamer performs a **COW
(copy-on-write)** copy. The `EglCudaMeta` (which provides the CUDA device
pointer on Jetson) is attached to this **copy**, not the original buffer.

**Problem:** If the `SurfaceView` (and thus the COW copy) is dropped **before**
the CUDA device pointer is used for the actual render copy, the pointer becomes
**invalid** — the EGL-CUDA mapping was destroyed with the copy.

**Solution:** `render_to_nvbuf` keeps the `SurfaceView` alive until **after**
`render_to_nvbuf_with_ptr` completes, then drops it explicitly:

```rust
let view = SurfaceView::from_buffer(owned, 0)?;
let data_ptr = view.data_ptr();
let pitch = view.pitch() as usize;
let result = self.render_to_nvbuf_with_ptr(dst_buf, data_ptr, pitch, config);
drop(view);  // safe: pointer already consumed
result
```

⚠ This does NOT affect dGPU — `dataPtr` is directly CUDA-addressable and does
not depend on `EglCudaMeta`. The issue is specific to Jetson's VIC-managed
`SurfaceArray` memory type.
