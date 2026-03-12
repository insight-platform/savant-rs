# Critical Caveats & Design Decisions

## 1. `as_gst_buffer()` Returns a Self-Contained Wrapper — NOT the Pool Buffer

### Uniform batches (`DsNvUniformSurfaceBuffer`)
`as_gst_buffer()` does **not** return the pool-allocated buffer directly.
It builds a new **system-memory** buffer with:

1. `NvBufSurface` header + all `NvBufSurfaceParams` entries **inlined**
   contiguously (header immediately followed by params array).
2. `surfaceList` pointer set to point within the buffer's own GstMemory.
3. `GstParentBufferMeta` referencing the original pool buffer — keeps GPU
   memory alive without any CUDA copy.

**Why:** Pool-allocated buffers have `surfaceList` pointing to memory managed
by the NvDs allocator. If the user clones the buffer and calls `make_mut()`
(COW), the copy shares the NvBufSurface header via GstMemory but the
`surfaceList` pointer becomes **dangling** once the original buffer returns
to the pool (when the struct is dropped). By inlining the params, the
returned buffer is fully self-contained and safe to outlive the struct.

⚠ **Never** return `&self.buffer` or `self.buffer.clone()` directly for
uniform batches — the `surfaceList` will dangle after struct drop.

### Non-uniform batches (`DsNvNonUniformSurfaceBuffer`)
Already uses system memory with inlined `surfaceList`, so `as_gst_buffer()`
simply returns `self.buffer.clone()` (refcount+1). Safe because:
- `surfaceList` is inside the buffer's own GstMemory
- `GstParentBufferMeta` entries keep all source GPU buffers alive

### Return type
Both return `gst::Buffer` (owned), **not** `&gst::Buffer`.
Callers do NOT need `.clone()`.

---

## 2. `extract_slot_view()` Uses the Same Pattern

`extract_slot_view()` creates a system-memory buffer with `batchSize=1`,
`numFilled=1`, inlined `surfaceList[0]` copied from the batch, and
`GstParentBufferMeta` keeping the batch alive. Same self-contained design.

---

## 3. `finalize()` Is Non-Consuming — State Guards Apply

| State | Allowed | Blocked |
|---|---|---|
| Before `finalize()` | `fill_slot`/`add`, `slot_ptr` (uniform only) | `as_gst_buffer`, `extract_slot_view` |
| After `finalize()` | `as_gst_buffer`, `extract_slot_view`, `is_finalized` | `fill_slot`/`add` |

Calling a blocked operation returns `NvBufSurfaceError::NotFinalized` or
`NvBufSurfaceError::AlreadyFinalized`. Double-finalize also returns
`AlreadyFinalized`.

---

## 4. Pool Buffer Lifecycle (Uniform)

```
pool.acquire_buffer()
  → gst::Buffer (refcount=1, pool-owned)
  → stored in DsNvUniformSurfaceBuffer.buffer
  → fill_slot() calls NvBufSurfTransform to write GPU data
  → finalize() sets numFilled + SavantIdMeta on the buffer
  → as_gst_buffer() creates a wrapper with GstParentBufferMeta
  → struct dropped → buffer.refcount decreases
  → if refcount reaches 0 → buffer returns to pool
  → wrapper keeps parent ref alive → pool buffer stays until wrapper is dropped
```

⚠ The pool has a finite number of buffers (`pool_size`). If wrapper buffers
leak (parent ref never released), the pool exhausts and
`acquire_batched_surface` blocks/fails. Smoke tests use `pool_size=2` with
50 iterations to catch this.

---

## 5. `with_mut_buffer_ref` — Mutating Buffers from Python

The PyO3 layer uses `with_mut_buffer_ref()` for functions that need
`&mut gst::BufferRef` (`set_num_filled`, `set_buffer_pts`, etc.).

- **`DsNvBufSurfaceGstBuffer` path:** uses `gst::Buffer::make_mut()` (COW)
  so the caller always gets a writable reference regardless of refcount.
- **Raw `usize` pointer path:** checks `gst_mini_object_is_writable()` via
  FFI before calling `from_mut_ptr`. Returns a clear Python error if
  refcount > 1, advising the user to pass a `DsNvBufSurfaceGstBuffer`
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
cargo test -p deepstream_nvbufsurface

# Python tests (GPU required, full build)
SAVANT_FEATURES=deepstream make release install && make sp-pytest

# Clippy
cargo clippy --features deepstream --all-targets
```

---

## 10. Jetson vs dGPU Memory Access

- **Jetson (aarch64):** `NvBufSurfaceMemType::Default` maps to `SurfaceArray`
  (VIC-managed), which is **not** CUDA-addressable. Direct use of
  `cuMemsetD8_v2` / `cudaMemset2DAsync` / `cuMemcpyHtoD_v2` would fail
  with CUDA error 1 (`cudaErrorInvalidValue`). Instead, use
  `NvBufSurfaceMap` → CPU write → `NvBufSurfaceSyncForDevice` →
  `NvBufSurfaceUnMap`.

- **dGPU:** `Default` maps to `CudaDevice`, and `cuMemsetD8_v2` /
  `cudaMemset2DAsync` / `cuMemcpyHtoD_v2` work directly.

- **nvinfer on Jetson:** `scaling-compute-hw=1` is needed in nvinfer config to
  avoid VIC limitations with small surfaces (< 16×16 pixels).

- This platform difference is handled via `cfg(target_arch = "aarch64")` in:
  - `surface_ops` module (`memset_surface`, `upload_to_surface`)
  - `transform.rs` (`clear_surface_black` / `clear_surface_black_mapped` for
    letterbox padding)

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

## 14. upload_to_surface Takes 5 Arguments

```rust
pub unsafe fn upload_to_surface(
    buf: &gst::Buffer,
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,    // 4 for RGBA, 3 for RGB
) → Result<(), NvBufSurfaceError>
```

The `channels` parameter was added for multi-format support. Commonly
forgotten when calling from tests. Use `4` for RGBA, `3` for RGB.

Platform-aware internally: uses CUDA driver API (`cuMemcpyHtoD_v2`) on dGPU,
`NvBufSurfaceMap` → CPU write → `NvBufSurfaceSyncForDevice` on Jetson.

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

## 16. Skia Renderer Jetson Map/Unmap for CUDA-GL Interop

The Skia renderer (`skia_renderer.rs`) uses CUDA-GL interop to copy pixels
between NvBufSurface and an OpenGL texture (via `cudaArray`). On **dGPU**,
this is a direct GPU-to-GPU copy (`cudaMemcpyDeviceToDevice`). On **Jetson**
(`SurfaceArray`, VIC-managed), `dataPtr` is not CUDA-addressable, so the
same `cudaMemcpy2DToArray` / `cudaMemcpy2DFromArray` calls fail with
CUDA error 1.

**Fix:** Platform-aware paths via `cfg(target_arch = "aarch64")`:

- **Load (NvBuf → GL texture):** `NvBufSurfaceMap` → `NvBufSurfaceSyncForCpu`
  → `cudaMemcpy2DToArray(mappedAddr, ..., HostToDevice)` → `NvBufSurfaceUnMap`

- **Write (GL texture → NvBuf):** `NvBufSurfaceMap` →
  `cudaMemcpy2DFromArray(mappedAddr, ..., DeviceToHost)` →
  `NvBufSurfaceSyncForDevice` → `NvBufSurfaceUnMap`

API changes:
- `load_from_nvbuf` and `from_nvbuf` now accept `&gst::BufferRef` (not
  raw `data_ptr`/`pitch`) so the NvBufSurface can be mapped on Jetson.
- `copy_gl_to_nvbuf` accepts `*mut ffi::NvBufSurface` internally.
- `render_to_nvbuf_raw` is `#[cfg(not(target_arch = "aarch64"))]` (dGPU only,
  no callers in production code).

---

## 11. `TransformConfig` Is Move-Only in Loops

`TransformConfig` implements `Clone` but NOT `Copy`. In loops, create a
fresh `TransformConfig::default()` per iteration or call `.clone()`:
```rust
for _ in 0..N {
    let mut batch = gen.acquire_batched_surface(TransformConfig::default()).unwrap();
    // ...
}
```
