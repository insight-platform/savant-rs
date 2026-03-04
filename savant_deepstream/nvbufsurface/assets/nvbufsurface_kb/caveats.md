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

## 10. `TransformConfig` Is Move-Only in Loops

`TransformConfig` implements `Clone` but NOT `Copy`. In loops, create a
fresh `TransformConfig::default()` per iteration or call `.clone()`:
```rust
for _ in 0..N {
    let mut batch = gen.acquire_batched_surface(TransformConfig::default()).unwrap();
    // ...
}
```
