# Test Patterns & Templates

## Test Categories

All integration tests require a GPU (CUDA + NvBufSurface runtime).
Unit tests in `surface_view.rs` run without GPU using `gst::init()` only.

---

## Cargo.toml Dev Dependencies
```toml
[dev-dependencies]
env_logger = "0.11"
clap = { workspace = true }
ctrlc = { workspace = true }
nvidia_gpu_utils = { workspace = true }   # has_nvenc(), jetson_model() for test guards
```

## Test File Location
Integration tests: `savant_deepstream/nvbufsurface/tests/*.rs`
Unit tests: inline `#[cfg(test)] mod tests` in source files

---

## Common Helpers

### Shared Init (tests/common/mod.rs)
```rust
use deepstream_nvbufsurface::cuda_init;
use gstreamer as gst;
use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        let _ = env_logger::try_init();
        gst::init().unwrap();
        cuda_init(0).expect("Failed to initialize CUDA - is a GPU available?");
    });
}
```

All test files begin with `mod common;` and call `common::init()` in every test.

### Source Generator Helper
```rust
fn make_src_gen(format: VideoFormat, w: u32, h: u32) -> DsNvSurfaceBufferGenerator {
    DsNvSurfaceBufferGenerator::builder(format, w, h)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(4)
        .max_buffers(4)
        .build()
        .expect("failed to build source generator")
}
```

### memset_surface / upload_to_surface
```rust
// Create view first (by value for input buffers)
let view = SurfaceView::from_buffer(buf, 0).unwrap();

// Fill surface with zeros (no longer unsafe)
deepstream_nvbufsurface::memset_surface(&view, 0x00).unwrap();

// Upload RGBA pixel data (channels must match surface colour format)
let pixels: Vec<u8> = vec![0xFF; 640 * 480 * 4]; // white RGBA
deepstream_nvbufsurface::upload_to_surface(&view, &pixels, 640, 480, 4).unwrap();

// Recover buffer for downstream if sole owner
let buf = view.into_buffer().expect("no sibling views");
```

### Batched buffer usage with SharedMutableGstBuffer
```rust
// Wrap batch buffer in shared handle
let shared = SharedMutableGstBuffer::from(batch_buf);

// Create one SurfaceView per slot (borrow shared; clones Arc internally)
let view0 = SurfaceView::from_shared(&shared, 0).unwrap();
let view1 = SurfaceView::from_shared(&shared, 1).unwrap();

// Each view has distinct data_ptr for its slot
deepstream_nvbufsurface::memset_surface(&view0, 0x00).unwrap();
deepstream_nvbufsurface::memset_surface(&view1, 0xFF).unwrap();

// Pass buffer to encoder without consuming a view
let shared_for_encoder = view0.shared_buffer();
// ... submit shared_for_encoder.lock().as_ref() to encoder ...

// Extract buffer only when sole owner (drop sibling views first)
drop(view1);
let buf = view0.into_buffer().expect("sole owner after dropping view1");
```

---

### Batched Generator Helper
```rust
fn make_batched_gen(
    format: VideoFormat, w: u32, h: u32, batch: u32, pool: u32,
) -> DsNvUniformSurfaceBufferGenerator {
    DsNvUniformSurfaceBufferGenerator::new(
        format, w, h, batch, pool, 0, NvBufSurfaceMemType::Default,
    ).expect("failed to build batched generator")
}
```

### Build Uniform Batch (with timestamps and IDs)
```rust
fn build_uniform_batch(ids: &[i64]) -> SharedMutableGstBuffer {
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, ids.len() as u32, 2);
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
    let ids_vec: Vec<SavantIdMetaKind> = ids.iter().map(|&id| SavantIdMetaKind::Frame(id)).collect();
    for &id in ids {
        let src_shared = src_gen.acquire_buffer(Some(id)).unwrap();
        batch.fill_slot(&*src_shared.lock(), None, Some(id)).unwrap();
    }
    batch.finalize(ids.len() as u32, ids_vec).unwrap();
    let shared = batch.shared_buffer();
    {
        let mut guard = shared.lock();
        let buf_ref = guard.make_mut();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(1_000_000));
        buf_ref.set_dts(gst::ClockTime::from_nseconds(2_000_000));
        buf_ref.set_duration(gst::ClockTime::from_nseconds(33_333_333));
        buf_ref.set_offset(42);
        buf_ref.set_offset_end(43);
    }
    shared
}
```

### Build Heterogeneous Batch
```rust
fn build_heterogeneous_batch(resolutions: &[(u32, u32)], ids: &[i64]) -> SharedMutableGstBuffer {
    let mut batch = DsNvNonUniformSurfaceBuffer::new(0);
    for (i, &(w, h)) in resolutions.iter().enumerate() {
        let gen = make_src_gen(VideoFormat::RGBA, w, h);
        let shared = gen.acquire_buffer(Some(ids[i])).unwrap();
        let view = SurfaceView::from_shared(&shared, 0).unwrap();
        batch.add(&view, Some(ids[i])).unwrap();
    }
    let ids_vec: Vec<SavantIdMetaKind> = ids.iter().map(|&id| SavantIdMetaKind::Frame(id)).collect();
    let shared = batch.finalize(ids_vec).unwrap();
    {
        let mut guard = shared.lock();
        let buf_ref = guard.make_mut();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(5_000_000));
        buf_ref.set_dts(gst::ClockTime::from_nseconds(6_000_000));
        buf_ref.set_duration(gst::ClockTime::from_nseconds(16_666_667));
        buf_ref.set_offset(99);
        buf_ref.set_offset_end(100);
    }
    shared
}
```

---

## Test Templates

### Basic Slot View via SurfaceView::from_shared
```rust
#[test]
fn test_uniform_slot_view() {
    common::init();
    let shared = build_uniform_batch(&[10, 20, 30]);
    let view = SurfaceView::from_shared(&shared, 0).unwrap();

    assert!(!view.data_ptr().is_null());
    assert!(view.pitch() > 0);
    assert_eq!(view.width(), 640);
    assert_eq!(view.height(), 640);
}
```

### Safety: Buffer Valid After Struct Drop
```rust
#[test]
fn test_uniform_buffer_valid_after_struct_drop() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 3, 2);

    let shared = {
        let mut batch = batched_gen.acquire_batched_surface(TransformConfig::default()).unwrap();
        let ids: Vec<SavantIdMetaKind> = (0..3).map(|i| SavantIdMetaKind::Frame(i)).collect();
        for i in 0..3 {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(i)).unwrap();
        }
        batch.finalize(3, ids).unwrap();
        batch.shared_buffer()
        // batch dropped here — Arc still holds buffer
    };

    // Verify buffer is still valid via SurfaceView
    let view = SurfaceView::from_shared(&shared, 0).unwrap();
    assert!(!view.data_ptr().is_null());
    assert_eq!(view.width(), 640);
}
```

### Safety: Buffer Valid After COW
```rust
#[test]
fn test_uniform_buffer_valid_after_cow() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);

    let shared = {
        let mut batch = batched_gen.acquire_batched_surface(TransformConfig::default()).unwrap();
        let ids: Vec<SavantIdMetaKind> = (0..2).map(|i| SavantIdMetaKind::Frame(i)).collect();
        for i in 0..2 {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(i)).unwrap();
        }
        batch.finalize(2, ids).unwrap();
        let shared = batch.shared_buffer();
        {
            let mut guard = shared.lock();
            let buf_ref = guard.make_mut();
            buf_ref.set_pts(gst::ClockTime::from_nseconds(42));
        }
        shared
    };

    let view = SurfaceView::from_shared(&shared, 0).unwrap();
    assert!(!view.data_ptr().is_null());
}
```

### Leak Smoke Test (Uniform, pool_size=2)
```rust
#[test]
fn test_uniform_no_pool_leak() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 160, 160, 2, 2);

    for _ in 0..50 {
        let mut batch = batched_gen
            .acquire_batched_surface(TransformConfig::default())
            .unwrap();
        let ids: Vec<SavantIdMetaKind> = vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(2)];
        for _ in 0..2 {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(1)).unwrap();
        }
        batch.finalize(2, ids).unwrap();
        let _shared = batch.shared_buffer();
        // batch + _shared dropped here — pool buffers must return
    }
}
```

### Leak Smoke Test (COW variant)
```rust
#[test]
fn test_uniform_cow_no_pool_leak() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 160, 160, 2, 2);

    for _ in 0..50 {
        let mut batch = batched_gen
            .acquire_batched_surface(TransformConfig::default())
            .unwrap();
        let ids = vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(2)];
        for _ in 0..2 {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(1)).unwrap();
        }
        batch.finalize(2, ids).unwrap();
        let shared = batch.shared_buffer();
        drop(batch);
        {
            let mut guard = shared.lock();
            let buf_ref = guard.make_mut(); // trigger COW
            buf_ref.set_pts(gst::ClockTime::from_nseconds(1));
        }
        drop(shared);
    }
}
```

### Leak Smoke Test (Heterogeneous)
```rust
#[test]
fn test_heterogeneous_no_leak() {
    common::init();
    let gen = make_src_gen(VideoFormat::RGBA, 320, 240);

    for _ in 0..50 {
        let mut batch = DsNvNonUniformSurfaceBuffer::new(0);
        for _ in 0..2 {
            let buf = gen.acquire_surface(None).unwrap();
            let view = SurfaceView::from_buffer(buf, 0).unwrap();
            batch.add(&view, Some(1)).unwrap();
        }
        let ids = vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(2)];
        let _shared = batch.finalize(ids).unwrap();
    }
}
```

### State Guard Tests
```rust
#[test]
fn test_uniform_already_finalized_guards() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);
    let mut batch = batched_gen.acquire_batched_surface(TransformConfig::default()).unwrap();
    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, Some(1)).unwrap();
    batch.finalize(1, vec![SavantIdMetaKind::Frame(1)]).unwrap();

    assert!(batch.is_finalized());
    assert!(matches!(
        batch.finalize(1, vec![SavantIdMetaKind::Frame(1)]),
        Err(NvBufSurfaceError::AlreadyFinalized)
    ));
    assert!(matches!(
        batch.fill_slot(&src, None, Some(2)),
        Err(NvBufSurfaceError::AlreadyFinalized)
    ));
}
```

---

## Key Imports for Tests
```rust
use deepstream_nvbufsurface::{
    cuda_init,
    extract_nvbufsurface,
    DsNvNonUniformSurfaceBuffer,
    DsNvSurfaceBufferGenerator,
    DsNvUniformSurfaceBufferGenerator,
    NvBufSurfaceError,
    NvBufSurfaceMemType,
    SavantIdMeta,
    SavantIdMetaKind,
    SharedMutableGstBuffer,
    SurfaceView,
    TransformConfig,
    VideoFormat,
};
use gstreamer as gst;
```

---

## Running Tests

```bash
# All nvbufsurface tests (GPU required)
cargo test -p deepstream_nvbufsurface

# Single test file
cargo test -p deepstream_nvbufsurface --test slot_view

# Single test
cargo test -p deepstream_nvbufsurface --test slot_view test_uniform_extract_first_slot

# With logging
RUST_LOG=debug cargo test -p deepstream_nvbufsurface --test slot_view -- --nocapture

# Python tests (full build cycle)
SAVANT_FEATURES=deepstream make release install && make sp-pytest
```

⚠ Do **not** run `cargo test --features deepstream` on the entire workspace —
the `savant_rs` Python extension crate has linking issues in test mode.
Always test individual crates: `-p deepstream_nvbufsurface`, `-p nvinfer`.

---

## Hardware-Gated Test Pattern (bridge_meta.rs)

Encoder bridge tests must check hardware availability at the start:

```rust
#[test]
fn test_bridge_meta_nvv4l2h264enc() {
    // NVENC guard: Orin Nano and datacenter GPUs lack NVENC
    if !nvidia_gpu_utils::has_nvenc(0).unwrap_or(false) {
        eprintln!("NVENC not available — skipping");
        return;
    }
    run_pipeline_bridge_test(&EncoderTestConfig {
        format: VideoFormat::NV12,
        enc_name: "nvv4l2h264enc",
        parser: Some("h264parse"),
        pre_encoder: None,
    }, 30);
}

#[test]
fn test_bridge_meta_nvjpegenc() {
    common::init();  // GStreamer must be init'd before ElementFactory::find
    if gst::ElementFactory::find("nvjpegenc").is_none() {
        eprintln!("nvjpegenc not available — skipping");
        return;
    }
    run_pipeline_bridge_test(&EncoderTestConfig {
        format: VideoFormat::I420,
        enc_name: "nvjpegenc",
        parser: Some("jpegparse"),
        // On Jetson, nvjpegenc needs surfaces re-allocated by nvvideoconvert
        pre_encoder: if cfg!(target_arch = "aarch64") {
            Some("nvvideoconvert")
        } else {
            None
        },
    }, 30);
}
```

When inserting `nvvideoconvert` as `pre_encoder`, set `disable-passthrough=true`
on it — otherwise it passes through buffers when caps match, defeating the
purpose. `SavantIdMeta` survives through `nvvideoconvert` because it has a
proper `savant_id_meta_transform` function.

---

---

## SurfaceView GPU Test Templates (tests/surface_view_gpu.rs)

### CUDA Addressability
```rust
#[test]
fn test_data_ptr_is_cuda_addressable() {
    common::init();
    let gen = DsNvSurfaceBufferGenerator::new(
        VideoFormat::RGBA, 320, 240, 30, 1, 0, NvBufSurfaceMemType::Default,
    ).unwrap();
    let buf = gen.acquire_surface(Some(0)).unwrap();
    let view = SurfaceView::from_buffer(buf, 0).unwrap();
    assert!(!view.data_ptr().is_null());
    assert!(view.pitch() >= view.width() * view.channels());
    // Verify CUDA accessibility via cudaMemcpy2D readback
}
```

### Write/Read Roundtrip
```rust
#[test]
fn test_write_read_roundtrip() {
    common::init();
    let gen = /* ... */;
    let buf = gen.acquire_surface(Some(0)).unwrap();
    let view = SurfaceView::from_buffer(buf, 0).unwrap();
    deepstream_nvbufsurface::memset_surface(&view, 0xAB).unwrap();
    // cudaMemcpy2D readback and verify bytes == 0xAB
}
```

### Uniform Batch Slot Views
```rust
#[test]
fn test_uniform_batch_slot_views_distinct() {
    common::init();
    let src_gen = /* ... */;
    let batch_gen = /* ... */;
    let mut batch = batch_gen.acquire_batched_surface(TransformConfig::default()).unwrap();
    // Fill slots, finalize(num_filled, ids)
    let shared = batch.shared_buffer();
    // Create one SurfaceView per slot via from_shared (takes &shared)
    let views: Vec<_> = (0..N)
        .map(|i| SurfaceView::from_shared(&shared, i).unwrap())
        .collect();
    // Assert distinct data_ptr across views
}
```

### EglCudaMeta Tracking Tests (aarch64 only)
```rust
#[cfg(target_arch = "aarch64")]
mod tracking {
    static LOCK: Mutex<()> = Mutex::new(());  // Serialize tracking tests

    #[test]
    fn test_meta_deregistration_on_pool_destroy() {
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        common::init();
        let (base_reg, base_dereg) = deepstream_nvbufsurface::egl_cuda_meta::tracking_counts();
        // Create pool, acquire buffer, create SurfaceView (triggers registration)
        // Drop everything
        // Assert: new_reg >= 2, new_dereg >= 2 (relaxed — concurrent tests may
        // inject extra registrations; we only assert our minimum)
    }
}
```

**Note:** The `map_unmap_cycle` verification test module (`tests/surface_view_gpu.rs`)
confirms Jetson behavior: `cuGraphicsEGLRegisterImage` creates an implicit permanent
mapping; `cuGraphicsUnmapResources` returns error 999. No RAII map/unmap cycle.

---

## Benchmarks

See `benches/surface_view_mapping.rs`:
- `bench_registration_plus_first_view` — fresh buffer (includes EGL-CUDA registration on Jetson)
- `bench_recycled_buffer_view` — recycled pool buffer; POOLED meta survives recycle, no re-registration

Use `from_buffer(buf, 0)` for input buffers — no `wrap` workaround needed. POOLED meta
prevents re-registration on recycle.

```bash
cargo bench -p deepstream_nvbufsurface --bench surface_view_mapping
```

---

## Timing Guidance
- Integration tests complete in <1s per test (GPU transforms are fast)
- Leak smoke tests (50 iterations, small pools) complete in ~2-5s
- No sleeps needed — all operations are synchronous
