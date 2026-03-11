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
// Fill surface with zeros
unsafe { deepstream_nvbufsurface::memset_surface(&buf, 0x00).unwrap(); }

// Upload RGBA pixel data
let pixels: Vec<u8> = vec![0xFF; 640 * 480 * 4]; // white RGBA
unsafe { deepstream_nvbufsurface::upload_to_surface(&buf, &pixels, 640, 480).unwrap(); }
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
fn build_uniform_batch(ids: &[i64]) -> gst::Buffer {
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, ids.len() as u32, 2);
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
    for &id in ids {
        let src = src_gen.acquire_surface(None).unwrap();
        batch.fill_slot(&src, None, Some(id)).unwrap();
    }
    batch.finalize().unwrap();
    let mut buf = batch.as_gst_buffer().unwrap();
    {
        let buf_ref = buf.make_mut();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(1_000_000));
        buf_ref.set_dts(gst::ClockTime::from_nseconds(2_000_000));
        buf_ref.set_duration(gst::ClockTime::from_nseconds(33_333_333));
        buf_ref.set_offset(42);
        buf_ref.set_offset_end(43);
    }
    buf
}
```

### Build Heterogeneous Batch
```rust
fn build_heterogeneous_batch(resolutions: &[(u32, u32)], ids: &[i64]) -> gst::Buffer {
    let mut batch = DsNvNonUniformSurfaceBuffer::new(resolutions.len() as u32, 0).unwrap();
    for (i, &(w, h)) in resolutions.iter().enumerate() {
        let gen = make_src_gen(VideoFormat::RGBA, w, h);
        let buf = gen.acquire_surface(None).unwrap();
        batch.add(&buf, Some(ids[i])).unwrap();
    }
    batch.finalize().unwrap();
    let mut buf = batch.as_gst_buffer().unwrap();
    {
        let buf_ref = buf.make_mut();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(5_000_000));
        buf_ref.set_dts(gst::ClockTime::from_nseconds(6_000_000));
        buf_ref.set_duration(gst::ClockTime::from_nseconds(16_666_667));
        buf_ref.set_offset(99);
        buf_ref.set_offset_end(100);
    }
    buf
}
```

---

## Test Templates

### Basic Slot View Extraction
```rust
#[test]
fn test_uniform_extract_first_slot() {
    common::init();
    let batch = build_uniform_batch(&[10, 20, 30]);
    let view = extract_slot_view(&batch, 0).unwrap();

    let surf_ptr = unsafe { extract_nvbufsurface(view.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.batchSize, 1);
    assert_eq!(surf.numFilled, 1);

    let params = unsafe { &*surf.surfaceList };
    assert!(!params.dataPtr.is_null());
    assert!(params.pitch > 0);
    assert_eq!(params.width, 640);
    assert_eq!(params.height, 640);
}
```

### Safety: Buffer Valid After Struct Drop
```rust
#[test]
fn test_uniform_buffer_valid_after_struct_drop() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 3, 2);

    let buf = {
        let mut batch = batched_gen.acquire_batched_surface(TransformConfig::default()).unwrap();
        for _ in 0..3 {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(1)).unwrap();
        }
        batch.finalize().unwrap();
        batch.as_gst_buffer().unwrap()
        // batch dropped here — pool buffer returned
    };

    // Verify buffer is still valid
    let surf_ptr = unsafe { extract_nvbufsurface(buf.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.numFilled, 3);
    for i in 0..3 {
        let params = unsafe { &*surf.surfaceList.add(i) };
        assert!(!params.dataPtr.is_null());
        assert_eq!(params.width, 640);
    }
}
```

### Safety: Buffer Valid After COW
```rust
#[test]
fn test_uniform_buffer_valid_after_cow() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);

    let buf = {
        let mut batch = batched_gen.acquire_batched_surface(TransformConfig::default()).unwrap();
        for _ in 0..2 {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(1)).unwrap();
        }
        batch.finalize().unwrap();
        let mut b = batch.as_gst_buffer().unwrap();
        // Force COW
        let buf_ref = b.make_mut();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(42));
        b
    };

    let surf_ptr = unsafe { extract_nvbufsurface(buf.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    for i in 0..2 {
        let params = unsafe { &*surf.surfaceList.add(i) };
        assert!(!params.dataPtr.is_null());
    }
}
```

### Leak Smoke Test (Uniform, pool_size=2)
```rust
#[test]
fn test_uniform_as_gst_buffer_no_pool_leak() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 160, 160, 2, 2);

    for _ in 0..50 {
        let mut batch = batched_gen
            .acquire_batched_surface(TransformConfig::default())
            .unwrap();
        for _ in 0..2 {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(1)).unwrap();
        }
        batch.finalize().unwrap();
        let _buf = batch.as_gst_buffer().unwrap();
        // batch + _buf dropped here — pool buffers must return
    }
}
```

### Leak Smoke Test (COW variant)
```rust
#[test]
fn test_uniform_as_gst_buffer_cow_no_pool_leak() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 160, 160, 2, 2);

    for _ in 0..50 {
        let mut batch = batched_gen
            .acquire_batched_surface(TransformConfig::default())
            .unwrap();
        for _ in 0..2 {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(1)).unwrap();
        }
        batch.finalize().unwrap();
        let mut buf = batch.as_gst_buffer().unwrap();
        drop(batch);
        {
            let buf_ref = buf.make_mut(); // trigger COW
            buf_ref.set_pts(gst::ClockTime::from_nseconds(1));
        }
        drop(buf);
    }
}
```

### Leak Smoke Test (Heterogeneous)
```rust
#[test]
fn test_heterogeneous_as_gst_buffer_no_leak() {
    common::init();
    let gen = make_src_gen(VideoFormat::RGBA, 320, 240);

    for _ in 0..50 {
        let mut batch = DsNvNonUniformSurfaceBuffer::new(2, 0).unwrap();
        for _ in 0..2 {
            let buf = gen.acquire_surface(None).unwrap();
            batch.add(&buf, Some(1)).unwrap();
        }
        batch.finalize().unwrap();
        let _buf = batch.as_gst_buffer().unwrap();
    }
}
```

### State Guard Tests
```rust
#[test]
fn test_uniform_not_finalized_guards() {
    common::init();
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);
    let batch = batched_gen
        .acquire_batched_surface(TransformConfig::default())
        .unwrap();

    assert!(matches!(batch.as_gst_buffer(), Err(NvBufSurfaceError::NotFinalized)));
    assert!(matches!(batch.extract_slot_view(0), Err(NvBufSurfaceError::NotFinalized)));
    assert!(!batch.is_finalized());
}

#[test]
fn test_uniform_already_finalized_guards() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);
    let mut batch = batched_gen.acquire_batched_surface(TransformConfig::default()).unwrap();
    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, Some(1)).unwrap();
    batch.finalize().unwrap();

    assert!(batch.is_finalized());
    assert!(matches!(batch.finalize(), Err(NvBufSurfaceError::AlreadyFinalized)));
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
    extract_slot_view,
    DsNvNonUniformSurfaceBuffer,
    DsNvSurfaceBufferGenerator,
    DsNvUniformSurfaceBufferGenerator,
    NvBufSurfaceError,
    NvBufSurfaceMemType,
    SavantIdMeta,
    SavantIdMetaKind,
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

## Timing Guidance
- Integration tests complete in <1s per test (GPU transforms are fast)
- Leak smoke tests (50 iterations, small pools) complete in ~2-5s
- No sleeps needed — all operations are synchronous
