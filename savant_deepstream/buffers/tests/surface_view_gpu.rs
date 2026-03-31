//! GPU-level integration tests for [`SurfaceView`].
//!
//! All tests are `#[serial]` because:
//! - EGL-CUDA operations share per-process driver state (context, EGL display).
//! - Tracking counters (`REGISTRATIONS` / `DEREGISTRATIONS`) are global atomics
//!   that would be corrupted by concurrent `SurfaceView` construction.

mod common;

use deepstream_buffers::{
    ffi, BufferGenerator, ComputeMode, Interpolation, NvBufSurfaceMemType, Padding, Rect,
    SavantIdMetaKind, SurfaceView, TransformConfig, UniformBatchGenerator, VideoFormat,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serial_test::serial;

fn make_gen(pool_size: u32) -> BufferGenerator {
    BufferGenerator::builder(VideoFormat::RGBA, 64, 48)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(pool_size)
        .max_buffers(pool_size)
        .build()
        .expect("failed to build generator")
}

// ─── Core SurfaceView GPU tests ──────────────────────────────────────────────

#[test]
#[serial]
fn test_data_ptr_is_cuda_addressable() {
    common::init();
    let gen = make_gen(1);
    let shared = gen.acquire(None).unwrap();
    let view = SurfaceView::from_buffer(&shared, 0).unwrap();

    assert!(!view.data_ptr().is_null(), "data_ptr should not be null");
    assert!(view.pitch() > 0);

    let row_bytes = (view.width() * view.channels()) as usize;
    let mut host = vec![0u8; row_bytes];
    let rc = unsafe {
        ffi::cudaMemcpy2D(
            host.as_mut_ptr() as *mut std::ffi::c_void,
            row_bytes,
            view.data_ptr() as *const std::ffi::c_void,
            view.pitch() as usize,
            row_bytes,
            1,
            ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    };
    assert_eq!(rc, 0, "cudaMemcpy2D(D2H) failed with rc={rc}");
}

#[test]
#[serial]
fn test_write_read_roundtrip() {
    common::init();
    let gen = make_gen(1);
    let shared = gen.acquire(None).unwrap();
    let view = SurfaceView::from_buffer(&shared, 0).unwrap();

    view.memset(0xAB).expect("memset failed");

    let row_bytes = (view.width() * view.channels()) as usize;
    let total = row_bytes * view.height() as usize;
    let mut host = vec![0u8; total];

    let rc = unsafe {
        ffi::cudaMemcpy2D(
            host.as_mut_ptr() as *mut std::ffi::c_void,
            row_bytes,
            view.data_ptr() as *const std::ffi::c_void,
            view.pitch() as usize,
            row_bytes,
            view.height() as usize,
            ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    };
    assert_eq!(rc, 0, "cudaMemcpy2D(D2H) failed with rc={rc}");
    assert!(
        host.iter().all(|&b| b == 0xAB),
        "expected all bytes 0xAB, got first divergent at {:?}",
        host.iter().position(|&b| b != 0xAB)
    );
}

#[test]
#[serial]
fn test_recycled_buffer_keeps_mapping() {
    common::init();
    let gen = make_gen(1);

    let first_ptr;
    {
        let shared = gen.acquire(Some(1)).unwrap();
        let view = SurfaceView::from_buffer(&shared, 0).unwrap();
        first_ptr = view.data_ptr();
        assert!(!first_ptr.is_null());
    }

    {
        let shared = gen.acquire(Some(2)).unwrap();
        let view = SurfaceView::from_buffer(&shared, 0).unwrap();
        assert!(
            !view.data_ptr().is_null(),
            "recycled buffer should still have a valid data_ptr"
        );

        let row_bytes = (view.width() * view.channels()) as usize;
        let mut host = vec![0u8; row_bytes];
        let rc = unsafe {
            ffi::cudaMemcpy2D(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                row_bytes,
                view.data_ptr() as *const std::ffi::c_void,
                view.pitch() as usize,
                row_bytes,
                1,
                ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(
            rc, 0,
            "cudaMemcpy2D(D2H) on recycled buffer failed: rc={rc}"
        );
    }
}

#[test]
#[serial]
fn test_pitch_matches_surface() {
    common::init();
    let gen = make_gen(1);
    let shared = gen.acquire(None).unwrap();
    let view = SurfaceView::from_buffer(&shared, 0).unwrap();

    let min_pitch = view.width() * view.channels();
    assert!(
        view.pitch() >= min_pitch,
        "pitch {} < width*channels {}",
        view.pitch(),
        min_pitch
    );
}

#[test]
#[serial]
fn test_drop_view_releases_shared_ref() {
    common::init();
    let gen = make_gen(1);
    let shared = gen.acquire(None).unwrap();
    let view = SurfaceView::from_buffer(&shared, 0).unwrap();
    assert_eq!(view.slot_index(), 0);
    assert_eq!(shared.strong_count(), 2);
    drop(view);
    assert_eq!(shared.strong_count(), 1);
}

// ─── UniformBatchBuffer slot tests via SurfaceView ───────────────────────────

fn make_batched_gen(batch_size: u32) -> UniformBatchGenerator {
    UniformBatchGenerator::new(
        VideoFormat::RGBA,
        64,
        48,
        batch_size,
        2,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("failed to build batched generator")
}

fn make_src_gen() -> BufferGenerator {
    BufferGenerator::new(
        VideoFormat::RGBA,
        128,
        96,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("failed to build source generator")
}

#[test]
#[serial]
fn test_uniform_batch_slot_views_distinct() {
    common::init();
    let src_gen = make_src_gen();
    let batched_gen = make_batched_gen(4);
    let ids: Vec<_> = (0..4).map(|i| SavantIdMetaKind::Frame(i as u128)).collect();
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batch(config, ids).unwrap();

    for i in 0..4 {
        let src = src_gen.acquire(None).unwrap();
        let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
        batch.transform_slot(i as u32, &src_view, None).unwrap();
    }
    batch.finalize().unwrap();
    let shared = batch.shared_buffer();

    let views: Vec<_> = (0..4u32)
        .map(|i| {
            let view = SurfaceView::from_buffer(&shared, i).unwrap();
            assert!(!view.data_ptr().is_null(), "slot {i} data_ptr is null");
            assert_eq!(view.width(), 64);
            assert_eq!(view.height(), 48);
            assert_eq!(view.channels(), 4);
            assert_eq!(view.slot_index(), i);
            view
        })
        .collect();

    for i in 0..4 {
        for j in (i + 1)..4 {
            assert_ne!(
                views[i].data_ptr(),
                views[j].data_ptr(),
                "slot {i} and {j} should have distinct data_ptrs"
            );
        }
    }
}

#[test]
#[serial]
fn test_uniform_batch_slot_cuda_readback() {
    common::init();
    let src_gen = make_src_gen();
    let batched_gen = make_batched_gen(2);
    let ids: Vec<_> = (0..2).map(|i| SavantIdMetaKind::Frame(i as u128)).collect();
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batch(config, ids).unwrap();

    for i in 0..2 {
        let src = src_gen.acquire(None).unwrap();
        let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
        batch.transform_slot(i as u32, &src_view, None).unwrap();
    }
    batch.finalize().unwrap();
    let shared = batch.shared_buffer();

    for i in 0..2u32 {
        let view = SurfaceView::from_buffer(&shared, i).unwrap();
        let row_bytes = (view.width() * view.channels()) as usize;
        let mut host = vec![0u8; row_bytes];
        let rc = unsafe {
            ffi::cudaMemcpy2D(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                row_bytes,
                view.data_ptr() as *const std::ffi::c_void,
                view.pitch() as usize,
                row_bytes,
                1,
                ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(rc, 0, "slot {i}: cudaMemcpy2D(D2H) failed: rc={rc}");
    }
}

#[test]
#[serial]
fn test_uniform_batch_slot_out_of_bounds() {
    common::init();
    let src_gen = make_src_gen();
    let batched_gen = make_batched_gen(2);
    let config = TransformConfig::default();
    let mut batch = batched_gen
        .acquire_batch(config, vec![SavantIdMetaKind::Frame(1)])
        .unwrap();

    let src = src_gen.acquire(None).unwrap();
    let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
    batch.transform_slot(0, &src_view, None).unwrap();
    batch.finalize().unwrap();
    let shared = batch.shared_buffer();

    assert!(
        SurfaceView::from_buffer(&shared, 1).is_err(),
        "slot 1 should be out of bounds (numFilled=1)"
    );
}

#[test]
#[serial]
fn test_shared_buffer_strong_count_with_siblings() {
    common::init();
    let gen = make_gen(1);
    let shared = gen.acquire(None).unwrap();
    let view = SurfaceView::from_buffer(&shared, 0).unwrap();
    let sibling = view.shared_buffer();
    assert_eq!(shared.strong_count(), 3);
    drop(sibling);
    assert_eq!(shared.strong_count(), 2);
    drop(view);
    assert_eq!(shared.strong_count(), 1);
}

#[test]
#[serial]
fn test_upload_crop_transform_readback_consistency() {
    common::init();

    const SRC_W: u32 = 1920;
    const SRC_H: u32 = 1080;
    const DST_W: u32 = 600;
    const DST_H: u32 = 600;
    const CHANNELS: u32 = 4;
    const CROP_LEFT: u32 = 100;
    const CROP_TOP: u32 = 200;
    const ITERATIONS: u64 = 10;

    let src_gen = BufferGenerator::builder(VideoFormat::RGBA, SRC_W, SRC_H)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(1)
        .max_buffers(1)
        .build()
        .expect("failed to build source generator");

    let dst_gen = BufferGenerator::builder(VideoFormat::RGBA, DST_W, DST_H)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(1)
        .max_buffers(1)
        .build()
        .expect("failed to build destination generator");

    let compute_mode = if cfg!(target_arch = "aarch64") {
        ComputeMode::Gpu
    } else {
        ComputeMode::Default
    };
    let config = TransformConfig {
        padding: Padding::None,
        interpolation: Interpolation::Nearest,
        compute_mode,
        ..Default::default()
    };
    let crop = Rect::new(CROP_LEFT, CROP_TOP, DST_W, DST_H);

    let src_row_bytes = (SRC_W * CHANNELS) as usize;
    let dst_row_bytes = (DST_W * CHANNELS) as usize;

    for iter in 0..ITERATIONS {
        let mut src_host = vec![0u8; src_row_bytes * SRC_H as usize];
        let mut rng = StdRng::seed_from_u64(0xD15EA5E_u64 ^ iter);
        rng.fill(src_host.as_mut_slice());

        let src_shared = src_gen.acquire(Some(iter as u128)).unwrap();
        src_shared
            .with_view(0, |view| view.upload(&src_host, SRC_W, SRC_H, CHANNELS))
            .expect("upload failed");

        let dst_shared = dst_gen.acquire(Some((1000 + iter) as u128)).unwrap();
        let src_view = SurfaceView::from_buffer(&src_shared, 0).unwrap();
        let dst_view = SurfaceView::from_buffer(&dst_shared, 0).unwrap();
        src_view
            .transform_into(&dst_view, &config, Some(&crop))
            .expect("transform_into failed");

        let mut dst_host = vec![0u8; dst_row_bytes * DST_H as usize];
        let rc = unsafe {
            ffi::cudaMemcpy2D(
                dst_host.as_mut_ptr() as *mut std::ffi::c_void,
                dst_row_bytes,
                dst_view.data_ptr() as *const std::ffi::c_void,
                dst_view.pitch() as usize,
                dst_row_bytes,
                DST_H as usize,
                ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(
            rc, 0,
            "iteration {iter}: cudaMemcpy2D(D2H) failed with rc={rc}"
        );

        for y in 0..DST_H as usize {
            let src_start = (((CROP_TOP as usize + y) * SRC_W as usize) + CROP_LEFT as usize)
                * CHANNELS as usize;
            let src_end = src_start + dst_row_bytes;
            let dst_start = y * dst_row_bytes;
            let dst_end = dst_start + dst_row_bytes;
            assert_eq!(
                &dst_host[dst_start..dst_end],
                &src_host[src_start..src_end],
                "iteration {iter}: row {y} mismatch",
            );
        }
    }
}

// ─── EGL-CUDA map/unmap cycle verification (Jetson only) ─────────────────────

#[cfg(target_arch = "aarch64")]
mod map_unmap_cycle {
    use super::*;

    extern "C" {
        fn eglGetDisplay(display_id: *const std::ffi::c_void) -> *mut std::ffi::c_void;
        fn eglInitialize(display: *mut std::ffi::c_void, major: *mut i32, minor: *mut i32) -> u32;
    }

    #[test]
    #[serial]
    fn test_implicit_map_is_permanent() {
        common::init();

        unsafe {
            let display = eglGetDisplay(std::ptr::null());
            assert!(!display.is_null(), "eglGetDisplay failed");
            let mut major = 0i32;
            let mut minor = 0i32;
            eglInitialize(display, &mut major, &mut minor);
        }

        let gen = make_gen(1);
        let shared = gen.acquire(None).unwrap();
        let guard = shared.lock();
        let surf_ptr = unsafe { deepstream_buffers::extract_nvbufsurface(guard.as_ref()).unwrap() };

        unsafe {
            let rc = ffi::NvBufSurfaceMapEglImage(surf_ptr, 0);
            assert_eq!(rc, 0, "NvBufSurfaceMapEglImage failed: {rc}");

            let surf_list = (*surf_ptr).surfaceList;
            let egl_image: *mut std::ffi::c_void = std::ptr::read_volatile(
                &(*surf_list).mappedAddr.eglImage as *const *mut std::ffi::c_void,
            );
            assert!(!egl_image.is_null(), "eglImage is null");

            let mut resource: ffi::CUgraphicsResource = std::ptr::null_mut();
            let rc = ffi::cuGraphicsEGLRegisterImage(
                &mut resource,
                egl_image,
                ffi::CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE,
            );
            assert_eq!(rc, 0, "cuGraphicsEGLRegisterImage failed: {rc}");

            let rc = ffi::cuGraphicsUnmapResources(1, &mut resource, std::ptr::null_mut());
            assert_ne!(rc, 0, "cuGraphicsUnmapResources unexpectedly succeeded");

            let mut egl_frame = std::mem::MaybeUninit::<ffi::CUeglFrame>::zeroed();
            let rc =
                ffi::cuGraphicsResourceGetMappedEglFrame(egl_frame.as_mut_ptr(), resource, 0, 0);
            assert_eq!(rc, 0, "cuGraphicsResourceGetMappedEglFrame failed: {rc}");
            let egl_frame = egl_frame.assume_init();
            assert_eq!(egl_frame.frame_type, ffi::CU_EGL_FRAME_TYPE_PITCH);

            let data_ptr = egl_frame.frame.p_pitch[0];
            assert!(!data_ptr.is_null(), "CUDA pointer is null");

            let mut host = vec![0u8; 64];
            let rc = ffi::cudaMemcpy2D(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                64,
                data_ptr as *const std::ffi::c_void,
                egl_frame.pitch as usize,
                64,
                1,
                ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            assert_eq!(rc, 0, "cudaMemcpy2D(D2H) with mapped pointer failed: {rc}");

            ffi::cuGraphicsUnregisterResource(resource);
            ffi::NvBufSurfaceUnMapEglImage(surf_ptr, 0);
        }
    }
}

// ─── EGL-CUDA meta tracking (Jetson only) ────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod tracking {
    use super::*;

    #[test]
    #[serial]
    fn test_meta_deregistration_on_pool_destroy() {
        common::init();
        deepstream_buffers::egl_cuda_meta::reset_tracking();

        {
            let gen = make_gen(2);
            let shared_a = gen.acquire(Some(1)).unwrap();
            let shared_b = gen.acquire(Some(2)).unwrap();

            let _view_a = SurfaceView::from_buffer(&shared_a, 0).unwrap();
            let _view_b = SurfaceView::from_buffer(&shared_b, 0).unwrap();
        }

        let (reg, dereg) = deepstream_buffers::egl_cuda_meta::tracking_counts();
        assert_eq!(reg, 2, "expected exactly 2 registrations, got {reg}");
        assert_eq!(dereg, 2, "expected exactly 2 deregistrations, got {dereg}");
    }

    #[test]
    #[serial]
    fn test_meta_balanced_across_recycles() {
        common::init();
        deepstream_buffers::egl_cuda_meta::reset_tracking();

        let gen = make_gen(1);
        let n_cycles = 5;

        for i in 0..n_cycles {
            let shared = gen.acquire(Some(i as u128)).unwrap();
            let view = SurfaceView::from_buffer(&shared, 0).unwrap();
            assert!(!view.data_ptr().is_null(), "cycle {i}: null data_ptr");
        }

        let (reg_before_drop, _) = deepstream_buffers::egl_cuda_meta::tracking_counts();
        assert_eq!(
            reg_before_drop, 1,
            "expected 1 registration (POOLED meta survives recycles), got {reg_before_drop}"
        );

        drop(gen);
        let (reg, dereg) = deepstream_buffers::egl_cuda_meta::tracking_counts();
        assert_eq!(reg, 1, "expected 1 registration total, got {reg}");
        assert_eq!(
            dereg, 1,
            "expected 1 deregistration (pool destroy), got {dereg}"
        );
    }

    #[test]
    #[serial]
    fn test_registration_persists_across_views() {
        common::init();
        deepstream_buffers::egl_cuda_meta::reset_tracking();

        let gen = make_gen(1);

        let ptr_first;
        {
            let shared = gen.acquire(Some(1)).unwrap();
            let view = SurfaceView::from_buffer(&shared, 0).unwrap();
            ptr_first = view.data_ptr();
            assert!(!ptr_first.is_null());
        }

        let (reg1, _) = deepstream_buffers::egl_cuda_meta::tracking_counts();
        assert_eq!(reg1, 1, "first view should register once");

        {
            let shared2 = gen.acquire(Some(2)).unwrap();
            let view2 = SurfaceView::from_buffer(&shared2, 0).unwrap();
            assert_eq!(
                view2.data_ptr(),
                ptr_first,
                "same physical buffer should yield the same CUDA pointer"
            );
        }

        let (reg2, dereg2) = deepstream_buffers::egl_cuda_meta::tracking_counts();
        assert_eq!(
            reg2, 1,
            "second view on recycled buffer should NOT re-register"
        );
        assert_eq!(dereg2, 0, "no deregistrations should happen yet");

        drop(gen);
        let (_, dereg_final) = deepstream_buffers::egl_cuda_meta::tracking_counts();
        assert_eq!(dereg_final, 1, "pool destroy should deregister once");
    }

    #[test]
    #[serial]
    fn test_multi_slot_meta_lifecycle() {
        common::init();
        deepstream_buffers::egl_cuda_meta::reset_tracking();

        {
            let src_gen = make_src_gen();
            let batched_gen = make_batched_gen(4);
            let ids: Vec<_> = (0..4).map(|i| SavantIdMetaKind::Frame(i as u128)).collect();
            let config = TransformConfig::default();
            let mut batch = batched_gen.acquire_batch(config, ids).unwrap();

            for i in 0..4 {
                let src = src_gen.acquire(None).unwrap();
                let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
                batch.transform_slot(i as u32, &src_view, None).unwrap();
            }
            batch.finalize().unwrap();
            let shared = batch.shared_buffer();

            for i in 0..4u32 {
                let view = SurfaceView::from_buffer(&shared, i).unwrap();
                assert!(!view.data_ptr().is_null(), "slot {i} data_ptr is null");
            }

            let (reg, dereg) = deepstream_buffers::egl_cuda_meta::tracking_counts();
            assert_eq!(reg, 4, "expected 4 slot registrations, got {reg}");
            assert_eq!(dereg, 0, "no deregistrations yet while buffer alive");
        }

        let (reg, dereg) = deepstream_buffers::egl_cuda_meta::tracking_counts();
        assert_eq!(reg, 4, "registration count should not change after drop");
        assert_eq!(
            dereg, 4,
            "all 4 slots should be deregistered on buffer drop"
        );
    }
}
