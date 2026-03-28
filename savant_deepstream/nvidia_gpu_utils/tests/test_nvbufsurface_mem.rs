//! Integration test: verify that `gpu_mem_used_mib` reflects GPU memory changes
//! when allocating and releasing NvBufSurface buffers via `deepstream_buffers`.
//!
//! Run with: `cargo test -p nvidia_gpu_utils --test test_nvbufsurface_mem -- --nocapture`

use deepstream_buffers::{cuda_init, BufferGenerator, NvBufSurfaceMemType, VideoFormat};
use nvidia_gpu_utils::gpu_mem_used_mib;
use serial_test::serial;

#[test]
#[serial]
fn gpu_mem_reflects_surface_allocation() {
    gstreamer::init().unwrap();
    cuda_init(0).unwrap();

    let before = gpu_mem_used_mib(0).expect("gpu_mem_used_mib should succeed");

    // Allocate several large surfaces (10x 1920x1080 RGBA ≈ 80 MiB)
    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        1920,
        1080,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("Failed to create BufferGenerator");

    let buffers: Vec<_> = (0..10)
        .map(|i| gen.acquire(Some(i)).expect("acquire should succeed"))
        .collect();

    let during = gpu_mem_used_mib(0).expect("gpu_mem_used_mib should succeed");
    assert!(
        during > before,
        "GPU memory should increase after allocation: before={} MiB, during={} MiB",
        before,
        during
    );

    // Drop all surfaces
    drop(buffers);
    drop(gen);

    let after = gpu_mem_used_mib(0).expect("gpu_mem_used_mib should succeed");
    // Memory should return close to baseline (within tolerance for pool caching, etc.)
    assert!(
        after <= before + 32,
        "GPU memory should return near baseline after release: before={} MiB, after={} MiB",
        before,
        after
    );
}
