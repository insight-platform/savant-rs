//! Memory leak smoke tests for SidecarNvInfer.

mod common;

use deepstream_nvbufsurface::{
    BatchedNvBufSurfaceGenerator, NvBufSurfaceGenerator, NvBufSurfaceMemType, TransformConfig,
    VideoFormat,
};
use nvidia_gpu_utils::gpu_mem_used_mib;
use serial_test::serial;
use sidecar_nvinfer::{SidecarConfig, SidecarNvInfer};

fn make_identity_batch(num_frames: u32) -> gstreamer::Buffer {
    common::init();

    let src_gen = NvBufSurfaceGenerator::builder(VideoFormat::RGBA, 12, 12)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(4)
        .max_buffers(4)
        .build()
        .expect("src generator");

    let batched_gen = BatchedNvBufSurfaceGenerator::new(
        VideoFormat::RGBA,
        12,
        12,
        16,
        2,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("batched generator");

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    for i in 0..num_frames {
        let src = src_gen.acquire_surface(Some(i as i64)).unwrap();
        batch.fill_slot(&src, None, Some(i as i64)).unwrap();
    }

    batch.finalize()
}

#[test]
#[serial]
fn test_memory_no_leak() {
    common::init();

    let onnx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/identity.onnx");
    if !onnx.exists() {
        eprintln!("Skipping: identity.onnx not found at {:?}", onnx);
        return;
    }

    let before = gpu_mem_used_mib(0).expect("gpu_mem_used_mib");

    let props = common::identity_properties();
    let config = SidecarConfig::new(props, "RGBA", 12, 12);
    let callback = Box::new(|_| {});
    let sidecar = SidecarNvInfer::new(config, callback).expect("create sidecar");

    for i in 0..20 {
        let batch = make_identity_batch(2);
        let _ = sidecar.infer_sync(batch, i);
    }

    drop(sidecar);

    let after = gpu_mem_used_mib(0).expect("gpu_mem_used_mib");
    assert!(
        after <= before + 32,
        "GPU memory should return near baseline: before={} MiB, after={} MiB",
        before,
        after
    );
}
