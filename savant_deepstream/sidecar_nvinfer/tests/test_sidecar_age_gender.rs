//! Integration tests for SidecarNvInfer with the age_gender model (multi-output).

mod common;

use deepstream_nvbufsurface::{
    BatchedNvBufSurfaceGenerator, NvBufSurfaceGenerator, NvBufSurfaceMemType, TransformConfig,
    VideoFormat,
};
use sidecar_nvinfer::{SidecarConfig, SidecarNvInfer};
use std::path::PathBuf;

fn age_gender_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/age_gender_nvinfer.txt")
}

fn make_age_gender_batch(num_frames: u32) -> gstreamer::Buffer {
    common::init();

    let src_gen = NvBufSurfaceGenerator::builder(VideoFormat::RGBA, 112, 112)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(4)
        .max_buffers(4)
        .build()
        .expect("src generator");

    let batched_gen = BatchedNvBufSurfaceGenerator::new(
        VideoFormat::RGBA,
        112,
        112,
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
fn test_multi_output_layer_names() {
    common::init();

    let config_path = age_gender_config_path();
    if !config_path.exists() {
        eprintln!("Skipping: config not found at {:?}", config_path);
        return;
    }
    let onnx_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/age_gender_mobilenet_v2_dynBatch.onnx");
    if !onnx_path.exists() {
        eprintln!("Skipping: age_gender model not found at {:?}", onnx_path);
        return;
    }

    let config = SidecarConfig::new(config_path, "RGBA", 112, 112);
    let callback = Box::new(|_| {});
    let sidecar = SidecarNvInfer::new(config, callback).expect("create sidecar");

    let batch = make_age_gender_batch(1);
    let output = sidecar.infer_sync(batch, 1).expect("infer_sync");

    assert_eq!(output.num_elements(), 1);
    let elem = &output.elements()[0];
    assert!(
        elem.tensors.len() >= 2,
        "age_gender model should have at least 2 output tensors, got {}",
        elem.tensors.len()
    );
}
