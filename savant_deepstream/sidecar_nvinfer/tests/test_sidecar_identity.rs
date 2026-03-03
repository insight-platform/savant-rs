//! Integration tests for SidecarNvInfer with the identity model.

mod common;

use deepstream_nvbufsurface::{
    BatchedNvBufSurfaceGenerator, NvBufSurfaceGenerator, NvBufSurfaceMemType, TransformConfig,
    VideoFormat,
};
use sidecar_nvinfer::{attach_batch_meta, DataType, SidecarConfig, SidecarNvInfer};

#[link(name = "cuda")]
extern "C" {
    fn cuMemsetD8_v2(dst: u64, value: u8, count: usize) -> u32;
}

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

/// Build a batch where every source surface is memset to `fill_byte`.
///
/// Returns the batch buffer and the expected per-element output sum.
/// The identity model with `net-scale-factor=1.0` converts each RGBA u8 pixel
/// to RGB float32: `output_value = pixel_byte * 1.0`.
/// Shape is [3, 12, 12] = 432 elements, each equal to `fill_byte as f64`.
fn make_identity_batch_known(num_frames: u32, fill_byte: u8) -> (gstreamer::Buffer, f64) {
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
        let (src, data_ptr, pitch) = src_gen.acquire_surface_with_ptr(Some(i as i64)).unwrap();
        // Fill the entire GPU allocation (pitch × height) with the known byte.
        let fill_size = (pitch * 12) as usize;
        let ret = unsafe { cuMemsetD8_v2(data_ptr as u64, fill_byte, fill_size) };
        assert_eq!(ret, 0, "cuMemsetD8_v2 failed with code {}", ret);
        batch.fill_slot(&src, None, Some(i as i64)).unwrap();
    }

    let expected_sum = 3.0 * 12.0 * 12.0 * (fill_byte as f64);
    (batch.finalize(), expected_sum)
}

fn identity_sidecar() -> Option<SidecarNvInfer> {
    let config_path = common::identity_config_path();
    if !config_path.exists() {
        eprintln!("Skipping: config not found at {:?}", config_path);
        return None;
    }
    let config = SidecarConfig::new(config_path, "RGBA", 12, 12);
    let callback = Box::new(|_| {});
    Some(SidecarNvInfer::new(config, callback).expect("create sidecar"))
}

/// Sum all f32 values in a host tensor slice.
fn tensor_f32_sum(tensor: &sidecar_nvinfer::TensorView) -> f64 {
    let slice: &[f32] = unsafe { tensor.as_slice() };
    slice.iter().map(|&v| v as f64).sum()
}

#[test]
fn test_attach_batch_meta() {
    common::init();
    let mut buffer = make_identity_batch(2);
    let buf_ref = buffer.get_mut().expect("buffer writable");
    attach_batch_meta(buf_ref, 2, 16).unwrap();
}

#[test]
fn test_sync_single_frame() {
    common::init();
    let sidecar = match identity_sidecar() {
        Some(s) => s,
        None => return,
    };

    let (batch, expected_sum) = make_identity_batch_known(1, 128);
    let output = sidecar.infer_sync(batch, 1).expect("infer_sync");

    assert_eq!(output.batch_id(), 1);
    assert_eq!(output.num_elements(), 1);

    let elem = &output.elements()[0];
    assert!(
        !elem.tensors.is_empty(),
        "identity model must produce at least one output tensor"
    );

    let t = &elem.tensors[0];
    assert_eq!(
        t.data_type,
        DataType::Float,
        "identity output should be f32"
    );
    assert_eq!(
        t.dims.dimensions,
        vec![3, 12, 12],
        "expected [3,12,12] output shape"
    );
    assert_eq!(t.dims.num_elements, 432);
    assert_eq!(t.byte_length, 432 * 4);

    let actual_sum = tensor_f32_sum(t);
    let rel_err = (actual_sum - expected_sum).abs() / expected_sum;
    assert!(
        rel_err < 0.01,
        "identity output sum should match input sum: expected={expected_sum}, actual={actual_sum}, rel_err={rel_err:.6}"
    );
}

#[test]
fn test_sync_uniform_batch() {
    common::init();
    let sidecar = match identity_sidecar() {
        Some(s) => s,
        None => return,
    };

    let (batch, expected_sum) = make_identity_batch_known(4, 200);
    let output = sidecar.infer_sync(batch, 42).expect("infer_sync");

    assert_eq!(output.batch_id(), 42);
    assert_eq!(output.num_elements(), 4);

    for (i, elem) in output.elements().iter().enumerate() {
        assert!(
            !elem.tensors.is_empty(),
            "element {i} must have output tensors"
        );
        let t = &elem.tensors[0];
        assert_eq!(t.data_type, DataType::Float);
        assert_eq!(t.dims.num_elements, 432);

        let actual_sum = tensor_f32_sum(t);
        let rel_err = (actual_sum - expected_sum).abs() / expected_sum;
        assert!(
            rel_err < 0.01,
            "element {i}: expected_sum={expected_sum}, actual_sum={actual_sum}, rel_err={rel_err:.6}"
        );
    }
}

#[test]
fn test_identity_different_fill_values() {
    common::init();
    let sidecar = match identity_sidecar() {
        Some(s) => s,
        None => return,
    };

    for fill in [64u8, 128, 200] {
        let (batch, expected_sum) = make_identity_batch_known(1, fill);
        let output = sidecar.infer_sync(batch, fill as u64).expect("infer_sync");
        let actual_sum = tensor_f32_sum(&output.elements()[0].tensors[0]);
        let rel_err = (actual_sum - expected_sum).abs() / expected_sum;
        assert!(
            rel_err < 0.01,
            "fill={fill}: expected={expected_sum}, actual={actual_sum}, rel_err={rel_err:.6}"
        );
    }
}

#[test]
fn test_element_ids_preserved() {
    common::init();
    let sidecar = match identity_sidecar() {
        Some(s) => s,
        None => return,
    };

    // Use non-trivial IDs with gaps and negative values to prove the actual
    // user-supplied IDs propagate, not just frame indices.
    let ids_to_send: Vec<i64> = vec![42, -7, 1000, 0];

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
    for &id in &ids_to_send {
        let src = src_gen.acquire_surface(Some(id)).unwrap();
        batch.fill_slot(&src, None, Some(id)).unwrap();
    }
    let buffer = batch.finalize();

    let output = sidecar.infer_sync(buffer, 99).expect("infer_sync");

    assert_eq!(output.batch_id(), 99);
    assert_eq!(output.num_elements(), ids_to_send.len());
    let received_ids: Vec<Option<i64>> = output.elements().iter().map(|e| e.id).collect();
    assert_eq!(
        received_ids,
        ids_to_send.iter().map(|&id| Some(id)).collect::<Vec<_>>(),
        "IDs must propagate end-to-end through nvinfer"
    );
}

#[test]
fn test_async_callback() {
    common::init();

    let config_path = common::identity_config_path();
    if !config_path.exists() {
        eprintln!("Skipping: config not found at {:?}", config_path);
        return;
    }

    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    let received = Arc::new(AtomicU64::new(0));
    let received_clone = received.clone();

    let config = SidecarConfig::new(config_path, "RGBA", 12, 12).queue_depth(2);
    let callback = Box::new(move |output: sidecar_nvinfer::BatchInferenceOutput| {
        received_clone.store(output.batch_id(), Ordering::SeqCst);
    });
    let sidecar = SidecarNvInfer::new(config, callback).expect("create sidecar");

    for batch_id in 1..=3u64 {
        let batch = make_identity_batch(1);
        sidecar.submit(batch, batch_id).expect("submit");
    }

    std::thread::sleep(std::time::Duration::from_secs(5));
    assert!(
        received.load(Ordering::SeqCst) > 0,
        "callback should have been invoked"
    );
}
