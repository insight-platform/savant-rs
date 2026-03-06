//! Integration tests for NvInfer with the identity model.

mod common;

use deepstream_nvbufsurface::{
    DsNvSurfaceBufferGenerator, DsNvUniformSurfaceBufferGenerator, NvBufSurfaceMemType,
    TransformConfig, VideoFormat,
};
use nvinfer::{
    attach_batch_meta_with_rois, DataType, MetaClearPolicy, NvInfer, NvInferConfig, Rect, Roi,
};
use std::collections::HashMap;

#[link(name = "cuda")]
extern "C" {
    fn cuMemsetD8_v2(dst: u64, value: u8, count: usize) -> u32;
}

fn make_identity_batch(num_frames: u32) -> gstreamer::Buffer {
    common::init();

    let src_gen = DsNvSurfaceBufferGenerator::builder(VideoFormat::RGBA, 12, 12)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(4)
        .max_buffers(4)
        .build()
        .expect("src generator");

    let batched_gen = DsNvUniformSurfaceBufferGenerator::new(
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

    batch.finalize().unwrap();
    batch.as_gst_buffer().unwrap()
}

/// Build a batch where every source surface is memset to `fill_byte`.
///
/// Returns the batch buffer and the expected per-element output sum.
/// The identity model with `net-scale-factor=1.0` converts each RGBA u8 pixel
/// to RGB float32: `output_value = pixel_byte * 1.0`.
/// Shape is [3, 12, 12] = 432 elements, each equal to `fill_byte as f64`.
fn make_identity_batch_known(num_frames: u32, fill_byte: u8) -> (gstreamer::Buffer, f64) {
    let (buf, sums) = make_identity_batch_per_frame(&vec![fill_byte; num_frames as usize]);
    (buf, sums[0])
}

/// Build a batch where each frame `i` is memset to `fill_bytes[i]`.
///
/// Returns the batch buffer and a vector of expected per-element output sums.
fn make_identity_batch_per_frame(fill_bytes: &[u8]) -> (gstreamer::Buffer, Vec<f64>) {
    common::init();

    let num_frames = fill_bytes.len() as u32;
    let src_gen = DsNvSurfaceBufferGenerator::builder(VideoFormat::RGBA, 12, 12)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(num_frames.max(4))
        .max_buffers(num_frames.max(4))
        .build()
        .expect("src generator");

    let batched_gen = DsNvUniformSurfaceBufferGenerator::new(
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

    for (i, &fill_byte) in fill_bytes.iter().enumerate() {
        let (src, data_ptr, pitch) = src_gen.acquire_surface_with_ptr(Some(i as i64)).unwrap();
        let fill_size = (pitch * 12) as usize;
        let ret = unsafe { cuMemsetD8_v2(data_ptr as u64, fill_byte, fill_size) };
        assert_eq!(ret, 0, "cuMemsetD8_v2 failed with code {}", ret);
        batch.fill_slot(&src, None, Some(i as i64)).unwrap();
    }

    let expected_sums: Vec<f64> = fill_bytes
        .iter()
        .map(|&b| 3.0 * 12.0 * 12.0 * (b as f64))
        .collect();
    batch.finalize().unwrap();
    let buf = batch.as_gst_buffer().unwrap();
    (buf, expected_sums)
}

fn identity_engine() -> Option<NvInfer> {
    use std::path::Path;
    let onnx = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/identity.onnx");
    if !onnx.exists() {
        eprintln!("Skipping: identity.onnx not found at {:?}", onnx);
        return None;
    }
    let props = common::identity_properties();
    let config = NvInferConfig::new(props, "RGBA", 12, 12);
    let callback = Box::new(|_| {});
    Some(NvInfer::new(config, callback).expect("create NvInfer"))
}

/// Sum all f32 values in a host tensor slice.
fn tensor_f32_sum(tensor: &nvinfer::TensorView) -> f64 {
    let slice: &[f32] = unsafe { tensor.as_slice() };
    slice.iter().map(|&v| v as f64).sum()
}

#[test]
fn test_attach_batch_meta() {
    common::init();
    let mut buffer = make_identity_batch(2);
    let buf_ref = buffer.get_mut().expect("buffer writable");
    attach_batch_meta_with_rois(buf_ref, 2, 16, MetaClearPolicy::Before, None, 12, 12).unwrap();
}

#[test]
fn test_sync_single_frame() {
    common::init();
    let engine = match identity_engine() {
        Some(s) => s,
        None => return,
    };

    let (batch, expected_sum) = make_identity_batch_known(1, 128);
    let output = engine.infer_sync(batch, 1, None).expect("infer_sync");

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
    let engine = match identity_engine() {
        Some(s) => s,
        None => return,
    };

    let (batch, expected_sum) = make_identity_batch_known(4, 200);
    let output = engine.infer_sync(batch, 42, None).expect("infer_sync");

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
    let engine = match identity_engine() {
        Some(s) => s,
        None => return,
    };

    let fills: Vec<u8> = vec![64, 128, 200];
    let (batch, expected_sums) = make_identity_batch_per_frame(&fills);
    let output = engine.infer_sync(batch, 7, None).expect("infer_sync");

    assert_eq!(output.num_elements(), fills.len());
    for (i, (elem, &expected_sum)) in output.elements().iter().zip(&expected_sums).enumerate() {
        assert!(!elem.tensors.is_empty(), "element {i} must have tensors");
        let actual_sum = tensor_f32_sum(&elem.tensors[0]);
        let rel_err = (actual_sum - expected_sum).abs() / expected_sum;
        assert!(
            rel_err < 0.01,
            "element {i} (fill={}): expected={expected_sum}, actual={actual_sum}, rel_err={rel_err:.6}",
            fills[i]
        );
    }
}

#[test]
fn test_element_ids_preserved() {
    common::init();
    let engine = match identity_engine() {
        Some(s) => s,
        None => return,
    };

    // Use non-trivial IDs with gaps and negative values to prove the actual
    // user-supplied IDs propagate, not just frame indices.
    let ids_to_send: Vec<i64> = vec![42, -7, 1000, 0];

    let src_gen = DsNvSurfaceBufferGenerator::builder(VideoFormat::RGBA, 12, 12)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(4)
        .max_buffers(4)
        .build()
        .expect("src generator");

    let batched_gen = DsNvUniformSurfaceBufferGenerator::new(
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
    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();

    let output = engine.infer_sync(buffer, 99, None).expect("infer_sync");

    assert_eq!(output.batch_id(), 99);
    assert_eq!(output.num_elements(), ids_to_send.len());
    let received_ids: Vec<Option<i64>> = output.elements().iter().map(|e| e.frame_id).collect();
    assert_eq!(
        received_ids,
        ids_to_send.iter().map(|&id| Some(id)).collect::<Vec<_>>(),
        "IDs must propagate end-to-end through nvinfer"
    );
}

#[test]
fn test_async_callback() {
    common::init();

    let onnx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/identity.onnx");
    if !onnx.exists() {
        eprintln!("Skipping: identity.onnx not found at {:?}", onnx);
        return;
    }

    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    let received = Arc::new(AtomicU64::new(0));
    let received_clone = received.clone();

    let props = common::identity_properties();
    let config = NvInferConfig::new(props, "RGBA", 12, 12).queue_depth(2);
    let callback = Box::new(move |output: nvinfer::BatchInferenceOutput| {
        received_clone.store(output.batch_id(), Ordering::SeqCst);
    });
    let engine = NvInfer::new(config, callback).expect("create NvInfer");

    for batch_id in 1..=3u64 {
        let batch = make_identity_batch(1);
        engine.submit(batch, batch_id, None).expect("submit");
    }

    std::thread::sleep(std::time::Duration::from_secs(5));
    assert!(
        received.load(Ordering::SeqCst) > 0,
        "callback should have been invoked"
    );
}

fn expect_new_fails_with(props: HashMap<String, String>, expected_substring: &str) {
    match NvInfer::new(NvInferConfig::new(props, "RGBA", 12, 12), Box::new(|_| {})) {
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains(expected_substring),
                "error should mention '{expected_substring}', got: {msg}"
            );
        }
        Ok(_) => panic!("expected NvInfer::new to fail for '{expected_substring}'"),
    }
}

#[test]
fn test_config_rejects_wrong_process_mode() {
    common::init();

    let mut props = common::identity_properties();
    // process-mode=1 (primary) is not allowed; NvInfer requires mode=2 (secondary).
    props.insert("process-mode".into(), "1".into());
    props.insert("output-tensor-meta".into(), "1".into());
    expect_new_fails_with(props, "process-mode");
}

#[test]
fn test_config_rejects_wrong_output_tensor_meta_value() {
    common::init();

    let mut props = common::identity_properties();
    props.insert("process-mode".into(), "2".into());
    props.insert("output-tensor-meta".into(), "2".into());
    expect_new_fails_with(props, "output-tensor-meta");
}

#[test]
fn test_config_rejects_disabled_output_tensor_meta() {
    common::init();

    let mut props = common::identity_properties();
    props.insert("process-mode".into(), "2".into());
    props.insert("output-tensor-meta".into(), "0".into());
    expect_new_fails_with(props, "output-tensor-meta");
}

/// Two ROIs with identical geometry on one frame must produce identical tensor
/// outputs – the identity model should return the same values regardless of
/// which ROI ID is assigned.
#[test]
fn test_two_rois_same_rect_same_output() {
    common::init();
    let engine = match identity_engine() {
        Some(s) => s,
        None => return,
    };

    let (batch, _) = make_identity_batch_known(1, 100);
    let full_rect = Rect {
        left: 0,
        top: 0,
        width: 12,
        height: 12,
    };
    let rois: HashMap<u32, Vec<Roi>> = [(
        0,
        vec![
            Roi {
                id: 10,
                rect: full_rect.clone(),
            },
            Roi {
                id: 20,
                rect: full_rect,
            },
        ],
    )]
    .into();

    let output = engine
        .infer_sync(batch, 55, Some(&rois))
        .expect("infer_sync");

    // Both ROIs should produce one ElementOutput each.
    assert_eq!(
        output.num_elements(),
        2,
        "expected one output per ROI, got {}",
        output.num_elements()
    );

    let elems = output.elements();
    assert_eq!(elems[0].roi_id, Some(10), "first roi_id mismatch");
    assert_eq!(elems[1].roi_id, Some(20), "second roi_id mismatch");

    // Both share the same frame_id.
    assert_eq!(elems[0].frame_id, elems[1].frame_id);

    // Tensors from the same input region must be identical.
    assert_eq!(
        elems[0].tensors.len(),
        elems[1].tensors.len(),
        "tensor count must match"
    );
    for (t0, t1) in elems[0].tensors.iter().zip(elems[1].tensors.iter()) {
        assert_eq!(t0.dims, t1.dims, "tensor dims must match");
        let s0: &[f32] = unsafe { t0.as_slice() };
        let s1: &[f32] = unsafe { t1.as_slice() };
        assert_eq!(s0, s1, "tensor values for same-rect ROIs must be identical");
    }
}
