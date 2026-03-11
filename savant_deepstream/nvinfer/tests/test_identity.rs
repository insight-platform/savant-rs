//! Integration tests for NvInfer with the identity model.

mod common;

use deepstream_nvbufsurface::{
    DsNvNonUniformSurfaceBuffer, DsNvSurfaceBufferGenerator, DsNvUniformSurfaceBufferGenerator,
    NvBufSurfaceMemType, VideoFormat,
};
use nvinfer::{
    attach_batch_meta_with_rois, DataType, MetaClearPolicy, NvInfer, NvInferConfig, Roi,
};
use savant_core::primitives::RBBox;
use serial_test::serial;
use std::collections::HashMap;

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

    let config = common::platform_transform_config();
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

    let config = common::platform_transform_config();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    for (i, &fill_byte) in fill_bytes.iter().enumerate() {
        let src = src_gen.acquire_surface(Some(i as i64)).unwrap();
        unsafe {
            deepstream_nvbufsurface::memset_surface(&src, fill_byte).expect("memset_surface");
        }
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

fn has_identity_onnx() -> bool {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets/identity.onnx")
        .exists()
}

fn identity_engine() -> Option<NvInfer> {
    if !has_identity_onnx() {
        eprintln!("Skipping: identity.onnx not found");
        return None;
    }
    let props = common::identity_properties();
    let config = NvInferConfig::new(props, "RGBA", 12, 12);
    Some(NvInfer::new(config, Box::new(|_| {})).expect("create NvInfer"))
}

fn identity_engine_flexible() -> Option<NvInfer> {
    if !has_identity_onnx() {
        eprintln!("Skipping: identity.onnx not found");
        return None;
    }
    let props = common::identity_properties();
    let config = NvInferConfig::new_flexible(props, "RGBA");
    Some(NvInfer::new(config, Box::new(|_| {})).expect("create NvInfer (flexible)"))
}

/// Build a non-uniform batch from per-frame specs `(width, height, fill_byte, frame_id)`.
///
/// Each frame is a separate GPU surface filled with a constant byte.
/// The returned buffer is writable (the intermediate `DsNvNonUniformSurfaceBuffer`
/// is dropped before the buffer is returned).
fn make_nonuniform_identity_batch(frames: &[(u32, u32, u8, i64)]) -> gstreamer::Buffer {
    common::init();
    let mut batch =
        DsNvNonUniformSurfaceBuffer::new(frames.len() as u32, 0).expect("create non-uniform batch");

    for &(w, h, fill, id) in frames {
        let gen = DsNvSurfaceBufferGenerator::builder(VideoFormat::RGBA, w, h)
            .gpu_id(0)
            .mem_type(NvBufSurfaceMemType::Default)
            .min_buffers(1)
            .max_buffers(1)
            .build()
            .expect("src generator");

        let src = gen.acquire_surface(Some(id)).unwrap();
        unsafe {
            deepstream_nvbufsurface::memset_surface(&src, fill).expect("memset_surface");
        }

        batch.add(&src, Some(id)).unwrap();
    }

    batch.finalize().unwrap();
    batch.as_gst_buffer().unwrap()
}

/// Build full-frame ROIs for a non-uniform batch.
///
/// `slots` is `&[(width, height, &[roi_id])]`. Returns a map keyed by slot index.
fn full_frame_rois(slots: &[(u32, u32, &[i64])]) -> HashMap<u32, Vec<Roi>> {
    slots
        .iter()
        .enumerate()
        .map(|(slot, &(w, h, ids))| {
            let bbox = RBBox::ltwh(0.0, 0.0, w as f32, h as f32).unwrap();
            (
                slot as u32,
                ids.iter()
                    .map(|&id| Roi {
                        id,
                        bbox: bbox.clone(),
                    })
                    .collect(),
            )
        })
        .collect()
}

/// Sum all f32 values in a host tensor slice.
fn tensor_f32_sum(tensor: &nvinfer::TensorView) -> f64 {
    let slice: &[f32] = unsafe { tensor.as_slice() };
    slice.iter().map(|&v| v as f64).sum()
}

#[test]
#[serial]
fn test_attach_batch_meta() {
    common::init();
    let mut buffer = make_identity_batch(2);
    let buf_ref = buffer.get_mut().expect("buffer writable");
    attach_batch_meta_with_rois(buf_ref, 2, 16, MetaClearPolicy::Before, None, 12, 12).unwrap();
}

#[test]
#[serial]
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
#[serial]
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
#[serial]
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
#[serial]
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

    let config = common::platform_transform_config();
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
#[serial]
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
#[serial]
fn test_config_rejects_wrong_process_mode() {
    common::init();

    let mut props = common::identity_properties();
    // process-mode=1 (primary) is not allowed; NvInfer requires mode=2 (secondary).
    props.insert("process-mode".into(), "1".into());
    props.insert("output-tensor-meta".into(), "1".into());
    expect_new_fails_with(props, "process-mode");
}

#[test]
#[serial]
fn test_config_rejects_wrong_output_tensor_meta_value() {
    common::init();

    let mut props = common::identity_properties();
    props.insert("process-mode".into(), "2".into());
    props.insert("output-tensor-meta".into(), "2".into());
    expect_new_fails_with(props, "output-tensor-meta");
}

#[test]
#[serial]
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
#[serial]
fn test_two_rois_same_rect_same_output() {
    common::init();
    let engine = match identity_engine() {
        Some(s) => s,
        None => return,
    };

    let (batch, _) = make_identity_batch_known(1, 100);
    let full_bbox = RBBox::ltwh(0.0, 0.0, 12.0, 12.0).unwrap();
    let rois: HashMap<u32, Vec<Roi>> = [(
        0,
        vec![
            Roi {
                id: 10,
                bbox: full_bbox.clone(),
            },
            Roi {
                id: 20,
                bbox: full_bbox,
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

/// Non-uniform batch (two frames of different spatial dimensions) with two
/// full-frame ROIs on each frame.  Validates element count, ID propagation,
/// per-ROI tensor correctness, and cross-frame distinctness.
#[test]
#[serial]
fn test_nonuniform_batch_two_rois_each() {
    common::init();
    let engine = match identity_engine_flexible() {
        Some(e) => e,
        None => return,
    };

    let frames: &[(u32, u32, u8, i64)] = &[(24, 24, 80, 10), (36, 36, 200, 20)];
    let buffer = make_nonuniform_identity_batch(frames);

    let rois = full_frame_rois(&[(24, 24, &[100, 101]), (36, 36, &[200, 201])]);

    let output = engine
        .infer_sync(buffer, 77, Some(&rois))
        .expect("infer_sync");

    assert_eq!(output.batch_id(), 77, "batch_id must survive round-trip");
    assert_eq!(
        output.num_elements(),
        4,
        "expected 2 frames x 2 ROIs = 4 elements, got {}",
        output.num_elements()
    );

    let elems = output.elements();
    let roi_ids_per_frame: &[&[i64]] = &[&[100, 101], &[200, 201]];

    for (i, (&(w, h, fill, fid), roi_ids)) in frames.iter().zip(roi_ids_per_frame).enumerate() {
        let _ = (w, h); // used only in ROI construction
        let e0 = &elems[i * 2];
        let e1 = &elems[i * 2 + 1];

        assert_eq!(e0.frame_id, Some(fid), "frame {i} elem 0: frame_id");
        assert_eq!(e1.frame_id, Some(fid), "frame {i} elem 1: frame_id");
        assert_eq!(e0.roi_id, Some(roi_ids[0]), "frame {i} elem 0: roi_id");
        assert_eq!(e1.roi_id, Some(roi_ids[1]), "frame {i} elem 1: roi_id");

        // Both ROIs cover the full frame with uniform fill, so tensors must match.
        for (t0, t1) in e0.tensors.iter().zip(e1.tensors.iter()) {
            let s0: &[f32] = unsafe { t0.as_slice() };
            let s1: &[f32] = unsafe { t1.as_slice() };
            assert_eq!(
                s0, s1,
                "frame {i}: same-rect ROIs must produce identical tensors"
            );
        }

        let expected_sum = 3.0 * 12.0 * 12.0 * (fill as f64);
        let actual_sum = tensor_f32_sum(&e0.tensors[0]);
        let rel_err = (actual_sum - expected_sum).abs() / expected_sum;
        assert!(
            rel_err < 0.01,
            "frame {i} (fill={fill}): expected={expected_sum}, actual={actual_sum}, \
             rel_err={rel_err:.6}"
        );
    }

    let sum_f0 = tensor_f32_sum(&elems[0].tensors[0]);
    let sum_f1 = tensor_f32_sum(&elems[2].tensors[0]);
    assert!(
        (sum_f0 - sum_f1).abs() > 1.0,
        "different fill bytes must yield distinguishable sums: f0={sum_f0}, f1={sum_f1}"
    );
}

/// Non-uniform batch with unequal ROI counts per frame: frame 0 has 1 ROI,
/// frame 1 has 3.  Validates correct element ordering and frame/ROI ID
/// propagation when the mapping is non-rectangular.
#[test]
#[serial]
fn test_nonuniform_batch_unequal_roi_counts() {
    common::init();
    let engine = match identity_engine_flexible() {
        Some(e) => e,
        None => return,
    };

    let frames: &[(u32, u32, u8, i64)] = &[(24, 24, 100, 1), (36, 36, 180, 2)];
    let buffer = make_nonuniform_identity_batch(frames);

    let rois = full_frame_rois(&[(24, 24, &[10]), (36, 36, &[20, 21, 22])]);

    let output = engine
        .infer_sync(buffer, 88, Some(&rois))
        .expect("infer_sync");

    assert_eq!(output.batch_id(), 88);
    assert_eq!(
        output.num_elements(),
        4,
        "expected 1 + 3 = 4 elements, got {}",
        output.num_elements()
    );

    let elems = output.elements();

    // Frame 0: 1 element.
    assert_eq!(elems[0].frame_id, Some(1), "elem 0: frame_id");
    assert_eq!(elems[0].roi_id, Some(10), "elem 0: roi_id");

    // Frame 1: 3 elements.
    for (j, &expected_roi) in [20i64, 21, 22].iter().enumerate() {
        let idx = 1 + j;
        assert_eq!(elems[idx].frame_id, Some(2), "elem {idx}: frame_id");
        assert_eq!(elems[idx].roi_id, Some(expected_roi), "elem {idx}: roi_id");
    }

    // Tensor sums: all ROIs on the same frame should produce the same sum.
    let expected_sum_f0 = 3.0 * 12.0 * 12.0 * 100.0;
    let expected_sum_f1 = 3.0 * 12.0 * 12.0 * 180.0;

    let sum_0 = tensor_f32_sum(&elems[0].tensors[0]);
    let rel_0 = (sum_0 - expected_sum_f0).abs() / expected_sum_f0;
    assert!(
        rel_0 < 0.01,
        "frame 0: expected={expected_sum_f0}, actual={sum_0}"
    );

    for j in 0..3 {
        let sum_j = tensor_f32_sum(&elems[1 + j].tensors[0]);
        let rel_j = (sum_j - expected_sum_f1).abs() / expected_sum_f1;
        assert!(
            rel_j < 0.01,
            "frame 1 roi {j}: expected={expected_sum_f1}, actual={sum_j}"
        );
    }

    // All three ROIs on frame 1 share the same full-frame rect and fill,
    // so their tensors must be byte-identical.
    let s_ref: &[f32] = unsafe { elems[1].tensors[0].as_slice() };
    for j in 1..3 {
        let s_j: &[f32] = unsafe { elems[1 + j].tensors[0].as_slice() };
        assert_eq!(s_ref, s_j, "frame 1: ROI 0 vs ROI {j} tensors must match");
    }
}

/// Interleave uniform and non-uniform batches through the same flexible
/// engine.  Verifies the pipeline handles both `DsNvUniformSurfaceBuffer`
/// and `DsNvNonUniformSurfaceBuffer` without renegotiating or leaking state.
#[test]
#[serial]
fn test_mixed_uniform_nonuniform_sequential() {
    common::init();
    let engine = match identity_engine_flexible() {
        Some(e) => e,
        None => return,
    };

    // --- Batch 1: non-uniform 24x24 + 36x36, 1 ROI each ---
    {
        let frames: &[(u32, u32, u8, i64)] = &[(24, 24, 50, 1), (36, 36, 150, 2)];
        let buffer = make_nonuniform_identity_batch(frames);
        let rois = full_frame_rois(&[(24, 24, &[10]), (36, 36, &[20])]);
        let output = engine
            .infer_sync(buffer, 1, Some(&rois))
            .expect("batch 1 infer_sync");
        assert_eq!(output.batch_id(), 1);
        assert_eq!(output.num_elements(), 2);
        for (i, &fill) in [50u8, 150].iter().enumerate() {
            let expected_sum = 3.0 * 12.0 * 12.0 * (fill as f64);
            let actual_sum = tensor_f32_sum(&output.elements()[i].tensors[0]);
            let rel = (actual_sum - expected_sum).abs() / expected_sum;
            assert!(
                rel < 0.01,
                "batch 1 elem {i}: expected={expected_sum}, actual={actual_sum}"
            );
        }
    }

    // --- Batch 2: uniform 2x 12x12 (fill=100), 1 ROI each ---
    {
        let (buffer, expected_sum) = make_identity_batch_known(2, 100);
        let rois = full_frame_rois(&[(12, 12, &[30]), (12, 12, &[31])]);
        let output = engine
            .infer_sync(buffer, 2, Some(&rois))
            .expect("batch 2 infer_sync");
        assert_eq!(output.batch_id(), 2);
        assert_eq!(output.num_elements(), 2);
        for (i, elem) in output.elements().iter().enumerate() {
            let actual_sum = tensor_f32_sum(&elem.tensors[0]);
            let rel = (actual_sum - expected_sum).abs() / expected_sum;
            assert!(
                rel < 0.01,
                "batch 2 elem {i}: expected={expected_sum}, actual={actual_sum}"
            );
        }
    }

    // --- Batch 3: non-uniform 48x48 + 12x12, 1 ROI each ---
    {
        let frames: &[(u32, u32, u8, i64)] = &[(48, 48, 200, 5), (12, 12, 80, 6)];
        let buffer = make_nonuniform_identity_batch(frames);
        let rois = full_frame_rois(&[(48, 48, &[50]), (12, 12, &[60])]);
        let output = engine
            .infer_sync(buffer, 3, Some(&rois))
            .expect("batch 3 infer_sync");
        assert_eq!(output.batch_id(), 3);
        assert_eq!(output.num_elements(), 2);
        for (i, &fill) in [200u8, 80].iter().enumerate() {
            let expected_sum = 3.0 * 12.0 * 12.0 * (fill as f64);
            let actual_sum = tensor_f32_sum(&output.elements()[i].tensors[0]);
            let rel = (actual_sum - expected_sum).abs() / expected_sum;
            assert!(
                rel < 0.01,
                "batch 3 elem {i}: expected={expected_sum}, actual={actual_sum}"
            );
        }
    }

    // --- Batch 4: uniform 1x 12x12 (fill=255), 1 ROI ---
    {
        let (buffer, expected_sum) = make_identity_batch_known(1, 255);
        let rois = full_frame_rois(&[(12, 12, &[70])]);
        let output = engine
            .infer_sync(buffer, 4, Some(&rois))
            .expect("batch 4 infer_sync");
        assert_eq!(output.batch_id(), 4);
        assert_eq!(output.num_elements(), 1);
        let actual_sum = tensor_f32_sum(&output.elements()[0].tensors[0]);
        let rel = (actual_sum - expected_sum).abs() / expected_sum;
        assert!(
            rel < 0.01,
            "batch 4: expected={expected_sum}, actual={actual_sum}"
        );
    }
}

/// Regression test: `NvInferConfig::new_flexible` with `rois=None` must
/// still produce correct inference output. Previously, the flexible config
/// set `input_width=0, input_height=0`, which caused a 0x0 sentinel ROI
/// to be attached. nvinfer silently skipped inference on the zero-area
/// object, returning incorrect (often zero) tensor values.
#[test]
#[serial]
fn test_flexible_engine_no_rois_runs_inference() {
    let engine = match identity_engine_flexible() {
        Some(e) => e,
        None => return,
    };

    let fill_byte: u8 = 42;
    let (buffer, expected_sum) = make_identity_batch_known(2, fill_byte);
    assert!(
        expected_sum > 0.0,
        "precondition: expected_sum must be nonzero"
    );

    let output = engine
        .infer_sync(buffer, 99, None)
        .expect("infer_sync with flexible engine and no ROIs");

    assert_eq!(output.batch_id(), 99);
    assert_eq!(
        output.num_elements(),
        2,
        "flexible engine with no ROIs must still produce one element per frame"
    );

    for (i, elem) in output.elements().iter().enumerate() {
        assert!(!elem.tensors.is_empty(), "element {i} must have tensors");
        let actual_sum = tensor_f32_sum(&elem.tensors[0]);
        let rel = (actual_sum - expected_sum).abs() / expected_sum;
        assert!(
            rel < 0.01,
            "element {i}: expected_sum={expected_sum}, actual_sum={actual_sum}, rel_err={rel}"
        );
    }
}
