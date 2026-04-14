//! Graceful shutdown tests (GPU + DeepStream required).

mod common;

use deepstream_buffers::{
    BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, SharedBuffer, UniformBatchGenerator,
    VideoFormat,
};
use gstreamer as gst;
use nvinfer::{ModelColorFormat, NvInfer, NvInferConfig, NvInferError, NvInferOutput};
use serial_test::serial;
use std::time::Duration;

const FRAMES_PER_BATCH: u32 = 2;

fn make_identity_batch(num_frames: u32) -> SharedBuffer {
    common::init();

    let min_bufs = num_frames.max(4);
    let src_gen = BufferGenerator::builder(VideoFormat::RGBA, 12, 12)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(min_bufs)
        .max_buffers(min_bufs)
        .build()
        .expect("src generator");

    let batched_gen = UniformBatchGenerator::new(
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
    let ids: Vec<SavantIdMetaKind> = (0..num_frames)
        .map(|i| SavantIdMetaKind::Frame(i as u128))
        .collect();
    let mut batch = batched_gen.acquire_batch(config, ids).unwrap();

    for i in 0..num_frames {
        let src_shared = src_gen.acquire(Some(i as u128)).unwrap();
        let src_view = deepstream_buffers::SurfaceView::from_buffer(&src_shared, 0).unwrap();
        batch.transform_slot(i, &src_view, None).unwrap();
    }

    batch.finalize().unwrap();
    batch.into_shared_buffer()
}

fn identity_engine() -> Option<NvInfer> {
    let onnx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/identity.onnx");
    if !onnx.exists() {
        eprintln!("Skipping: identity.onnx not found at {:?}", onnx);
        return None;
    }
    let props = common::identity_properties();
    let config = NvInferConfig::new(props, VideoFormat::RGBA, 12, 12, ModelColorFormat::RGB);
    let engine = NvInfer::new(config).expect("create NvInfer");
    common::promote_built_engine("identity.onnx", 16);
    Some(engine)
}

fn count_inference(outputs: &[NvInferOutput]) -> usize {
    outputs
        .iter()
        .filter(|o| matches!(o, NvInferOutput::Inference(_)))
        .count()
}

#[test]
#[serial]
fn test_graceful_shutdown_drains_all() {
    common::init();
    let engine = match identity_engine() {
        Some(e) => e,
        None => return,
    };

    const N: usize = 3;
    for _ in 0..N {
        let shared = make_identity_batch(FRAMES_PER_BATCH);
        engine.submit(shared, None).expect("submit");
    }

    let drained = engine
        .graceful_shutdown(Duration::from_secs(30))
        .expect("graceful_shutdown");
    assert_eq!(
        count_inference(&drained),
        N,
        "expected {N} inference outputs, got {drained:?}"
    );
}

#[test]
#[serial]
fn test_graceful_shutdown_timeout() {
    common::init();
    let engine = match identity_engine() {
        Some(e) => e,
        None => return,
    };

    let shared = make_identity_batch(FRAMES_PER_BATCH);
    engine.submit(shared, None).expect("submit");

    let drained = engine
        .graceful_shutdown(Duration::ZERO)
        .expect("graceful_shutdown");
    assert!(
        count_inference(&drained) < 1,
        "zero timeout should not fully drain inference: {drained:?}"
    );
}

#[test]
#[serial]
fn test_graceful_shutdown_submit_after_returns_shutting_down() {
    common::init();
    let engine = match identity_engine() {
        Some(e) => e,
        None => return,
    };

    let _ = engine
        .graceful_shutdown(Duration::from_secs(30))
        .expect("graceful_shutdown");

    let shared = make_identity_batch(FRAMES_PER_BATCH);
    let err = engine
        .submit(shared, None)
        .expect_err("submit after shutdown");
    assert!(matches!(err, NvInferError::ShuttingDown));

    let structure = gst::Structure::builder("test").build();
    let ev = gst::event::CustomDownstream::new(structure);
    let err = engine
        .send_event(ev)
        .expect_err("send_event after shutdown");
    assert!(matches!(err, NvInferError::ShuttingDown));

    let err = engine.send_eos("s1").expect_err("send_eos after shutdown");
    assert!(matches!(err, NvInferError::ShuttingDown));
}
