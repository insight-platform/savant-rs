//! E2E test for source-scoped EOS propagation through NvInferBatchingOperator callback.

mod common;

use deepstream_buffers::VideoFormat;
use deepstream_nvinfer::{
    BatchFormationResult, ModelColorFormat, NvInferBatchingOperator, NvInferBatchingOperatorConfig,
    NvInferConfig, OperatorOutput, OperatorResultCallback, RoiKind,
};
use serial_test::serial;
use std::sync::{mpsc, Arc};
use std::time::Duration;

#[test]
#[serial]
fn batching_operator_callback_receives_source_eos() {
    common::init();

    let assets = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    if !assets.join("identity.onnx").exists() {
        eprintln!("Skipping: identity.onnx not found");
        return;
    }

    let nvinfer_config = NvInferConfig::new(
        common::identity_properties(),
        VideoFormat::RGBA,
        12,
        12,
        ModelColorFormat::RGB,
    );
    let operator_config = NvInferBatchingOperatorConfig {
        max_batch_size: 2,
        max_batch_wait: Duration::from_secs(5),
        nvinfer: nvinfer_config,
        pending_batch_timeout: Duration::from_secs(60),
    };

    let batch_formation: deepstream_nvinfer::BatchFormationCallback = Arc::new(|frames| {
        let ids = frames
            .iter()
            .map(|_| deepstream_buffers::SavantIdMetaKind::Frame(0))
            .collect();
        let rois = frames.iter().map(|_| RoiKind::FullFrame).collect();
        BatchFormationResult { ids, rois }
    });

    let (tx, rx) = mpsc::channel::<String>();
    let result_callback: OperatorResultCallback = Box::new(move |output| match output {
        OperatorOutput::Eos { source_id } => {
            tx.send(source_id).expect("send eos source id");
        }
        OperatorOutput::Inference(_) => panic!("unexpected inference output"),
        OperatorOutput::Error(e) => panic!("unexpected pipeline error: {e}"),
    });

    let mut operator =
        NvInferBatchingOperator::new(operator_config, batch_formation, result_callback)
            .expect("create NvInferBatchingOperator");

    common::promote_built_engine("identity.onnx", 16);

    let source_id = "stream-A";
    operator.send_eos(source_id).expect("send source eos");

    let got = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("timeout waiting for source eos callback");
    assert_eq!(got, source_id);

    operator.shutdown().expect("operator shutdown");
}
