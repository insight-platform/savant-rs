//! Tests for logical per-source EOS propagation through NvInfer.

mod common;

use deepstream_nvinfer::{ModelColorFormat, NvInfer, NvInferConfig, NvInferOutput, VideoFormat};
use serial_test::serial;
use std::time::Duration;

fn identity_engine() -> Option<NvInfer> {
    let assets = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    if !assets.join("identity.onnx").exists() {
        eprintln!("Skipping: identity.onnx not found");
        return None;
    }
    let props = common::identity_properties();
    let config = NvInferConfig::new(props, VideoFormat::RGBA, 12, 12, ModelColorFormat::RGB);
    let engine = NvInfer::new(config).expect("create identity NvInfer");
    common::promote_built_engine("identity.onnx", 16);
    Some(engine)
}

#[test]
#[serial]
fn source_eos_propagates_with_source_id() {
    common::init();
    let engine = match identity_engine() {
        Some(engine) => engine,
        None => return,
    };

    let source_id = "camera-7";
    engine.send_eos(source_id).expect("send source eos");

    for _ in 0..64 {
        match engine
            .recv_timeout(Duration::from_secs(2))
            .expect("recv timeout call")
        {
            Some(NvInferOutput::Eos { source_id: got }) => {
                assert_eq!(got, source_id);
                engine.shutdown().expect("shutdown");
                return;
            }
            Some(NvInferOutput::Event(_)) => continue,
            Some(NvInferOutput::Inference(_)) => panic!("unexpected inference output"),
            Some(NvInferOutput::Error(e)) => panic!("unexpected pipeline error: {e}"),
            None => continue,
        }
    }

    panic!("did not receive logical source EOS output");
}
