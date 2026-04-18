//! Integration tests for the per-source EOS marker delivered in-band via
//! [`NvEncoder::send_source_eos`].
//!
//! Verifies that the event is ordered with encoded frames and surfaces as
//! [`NvEncoderOutput::SourceEos`] on the output channel.

mod common;

use std::time::Duration;

use common::*;
use deepstream_encoders::prelude::*;
use serial_test::serial;

const FRAME_DUR_NS: u64 = 33_333_333;

/// Helper: drive encoder → encode N frames, then send source_eos, then
/// encode M more frames.  Run graceful_shutdown and collect outputs in
/// order.
fn encode_frames_with_source_eos(
    encoder_cfg: EncoderConfig,
    n_before: usize,
    n_after: usize,
    source_id: &str,
) -> Vec<NvEncoderOutput> {
    let encoder = NvEncoder::new(test_nv_encoder_config(encoder_cfg)).expect("encoder creation");

    let mut pts = 0u64;
    for i in 0..n_before {
        let buf = acquire_buffer(&encoder, i as u128);
        encoder
            .submit_frame(buf, i as u128, pts, Some(FRAME_DUR_NS))
            .expect("submit_frame");
        pts += FRAME_DUR_NS;
    }
    encoder.send_source_eos(source_id).expect("send_source_eos");
    for i in n_before..n_before + n_after {
        let buf = acquire_buffer(&encoder, i as u128);
        encoder
            .submit_frame(buf, i as u128, pts, Some(FRAME_DUR_NS))
            .expect("submit_frame");
        pts += FRAME_DUR_NS;
    }

    let mut outputs = Vec::new();
    encoder
        .graceful_shutdown(Some(Duration::from_secs(5)), |out| {
            outputs.push(out);
        })
        .expect("graceful_shutdown");
    outputs
}

#[test]
#[serial]
fn source_eos_ordered_between_frames() {
    init();
    // PNG pipeline is always available (CPU-based pngenc) so this test
    // does not depend on hardware.
    let cfg = EncoderConfig::Png(PngEncoderConfig::new(128, 128));
    let outputs = encode_frames_with_source_eos(cfg, 2, 2, "stream-42");

    let mut frame_count_before_eos = 0usize;
    let mut frame_count_after_eos = 0usize;
    let mut seen_source_eos = false;
    for out in outputs {
        match out {
            NvEncoderOutput::Frame(_) => {
                if seen_source_eos {
                    frame_count_after_eos += 1;
                } else {
                    frame_count_before_eos += 1;
                }
            }
            NvEncoderOutput::SourceEos { source_id } => {
                assert!(!seen_source_eos, "SourceEos delivered twice");
                assert_eq!(source_id, "stream-42");
                seen_source_eos = true;
            }
            NvEncoderOutput::Error(e) => panic!("encoder error: {e}"),
            NvEncoderOutput::Event(_) | NvEncoderOutput::Eos => {}
        }
    }
    assert!(seen_source_eos, "SourceEos was not delivered");
    assert_eq!(
        frame_count_before_eos, 2,
        "expected 2 frames before source_eos"
    );
    assert_eq!(
        frame_count_after_eos, 2,
        "expected 2 frames after source_eos"
    );
}

#[test]
#[serial]
fn source_eos_does_not_stop_encoder() {
    init();
    let cfg = EncoderConfig::Png(PngEncoderConfig::new(64, 64));
    let encoder = NvEncoder::new(test_nv_encoder_config(cfg)).unwrap();

    let buf = acquire_buffer(&encoder, 0);
    encoder.submit_frame(buf, 0, 0, Some(FRAME_DUR_NS)).unwrap();
    encoder.send_source_eos("alpha").unwrap();
    // Encoder must still accept submissions after source_eos (it is a
    // logical marker, not a real EOS).
    let buf = acquire_buffer(&encoder, 1);
    encoder
        .submit_frame(buf, 1, FRAME_DUR_NS, Some(FRAME_DUR_NS))
        .unwrap();

    let frames = drain_to_frames(&encoder);
    assert_eq!(frames.len(), 2);
}
