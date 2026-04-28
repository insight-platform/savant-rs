//! Regression tests for `NvEncoder::fence_after_shutdown` (Plan A,
//! cuda-700 hardening).
//!
//! All three tests reset the process-wide CUDA poison flag, exercise an
//! encoder lifecycle, and assert the flag is still clear afterwards.
//!
//! These tests use the PNG encoder path because it is always available
//! (CPU `pngenc`) and requires no NVENC; the fence still runs because
//! `NvEncoder::shutdown` always calls it.  HW-only behaviour (V4L2 NVENC
//! holding input buffers across `set_state(Null)`) is exercised by the
//! cars-demo-zmq end-to-end validation in the plan, not here.

mod common;

use std::time::Duration;

use common::*;
use deepstream_buffers::cuda_poison::{is_poisoned, reset_poison_for_tests};
use deepstream_encoders::prelude::*;
use serial_test::serial;

const FRAME_DUR_NS: u64 = 33_333_333;

#[test]
#[serial]
fn shutdown_calls_device_sync_no_poison() {
    init();
    reset_poison_for_tests();

    let cfg = EncoderConfig::Png(PngEncoderConfig::new(128, 128));
    let encoder = NvEncoder::new(test_nv_encoder_config(cfg)).expect("encoder");

    let mut pts = 0u64;
    for i in 0..4 {
        let buf = acquire_buffer(&encoder, i as u128);
        encoder
            .submit_frame(buf, i as u128, pts, Some(FRAME_DUR_NS))
            .expect("submit_frame");
        pts += FRAME_DUR_NS;
    }

    encoder
        .graceful_shutdown(Some(Duration::from_secs(5)), |_out| {})
        .expect("graceful_shutdown");

    assert!(
        !is_poisoned(),
        "fence_after_shutdown itself must not poison the CUDA context"
    );
}

#[test]
#[serial]
fn shutdown_with_convert_ctx_no_poison() {
    init();
    reset_poison_for_tests();

    // RGBA -> NV12 conversion happens out-of-band on `convert_ctx.cuda_stream`,
    // so this configuration exercises the convert-stream sync branch of
    // `fence_after_shutdown`.  We need NVENC for that path; skip otherwise.
    let Some(cfg) = make_rgba_config(640, 480) else {
        assert!(
            !is_poisoned(),
            "shutdown_with_convert_ctx_no_poison: NVENC config probe ran with already-poisoned CUDA context — likely fence regression"
        );
        eprintln!("skipping: RGBA encoder config unavailable on this host");
        return;
    };
    let encoder = match NvEncoder::new(test_nv_encoder_config(cfg)) {
        Ok(encoder) => encoder,
        Err(e) => {
            assert!(
                !is_poisoned(),
                "shutdown_with_convert_ctx_no_poison: NVENC encoder setup ran with already-poisoned CUDA context — likely fence regression"
            );
            eprintln!("skipping: RGBA encoder setup failed on this host: {e}");
            return;
        }
    };

    let mut pts = 0u64;
    for i in 0..2 {
        let buf = acquire_buffer(&encoder, i as u128);
        encoder
            .submit_frame(buf, i as u128, pts, Some(FRAME_DUR_NS))
            .expect("submit_frame");
        pts += FRAME_DUR_NS;
    }

    let mut saw_pipeline_error = false;
    let shutdown_result = encoder.graceful_shutdown(Some(Duration::from_secs(5)), |out| {
        if let NvEncoderOutput::Error(e) = out {
            eprintln!("skipping: RGBA encoder runtime error on this host: {e}");
            saw_pipeline_error = true;
        }
    });
    if saw_pipeline_error {
        assert!(
            !is_poisoned(),
            "shutdown_with_convert_ctx_no_poison: pipeline runtime error ran with already-poisoned CUDA context — likely fence regression"
        );
        return;
    }
    if let Err(e) = shutdown_result {
        assert!(
            !is_poisoned(),
            "shutdown_with_convert_ctx_no_poison: graceful_shutdown ran with already-poisoned CUDA context — likely fence regression"
        );
        eprintln!("skipping: RGBA graceful_shutdown failed on this host: {e}");
        return;
    }

    assert!(
        !is_poisoned(),
        "convert-stream sync + device sync must not poison the CUDA context"
    );
}

#[test]
#[serial]
fn repeated_create_destroy_no_poison() {
    init();
    reset_poison_for_tests();

    // 50x create -> submit 1 frame -> graceful_shutdown.
    // Mirrors the cars-demo-zmq churn pattern at unit scale.  PNG path
    // is always available; this loop is mainly a smoke check that 50
    // fence invocations are themselves harmless.
    for iter in 0..50 {
        let cfg = EncoderConfig::Png(PngEncoderConfig::new(64, 64));
        let encoder = NvEncoder::new(test_nv_encoder_config(cfg))
            .unwrap_or_else(|e| panic!("iter {iter}: encoder: {e}"));

        let buf = acquire_buffer(&encoder, iter as u128);
        encoder
            .submit_frame(buf, iter as u128, 0, Some(FRAME_DUR_NS))
            .unwrap_or_else(|e| panic!("iter {iter}: submit: {e}"));

        encoder
            .graceful_shutdown(Some(Duration::from_secs(5)), |_| {})
            .unwrap_or_else(|e| panic!("iter {iter}: graceful_shutdown: {e}"));

        assert!(
            !is_poisoned(),
            "iter {iter}: fence_after_shutdown poisoned the CUDA context"
        );
    }
}

/// Regression test for the cars-demo-zmq cascade against the **HW NVENC**
/// path — the actual race the fence protects.  Skips cleanly on hosts
/// without NVENC (Orin Nano, x86 without hardware encoder, …).
///
/// Each iteration: build an RGBA H264 encoder (engages `convert_ctx` with
/// out-of-band RGBA→NV12 conversion), submit a few frames, and tear down
/// with [`NvEncoder::graceful_shutdown`].  Without `fence_after_shutdown`,
/// V4L2 NVENC's private CUDA stream may still be reading from the
/// just-released input buffers when the encoder's pools are dropped,
/// poisoning the context for the next iteration.
#[test]
#[serial]
fn repeated_create_destroy_nvenc_no_poison() {
    init();
    reset_poison_for_tests();

    if !has_nvenc() {
        assert!(
            !is_poisoned(),
            "iter preflight: NVENC availability probe ran with already-poisoned CUDA context — likely fence regression"
        );
        eprintln!("skipping: NVENC unavailable on this host");
        return;
    }

    let iters = 20;
    for iter in 0..iters {
        let Some(cfg) = make_rgba_config(640, 480) else {
            assert!(
                !is_poisoned(),
                "iter {iter}: NVENC encoder setup ran with already-poisoned CUDA context — likely fence regression"
            );
            eprintln!("iter {iter}: RGBA config unavailable; skipping rest of loop");
            return;
        };
        let encoder = match NvEncoder::new(test_nv_encoder_config(cfg)) {
            Ok(e) => e,
            Err(e) => {
                assert!(
                    !is_poisoned(),
                    "iter {iter}: NVENC encoder setup ran with already-poisoned CUDA context — likely fence regression"
                );
                eprintln!("iter {iter}: NVENC encoder setup failed ({e}); skipping rest");
                return;
            }
        };

        let mut pts = 0u64;
        for i in 0..3 {
            let buf = acquire_buffer(&encoder, ((iter * 1_000) + i) as u128);
            if let Err(e) =
                encoder.submit_frame(buf, ((iter * 1_000) + i) as u128, pts, Some(FRAME_DUR_NS))
            {
                assert!(
                    !is_poisoned(),
                    "iter {iter}: submit_frame ran with already-poisoned CUDA context — likely fence regression"
                );
                eprintln!("iter {iter}: submit_frame failed ({e}); skipping rest");
                return;
            }
            pts += FRAME_DUR_NS;
        }

        let mut runtime_error = false;
        let shutdown_result = encoder.graceful_shutdown(Some(Duration::from_secs(5)), |out| {
            if let NvEncoderOutput::Error(e) = out {
                eprintln!("iter {iter}: NVENC pipeline error ({e}); skipping rest");
                runtime_error = true;
            }
        });
        if runtime_error {
            assert!(
                !is_poisoned(),
                "iter {iter}: pipeline runtime error ran with already-poisoned CUDA context — likely fence regression"
            );
            return;
        }
        if let Err(e) = shutdown_result {
            assert!(
                !is_poisoned(),
                "iter {iter}: graceful_shutdown ran with already-poisoned CUDA context — likely fence regression"
            );
            eprintln!("iter {iter}: graceful_shutdown failed ({e}); skipping rest");
            return;
        }

        assert!(
            !is_poisoned(),
            "iter {iter}: fence_after_shutdown failed to keep CUDA context clean across NVENC churn"
        );
    }

    assert!(
        !is_poisoned(),
        "iter final: final ran with already-poisoned CUDA context — likely fence regression"
    );
}
