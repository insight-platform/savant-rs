//! Integration tests for [`bridge_savant_id_meta`] — verifies that
//! `SavantIdMeta` is propagated through hardware encoders that would
//! otherwise drop custom GStreamer meta.
//!
//! Each test builds a minimal pipeline:
//!
//! ```text
//! appsrc (memory:NVMM) → encoder → [parser →] appsink
//! ```
//!
//! With [`bridge_savant_id_meta`] installed on the encoder element,
//! every output buffer at `appsink` must carry the same `SavantIdMeta`
//! that was attached at `appsrc`.

mod common;

use deepstream_nvbufsurface::{
    bridge_savant_id_meta, NvBufSurfaceGenerator, NvBufSurfaceMemType, SavantIdMeta, VideoFormat,
};
use gstreamer as gst;
use gstreamer::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Per-encoder test configuration.
struct EncoderTestConfig {
    /// NvBufSurface pixel format (e.g. `VideoFormat::NV12`, `VideoFormat::I420`).
    format: VideoFormat,
    /// GStreamer encoder element factory name.
    enc_name: &'static str,
    /// Optional parser element placed between encoder and appsink.
    parser: Option<&'static str>,
}

/// Run a full pipeline and assert that [`bridge_savant_id_meta`]
/// propagates `SavantIdMeta` across the given encoder.
fn run_pipeline_bridge_test(config: &EncoderTestConfig, num_frames: u32) {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        config.format,
        640,
        480,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("Failed to create generator");

    let pipeline = gst::Pipeline::new();

    let appsrc_elem = gst::ElementFactory::make("appsrc")
        .name("src")
        .build()
        .expect("appsrc");
    let enc = gst::ElementFactory::make(config.enc_name)
        .name("enc")
        .build()
        .unwrap_or_else(|_| panic!("Failed to create {}", config.enc_name));

    // Install PTS-keyed meta bridge on the encoder
    bridge_savant_id_meta(&enc);

    let sink_elem = gst::ElementFactory::make("appsink")
        .name("sink")
        .build()
        .expect("appsink");

    // Configure appsrc
    appsrc_elem.set_property("caps", &generator.nvmm_caps());
    appsrc_elem.set_property_from_str("format", "time");
    appsrc_elem.set_property_from_str("stream-type", "stream");

    // Configure appsink
    sink_elem.set_property("sync", false);
    sink_elem.set_property("emit-signals", true);

    // Build the element chain: appsrc → enc → [parser →] appsink
    let parser_elem = config.parser.map(|name| {
        gst::ElementFactory::make(name)
            .name("parse")
            .build()
            .unwrap_or_else(|_| panic!("Failed to create {}", name))
    });

    let mut chain: Vec<&gst::Element> = vec![&appsrc_elem, &enc];
    if let Some(ref p) = parser_elem {
        chain.push(p);
    }
    chain.push(&sink_elem);

    pipeline
        .add_many(chain.iter().copied())
        .expect("Failed to add elements");
    gst::Element::link_many(chain.iter().copied()).unwrap();

    // Set up appsink callback to count received buffers and meta
    let received = Arc::new(AtomicU32::new(0));
    let meta_ok = Arc::new(AtomicU32::new(0));

    let appsink = sink_elem.dynamic_cast::<gstreamer_app::AppSink>().unwrap();

    let r = received.clone();
    let m = meta_ok.clone();
    appsink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                let sample = sink.pull_sample().map_err(|_| gst::FlowError::Error)?;
                r.fetch_add(1, Ordering::Relaxed);
                if let Some(buffer) = sample.buffer() {
                    if buffer.meta::<SavantIdMeta>().is_some() {
                        m.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    // Start pipeline
    pipeline
        .set_state(gst::State::Playing)
        .expect("Failed to start pipeline");

    let appsrc = appsrc_elem.dynamic_cast::<gstreamer_app::AppSrc>().unwrap();

    let frame_duration_ns: u64 = 33_333_333;
    let mut pushed = 0u32;

    for i in 0..num_frames {
        let pts_ns = i as u64 * frame_duration_ns;
        match generator.push_to_appsrc(&appsrc, pts_ns, frame_duration_ns, Some(i as i64)) {
            Ok(()) => pushed += 1,
            Err(e) => panic!("Push failed at frame {}: {:?}", i, e),
        }
    }

    // EOS + drain
    NvBufSurfaceGenerator::send_eos(&appsrc).unwrap();
    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gst::ClockTime::from_seconds(30)) {
        match msg.view() {
            gst::MessageView::Eos(..) => break,
            gst::MessageView::Error(err) => {
                panic!(
                    "[{}] pipeline error from {:?}: {:?}",
                    config.enc_name,
                    err.src().map(|s| s.path_string()),
                    err.error()
                );
            }
            _ => {}
        }
    }

    pipeline.set_state(gst::State::Null).unwrap();

    let recv = received.load(Ordering::Relaxed);
    let meta = meta_ok.load(Ordering::Relaxed);

    eprintln!(
        "  [{}] pushed={}, received={}, meta_ok={}",
        config.enc_name, pushed, recv, meta,
    );

    assert_eq!(
        pushed, num_frames,
        "[{}] not all frames pushed",
        config.enc_name
    );
    assert!(recv > 0, "[{}] appsink received 0 buffers", config.enc_name);
    assert_eq!(
        meta, recv,
        "[{}] bridge_savant_id_meta: meta_ok ({}) != received ({})",
        config.enc_name, meta, recv,
    );
}

// ─── Per-encoder tests ───────────────────────────────────────────────────────

#[test]
fn test_bridge_meta_nvv4l2h265enc() {
    run_pipeline_bridge_test(
        &EncoderTestConfig {
            format: VideoFormat::NV12,
            enc_name: "nvv4l2h265enc",
            parser: Some("h265parse"),
        },
        30,
    );
}

#[test]
fn test_bridge_meta_nvv4l2h264enc() {
    run_pipeline_bridge_test(
        &EncoderTestConfig {
            format: VideoFormat::NV12,
            enc_name: "nvv4l2h264enc",
            parser: Some("h264parse"),
        },
        30,
    );
}

#[test]
fn test_bridge_meta_nvjpegenc() {
    run_pipeline_bridge_test(
        &EncoderTestConfig {
            format: VideoFormat::I420,
            enc_name: "nvjpegenc",
            parser: Some("jpegparse"),
        },
        30,
    );
}
