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

use deepstream_buffers::{
    bridge_savant_id_meta, BufferGenerator, NvBufSurfaceMemType, SavantIdMeta, VideoFormat,
};
use gstreamer as gst;
use gstreamer::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

// ─── Helpers ─────────────────────────────────────────────────────────────────

struct EncoderTestConfig {
    format: VideoFormat,
    enc_name: &'static str,
    parser: Option<&'static str>,
    pre_encoder: Option<&'static str>,
}

fn run_pipeline_bridge_test(config: &EncoderTestConfig, num_frames: u32) {
    common::init();

    let generator = BufferGenerator::builder(config.format, 640, 480)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(16)
        .max_buffers(16)
        .build()
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

    bridge_savant_id_meta(&enc).expect("bridge_savant_id_meta");

    let sink_elem = gst::ElementFactory::make("appsink")
        .name("sink")
        .build()
        .expect("appsink");

    let caps: gst::Caps = generator.nvmm_caps().parse().unwrap();
    appsrc_elem.set_property("caps", &caps);
    appsrc_elem.set_property_from_str("format", "time");
    appsrc_elem.set_property_from_str("stream-type", "stream");

    sink_elem.set_property("sync", false);
    sink_elem.set_property("emit-signals", true);

    let pre_enc_elem = config.pre_encoder.map(|name| {
        let elem = gst::ElementFactory::make(name)
            .name("pre_enc")
            .build()
            .unwrap_or_else(|_| panic!("Failed to create {}", name));
        if elem.has_property("disable-passthrough", None) {
            elem.set_property("disable-passthrough", true);
        }
        elem
    });
    let parser_elem = config.parser.map(|name| {
        gst::ElementFactory::make(name)
            .name("parse")
            .build()
            .unwrap_or_else(|_| panic!("Failed to create {}", name))
    });

    let mut chain: Vec<&gst::Element> = vec![&appsrc_elem];
    if let Some(ref c) = pre_enc_elem {
        chain.push(c);
    }
    chain.push(&enc);
    if let Some(ref p) = parser_elem {
        chain.push(p);
    }
    chain.push(&sink_elem);

    pipeline
        .add_many(chain.iter().copied())
        .expect("Failed to add elements");
    gst::Element::link_many(chain.iter().copied()).unwrap();

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

    pipeline
        .set_state(gst::State::Playing)
        .expect("Failed to start pipeline");

    let appsrc = appsrc_elem.dynamic_cast::<gstreamer_app::AppSrc>().unwrap();

    let frame_duration_ns: u64 = 33_333_333;
    let mut pushed = 0u32;

    for i in 0..num_frames {
        let pts_ns = i as u64 * frame_duration_ns;

        let shared = generator
            .acquire(Some(i as u128))
            .unwrap_or_else(|e| panic!("acquire failed at frame {}: {:?}", i, e));

        let mut buf = shared.into_buffer().unwrap_or_else(|_| {
            panic!("into_buffer failed at frame {} (outstanding references)", i)
        });
        {
            let buf_ref = buf.make_mut();
            buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
            buf_ref.set_duration(gst::ClockTime::from_nseconds(frame_duration_ns));
        }

        match appsrc.push_buffer(buf) {
            Ok(_) => pushed += 1,
            Err(e) => panic!("Push failed at frame {}: {:?}", i, e),
        }
    }

    appsrc.end_of_stream().unwrap();
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
    if !nvidia_gpu_utils::has_nvenc(0).unwrap_or(false) {
        eprintln!("NVENC not available on this GPU — skipping nvv4l2h265enc test");
        return;
    }
    run_pipeline_bridge_test(
        &EncoderTestConfig {
            format: VideoFormat::NV12,
            enc_name: "nvv4l2h265enc",
            parser: Some("h265parse"),
            pre_encoder: None,
        },
        30,
    );
}

#[test]
fn test_bridge_meta_nvv4l2h264enc() {
    if !nvidia_gpu_utils::has_nvenc(0).unwrap_or(false) {
        eprintln!("NVENC not available on this GPU — skipping nvv4l2h264enc test");
        return;
    }
    run_pipeline_bridge_test(
        &EncoderTestConfig {
            format: VideoFormat::NV12,
            enc_name: "nvv4l2h264enc",
            parser: Some("h264parse"),
            pre_encoder: None,
        },
        30,
    );
}

#[test]
fn test_bridge_meta_nvjpegenc() {
    common::init();
    if gst::ElementFactory::find("nvjpegenc").is_none() {
        eprintln!("nvjpegenc not available — skipping");
        return;
    }
    run_pipeline_bridge_test(
        &EncoderTestConfig {
            format: VideoFormat::I420,
            enc_name: "nvjpegenc",
            parser: Some("jpegparse"),
            pre_encoder: if cfg!(target_arch = "aarch64") {
                Some("nvvideoconvert")
            } else {
                None
            },
        },
        30,
    );
}
