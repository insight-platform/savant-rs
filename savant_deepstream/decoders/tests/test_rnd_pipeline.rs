//! R&D tests: validate GStreamer decoder pipeline data flow
//! with filesrc (known-working from CLI) vs appsrc approaches.

mod common;

use common::*;
use deepstream_decoders::prelude::*;
use deepstream_decoders::NvDecoderExt;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use serial_test::serial;

/// ── Test 1: filesrc pipeline (CLI equivalent) ────────────────────────
/// This MUST work because gst-launch-1.0 works.
#[test]
#[serial]
fn rnd_filesrc_h264_annexb() {
    init();

    let path = assets_dir().join("test_h264_annexb_ip.h264");
    let pipeline_str = format!(
        "filesrc location={} ! h264parse ! nvv4l2decoder ! appsink name=sink sync=false emit-signals=false",
        path.display()
    );
    let pipeline = gst::parse::launch(&pipeline_str).unwrap();
    let pipeline = pipeline.dynamic_cast::<gst::Pipeline>().unwrap();
    let sink = pipeline
        .by_name("sink")
        .unwrap()
        .dynamic_cast::<gst_app::AppSink>()
        .unwrap();

    pipeline.set_state(gst::State::Playing).unwrap();

    let mut count = 0u32;
    loop {
        match sink.try_pull_sample(gst::ClockTime::from_seconds(5)) {
            Some(sample) => {
                let buf = sample.buffer().unwrap();
                let pts = buf.pts().map(|t| t.nseconds()).unwrap_or(0);
                let size = buf.size();
                let caps = sample.caps();
                let is_nvmm = caps
                    .and_then(|c| c.features(0))
                    .map(|f| f.contains("memory:NVMM"))
                    .unwrap_or(false);
                eprintln!(
                    "  [filesrc] frame {count}: pts={pts} size={size} nvmm={is_nvmm} caps={:?}",
                    caps.map(|c| c.to_string())
                );
                count += 1;
            }
            None if sink.is_eos() => break,
            None => {
                eprintln!("  [filesrc] try_pull_sample returned None, not EOS yet");
                break;
            }
        }
    }

    pipeline.set_state(gst::State::Null).unwrap();
    eprintln!("[filesrc] total decoded frames: {count}");
    assert!(count > 0, "filesrc pipeline produced 0 frames");
    assert!(
        count >= 8,
        "expected at least 8 frames from test_h264_annexb_ip.h264, got {count}"
    );
}

/// ── Test 2: appsrc feeding whole file as single buffer ───────────────
/// If this fails, the issue is appsrc configuration, not AU splitting.
#[test]
#[serial]
fn rnd_appsrc_whole_file_h264() {
    init();

    let path = assets_dir().join("test_h264_annexb_ip.h264");
    let data = std::fs::read(&path).unwrap();
    eprintln!("[whole-file] file size: {} bytes", data.len());

    let pipeline = gst::Pipeline::new();
    let appsrc = gst::ElementFactory::make("appsrc")
        .name("src")
        .build()
        .unwrap()
        .dynamic_cast::<gst_app::AppSrc>()
        .unwrap();
    let parser = gst::ElementFactory::make("h264parse")
        .name("parse")
        .build()
        .unwrap();
    let dec = gst::ElementFactory::make("nvv4l2decoder")
        .name("dec")
        .build()
        .unwrap();
    let appsink = gst::ElementFactory::make("appsink")
        .name("sink")
        .build()
        .unwrap()
        .dynamic_cast::<gst_app::AppSink>()
        .unwrap();

    let caps = gst::Caps::builder("video/x-h264")
        .field("stream-format", "byte-stream")
        .field("alignment", "au")
        .build();
    let src_elem = appsrc.upcast_ref::<gst::Element>();
    src_elem.set_property("caps", &caps);
    src_elem.set_property_from_str("format", "time");
    src_elem.set_property_from_str("stream-type", "stream");
    src_elem.set_property("is-live", true);
    src_elem.set_property("min-latency", 0i64);

    let sink_elem = appsink.upcast_ref::<gst::Element>();
    sink_elem.set_property("sync", false);
    sink_elem.set_property("emit-signals", false);

    pipeline
        .add_many([appsrc.upcast_ref(), &parser, &dec, appsink.upcast_ref()])
        .unwrap();
    gst::Element::link_many([appsrc.upcast_ref(), &parser, &dec, appsink.upcast_ref()]).unwrap();

    pipeline.set_state(gst::State::Playing).unwrap();

    let mut buffer = gst::Buffer::from_mut_slice(data);
    {
        let buf = buffer.get_mut().unwrap();
        buf.set_pts(gst::ClockTime::ZERO);
    }
    eprintln!("[whole-file] pushing entire file as one buffer...");
    appsrc.push_buffer(buffer).unwrap();
    eprintln!("[whole-file] sending EOS...");
    appsrc.end_of_stream().unwrap();

    let mut count = 0u32;
    loop {
        match appsink.try_pull_sample(gst::ClockTime::from_seconds(5)) {
            Some(sample) => {
                let buf = sample.buffer().unwrap();
                let pts = buf.pts().map(|t| t.nseconds()).unwrap_or(0);
                let size = buf.size();
                eprintln!("  [whole-file] frame {count}: pts={pts} size={size}");
                count += 1;
            }
            None if appsink.is_eos() => break,
            None => {
                eprintln!("  [whole-file] timeout, not EOS");
                break;
            }
        }
    }

    pipeline.set_state(gst::State::Null).unwrap();
    eprintln!("[whole-file] total decoded frames: {count}");
    assert!(count > 0, "appsrc whole-file produced 0 frames");
}

/// ── Test 3: appsrc feeding individual AUs (same as annexb test) ──────
/// If test 2 passes but this fails, the AU splitting is wrong.
#[test]
#[serial]
fn rnd_appsrc_individual_aus_h264() {
    init();

    let path = assets_dir().join("test_h264_annexb_ip.h264");
    let data = std::fs::read(&path).unwrap();

    let nalus = split_annexb_nalus(&data, "h264");
    eprintln!("[per-AU] total NALUs: {}", nalus.len());
    for (i, n) in nalus.iter().enumerate() {
        let off = nal_payload_offset(n);
        let ntype = if off < n.len() { n[off] & 0x1f } else { 0 };
        eprintln!(
            "  NALU {i}: len={} type={ntype} vcl={}",
            n.len(),
            is_vcl_nalu("h264", n)
        );
    }

    let aus = group_nalus_to_access_units("h264", nalus);
    eprintln!("[per-AU] access units: {}", aus.len());
    for (i, au) in aus.iter().enumerate() {
        eprintln!("  AU {i}: len={}", au.len());
    }

    let pipeline = gst::Pipeline::new();
    let appsrc = gst::ElementFactory::make("appsrc")
        .name("src")
        .build()
        .unwrap()
        .dynamic_cast::<gst_app::AppSrc>()
        .unwrap();
    let parser = gst::ElementFactory::make("h264parse")
        .name("parse")
        .build()
        .unwrap();
    let dec = gst::ElementFactory::make("nvv4l2decoder")
        .name("dec")
        .build()
        .unwrap();
    let appsink = gst::ElementFactory::make("appsink")
        .name("sink")
        .build()
        .unwrap()
        .dynamic_cast::<gst_app::AppSink>()
        .unwrap();

    let caps = gst::Caps::builder("video/x-h264")
        .field("stream-format", "byte-stream")
        .field("alignment", "au")
        .build();
    let src_elem = appsrc.upcast_ref::<gst::Element>();
    src_elem.set_property("caps", &caps);
    src_elem.set_property_from_str("format", "time");
    src_elem.set_property_from_str("stream-type", "stream");
    src_elem.set_property("is-live", true);
    src_elem.set_property("min-latency", 0i64);

    let sink_elem = appsink.upcast_ref::<gst::Element>();
    sink_elem.set_property("sync", false);
    sink_elem.set_property("emit-signals", false);

    pipeline
        .add_many([appsrc.upcast_ref(), &parser, &dec, appsink.upcast_ref()])
        .unwrap();
    gst::Element::link_many([appsrc.upcast_ref(), &parser, &dec, appsink.upcast_ref()]).unwrap();

    pipeline.set_state(gst::State::Playing).unwrap();

    let dur = 33_333_333u64;
    for (i, au) in aus.iter().take(8).enumerate() {
        let pts = i as u64 * dur;
        let mut buffer = gst::Buffer::from_mut_slice(au.clone());
        {
            let buf = buffer.get_mut().unwrap();
            buf.set_pts(gst::ClockTime::from_nseconds(pts));
            buf.set_dts(gst::ClockTime::from_nseconds(pts));
            buf.set_duration(gst::ClockTime::from_nseconds(dur));
        }
        eprintln!("[per-AU] push AU {i}: {} bytes, pts={pts}", au.len());
        appsrc.push_buffer(buffer).unwrap();
    }
    eprintln!("[per-AU] sending EOS...");
    appsrc.end_of_stream().unwrap();

    let mut count = 0u32;
    loop {
        match appsink.try_pull_sample(gst::ClockTime::from_seconds(5)) {
            Some(sample) => {
                let buf = sample.buffer().unwrap();
                let pts = buf.pts().map(|t| t.nseconds()).unwrap_or(0);
                let size = buf.size();
                eprintln!("  [per-AU] frame {count}: pts={pts} size={size}");
                count += 1;
            }
            None if appsink.is_eos() => break,
            None => {
                eprintln!("  [per-AU] timeout after 5s, not EOS");
                break;
            }
        }
    }

    pipeline.set_state(gst::State::Null).unwrap();
    eprintln!("[per-AU] total decoded frames: {count}");
    assert!(count > 0, "appsrc per-AU produced 0 frames");
}

/// ── Test 4: appsrc with NvDecoder (same as real test) ────────────────
/// If test 3 passes but this fails, issue is in NvDecoder drain logic.
#[test]
#[serial]
fn rnd_nvdecoder_annexb_h264() {
    init();

    let platform_tag = current_platform_tag();
    eprintln!("[NvDecoder] platform tag: {platform_tag}");

    let path = assets_dir().join("test_h264_annexb_ip.h264");
    let data = std::fs::read(&path).unwrap();
    let nalus = split_annexb_nalus(&data, "h264");
    let aus = group_nalus_to_access_units("h264", nalus);

    let config = DecoderConfig::H264(H264DecoderConfig::new(H264StreamFormat::ByteStream));
    let decoder = NvDecoder::new(
        test_decoder_config(0, config),
        make_rgba_pool(320, 240),
        identity_transform_config(),
    )
    .unwrap();

    let dur = 33_333_333u64;
    for (i, au) in aus.iter().take(8).enumerate() {
        let pts = i as u64 * dur;
        eprintln!("[NvDecoder] submit AU {i}: {} bytes, pts={pts}", au.len());
        decoder
            .submit_packet(au, i as u128, pts, Some(pts), Some(dur))
            .unwrap_or_else(|e| panic!("submit failed at AU {i}: {e}"));
    }
    eprintln!("[NvDecoder] sending EOS...");
    decoder.send_eos().unwrap();

    let mut frames = 0u32;
    loop {
        match decoder.recv_timeout(std::time::Duration::from_secs(10)) {
            Ok(Some(NvDecoderOutput::Frame(f))) => {
                eprintln!("  [NvDecoder] Frame: id={:?} pts={}", f.frame_id, f.pts_ns);
                frames += 1;
            }
            Ok(Some(NvDecoderOutput::Eos)) => {
                eprintln!("  [NvDecoder] EOS");
                break;
            }
            Ok(Some(NvDecoderOutput::Error(e))) => panic!("[NvDecoder] error: {e}"),
            Ok(Some(NvDecoderOutput::Event(_) | NvDecoderOutput::SourceEos { .. })) => {}
            Ok(None) => panic!("[NvDecoder] timeout after 10s, got {frames} frames"),
            Err(e) => panic!("[NvDecoder] recv error: {e}"),
        }
    }
    eprintln!("[NvDecoder] total decoded frames: {frames}");
    assert_eq!(frames, 8);
}
