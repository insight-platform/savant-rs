//! Transform (scale + letterbox) pipeline example.
//!
//! Demonstrates Mode 1 of the NvBufSurface API: acquiring a source buffer
//! at one resolution, then using `NvBufSurfTransform` to scale and letterbox
//! it into a destination buffer at a different resolution before encoding.
//!
//! Pipeline:
//! ```text
//! [source gen 1920×1080] → gen.transform() → [dest 640×640 appsrc] → nvv4l2h265enc → h265parse → fakesink
//! ```
//!
//! # Usage
//!
//! ```sh
//! cargo run --example transform_pipeline -p deepstream_nvbufsurface -- \
//!     --src-width 1920 --src-height 1080 --dst-width 640 --dst-height 640
//! ```

use clap::Parser;
use deepstream_nvbufsurface::{
    bridge_savant_id_meta, cuda_init, Interpolation, NvBufSurfaceGenerator, NvBufSurfaceMemType,
    Padding, TransformConfig, VideoFormat,
};
use gstreamer as gst;
use gstreamer::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Transform pipeline: scale + letterbox from source to destination resolution.
#[derive(Parser, Debug)]
#[command(name = "transform_pipeline")]
struct Args {
    /// Source frame width.
    #[arg(long, default_value_t = 1920)]
    src_width: u32,

    /// Source frame height.
    #[arg(long, default_value_t = 1080)]
    src_height: u32,

    /// Destination frame width.
    #[arg(long, default_value_t = 640)]
    dst_width: u32,

    /// Destination frame height.
    #[arg(long, default_value_t = 640)]
    dst_height: u32,

    /// Padding mode: none, right_bottom, symmetric.
    #[arg(long, default_value = "symmetric")]
    padding: String,

    /// Interpolation: nearest, bilinear, cubic, super, lanczos.
    #[arg(long, default_value = "bilinear")]
    interpolation: String,

    /// Buffer pool size.
    #[arg(long, default_value_t = 4)]
    pool_size: u32,

    /// Number of frames (0 = infinite).
    #[arg(long, short, default_value_t = 0)]
    num_frames: u64,

    /// Video codec: h264, h265.
    #[arg(long, default_value = "h265")]
    codec: String,
}

fn main() {
    let args = Args::parse();

    // ── Init ──
    gst::init().unwrap();
    cuda_init(0).expect("CUDA init failed");

    // ── Source generator (at source resolution) ──
    let src_gen =
        NvBufSurfaceGenerator::builder(VideoFormat::RGBA, args.src_width, args.src_height)
            .gpu_id(0)
            .mem_type(NvBufSurfaceMemType::Default)
            .min_buffers(args.pool_size)
            .max_buffers(args.pool_size)
            .build()
            .expect("failed to create source generator");

    // ── Destination generator (at destination resolution) ──
    let dst_gen =
        NvBufSurfaceGenerator::builder(VideoFormat::RGBA, args.dst_width, args.dst_height)
            .gpu_id(0)
            .mem_type(NvBufSurfaceMemType::Default)
            .min_buffers(args.pool_size)
            .max_buffers(args.pool_size)
            .build()
            .expect("failed to create destination generator");

    // ── Parse config ──
    let padding = Padding::from_str_name(&args.padding)
        .unwrap_or_else(|| panic!("unknown padding: {}", args.padding));
    let interpolation = Interpolation::from_str_name(&args.interpolation)
        .unwrap_or_else(|| panic!("unknown interpolation: {}", args.interpolation));

    let transform_config = TransformConfig {
        padding,
        interpolation,
        src_rect: None,
        ..Default::default()
    };

    // ── Build pipeline ──
    let enc_name = match args.codec.as_str() {
        "h264" => "nvv4l2h264enc",
        "h265" | "hevc" => "nvv4l2h265enc",
        "jpeg" => "nvjpegenc",
        other => panic!("unsupported codec: {}", other),
    };
    let parse_name = match args.codec.as_str() {
        "h264" => "h264parse",
        "h265" | "hevc" => "h265parse",
        "jpeg" => "jpegparse",
        _ => unreachable!(),
    };

    let pipeline = gst::Pipeline::new();
    let appsrc = gst::ElementFactory::make("appsrc")
        .property("is-live", true)
        .property("do-timestamp", false)
        .build()
        .unwrap();
    appsrc.set_property("caps", dst_gen.nvmm_caps());

    let convert = gst::ElementFactory::make("nvvideoconvert").build().unwrap();
    let encoder = gst::ElementFactory::make(enc_name).build().unwrap();
    let parser = gst::ElementFactory::make(parse_name).build().unwrap();
    let sink = gst::ElementFactory::make("fakesink")
        .property("sync", false)
        .build()
        .unwrap();

    pipeline
        .add_many([&appsrc, &convert, &encoder, &parser, &sink])
        .unwrap();
    gst::Element::link_many([&appsrc, &convert, &encoder, &parser, &sink]).unwrap();

    // Bridge SavantIdMeta across the encoder
    bridge_savant_id_meta(&encoder);

    let appsrc = appsrc.dynamic_cast::<gstreamer_app::AppSrc>().unwrap();

    // ── Stats ──
    let frame_count = Arc::new(AtomicU64::new(0));
    let running = Arc::new(AtomicBool::new(true));

    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nCtrl+C received, stopping...");
        r.store(false, Ordering::SeqCst);
    })
    .unwrap();

    pipeline.set_state(gst::State::Playing).unwrap();

    // ── Stats thread ──
    let fc = frame_count.clone();
    let r2 = running.clone();
    std::thread::spawn(move || {
        let mut last_count: u64 = 0;
        let mut last_time = Instant::now();
        while r2.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_secs(1));
            let now = Instant::now();
            let count = fc.load(Ordering::Relaxed);
            let dt = now.duration_since(last_time).as_secs_f64();
            let fps = (count - last_count) as f64 / dt;
            eprintln!(
                "[transform_pipeline] frames={}, fps={:.1} ({}×{} → {}×{})",
                count,
                fps,
                0,
                0,
                0,
                0, // placeholder
            );
            last_count = count;
            last_time = now;
        }
    });

    // ── Push loop ──
    let frame_duration_ns = 33_333_333u64; // ~30fps
    let limit = if args.num_frames == 0 {
        u64::MAX
    } else {
        args.num_frames
    };
    let mut i: u64 = 0;

    while i < limit && running.load(Ordering::SeqCst) {
        // 1. Acquire source buffer (simulating an incoming frame)
        let src_buf = match src_gen.acquire_surface(None) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("src acquire failed: {}", e);
                break;
            }
        };

        // 2. Transform: src_gen resolution → dst_gen resolution
        let mut dst_buf = match dst_gen.transform(&src_buf, &transform_config, Some(i as i64)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("transform failed: {}", e);
                break;
            }
        };

        // 3. Set timestamps and push
        let pts_ns = i * frame_duration_ns;
        {
            let buf = dst_buf.make_mut();
            buf.set_pts(gst::ClockTime::from_nseconds(pts_ns));
            buf.set_duration(gst::ClockTime::from_nseconds(frame_duration_ns));
        }
        match appsrc.push_buffer(dst_buf) {
            Ok(_) => {
                frame_count.fetch_add(1, Ordering::Relaxed);
                i += 1;
            }
            Err(e) => {
                eprintln!("push failed: {}", e);
                break;
            }
        }
    }

    // ── Teardown ──
    running.store(false, Ordering::SeqCst);
    let _ = appsrc.end_of_stream();
    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gst::ClockTime::from_seconds(5)) {
        if let gst::MessageView::Eos(..) = msg.view() {
            break;
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();
    eprintln!("[transform_pipeline] done, {} frames processed", i);
}
