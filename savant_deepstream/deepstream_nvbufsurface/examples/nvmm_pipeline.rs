//! NVMM encoding pipeline example.
//!
//! Pushes NVMM frames via `NvBufSurfaceGenerator` through a DeepStream
//! encoding pipeline as fast as possible, printing throughput statistics
//! every second.
//!
//! The generator allocates buffers in the requested format (RGBA, NV12, etc.)
//! directly in GPU memory. When the format is not encoder-native (i.e. not
//! NV12 or I420), `nvvideoconvert` is inserted to do GPU-to-GPU conversion
//! before encoding -- no CPU involvement at any stage.
//!
//! Pipeline (NV12/I420, no output file):
//! ```text
//! appsrc (memory:NVMM) → nvv4l2h26Xenc → h26Xparse → fakesink
//! ```
//!
//! Pipeline (RGBA, with .mp4 output):
//! ```text
//! appsrc (memory:NVMM) → nvvideoconvert → nvv4l2h26Xenc → h26Xparse → qtmux → filesink
//! ```
//!
//! # Usage
//!
//! ```sh
//! # Infinite run, discard output (benchmark mode)
//! cargo run --example nvmm_pipeline -p deepstream_nvbufsurface -- --width 1920 --height 1080
//!
//! # 300 frames of RGBA → H.264 to an MP4 file
//! cargo run --example nvmm_pipeline -p deepstream_nvbufsurface -- \
//!     --format RGBA --codec h264 --num-frames 300 --output /tmp/test.mp4
//!
//! # 600 frames of NV12 → H.265 raw elementary stream
//! cargo run --example nvmm_pipeline -p deepstream_nvbufsurface -- \
//!     --num-frames 600 --output /tmp/test.h265
//!
//! # Infinite run to a Matroska container (Ctrl+C to stop)
//! cargo run --example nvmm_pipeline -p deepstream_nvbufsurface -- \
//!     --codec h265 --output /tmp/test.mkv
//! ```

use clap::Parser;
use deepstream_nvbufsurface::{bridge_savant_id_meta, cuda_init, NvBufSurfaceGenerator, NvBufSurfaceMemType};
use gstreamer as gst;
use gstreamer::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Infinite NVMM encoding pipeline -- pushes frames as fast as possible.
#[derive(Parser, Debug)]
#[command(name = "nvmm_pipeline")]
struct Args {
    /// Frame width in pixels.
    #[arg(long, default_value_t = 1920)]
    width: u32,

    /// Frame height in pixels.
    #[arg(long, default_value_t = 1080)]
    height: u32,

    /// Video format (e.g. NV12, RGBA).
    #[arg(long, default_value = "NV12")]
    format: String,

    /// Framerate numerator (used for PTS calculation only; pipeline runs without sync).
    #[arg(long, default_value_t = 30)]
    fps: i32,

    /// GPU device ID.
    #[arg(long, default_value_t = 0)]
    gpu_id: u32,

    /// NvBufSurface memory type (0=Default, 2=CudaDevice).
    #[arg(long, default_value_t = 0)]
    mem_type: u32,

    /// Buffer pool size (number of NVMM buffers).
    /// The pool blocks on acquire when all buffers are in-flight,
    /// providing natural backpressure to match encoder throughput.
    #[arg(long, default_value_t = 4)]
    pool_size: u32,

    /// Video codec: h264 or h265 (default: h265).
    #[arg(long, default_value = "h265")]
    codec: String,

    /// Output file path. When set, writes encoded video to this file instead
    /// of discarding it. The container format is chosen by extension:
    ///   .mp4        → qtmux (ISO MP4)
    ///   .mkv/.webm  → matroskamux
    ///   .ts         → mpegtsmux
    ///   anything else (e.g. .h264, .h265) → raw elementary stream
    #[arg(long, short)]
    output: Option<String>,

    /// Number of frames to generate. When omitted the pipeline runs
    /// indefinitely until Ctrl+C.
    #[arg(long, short)]
    num_frames: Option<u64>,
}

/// Optionally create a container-muxer element based on the output file extension.
///
/// Returns `Some(element)` for recognised container formats, `None` for raw
/// elementary streams (`.h264`, `.h265`, `.264`, `.265`, or unknown extensions).
fn container_mux_for_path(path: &str) -> Option<gst::Element> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let factory = match ext.as_str() {
        "mp4" | "m4v" => "qtmux",
        "mkv" | "webm" => "matroskamux",
        "ts" => "mpegtsmux",
        _ => return None, // raw elementary stream
    };

    Some(
        gst::ElementFactory::make(factory)
            .name("cmux")
            .build()
            .unwrap_or_else(|_| panic!("Failed to create {}", factory)),
    )
}

fn main() {
    let args = Args::parse();

    // ── Init ─────────────────────────────────────────────────────────────
    gst::init().expect("Failed to initialize GStreamer");
    cuda_init(args.gpu_id).expect("Failed to initialize CUDA");

    let frame_duration_ns: u64 = (1_000_000_000u64)
        .checked_div(args.fps as u64)
        .unwrap_or(33_333_333);

    // ── Generator ────────────────────────────────────────────────────────
    let generator = NvBufSurfaceGenerator::builder(&args.format, args.width, args.height)
        .fps(args.fps, 1)
        .gpu_id(args.gpu_id)
        .mem_type(NvBufSurfaceMemType::from(args.mem_type))
        .min_buffers(args.pool_size)
        .max_buffers(args.pool_size)
        .build()
        .expect("Failed to create NvBufSurfaceGenerator");

    println!(
        "Generator created: {}x{} {} @ {} fps (gpu {}, pool {})",
        args.width, args.height, args.format, args.fps, args.gpu_id, args.pool_size,
    );

    // ── Pipeline ─────────────────────────────────────────────────────────
    //
    // NVENC encoders only accept NV12/I420. For other formats (RGBA, BGRx, etc.)
    // we insert nvvideoconvert to perform GPU-to-GPU colour-space conversion.
    // The initial buffer is still allocated in the requested format in GPU memory
    // by the generator -- there is no CPU involvement.
    let needs_convert = !matches!(args.format.as_str(), "NV12" | "I420");

    let (enc_name, parse_name) = match args.codec.as_str() {
        "h264" => ("nvv4l2h264enc", "h264parse"),
        "h265" | "hevc" => ("nvv4l2h265enc", "h265parse"),
        other => {
            eprintln!("Unknown codec '{}'. Supported: h264, h265 (hevc)", other);
            std::process::exit(1);
        }
    };

    let pipeline = gst::Pipeline::new();

    let appsrc_elem = gst::ElementFactory::make("appsrc")
        .name("src")
        .build()
        .expect("Failed to create appsrc");
    let enc = gst::ElementFactory::make(enc_name)
        .name("enc")
        .build()
        .unwrap_or_else(|_| panic!("Failed to create {}", enc_name));

    // Bridge SavantIdMeta across the encoder (PTS-keyed pad probes)
    bridge_savant_id_meta(&enc);

    let parse = gst::ElementFactory::make(parse_name)
        .name("parse")
        .build()
        .unwrap_or_else(|_| panic!("Failed to create {}", parse_name));

    // ── Sink: fakesink or filesink (with optional container muxer) ───────
    let (sink, container_mux) = if let Some(ref path) = args.output {
        let fsink = gst::ElementFactory::make("filesink")
            .name("sink")
            .build()
            .expect("Failed to create filesink");
        fsink.set_property("location", path);

        let cmux = container_mux_for_path(path);
        (fsink, cmux)
    } else {
        let fsink = gst::ElementFactory::make("fakesink")
            .name("sink")
            .build()
            .expect("Failed to create fakesink");
        fsink.set_property("sync", false);
        (fsink, None)
    };

    // Configure appsrc
    appsrc_elem.set_property("caps", &generator.nvmm_caps());
    appsrc_elem.set_property_from_str("format", "time");
    appsrc_elem.set_property_from_str("stream-type", "stream");

    // Collect the tail elements after parse: [container_mux] → sink
    let mut tail: Vec<&gst::Element> = Vec::new();
    if let Some(ref cm) = container_mux {
        tail.push(cm);
    }
    tail.push(&sink);

    // Determine the sink label for printing
    let sink_label = if let Some(ref path) = args.output {
        let cmux_name = container_mux
            .as_ref()
            .map(|e| format!("{} -> ", e.factory().map_or("mux".into(), |f| f.name().to_string())));
        format!("{}filesink({})", cmux_name.unwrap_or_default(), path)
    } else {
        "fakesink".to_string()
    };

    // Assemble: appsrc → [convert →] enc → parse → [cmux →] sink
    if needs_convert {
        let convert = gst::ElementFactory::make("nvvideoconvert")
            .name("convert")
            .build()
            .expect("Failed to create nvvideoconvert");

        let mut all_elems: Vec<&gst::Element> =
            vec![&appsrc_elem, &convert, &enc, &parse];
        all_elems.extend(&tail);
        pipeline
            .add_many(all_elems.iter().copied())
            .expect("Failed to add elements to pipeline");

        let mut chain: Vec<&gst::Element> = vec![&appsrc_elem, &convert, &enc, &parse];
        chain.extend(&tail);
        gst::Element::link_many(chain).unwrap();

        println!(
            "Pipeline: appsrc({}) -> nvvideoconvert -> {} -> {} -> {}",
            args.format, enc_name, parse_name, sink_label,
        );
    } else {
        let mut all_elems: Vec<&gst::Element> = vec![&appsrc_elem, &enc, &parse];
        all_elems.extend(&tail);
        pipeline
            .add_many(all_elems.iter().copied())
            .expect("Failed to add elements to pipeline");

        let mut chain: Vec<&gst::Element> = vec![&appsrc_elem, &enc, &parse];
        chain.extend(&tail);
        gst::Element::link_many(chain).unwrap();

        println!(
            "Pipeline: appsrc({}) -> {} -> {} -> {}",
            args.format, enc_name, parse_name, sink_label,
        );
    }

    // ── Start ────────────────────────────────────────────────────────────
    pipeline
        .set_state(gst::State::Playing)
        .expect("Failed to start pipeline");

    let appsrc = appsrc_elem.dynamic_cast::<gstreamer_app::AppSrc>().unwrap();

    if let Some(n) = args.num_frames {
        println!("Pipeline running ({} frames)...\n", n);
    } else {
        println!("Pipeline running (Ctrl+C to stop)...\n");
    }

    // ── Ctrl+C handling ──────────────────────────────────────────────────
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Failed to set Ctrl+C handler");

    // ── Push loop ────────────────────────────────────────────────────────
    let frame_count = Arc::new(AtomicU64::new(0));
    let fc = frame_count.clone();

    // Stats reporter thread
    let stats_running = running.clone();
    let stats_handle = std::thread::spawn(move || {
        let mut last_count = 0u64;
        let mut last_time = Instant::now();
        while stats_running.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_secs(1));
            let now = Instant::now();
            let count = fc.load(Ordering::Relaxed);
            let elapsed = now.duration_since(last_time).as_secs_f64();
            let delta = count - last_count;
            let fps = delta as f64 / elapsed;
            let rss = rss_kb();
            println!(
                "frames: {:>8}  |  fps: {:>8.1}  |  RSS: {} MB",
                count,
                fps,
                rss / 1024,
            );
            last_count = count;
            last_time = now;
        }
    });

    let limit = args.num_frames.unwrap_or(u64::MAX);
    let mut i: u64 = 0;
    while i < limit && running.load(Ordering::SeqCst) {
        let pts_ns = i * frame_duration_ns;
        match generator.push_to_appsrc(&appsrc, pts_ns, frame_duration_ns, Some(i as i64)) {
            Ok(()) => {
                frame_count.fetch_add(1, Ordering::Relaxed);
                i += 1;
            }
            Err(e) => {
                eprintln!("Push failed at frame {}: {}", i, e);
                break;
            }
        }

        // Check pipeline bus for errors (non-blocking)
        if let Some(msg) = pipeline
            .bus()
            .and_then(|b| b.pop_filtered(&[gst::MessageType::Error]))
        {
            if let gst::MessageView::Error(err) = msg.view() {
                eprintln!(
                    "Pipeline error from {:?}: {}",
                    err.src().map(|s| s.path_string()),
                    err.error()
                );
                break;
            }
        }
    }

    // ── Shutdown ─────────────────────────────────────────────────────────
    println!("\nStopping...");
    running.store(false, Ordering::SeqCst);

    let _ = NvBufSurfaceGenerator::send_eos(&appsrc);

    // Wait briefly for EOS to propagate
    if let Some(bus) = pipeline.bus() {
        let _ = bus.timed_pop_filtered(
            gst::ClockTime::from_seconds(5),
            &[gst::MessageType::Eos, gst::MessageType::Error],
        );
    }

    pipeline.set_state(gst::State::Null).unwrap();
    let _ = stats_handle.join();

    let total = frame_count.load(Ordering::Relaxed);
    println!("Total frames pushed: {}", total);
}

/// Read VmRSS from /proc/self/status (Linux only).
fn rss_kb() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                return parts[1].parse().unwrap_or(0);
            }
        }
    }
    0
}
