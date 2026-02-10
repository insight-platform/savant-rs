//! Skia GPU-rendered NVMM encoding pipeline.
//!
//! Demonstrates the [`SkiaRenderer`] API: each frame is drawn with Skia
//! (GPU-accelerated via EGL/OpenGL), copied into an NvBufSurface via
//! CUDA-GL interop (GPU-to-GPU, no CPU), and encoded with NVENC.
//!
//! The demo renders an animated test pattern with:
//! - Gradient background
//! - Moving coloured rectangles
//! - Frame counter text
//!
//! # Pipeline
//!
//! ```text
//! Skia GPU -> CUDA-GL copy -> NvBufSurface (RGBA NVMM)
//!     -> appsrc -> nvvideoconvert -> nvv4l2h26Xenc -> h26Xparse -> sink
//! ```
//!
//! # Usage
//!
//! ```sh
//! # 300 frames to MP4
//! cargo run --example skia_pipeline -p deepstream_nvbufsurface --features skia -- \
//!     --num-frames 300 --output /tmp/skia_demo.mp4
//!
//! # Infinite run (Ctrl+C to stop)
//! cargo run --example skia_pipeline -p deepstream_nvbufsurface --features skia
//! ```

use clap::Parser;
use deepstream_nvbufsurface::{
    bridge_savant_id_meta, cuda_init, NvBufSurfaceGenerator, NvBufSurfaceMemType, SkiaRenderer,
};
use gstreamer as gst;
use gstreamer::prelude::*;
use skia_safe::{
    gradient_shader, Color, Font, FontMgr, FontStyle, Paint, PaintStyle, Point, Rect, TileMode,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Skia GPU-rendered NVMM encoding pipeline.
#[derive(Parser, Debug)]
#[command(name = "skia_pipeline")]
struct Args {
    /// Frame width in pixels.
    #[arg(long, default_value_t = 1920)]
    width: u32,

    /// Frame height in pixels.
    #[arg(long, default_value_t = 1080)]
    height: u32,

    /// Framerate numerator.
    #[arg(long, default_value_t = 30)]
    fps: i32,

    /// GPU device ID.
    #[arg(long, default_value_t = 0)]
    gpu_id: u32,

    /// Buffer pool size.
    #[arg(long, default_value_t = 4)]
    pool_size: u32,

    /// Video codec: h264 or h265 (default: h265).
    #[arg(long, default_value = "h265")]
    codec: String,

    /// Output file path (.mp4, .mkv, .h264, .h265, etc.).
    #[arg(long, short)]
    output: Option<String>,

    /// Number of frames to generate. Omit for infinite.
    #[arg(long, short)]
    num_frames: Option<u64>,
}

/// Optionally create a container-muxer element based on file extension.
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
        _ => return None,
    };
    Some(
        gst::ElementFactory::make(factory)
            .name("cmux")
            .build()
            .unwrap_or_else(|_| panic!("Failed to create {}", factory)),
    )
}

// ── Detection class definitions ──────────────────────────────────────────────

const NUM_BOXES: usize = 20;

struct DetectionClass {
    name: &'static str,
    color: Color, // used for box border, fill (semi-transparent), and legend dot
}

const CLASSES: &[DetectionClass] = &[
    DetectionClass { name: "person",  color: Color::from_argb(255, 255, 80, 80) },
    DetectionClass { name: "car",     color: Color::from_argb(255, 80, 200, 255) },
    DetectionClass { name: "truck",   color: Color::from_argb(255, 255, 180, 40) },
    DetectionClass { name: "bicycle", color: Color::from_argb(255, 80, 255, 120) },
    DetectionClass { name: "dog",     color: Color::from_argb(255, 220, 100, 255) },
    DetectionClass { name: "bus",     color: Color::from_argb(255, 255, 255, 80) },
    DetectionClass { name: "bike",    color: Color::from_argb(255, 80, 255, 255) },
    DetectionClass { name: "sign",    color: Color::from_argb(255, 255, 140, 140) },
];

/// Deterministic pseudo-random float in [0, 1) seeded by two u64 values.
fn pseudo_rand(seed1: u64, seed2: u64) -> f32 {
    let mut h = seed1.wrapping_mul(6364136223846793005).wrapping_add(seed2);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    (h & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
}

// ── Cached drawing resources (created once, reused every frame) ──────────────

struct BBox {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    class_idx: usize,
    confidence: f32,
    id: u32,
}

/// Pre-allocated, reusable drawing resources.  Created once before the
/// render loop and passed into [`draw_frame`] each iteration to avoid
/// per-frame font lookups, paint construction, and heap allocations.
struct DrawCtx {
    // Fonts (different sizes, cached typeface)
    label_font: Font,
    title_font: Font,
    legend_font: Font,
    footer_font: Font,

    // Reusable paints
    white_paint: Paint,
    black_paint: Paint,
    sidebar_bg_paint: Paint,
    separator_paint: Paint,
    divider_paint: Paint,
    footer_bg_paint: Paint,
    footer_text_paint: Paint,
    fill_paint: Paint,
    stroke_paint: Paint,
    label_bg_paint: Paint,
    dot_paint: Paint,
    legend_text_paint: Paint,
    bg_paint: Paint,

    // Reusable string buffer to reduce allocations
    buf: String,

    // Reusable boxes vec
    boxes: Vec<BBox>,
}

impl DrawCtx {
    fn new() -> Self {
        let font_mgr = FontMgr::default();
        let typeface_bold = font_mgr
            .match_family_style("monospace", FontStyle::bold())
            .or_else(|| font_mgr.match_family_style("sans-serif", FontStyle::bold()))
            .expect("No fonts available");
        let typeface_normal = font_mgr
            .match_family_style("monospace", FontStyle::normal())
            .or_else(|| font_mgr.match_family_style("sans-serif", FontStyle::normal()))
            .expect("No fonts available");

        let label_font = Font::from_typeface(&typeface_bold, 16.0);
        let title_font = Font::from_typeface(&typeface_bold, 18.0);
        let legend_font = Font::from_typeface(&typeface_normal, 13.0);
        let footer_font = Font::from_typeface(&typeface_bold, 14.0);

        let mut white_paint = Paint::default();
        white_paint.set_color(Color::WHITE);
        white_paint.set_anti_alias(true);

        let mut black_paint = Paint::default();
        black_paint.set_color(Color::BLACK);
        black_paint.set_anti_alias(true);

        let mut sidebar_bg_paint = Paint::default();
        sidebar_bg_paint.set_color(Color::from_argb(210, 15, 18, 25));

        let mut separator_paint = Paint::default();
        separator_paint.set_color(Color::from_argb(100, 255, 255, 255));
        separator_paint.set_stroke_width(1.0);

        let mut divider_paint = Paint::default();
        divider_paint.set_color(Color::from_argb(60, 255, 255, 255));
        divider_paint.set_stroke_width(1.0);

        let mut footer_bg_paint = Paint::default();
        footer_bg_paint.set_color(Color::from_argb(180, 0, 0, 0));

        let mut footer_text_paint = Paint::default();
        footer_text_paint.set_color(Color::from_argb(200, 200, 200, 200));
        footer_text_paint.set_anti_alias(true);

        let mut fill_paint = Paint::default();
        fill_paint.set_anti_alias(true);

        let mut stroke_paint = Paint::default();
        stroke_paint.set_style(PaintStyle::Stroke);
        stroke_paint.set_stroke_width(2.0);
        stroke_paint.set_anti_alias(true);

        let label_bg_paint = Paint::default();

        let mut dot_paint = Paint::default();
        dot_paint.set_anti_alias(true);

        let mut legend_text_paint = Paint::default();
        legend_text_paint.set_anti_alias(true);
        legend_text_paint.set_color(Color::from_argb(220, 255, 255, 255));

        let bg_paint = Paint::default();

        Self {
            label_font,
            title_font,
            legend_font,
            footer_font,
            white_paint,
            black_paint,
            sidebar_bg_paint,
            separator_paint,
            divider_paint,
            footer_bg_paint,
            footer_text_paint,
            fill_paint,
            stroke_paint,
            label_bg_paint,
            dot_paint,
            legend_text_paint,
            bg_paint,
            buf: String::with_capacity(128),
            boxes: Vec::with_capacity(NUM_BOXES),
        }
    }
}

/// Draw one frame of the detection-overlay demo.
///
/// Renders:
/// - Dark gradient background
/// - `NUM_BOXES` animated semi-transparent bounding boxes with class labels
///   and confidence scores
/// - A sidebar legend listing every box's class, id, and position
/// - Frame counter and resolution info
fn draw_frame(
    renderer: &mut SkiaRenderer,
    ctx: &mut DrawCtx,
    frame_idx: u64,
    width: f32,
    height: f32,
) {
    use std::fmt::Write as _;
    let canvas = renderer.canvas();

    // ── Background gradient ──────────────────────────────────────────────────
    let hue_shift = (frame_idx as f32 * 0.3) % 360.0;
    let bg1 = hsv_to_color(hue_shift, 0.15, 0.10);
    let bg2 = hsv_to_color(hue_shift + 40.0, 0.20, 0.18);
    if let Some(shader) = gradient_shader::linear(
        (Point::new(0.0, 0.0), Point::new(width, height)),
        [bg1, bg2].as_ref(),
        None,
        TileMode::Clamp,
        None,
        None,
    ) {
        ctx.bg_paint.set_shader(shader);
        canvas.draw_rect(Rect::from_xywh(0.0, 0.0, width, height), &ctx.bg_paint);
        ctx.bg_paint.set_shader(None);
    } else {
        canvas.clear(Color::from_argb(255, 18, 20, 28));
    }

    // ── Sidebar dimensions ───────────────────────────────────────────────────
    let sidebar_w = 340.0_f32.min(width * 0.22);
    let scene_w = width - sidebar_w;
    let t = frame_idx as f32 / 60.0;

    // ── Generate bounding boxes ──────────────────────────────────────────────
    ctx.boxes.clear();
    for i in 0..NUM_BOXES {
        let seed = i as u64;
        let cx_base = pseudo_rand(seed, 100) * scene_w * 0.7 + scene_w * 0.15;
        let cy_base = pseudo_rand(seed, 200) * height * 0.7 + height * 0.15;
        let orbit_rx = pseudo_rand(seed, 300) * scene_w * 0.12 + 20.0;
        let orbit_ry = pseudo_rand(seed, 400) * height * 0.10 + 15.0;
        let speed = 0.3 + pseudo_rand(seed, 500) * 0.7;
        let phase = pseudo_rand(seed, 600) * std::f32::consts::TAU;

        let cx = cx_base + (t * speed + phase).cos() * orbit_rx;
        let cy = cy_base + (t * speed * 0.8 + phase).sin() * orbit_ry;

        let bw = 50.0 + pseudo_rand(seed, 700) * 140.0;
        let bh = 40.0 + pseudo_rand(seed, 800) * 160.0;
        let class_idx = (pseudo_rand(seed, 900) * CLASSES.len() as f32) as usize % CLASSES.len();
        let confidence = 0.55 + pseudo_rand(seed, 1000) * 0.44;

        ctx.boxes.push(BBox {
            x: (cx - bw / 2.0).clamp(0.0, scene_w - bw),
            y: (cy - bh / 2.0).clamp(0.0, height - bh),
            w: bw,
            h: bh,
            class_idx,
            confidence,
            id: i as u32,
        });
    }

    // ── Draw bounding boxes ──────────────────────────────────────────────────
    for b in &ctx.boxes {
        let cls = &CLASSES[b.class_idx];
        let r = Rect::from_xywh(b.x, b.y, b.w, b.h);

        // Semi-transparent fill
        ctx.fill_paint.set_color(Color::from_argb(50, cls.color.r(), cls.color.g(), cls.color.b()));
        canvas.draw_rect(r, &ctx.fill_paint);

        // Border
        ctx.stroke_paint.set_color(cls.color);
        canvas.draw_rect(r, &ctx.stroke_paint);

        // Label background
        ctx.buf.clear();
        let _ = write!(ctx.buf, "{} #{} {:.0}%", cls.name, b.id, b.confidence * 100.0);
        let (tw, _) = ctx.label_font.measure_str(&ctx.buf, Some(&ctx.white_paint));
        let lh = 22.0_f32;
        let lx = b.x;
        let ly = if b.y >= lh + 2.0 { b.y - lh - 2.0 } else { b.y };

        ctx.label_bg_paint.set_color(Color::from_argb(200, cls.color.r(), cls.color.g(), cls.color.b()));
        canvas.draw_rect(Rect::from_xywh(lx, ly, tw + 10.0, lh), &ctx.label_bg_paint);

        // Label text
        canvas.draw_str(&ctx.buf, Point::new(lx + 5.0, ly + lh - 5.0), &ctx.label_font, &ctx.black_paint);
    }

    // ── Sidebar ──────────────────────────────────────────────────────────────
    let sx = scene_w;
    canvas.draw_rect(Rect::from_xywh(sx, 0.0, sidebar_w, height), &ctx.sidebar_bg_paint);
    canvas.draw_line(Point::new(sx, 0.0), Point::new(sx, height), &ctx.separator_paint);

    // Title
    canvas.draw_str("DETECTIONS", Point::new(sx + 12.0, 28.0), &ctx.title_font, &ctx.white_paint);
    canvas.draw_line(
        Point::new(sx + 8.0, 36.0),
        Point::new(sx + sidebar_w - 8.0, 36.0),
        &ctx.divider_paint,
    );

    // Legend entries
    let mut y_off = 52.0_f32;
    let row_h = 18.0_f32;

    for (i, b) in ctx.boxes.iter().enumerate() {
        if y_off + row_h > height - 40.0 {
            ctx.buf.clear();
            let _ = write!(ctx.buf, "... +{} more", NUM_BOXES - i);
            ctx.legend_text_paint.set_color(Color::from_argb(180, 255, 255, 255));
            canvas.draw_str(&ctx.buf, Point::new(sx + 12.0, y_off + 14.0), &ctx.legend_font, &ctx.legend_text_paint);
            break;
        }

        let cls = &CLASSES[b.class_idx];

        // Coloured dot
        ctx.dot_paint.set_color(cls.color);
        canvas.draw_circle(Point::new(sx + 16.0, y_off + 6.0), 4.0, &ctx.dot_paint);

        // Entry text
        ctx.buf.clear();
        let _ = write!(
            ctx.buf,
            "{:<8} #{:<2} ({:>4},{:>4}) {:>3.0}%",
            cls.name, b.id, b.x as i32, b.y as i32, b.confidence * 100.0,
        );
        ctx.legend_text_paint.set_color(Color::from_argb(220, 255, 255, 255));
        canvas.draw_str(&ctx.buf, Point::new(sx + 26.0, y_off + 10.0), &ctx.legend_font, &ctx.legend_text_paint);

        y_off += row_h;
    }

    // ── Footer ───────────────────────────────────────────────────────────────
    canvas.draw_rect(Rect::from_xywh(sx, height - 32.0, sidebar_w, 32.0), &ctx.footer_bg_paint);
    ctx.buf.clear();
    let _ = write!(ctx.buf, "F:{:>6} {}x{} {}obj", frame_idx, width as u32, height as u32, NUM_BOXES);
    canvas.draw_str(&ctx.buf, Point::new(sx + 10.0, height - 11.0), &ctx.footer_font, &ctx.footer_text_paint);
}

/// Simple HSV to sRGB Color conversion.
fn hsv_to_color(h: f32, s: f32, v: f32) -> Color {
    let h = ((h % 360.0) + 360.0) % 360.0;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Color::from_argb(
        255,
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

fn main() {
    let args = Args::parse();

    // -- Init -----------------------------------------------------------------
    gst::init().expect("Failed to initialize GStreamer");
    cuda_init(args.gpu_id).expect("Failed to initialize CUDA");

    let frame_duration_ns: u64 = (1_000_000_000u64)
        .checked_div(args.fps as u64)
        .unwrap_or(33_333_333);

    // -- Skia Renderer --------------------------------------------------------
    let mut renderer = SkiaRenderer::new(args.width, args.height, args.gpu_id)
        .expect("Failed to create SkiaRenderer");
    println!(
        "SkiaRenderer created: {}x{} (gpu {})",
        args.width, args.height, args.gpu_id,
    );

    // -- Generator (RGBA -- Skia's native format) -----------------------------
    let generator = NvBufSurfaceGenerator::builder("RGBA", args.width, args.height)
        .fps(args.fps, 1)
        .gpu_id(args.gpu_id)
        .mem_type(NvBufSurfaceMemType::from(0u32))
        .min_buffers(args.pool_size)
        .max_buffers(args.pool_size)
        .build()
        .expect("Failed to create NvBufSurfaceGenerator");

    // -- Pipeline -------------------------------------------------------------
    // Skia renders RGBA, so we always need nvvideoconvert before the encoder.
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
        .expect("appsrc");
    let convert = gst::ElementFactory::make("nvvideoconvert")
        .name("convert")
        .build()
        .expect("nvvideoconvert");
    let enc = gst::ElementFactory::make(enc_name)
        .name("enc")
        .build()
        .unwrap_or_else(|_| panic!("{}", enc_name));

    // Bridge SavantIdMeta across the encoder (PTS-keyed pad probes)
    bridge_savant_id_meta(&enc);

    let parse = gst::ElementFactory::make(parse_name)
        .name("parse")
        .build()
        .unwrap_or_else(|_| panic!("{}", parse_name));

    let (sink, container_mux) = if let Some(ref path) = args.output {
        let fsink = gst::ElementFactory::make("filesink")
            .name("sink")
            .build()
            .expect("filesink");
        fsink.set_property("location", path);
        (fsink, container_mux_for_path(path))
    } else {
        let fsink = gst::ElementFactory::make("fakesink")
            .name("sink")
            .build()
            .expect("fakesink");
        fsink.set_property("sync", false);
        (fsink, None)
    };

    // Configure appsrc
    appsrc_elem.set_property("caps", &generator.nvmm_caps());
    appsrc_elem.set_property_from_str("format", "time");
    appsrc_elem.set_property_from_str("stream-type", "stream");

    // Assemble: appsrc → convert → enc → parse → [cmux →] sink
    let mut tail: Vec<&gst::Element> = Vec::new();
    if let Some(ref cm) = container_mux {
        tail.push(cm);
    }
    tail.push(&sink);

    let mut all_elems: Vec<&gst::Element> = vec![&appsrc_elem, &convert, &enc, &parse];
    all_elems.extend(&tail);
    pipeline
        .add_many(all_elems.iter().copied())
        .expect("Failed to add elements");

    let mut chain: Vec<&gst::Element> = vec![&appsrc_elem, &convert, &enc, &parse];
    chain.extend(&tail);
    gst::Element::link_many(chain).unwrap();

    let sink_label = if let Some(ref path) = args.output {
        let cmux_name = container_mux.as_ref().map(|e| {
            format!(
                "{} -> ",
                e.factory().map_or("mux".into(), |f| f.name().to_string())
            )
        });
        format!("{}filesink({})", cmux_name.unwrap_or_default(), path)
    } else {
        "fakesink".into()
    };
    println!(
        "Pipeline: Skia GPU -> appsrc(RGBA) -> nvvideoconvert -> {} -> {} -> {}",
        enc_name, parse_name, sink_label,
    );

    // -- Start ----------------------------------------------------------------
    pipeline
        .set_state(gst::State::Playing)
        .expect("Failed to start pipeline");

    let appsrc = appsrc_elem.dynamic_cast::<gstreamer_app::AppSrc>().unwrap();

    if let Some(n) = args.num_frames {
        println!("Pipeline running ({} frames)...\n", n);
    } else {
        println!("Pipeline running (Ctrl+C to stop)...\n");
    }

    // -- Ctrl+C handling ------------------------------------------------------
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Ctrl+C handler");

    // -- Stats reporter -------------------------------------------------------
    let frame_count = Arc::new(AtomicU64::new(0));
    let fc = frame_count.clone();
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

    // -- Push loop ------------------------------------------------------------
    let limit = args.num_frames.unwrap_or(u64::MAX);
    let w = args.width as f32;
    let h = args.height as f32;
    let mut i: u64 = 0;
    let mut draw_ctx = DrawCtx::new();

    while i < limit && running.load(Ordering::SeqCst) {
        // 1. Draw the frame with Skia
        draw_frame(&mut renderer, &mut draw_ctx, i, w, h);

        // 2. Acquire NvBufSurface buffer from the pool
        let mut buffer = match generator.acquire_surface(Some(i as i64)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("acquire_surface failed at frame {}: {}", i, e);
                break;
            }
        };

        // 3. Copy rendered pixels to NvBufSurface (GPU-to-GPU)
        if let Err(e) = renderer.render_to_nvbuf(buffer.make_mut(), None) {
            eprintln!("render_to_nvbuf failed at frame {}: {}", i, e);
            break;
        }

        // 4. Set timestamps and push
        let pts_ns = i * frame_duration_ns;
        {
            let buf = buffer.make_mut();
            buf.set_pts(gst::ClockTime::from_nseconds(pts_ns));
            buf.set_duration(gst::ClockTime::from_nseconds(frame_duration_ns));
        }
        match appsrc.push_buffer(buffer) {
            Ok(_) => {
                frame_count.fetch_add(1, Ordering::Relaxed);
                i += 1;
            }
            Err(e) => {
                eprintln!("Push failed at frame {}: {}", i, e);
                break;
            }
        }

        // Check for pipeline errors
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

    // -- Shutdown -------------------------------------------------------------
    println!("\nStopping...");
    running.store(false, Ordering::SeqCst);

    let _ = NvBufSurfaceGenerator::send_eos(&appsrc);
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
