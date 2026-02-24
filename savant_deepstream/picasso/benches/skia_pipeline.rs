//! Benchmark exercising the **Picasso** pipeline end-to-end with multiple
//! concurrent sources.
//!
//! Feeds `NUM_FRAMES` FullHD (1920x1080) frames **per source** through
//! `PicassoEngine` using the `Encode` path with Skia rendering.  Each frame carries 20
//! detection objects whose bounding boxes, labels and central dots are drawn by
//! Picasso's internal `ObjectDrawSpec`-driven renderer.  The `on_render`
//! callback adds only the scene-level overlay (background gradient, sidebar
//! legend, footer) — object drawing is **not** duplicated in the callback.
//!
//! The number of sources defaults to 4 and can be overridden via the
//! `BENCH_NUM_SOURCES` environment variable.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p picasso --bench skia_pipeline
//! BENCH_NUM_SOURCES=8 cargo bench -p picasso --bench skia_pipeline
//! ```

use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::TransformConfig;
use picasso::prelude::*;
use savant_core::draw::{
    BoundingBoxDraw, ColorDraw, DotDraw, LabelDraw, LabelPosition, ObjectDraw, PaddingDraw,
};
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
};
use savant_core::primitives::RBBox;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const NUM_FRAMES: u64 = 100_000;
const NUM_BOXES: usize = 20;
const FPS: i32 = 30;
const FRAME_DURATION_NS: u64 = 1_000_000_000 / FPS as u64;
const DEFAULT_NUM_SOURCES: usize = 4;

fn num_sources() -> usize {
    std::env::var("BENCH_NUM_SOURCES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_NUM_SOURCES)
}

fn source_id(idx: usize) -> String {
    format!("src-{idx}")
}

// -----------------------------------------------------------------------
// Deterministic pseudo-random (matches the Python version)
// -----------------------------------------------------------------------

fn pseudo_rand(seed1: u64, seed2: u64) -> f64 {
    let mut h = seed1
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(seed2);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    (h & 0x00FF_FFFF) as f64 / 0x0100_0000 as f64
}

fn hsv_to_argb(h_deg: f64, s: f64, v: f64) -> u32 {
    let h = ((h_deg % 360.0) + 360.0) % 360.0;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let sector = (h / 60.0) as u32 % 6;
    let (r, g, b) = match sector {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (0xFF << 24)
        | (((r + m) * 255.0) as u32) << 16
        | (((g + m) * 255.0) as u32) << 8
        | ((b + m) * 255.0) as u32
}

fn with_alpha(c: u32, a: u8) -> skia_safe::Color {
    skia_safe::Color::from_argb(a, (c >> 16) as u8, (c >> 8) as u8, c as u8)
}

// -----------------------------------------------------------------------
// Detection classes
// -----------------------------------------------------------------------

struct DetClass {
    name: &'static str,
    color: u32,
}

const CLASSES: [DetClass; 8] = [
    DetClass {
        name: "person",
        color: 0xFFFF5050,
    },
    DetClass {
        name: "car",
        color: 0xFF50C8FF,
    },
    DetClass {
        name: "truck",
        color: 0xFFFFB428,
    },
    DetClass {
        name: "bicycle",
        color: 0xFF50FF78,
    },
    DetClass {
        name: "dog",
        color: 0xFFDC64FF,
    },
    DetClass {
        name: "bus",
        color: 0xFFFFFF50,
    },
    DetClass {
        name: "bike",
        color: 0xFF50FFFF,
    },
    DetClass {
        name: "sign",
        color: 0xFFFF8C8C,
    },
];

fn class_color_by_label(label: &str) -> u32 {
    CLASSES
        .iter()
        .find(|c| c.name == label)
        .map(|c| c.color)
        .unwrap_or(0xFFFFFFFF)
}

// -----------------------------------------------------------------------
// On-render callback — draws scene-level overlay only (background,
// sidebar, footer).  Object bounding boxes / labels / dots are drawn
// by Picasso's internal draw_object_on_canvas via ObjectDrawSpec.
// -----------------------------------------------------------------------

fn make_paint(f: impl FnOnce(&mut skia_safe::Paint)) -> skia_safe::Paint {
    let mut p = skia_safe::Paint::default();
    f(&mut p);
    p
}

/// Cached Skia resources for the scene overlay.
struct OverlayCtx {
    title_font: skia_safe::Font,
    legend_font: skia_safe::Font,
    footer_font: skia_safe::Font,
    white_paint: skia_safe::Paint,
    sidebar_bg: skia_safe::Paint,
    separator: skia_safe::Paint,
    divider: skia_safe::Paint,
    footer_bg: skia_safe::Paint,
    footer_text_paint: skia_safe::Paint,
    dot_paint: skia_safe::Paint,
    legend_text: skia_safe::Paint,
}

impl OverlayCtx {
    fn new() -> Self {
        let fm = skia_safe::FontMgr::default();
        let bold_tf = fm
            .match_family_style("monospace", skia_safe::FontStyle::bold())
            .expect("monospace bold not found");
        let normal_tf = fm
            .match_family_style("monospace", skia_safe::FontStyle::normal())
            .expect("monospace normal not found");

        Self {
            title_font: skia_safe::Font::from_typeface(bold_tf.clone(), 18.0),
            legend_font: skia_safe::Font::from_typeface(normal_tf, 13.0),
            footer_font: skia_safe::Font::from_typeface(bold_tf, 14.0),
            white_paint: make_paint(|p| {
                p.set_color(skia_safe::Color::WHITE);
                p.set_anti_alias(true);
            }),
            sidebar_bg: make_paint(|p| {
                p.set_color(skia_safe::Color::from_argb(210, 15, 18, 25));
            }),
            separator: make_paint(|p| {
                p.set_color(skia_safe::Color::from_argb(100, 255, 255, 255));
                p.set_stroke_width(1.0);
                p.set_style(skia_safe::PaintStyle::Stroke);
            }),
            divider: make_paint(|p| {
                p.set_color(skia_safe::Color::from_argb(60, 255, 255, 255));
                p.set_stroke_width(1.0);
                p.set_style(skia_safe::PaintStyle::Stroke);
            }),
            footer_bg: make_paint(|p| {
                p.set_color(skia_safe::Color::from_argb(180, 0, 0, 0));
            }),
            footer_text_paint: make_paint(|p| {
                p.set_color(skia_safe::Color::from_argb(200, 200, 200, 200));
                p.set_anti_alias(true);
            }),
            dot_paint: make_paint(|p| {
                p.set_anti_alias(true);
            }),
            legend_text: make_paint(|p| {
                p.set_anti_alias(true);
                p.set_color(skia_safe::Color::from_argb(220, 255, 255, 255));
            }),
        }
    }
}

struct BenchOnRender {
    ctx: parking_lot::Mutex<OverlayCtx>,
}

impl BenchOnRender {
    fn new() -> Self {
        Self {
            ctx: parking_lot::Mutex::new(OverlayCtx::new()),
        }
    }
}

impl OnRender for BenchOnRender {
    fn call(&self, _source_id: &str, canvas: &skia_safe::Canvas, frame: &VideoFrameProxy) {
        let frame_idx = frame.get_pts() as u64 / FRAME_DURATION_NS;
        let mut ctx = self.ctx.lock();
        draw_scene_overlay(canvas, &mut ctx, frame_idx, frame);
    }
}

/// Draws scene-level overlay: background gradient, sidebar with detection
/// list, and footer.  Object-level drawing (boxes, labels, dots) is left
/// entirely to Picasso's built-in draw spec pipeline.
fn draw_scene_overlay(
    canvas: &skia_safe::Canvas,
    ctx: &mut OverlayCtx,
    frame_idx: u64,
    frame: &VideoFrameProxy,
) {
    let width = WIDTH as f32;
    let height = HEIGHT as f32;

    // Background gradient
    let hue_shift = (frame_idx as f64 * 0.3) % 360.0;
    let bg1 = hsv_to_argb(hue_shift, 0.15, 0.10);
    let bg2 = hsv_to_argb(hue_shift + 40.0, 0.20, 0.18);
    let bg_colors: &[skia_safe::Color] = &[with_alpha(bg1, 0xFF), with_alpha(bg2, 0xFF)];
    if let Some(shader) = skia_safe::Shader::linear_gradient(
        (
            skia_safe::Point::new(0.0, 0.0),
            skia_safe::Point::new(width, height),
        ),
        bg_colors,
        None,
        skia_safe::TileMode::Clamp,
        None,
        None,
    ) {
        let mut bg_paint = skia_safe::Paint::default();
        bg_paint.set_shader(shader);
        canvas.draw_rect(skia_safe::Rect::from_wh(width, height), &bg_paint);
    } else {
        canvas.clear(skia_safe::Color::from_argb(255, 18, 20, 28));
    }

    let sidebar_w = (width * 0.22).min(340.0);
    let sx = width - sidebar_w;

    // Sidebar background
    canvas.draw_rect(
        skia_safe::Rect::from_xywh(sx, 0.0, sidebar_w, height),
        &ctx.sidebar_bg,
    );
    canvas.draw_line((sx, 0.0), (sx, height), &ctx.separator);
    canvas.draw_str(
        "DETECTIONS",
        (sx + 12.0, 28.0),
        &ctx.title_font,
        &ctx.white_paint,
    );
    canvas.draw_line((sx + 8.0, 36.0), (sx + sidebar_w - 8.0, 36.0), &ctx.divider);

    // Sidebar: iterate frame objects for the detection list
    let objects = frame.get_all_objects();
    let mut y_off = 52.0f32;
    let row_h = 18.0f32;

    for (i, obj) in objects.iter().enumerate() {
        if y_off + row_h > height - 40.0 {
            ctx.legend_text
                .set_color(skia_safe::Color::from_argb(180, 255, 255, 255));
            let more = format!("... +{} more", objects.len() - i);
            canvas.draw_str(
                &more,
                (sx + 12.0, y_off + 14.0),
                &ctx.legend_font,
                &ctx.legend_text,
            );
            break;
        }

        let label = obj.get_label();
        let det = obj.get_detection_box();
        let color = class_color_by_label(&label);

        ctx.dot_paint.set_color(with_alpha(color, 0xFF));
        canvas.draw_circle((sx + 16.0, y_off + 6.0), 4.0, &ctx.dot_paint);

        let entry = format!(
            "{:<8} #{:<2} ({:>4},{:>4})",
            label,
            i,
            det.get_xc() as i32,
            det.get_yc() as i32,
        );
        ctx.legend_text
            .set_color(skia_safe::Color::from_argb(220, 255, 255, 255));
        canvas.draw_str(
            &entry,
            (sx + 26.0, y_off + 10.0),
            &ctx.legend_font,
            &ctx.legend_text,
        );

        y_off += row_h;
    }

    // Footer
    canvas.draw_rect(
        skia_safe::Rect::from_xywh(sx, height - 32.0, sidebar_w, 32.0),
        &ctx.footer_bg,
    );
    let footer = format!(
        "F:{:>6} {}x{} {}obj",
        frame_idx,
        WIDTH,
        HEIGHT,
        objects.len()
    );
    canvas.draw_str(
        &footer,
        (sx + 10.0, height - 11.0),
        &ctx.footer_font,
        &ctx.footer_text_paint,
    );
}

// -----------------------------------------------------------------------
// Encoded-frame sink callback — collects stats
// -----------------------------------------------------------------------

struct BenchEncodedSink {
    count: Arc<AtomicUsize>,
    bytes: Arc<AtomicU64>,
}

impl OnEncodedFrame for BenchEncodedSink {
    fn call(&self, output: EncodedOutput) {
        let EncodedOutput::VideoFrame(frame) = output else {
            return;
        };

        let content = frame.get_content();
        let data = match content.as_ref() {
            savant_core::primitives::frame::VideoFrameContent::Internal(d) => d,
            _ => panic!("expected Internal content on encoded frame"),
        };
        assert!(!data.is_empty(), "encoded buffer must not be empty");

        assert!(
            frame.get_source_id().starts_with("src-"),
            "unexpected source_id in encoded output: {}",
            frame.get_source_id()
        );
        assert_eq!(frame.get_width(), WIDTH as i64, "width mismatch");
        assert_eq!(frame.get_height(), HEIGHT as i64, "height mismatch");
        assert_eq!(frame.get_framerate(), "30/1", "framerate mismatch");

        let objects = frame.get_all_objects();
        assert_eq!(
            objects.len(),
            NUM_BOXES,
            "object count mismatch: expected {NUM_BOXES}, got {}",
            objects.len()
        );

        self.count.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(data.len() as u64, Ordering::Relaxed);
    }
}

// -----------------------------------------------------------------------
// Frame + object generation helpers
// -----------------------------------------------------------------------

fn make_frame(source_id: &str, w: i64, h: i64) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        "30/1",
        w,
        h,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        &None,
        None,
        (1, 1_000_000_000),
        0,
        None,
        None,
    )
    .unwrap()
}

fn make_nvmm_buffer(gen: &NvBufSurfaceGenerator, frame_id: i64) -> gstreamer::Buffer {
    let mut buf = gen.acquire_surface(Some(frame_id)).unwrap();
    {
        let buf_ref = buf.make_mut();
        buf_ref.set_pts(gstreamer::ClockTime::from_nseconds(
            frame_id as u64 * FRAME_DURATION_NS,
        ));
        buf_ref.set_duration(gstreamer::ClockTime::from_nseconds(FRAME_DURATION_NS));
    }
    buf
}

fn add_objects_to_frame(frame: &VideoFrameProxy, frame_idx: u64) {
    let scene_w = WIDTH as f32 - (WIDTH as f32 * 0.22).min(340.0);
    let height = HEIGHT as f32;
    let t = frame_idx as f32 / 60.0;

    for i in 0..NUM_BOXES {
        let seed = i as u64;
        let cx_base = pseudo_rand(seed, 100) as f32 * scene_w * 0.7 + scene_w * 0.15;
        let cy_base = pseudo_rand(seed, 200) as f32 * height * 0.7 + height * 0.15;
        let orbit_rx = pseudo_rand(seed, 300) as f32 * scene_w * 0.12 + 20.0;
        let orbit_ry = pseudo_rand(seed, 400) as f32 * height * 0.10 + 15.0;
        let speed = 0.3 + pseudo_rand(seed, 500) as f32 * 0.7;
        let phase = pseudo_rand(seed, 600) as f32 * std::f32::consts::TAU;

        let cx = cx_base + (t * speed + phase).cos() * orbit_rx;
        let cy = cy_base + (t * speed * 0.8 + phase).sin() * orbit_ry;
        let bw = 50.0 + pseudo_rand(seed, 700) as f32 * 140.0;
        let bh = 40.0 + pseudo_rand(seed, 800) as f32 * 160.0;
        let class_idx = (pseudo_rand(seed, 900) * CLASSES.len() as f64) as usize % CLASSES.len();

        let cls_name = CLASSES[class_idx].name;
        let obj = VideoObjectBuilder::default()
            .id(0)
            .namespace("detector".to_string())
            .label(cls_name.to_string())
            .detection_box(RBBox::new(cx, cy, bw, bh, None))
            .build()
            .unwrap();
        let _ = frame.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId);
    }
}

// -----------------------------------------------------------------------
// ObjectDrawSpec builder — registers bounding box + label + dot per class
// -----------------------------------------------------------------------

fn build_draw_spec() -> ObjectDrawSpec {
    let mut spec = ObjectDrawSpec::new();
    for cls in &CLASSES {
        let r = ((cls.color >> 16) & 0xFF) as i64;
        let g = ((cls.color >> 8) & 0xFF) as i64;
        let b = (cls.color & 0xFF) as i64;

        let border = ColorDraw::new(r, g, b, 255).unwrap();
        let bg = ColorDraw::new(r, g, b, 50).unwrap();
        let bb = BoundingBoxDraw::new(border, bg, 2, PaddingDraw::default_padding()).unwrap();

        let dot_color = ColorDraw::new(r, g, b, 255).unwrap();
        let dot = DotDraw::new(dot_color, 4).unwrap();

        let font_color = ColorDraw::new(0, 0, 0, 255).unwrap();
        let label_bg = ColorDraw::new(r, g, b, 200).unwrap();
        let label_border = ColorDraw::new(0, 0, 0, 0).unwrap();
        let label = LabelDraw::new(
            font_color,
            label_bg,
            label_border,
            1.4,
            1,
            LabelPosition::default_position().unwrap(),
            PaddingDraw::new(4, 2, 4, 2).unwrap(),
            vec!["{label} #{id}".to_string(), "{confidence}".to_string()],
        )
        .unwrap();

        spec.insert(
            "detector",
            cls.name,
            ObjectDraw::new(Some(bb), Some(dot), Some(label), false),
        );
    }
    spec
}

fn build_encoder_config() -> EncoderConfig {
    let enc_props = EncoderProperties::H264Dgpu(H264DgpuProps {
        bitrate: Some(4_000_000),
        preset: Some(DgpuPreset::P1),
        tuning_info: Some(TuningPreset::LowLatency),
        iframeinterval: Some(30),
        ..Default::default()
    });
    EncoderConfig::new(Codec::H264, WIDTH, HEIGHT)
        .format(VideoFormat::RGBA)
        .fps(FPS, 1)
        .properties(enc_props)
}

fn build_source_spec() -> SourceSpec {
    SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(build_encoder_config()),
        },
        draw: build_draw_spec(),
        font_family: "monospace".to_string(),
        use_on_render: true,
        use_on_gpumat: false,
        ..Default::default()
    }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

fn main() {
    gstreamer::init().unwrap();
    cuda_init(0).unwrap();

    let num_src = num_sources();
    let total_frames = NUM_FRAMES * num_src as u64;

    println!("=== Picasso Pipeline Benchmark ===");
    println!(
        "Resolution: {}x{}, Frames/source: {}, Sources: {}, Total frames: {}",
        WIDTH, HEIGHT, NUM_FRAMES, num_src, total_frames,
    );
    println!("Mode: Encode with Skia (H.264), FPS: {}", FPS);

    // -- Encoded-frame counters -------------------------------------------
    let enc_count = Arc::new(AtomicUsize::new(0));
    let enc_bytes = Arc::new(AtomicU64::new(0));

    // -- Callbacks (shared across all sources) ----------------------------
    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(BenchEncodedSink {
            count: enc_count.clone(),
            bytes: enc_bytes.clone(),
        })),
        on_render: Some(Arc::new(BenchOnRender::new())),
        ..Default::default()
    };

    // -- Engine -----------------------------------------------------------
    let general = GeneralSpec {
        idle_timeout_secs: 300,
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    // -- Register all sources ---------------------------------------------
    let source_ids: Vec<String> = (0..num_src).map(source_id).collect();
    for sid in &source_ids {
        engine.set_source_spec(sid, build_source_spec()).unwrap();
    }
    println!(
        "{} sources registered, {} draw classes (bbox+label+dot), font=monospace",
        num_src,
        CLASSES.len()
    );

    // -- One NvBufSurface generator per source ----------------------------
    let generators: Vec<NvBufSurfaceGenerator> = (0..num_src)
        .map(|_| {
            NvBufSurfaceGenerator::new(
                VideoFormat::RGBA,
                WIDTH,
                HEIGHT,
                FPS,
                1,
                0,
                NvBufSurfaceMemType::Default,
            )
            .unwrap()
        })
        .collect();
    println!(
        "Input NvBufSurface generators created: {}x {}x{} RGBA",
        num_src, WIDTH, HEIGHT
    );

    // -- Warm-up: 10 frames per source ------------------------------------
    for i in 0..10i64 {
        for (s, sid) in source_ids.iter().enumerate() {
            let frame = make_frame(sid, WIDTH as i64, HEIGHT as i64);
            add_objects_to_frame(&frame, i as u64);
            let buf = make_nvmm_buffer(&generators[s], i);
            engine.send_frame(sid, frame, buf).unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(35));
    }
    std::thread::sleep(std::time::Duration::from_secs(1));
    enc_count.store(0, Ordering::SeqCst);
    enc_bytes.store(0, Ordering::SeqCst);
    println!("Warm-up complete (10 frames x {} sources)\n", num_src);

    // -- Main loop --------------------------------------------------------
    let start = Instant::now();
    let mut last_report = Instant::now();
    let mut last_frame = 0u64;
    let mut submitted: u64 = 0;
    println!("Starting render+encode loop...\n");

    for i in 0..NUM_FRAMES {
        let frame_id = (i + 10) as i64;

        for (s, sid) in source_ids.iter().enumerate() {
            let frame = make_frame(sid, WIDTH as i64, HEIGHT as i64);
            let mut frame_mut = frame.clone();
            frame_mut.set_pts((i * FRAME_DURATION_NS) as i64).unwrap();
            add_objects_to_frame(&frame, i);
            let buf = make_nvmm_buffer(&generators[s], frame_id);
            engine.send_frame(sid, frame, buf).unwrap();
            submitted += 1;
        }

        let now = Instant::now();
        if now.duration_since(last_report).as_secs() >= 2 {
            let elapsed = now.duration_since(last_report).as_secs_f64();
            let delta = i - last_frame;
            let submit_fps = (delta * num_src as u64) as f64 / elapsed;
            let total_enc = enc_count.load(Ordering::Relaxed);
            let total_bytes = enc_bytes.load(Ordering::Relaxed);
            let rss = rss_kb();
            let gpu = gpu_mem_mib();
            println!(
                "frame {:>6}/{} | src {} | enc {:>6} | {:>7.1} fps | bitstream {:>8} KB | RSS {:>5} MB | GPU {} MiB",
                i + 1, NUM_FRAMES, num_src, total_enc, submit_fps, total_bytes / 1024, rss / 1024, gpu,
            );
            last_report = now;
            last_frame = i;
        }
    }

    // -- Drain: send EOS to all sources -----------------------------------
    for sid in &source_ids {
        engine.send_eos(sid).unwrap();
    }
    std::thread::sleep(std::time::Duration::from_secs(1));

    let total_elapsed = start.elapsed();
    let total_encoded = enc_count.load(Ordering::SeqCst);
    let total_bytes = enc_bytes.load(Ordering::SeqCst);
    let agg_fps = submitted as f64 / total_elapsed.as_secs_f64();
    let per_src_fps = agg_fps / num_src as f64;

    println!("\n=== Results ===");
    println!("Sources:            {}", num_src);
    println!("Frames/source:      {}", NUM_FRAMES);
    println!("Total submitted:    {}", submitted);
    println!("Total encoded:      {}", total_encoded);
    println!("Bitstream:          {} KB", total_bytes / 1024);
    println!("Wall time:          {:.2} s", total_elapsed.as_secs_f64());
    println!("Aggregate FPS:      {:.1}", agg_fps);
    println!("Per-source FPS:     {:.1}", per_src_fps);
    println!(
        "Per-frame avg:      {:.3} ms",
        total_elapsed.as_secs_f64() * 1000.0 / submitted as f64
    );
    println!("RSS:                {} MB", rss_kb() / 1024);
    println!("GPU memory:         {} MiB", gpu_mem_mib());

    engine.shutdown();
}

fn rss_kb() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            return parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        }
    }
    0
}

fn gpu_mem_mib() -> u64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
        .unwrap_or(0)
}
