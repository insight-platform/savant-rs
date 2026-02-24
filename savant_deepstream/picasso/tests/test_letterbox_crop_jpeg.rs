//! Letterbox test: crop + fill with JPEG codec.
//!
//! **Source** – 1280×720 frame with a single 100×100 green bounding box at the
//! center (640, 360).
//!
//! **Crop** – 500×300 central area (left=390, top=210).
//!
//! **Fill** – 720×1280 destination:
//!   (a) Symmetric padding — content scaled to fit, black bars top/bottom.
//!   (b) Stretching — crop stretched to fill the entire destination.
//!
//! **CPU path** – Skia raster surface at 720×1280, object coordinates
//! transformed via `rewrite_frame_transformations`, encoded as JPEG.
//!
//! **GPU JPEG-encoder path** – 1280×720 `NvBufSurface` fed through
//! `PicassoEngine` with JPEG codec, crop + padding/stretch transforms.
//!
//! All JPEGs are saved to `CARGO_TARGET_TMPDIR/letterbox/` for manual
//! inspection.  Dimensions are verified programmatically.

use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::{Padding, Rect, TransformConfig};
use picasso::prelude::*;
use picasso::rewrite_frame_transformations;
use picasso::skia::context::DrawContext;
use picasso::skia::object::draw_object;
use savant_core::draw::{BoundingBoxDraw, ColorDraw, ObjectDraw, PaddingDraw};
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
};
use savant_core::primitives::RBBox;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, Once};
use std::time::Duration;

const SRC_W: u32 = 1280;
const SRC_H: u32 = 720;
const DST_W: u32 = 720;
const DST_H: u32 = 1280;
const CROP_W: u32 = 500;
const CROP_H: u32 = 300;
const CROP_LEFT: u32 = (SRC_W - CROP_W) / 2;
const CROP_TOP: u32 = (SRC_H - CROP_H) / 2;
const FONT: &str = "sans-serif";

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(|| {
        gstreamer::init().unwrap();
        cuda_init(0).unwrap();
    });
}

// -----------------------------------------------------------------------
// Output helpers
// -----------------------------------------------------------------------

fn output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("letterbox");
    std::fs::create_dir_all(&dir).expect("create letterbox dir");
    dir
}

fn skia_jpeg(surface: &mut skia_safe::Surface, quality: u32) -> Vec<u8> {
    let img = surface.image_snapshot();
    let data = img
        .encode(None, skia_safe::EncodedImageFormat::JPEG, Some(quality))
        .expect("JPEG encode failed");
    data.as_bytes().to_vec()
}

// -----------------------------------------------------------------------
// Object + draw spec
// -----------------------------------------------------------------------

fn build_draw_spec() -> ObjectDrawSpec {
    let mut spec = ObjectDrawSpec::new();
    let border = ColorDraw::new(0, 255, 0, 255).unwrap();
    let bg = ColorDraw::new(0, 255, 0, 40).unwrap();
    let bb = BoundingBoxDraw::new(border, bg, 3, PaddingDraw::default_padding()).unwrap();
    spec.insert("det", "box", ObjectDraw::new(Some(bb), None, None, false));
    spec
}

fn create_frame(source_id: &str) -> VideoFrameProxy {
    let frame = VideoFrameProxy::new(
        source_id,
        "30/1",
        SRC_W as i64,
        SRC_H as i64,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        &None,
        None,
        (1, 1_000_000_000),
        0,
        None,
        None,
    )
    .unwrap();
    let obj = VideoObjectBuilder::default()
        .id(0)
        .namespace("det".to_string())
        .label("box".to_string())
        .detection_box(RBBox::new(
            SRC_W as f32 / 2.0,
            SRC_H as f32 / 2.0,
            100.0,
            100.0,
            None,
        ))
        .confidence(Some(0.99))
        .build()
        .unwrap();
    let _ = frame.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId);
    frame
}

// -----------------------------------------------------------------------
// Transform config
// -----------------------------------------------------------------------

fn crop_rect() -> Rect {
    Rect {
        left: CROP_LEFT,
        top: CROP_TOP,
        width: CROP_W,
        height: CROP_H,
    }
}

fn transform_config(padding: Padding) -> TransformConfig {
    TransformConfig {
        padding,
        src_rect: Some(crop_rect()),
        ..Default::default()
    }
}

// -----------------------------------------------------------------------
// CPU rendering path
// -----------------------------------------------------------------------

fn render_cpu(padding: Padding) -> Vec<u8> {
    let mut surface = skia_safe::surfaces::raster_n32_premul((DST_W as i32, DST_H as i32))
        .expect("raster surface");
    surface.canvas().clear(skia_safe::Color::BLACK);

    let frame = create_frame("cpu");
    let config = transform_config(padding);
    rewrite_frame_transformations(&frame, DST_W, DST_H, &config).unwrap();

    let draw_spec = build_draw_spec();
    let mut ctx = DrawContext::new(FONT);
    ctx.rebuild_template_cache(&draw_spec);

    let objects = frame.get_all_objects();
    for obj in &objects {
        let ns = obj.get_namespace();
        let label = obj.get_label();
        if let Some(d) = draw_spec.lookup(&ns, &label) {
            let tmpl = ctx.resolve_templates(&ns, &label, d).cloned();
            draw_object(surface.canvas(), obj, d, tmpl.as_ref(), &mut ctx);
        }
    }

    skia_jpeg(&mut surface, 95)
}

// -----------------------------------------------------------------------
// GPU JPEG-encoder path
// -----------------------------------------------------------------------

struct JpegCapture {
    data: Mutex<Option<Vec<u8>>>,
    ready: Condvar,
}

impl OnEncodedFrame for JpegCapture {
    fn call(&self, output: EncodedOutput) {
        let EncodedOutput::VideoFrame(frame) = output else {
            return;
        };
        let mut guard = self.data.lock().unwrap();
        if guard.is_some() {
            return;
        }
        let content = frame.get_content();
        if let savant_core::primitives::frame::VideoFrameContent::Internal(data) = content.as_ref()
        {
            *guard = Some(data.clone());
            self.ready.notify_all();
        }
    }
}

fn render_gpu_jpeg(source_id: &str, padding: Padding) -> Vec<u8> {
    ensure_init();

    let capture = Arc::new(JpegCapture {
        data: Mutex::new(None),
        ready: Condvar::new(),
    });

    let callbacks = Callbacks {
        on_encoded_frame: Some(capture.clone()),
        ..Default::default()
    };

    let general = GeneralSpec {
        idle_timeout_secs: 300,
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    let enc_config = EncoderConfig::new(Codec::Jpeg, DST_W, DST_H)
        .format(VideoFormat::RGBA)
        .fps(30, 1)
        .properties(EncoderProperties::Jpeg(JpegProps { quality: Some(95) }));

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: transform_config(padding),
            encoder: Box::new(enc_config),
        },
        draw: build_draw_spec(),
        font_family: FONT.to_string(),
        use_on_render: false,
        use_on_gpumat: false,
        ..Default::default()
    };
    engine.set_source_spec(source_id, spec).unwrap();

    let gen = NvBufSurfaceGenerator::new(
        VideoFormat::RGBA,
        SRC_W,
        SRC_H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let frame = create_frame(source_id);
    let mut buf = gen.acquire_surface(Some(0)).unwrap();
    {
        let buf_ref = buf.make_mut();
        buf_ref.set_pts(gstreamer::ClockTime::ZERO);
        buf_ref.set_duration(gstreamer::ClockTime::from_nseconds(33_333_333));
    }

    engine.send_frame(source_id, frame, buf).unwrap();
    engine.send_eos(source_id).unwrap();

    let guard = capture.data.lock().unwrap();
    let (guard, timeout) = capture
        .ready
        .wait_timeout_while(guard, Duration::from_secs(10), |d| d.is_none())
        .unwrap();
    assert!(
        !timeout.timed_out(),
        "JPEG capture timed out after 10 s for {source_id}"
    );
    let jpeg_bytes = guard.as_ref().unwrap().clone();
    drop(guard);

    std::thread::sleep(Duration::from_millis(200));
    engine.shutdown();

    jpeg_bytes
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

fn assert_jpeg_dimensions(jpeg: &[u8], expected_w: u32, expected_h: u32, label: &str) {
    let img = image::load_from_memory(jpeg).unwrap_or_else(|e| {
        panic!("{label}: failed to decode JPEG: {e}");
    });
    let (w, h) = (img.width(), img.height());
    assert_eq!(
        (w, h),
        (expected_w, expected_h),
        "{label}: expected {expected_w}×{expected_h}, got {w}×{h}"
    );
}

/// Count pixels where the green channel dominates (G > 128, G > R+64, G > B+64).
/// The bounding box is drawn with green border/fill, so a correctly rendered
/// image must contain a meaningful number of such pixels.
fn count_green_pixels(jpeg: &[u8]) -> usize {
    use image::GenericImageView;
    let img = image::load_from_memory(jpeg).expect("decode JPEG");
    img.pixels()
        .filter(|(_, _, px)| {
            let [r, g, b, _] = px.0;
            g > 128 && g > r.saturating_add(64) && g > b.saturating_add(64)
        })
        .count()
}

fn assert_has_green_bbox(jpeg: &[u8], label: &str) {
    let green = count_green_pixels(jpeg);
    eprintln!("{label}: green pixels = {green}");
    assert!(
        green >= 50,
        "{label}: expected green bbox but found only {green} green pixels — image is empty or corrupted",
    );
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

/// Crop 500×300 center of 1280×720 → 720×1280 with symmetric padding.
#[test]
fn letterbox_crop_symmetric_padding_jpeg() {
    let dir = output_dir();

    let cpu_jpeg = render_cpu(Padding::Symmetric);
    let gpu_jpeg = render_gpu_jpeg("lb-pad", Padding::Symmetric);

    let cpu_path = dir.join("cpu_symmetric.jpg");
    let gpu_path = dir.join("gpu_symmetric.jpg");
    std::fs::write(&cpu_path, &cpu_jpeg).unwrap();
    std::fs::write(&gpu_path, &gpu_jpeg).unwrap();
    eprintln!("CPU symmetric: {}", cpu_path.display());
    eprintln!("GPU symmetric: {}", gpu_path.display());

    assert_jpeg_dimensions(&cpu_jpeg, DST_W, DST_H, "CPU symmetric");
    assert_jpeg_dimensions(&gpu_jpeg, DST_W, DST_H, "GPU symmetric");
    assert_has_green_bbox(&cpu_jpeg, "CPU symmetric");
    assert_has_green_bbox(&gpu_jpeg, "GPU symmetric");
}

/// Crop 500×300 center of 1280×720 → 720×1280 with stretching (no padding).
#[test]
fn letterbox_crop_stretch_jpeg() {
    let dir = output_dir();

    let cpu_jpeg = render_cpu(Padding::None);
    let gpu_jpeg = render_gpu_jpeg("lb-str", Padding::None);

    let cpu_path = dir.join("cpu_stretch.jpg");
    let gpu_path = dir.join("gpu_stretch.jpg");
    std::fs::write(&cpu_path, &cpu_jpeg).unwrap();
    std::fs::write(&gpu_path, &gpu_jpeg).unwrap();
    eprintln!("CPU stretch:   {}", cpu_path.display());
    eprintln!("GPU stretch:   {}", gpu_path.display());

    assert_jpeg_dimensions(&cpu_jpeg, DST_W, DST_H, "CPU stretch");
    assert_jpeg_dimensions(&gpu_jpeg, DST_W, DST_H, "GPU stretch");
    assert_has_green_bbox(&cpu_jpeg, "CPU stretch");
    assert_has_green_bbox(&gpu_jpeg, "GPU stretch");
}

// -----------------------------------------------------------------------
// Two-source-in-one-engine test
// -----------------------------------------------------------------------

use std::collections::HashMap;

/// Per-source JPEG capture that stores one frame per source_id.
struct MultiCapture {
    data: Mutex<HashMap<String, Vec<u8>>>,
    ready: Condvar,
}

impl OnEncodedFrame for MultiCapture {
    fn call(&self, output: EncodedOutput) {
        let EncodedOutput::VideoFrame(frame) = output else {
            return;
        };
        let source_id = frame.get_source_id();
        let mut guard = self.data.lock().unwrap();
        if guard.contains_key(&source_id) {
            return;
        }
        let content = frame.get_content();
        if let savant_core::primitives::frame::VideoFrameContent::Internal(data) = content.as_ref()
        {
            guard.insert(source_id, data.clone());
            self.ready.notify_all();
        }
    }
}

/// Two sources (symmetric + stretch) processed by a single `PicassoEngine`.
///
/// This exercises the per-engine GPU lock under real contention — both worker
/// threads share the same lock and compete for GPU resources.
#[test]
fn letterbox_crop_two_sources_one_engine() {
    ensure_init();

    let dir = output_dir();

    let capture = Arc::new(MultiCapture {
        data: Mutex::new(HashMap::new()),
        ready: Condvar::new(),
    });

    let callbacks = Callbacks {
        on_encoded_frame: Some(capture.clone()),
        ..Default::default()
    };

    let general = GeneralSpec {
        idle_timeout_secs: 300,
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    let make_spec = |padding: Padding| -> SourceSpec {
        let enc_config = EncoderConfig::new(Codec::Jpeg, DST_W, DST_H)
            .format(VideoFormat::RGBA)
            .fps(30, 1)
            .properties(EncoderProperties::Jpeg(JpegProps { quality: Some(95) }));
        SourceSpec {
            codec: CodecSpec::Encode {
                transform: transform_config(padding),
                encoder: Box::new(enc_config),
            },
            draw: build_draw_spec(),
            font_family: FONT.to_string(),
            use_on_render: false,
            use_on_gpumat: false,
            ..Default::default()
        }
    };

    engine
        .set_source_spec("dual-pad", make_spec(Padding::Symmetric))
        .unwrap();
    engine
        .set_source_spec("dual-str", make_spec(Padding::None))
        .unwrap();

    let gen = NvBufSurfaceGenerator::new(
        VideoFormat::RGBA,
        SRC_W,
        SRC_H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    for src in ["dual-pad", "dual-str"] {
        let frame = create_frame(src);
        let mut buf = gen.acquire_surface(Some(0)).unwrap();
        {
            let buf_ref = buf.make_mut();
            buf_ref.set_pts(gstreamer::ClockTime::ZERO);
            buf_ref.set_duration(gstreamer::ClockTime::from_nseconds(33_333_333));
        }
        engine.send_frame(src, frame, buf).unwrap();
        engine.send_eos(src).unwrap();
    }

    let guard = capture.data.lock().unwrap();
    let (guard, timeout) = capture
        .ready
        .wait_timeout_while(guard, Duration::from_secs(10), |d| d.len() < 2)
        .unwrap();
    assert!(
        !timeout.timed_out(),
        "timed out waiting for 2 JPEGs, got {}",
        guard.len()
    );
    let gpu_pad = guard["dual-pad"].clone();
    let gpu_str = guard["dual-str"].clone();
    drop(guard);

    engine.shutdown();

    let _cpu_pad = render_cpu(Padding::Symmetric);
    let _cpu_str = render_cpu(Padding::None);

    std::fs::write(dir.join("dual_gpu_symmetric.jpg"), &gpu_pad).unwrap();
    std::fs::write(dir.join("dual_gpu_stretch.jpg"), &gpu_str).unwrap();
    eprintln!(
        "dual GPU symmetric: {}",
        dir.join("dual_gpu_symmetric.jpg").display()
    );
    eprintln!(
        "dual GPU stretch:   {}",
        dir.join("dual_gpu_stretch.jpg").display()
    );

    assert_jpeg_dimensions(&gpu_pad, DST_W, DST_H, "dual symmetric");
    assert_jpeg_dimensions(&gpu_str, DST_W, DST_H, "dual stretch");
    assert_has_green_bbox(&gpu_pad, "dual GPU symmetric");
    assert_has_green_bbox(&gpu_str, "dual GPU stretch");
}
