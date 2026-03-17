//! Integration test: CPU-rendered image vs GPU-rendered (Picasso pipeline)
//! image compared via perceptual hashing.
//!
//! **CPU path** – Skia raster surface at 1920×1080 (black), object bboxes
//! transformed from 800×800 via the letterbox affine, drawn with the same
//! render modules as Picasso, encoded as JPEG.
//!
//! **GPU canvas path** – 800×800 NvBufSurface fed through `PicassoEngine`
//! with `Encode` + Skia rendering (H.264, → 1920×1080, symmetric padding).  The
//! `OnRender` callback captures the canvas pixels after internal object
//! drawing.
//!
//! **GPU JPEG-encoder path** – Same setup but with `Codec::Jpeg`.  The
//! encoded JPEG frame is captured from `OnEncodedFrame`.
//!
//! All images are saved to `CARGO_TARGET_TMPDIR/compare/` for manual
//! inspection.  Perceptual hashes are compared; a small Hamming distance
//! means the images are structurally equivalent.

mod common;

use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::{Padding, TransformConfig};
use picasso::prelude::*;
use picasso::skia::context::DrawContext;
use picasso::skia::object::draw_object;
use picasso::transform::compute_letterbox_params;
use savant_core::draw::{
    BoundingBoxDraw, ColorDraw, DotDraw, LabelDraw, LabelPosition, ObjectDraw, PaddingDraw,
};
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod, VideoFrameTransformation,
};
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
};
use savant_core::primitives::RBBox;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const SRC_W: u32 = 800;
const SRC_H: u32 = 800;
const DST_W: u32 = 1920;
const DST_H: u32 = 1080;
const FONT: &str = "sans-serif";

// -----------------------------------------------------------------------
// Output helpers
// -----------------------------------------------------------------------

fn output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("compare");
    std::fs::create_dir_all(&dir).expect("create compare dir");
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
// Object definitions (in 800×800 source space)
// -----------------------------------------------------------------------

struct ObjDef {
    ns: &'static str,
    label: &'static str,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    r: i64,
    g: i64,
    b: i64,
}

const OBJECTS: &[ObjDef] = &[
    ObjDef {
        ns: "det",
        label: "person",
        cx: 200.0,
        cy: 200.0,
        w: 100.0,
        h: 180.0,
        r: 255,
        g: 80,
        b: 80,
    },
    ObjDef {
        ns: "det",
        label: "car",
        cx: 500.0,
        cy: 400.0,
        w: 200.0,
        h: 120.0,
        r: 80,
        g: 200,
        b: 255,
    },
    ObjDef {
        ns: "det",
        label: "truck",
        cx: 650.0,
        cy: 600.0,
        w: 150.0,
        h: 100.0,
        r: 255,
        g: 180,
        b: 40,
    },
    ObjDef {
        ns: "det",
        label: "bicycle",
        cx: 100.0,
        cy: 600.0,
        w: 80.0,
        h: 120.0,
        r: 80,
        g: 255,
        b: 120,
    },
    ObjDef {
        ns: "det",
        label: "dog",
        cx: 400.0,
        cy: 150.0,
        w: 120.0,
        h: 90.0,
        r: 220,
        g: 100,
        b: 255,
    },
];

fn build_draw_spec() -> ObjectDrawSpec {
    let mut spec = ObjectDrawSpec::new();
    for o in OBJECTS {
        let border = ColorDraw::new(o.r, o.g, o.b, 255).unwrap();
        let bg = ColorDraw::new(o.r, o.g, o.b, 50).unwrap();
        let bb = BoundingBoxDraw::new(border, bg, 2, PaddingDraw::default_padding()).unwrap();

        let dot_c = ColorDraw::new(o.r, o.g, o.b, 255).unwrap();
        let dot = DotDraw::new(dot_c, 4).unwrap();

        let fc = ColorDraw::new(0, 0, 0, 255).unwrap();
        let lbg = ColorDraw::new(o.r, o.g, o.b, 200).unwrap();
        let lbc = ColorDraw::new(0, 0, 0, 0).unwrap();
        let ld = LabelDraw::new(
            fc,
            lbg,
            lbc,
            1.4,
            1,
            LabelPosition::default_position().unwrap(),
            PaddingDraw::new(4, 2, 4, 2).unwrap(),
            vec!["{label}".to_string()],
        )
        .unwrap();

        spec.insert(
            o.ns,
            o.label,
            ObjectDraw::new(Some(bb), Some(dot), Some(ld), false),
        );
    }
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
    for o in OBJECTS {
        let obj = VideoObjectBuilder::default()
            .id(0)
            .namespace(o.ns.to_string())
            .label(o.label.to_string())
            .detection_box(RBBox::new(o.cx, o.cy, o.w, o.h, None))
            .confidence(Some(0.95))
            .build()
            .unwrap();
        let _ = frame.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId);
    }
    frame
}

// -----------------------------------------------------------------------
// CPU rendering path
// -----------------------------------------------------------------------

fn render_cpu() -> Vec<u8> {
    let mut surface = skia_safe::surfaces::raster_n32_premul((DST_W as i32, DST_H as i32))
        .expect("raster surface");
    surface.canvas().clear(skia_safe::Color::BLACK);

    let frame = create_frame("cpu");
    let mut fm = frame.clone();
    let (ow, oh, pl, pt, pr, pb) = compute_letterbox_params(
        SRC_W as u64,
        SRC_H as u64,
        DST_W as u64,
        DST_H as u64,
        Padding::Symmetric,
        None,
    )
    .expect("compute_letterbox_params");
    fm.add_transformation(VideoFrameTransformation::LetterBox(ow, oh, pl, pt, pr, pb));
    fm.transform_forward().unwrap();

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
// GPU rendering path (Picasso pipeline)
// -----------------------------------------------------------------------

struct PixelCapture {
    pixels: Mutex<Option<Vec<u8>>>,
    ready: Condvar,
}

impl OnRender for PixelCapture {
    fn call(
        &self,
        _source_id: &str,
        renderer: &mut deepstream_nvbufsurface::SkiaRenderer,
        _frame: &VideoFrameProxy,
    ) {
        let canvas = renderer.canvas();
        let info = skia_safe::ImageInfo::new(
            (DST_W as i32, DST_H as i32),
            skia_safe::ColorType::RGBA8888,
            skia_safe::AlphaType::Premul,
            None,
        );
        let row_bytes = DST_W as usize * 4;
        let mut buf = vec![0u8; row_bytes * DST_H as usize];
        if canvas.read_pixels(&info, &mut buf, row_bytes, (0, 0)) {
            *self.pixels.lock().unwrap() = Some(buf);
            self.ready.notify_all();
        } else {
            eprintln!("WARNING: canvas.read_pixels failed");
        }
    }
}

struct CountSink(Arc<AtomicUsize>);
impl OnEncodedFrame for CountSink {
    fn call(&self, output: OutputMessage) {
        if matches!(output, OutputMessage::VideoFrame(_)) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn render_gpu() -> Vec<u8> {
    gstreamer::init().unwrap();
    cuda_init(0).unwrap();

    let capture = Arc::new(PixelCapture {
        pixels: Mutex::new(None),
        ready: Condvar::new(),
    });
    let enc_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountSink(enc_count.clone()))),
        on_render: Some(capture.clone()),
        ..Default::default()
    };

    let general = GeneralSpec {
        idle_timeout_secs: 300,
        ..Default::default()
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    let enc_config = common::h264_encoder_config(DST_W, DST_H);

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig {
                padding: Padding::Symmetric,
                ..Default::default()
            },
            encoder: Box::new(enc_config),
        },
        draw: build_draw_spec(),
        font_family: FONT.to_string(),
        use_on_render: true,
        use_on_gpumat: false,
        ..Default::default()
    };
    engine.set_source_spec("gpu", spec).unwrap();

    let gen = DsNvUniformSurfaceBufferGenerator::builder(VideoFormat::RGBA, SRC_W, SRC_H, 1)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .pool_size(32)
        .build()
        .unwrap();

    let frame = create_frame("gpu");
    let shared = gen.acquire_buffer(Some(0)).unwrap();
    {
        let mut guard = shared.lock();
        let buf_ref = guard.make_mut();
        buf_ref.set_pts(gstreamer::ClockTime::ZERO);
        buf_ref.set_duration(gstreamer::ClockTime::from_nseconds(33_333_333));
    }

    let view = deepstream_nvbufsurface::SurfaceView::from_shared(&shared, 0).unwrap();
    engine.send_frame("gpu", frame, view, None).unwrap();

    // Wait for the OnRender capture (timeout 10 s)
    let guard = capture.pixels.lock().unwrap();
    let (guard, timeout) = capture
        .ready
        .wait_timeout_while(guard, Duration::from_secs(10), |px| px.is_none())
        .unwrap();
    assert!(!timeout.timed_out(), "GPU capture timed out after 10 s");
    let rgba_pixels = guard.as_ref().unwrap().clone();
    drop(guard);

    engine.send_eos("gpu").unwrap();
    std::thread::sleep(Duration::from_millis(500));
    engine.shutdown();

    // Build a Skia raster image from the captured RGBA pixels and encode JPEG
    let info = skia_safe::ImageInfo::new(
        (DST_W as i32, DST_H as i32),
        skia_safe::ColorType::RGBA8888,
        skia_safe::AlphaType::Premul,
        None,
    );
    let row_bytes = DST_W as usize * 4;
    let data = skia_safe::Data::new_copy(&rgba_pixels);
    let img = skia_safe::images::raster_from_data(&info, data, row_bytes)
        .expect("create image from captured pixels");
    let jpeg = img
        .encode(None, skia_safe::EncodedImageFormat::JPEG, Some(95u32))
        .expect("JPEG encode captured image");
    jpeg.as_bytes().to_vec()
}

// -----------------------------------------------------------------------
// GPU JPEG-encoder path (Picasso pipeline with Codec::Jpeg)
// -----------------------------------------------------------------------

struct JpegCapture {
    data: Mutex<Option<Vec<u8>>>,
    ready: Condvar,
}

impl OnEncodedFrame for JpegCapture {
    fn call(&self, output: OutputMessage) {
        let OutputMessage::VideoFrame(frame) = output else {
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

fn render_gpu_jpeg_encoded() -> Vec<u8> {
    gstreamer::init().unwrap();
    cuda_init(0).unwrap();

    let jpeg_capture = Arc::new(JpegCapture {
        data: Mutex::new(None),
        ready: Condvar::new(),
    });

    let callbacks = Callbacks {
        on_encoded_frame: Some(jpeg_capture.clone()),
        ..Default::default()
    };

    let general = GeneralSpec {
        idle_timeout_secs: 300,
        ..Default::default()
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    let enc_config = EncoderConfig::new(Codec::Jpeg, DST_W, DST_H)
        .format(VideoFormat::RGBA)
        .fps(30, 1)
        .properties(EncoderProperties::Jpeg(JpegProps { quality: Some(95) }));

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig {
                padding: Padding::Symmetric,
                ..Default::default()
            },
            encoder: Box::new(enc_config),
        },
        draw: build_draw_spec(),
        font_family: FONT.to_string(),
        use_on_render: false,
        use_on_gpumat: false,
        ..Default::default()
    };
    engine.set_source_spec("gpu-jpeg", spec).unwrap();

    let gen = DsNvUniformSurfaceBufferGenerator::builder(VideoFormat::RGBA, SRC_W, SRC_H, 1)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .pool_size(32)
        .build()
        .unwrap();

    let frame = create_frame("gpu-jpeg");
    let shared = gen.acquire_buffer(Some(0)).unwrap();
    {
        let mut guard = shared.lock();
        let buf_ref = guard.make_mut();
        buf_ref.set_pts(gstreamer::ClockTime::ZERO);
        buf_ref.set_duration(gstreamer::ClockTime::from_nseconds(33_333_333));
    }

    let view = deepstream_nvbufsurface::SurfaceView::from_shared(&shared, 0).unwrap();
    engine.send_frame("gpu-jpeg", frame, view, None).unwrap();

    // EOS triggers encoder flush which produces the pending JPEG frame.
    engine.send_eos("gpu-jpeg").unwrap();

    // Wait for the JPEG frame from the encoder (timeout 10 s)
    let guard = jpeg_capture.data.lock().unwrap();
    let (guard, timeout) = jpeg_capture
        .ready
        .wait_timeout_while(guard, Duration::from_secs(10), |d| d.is_none())
        .unwrap();
    assert!(
        !timeout.timed_out(),
        "JPEG encoder capture timed out after 10 s"
    );
    let jpeg_bytes = guard.as_ref().unwrap().clone();
    drop(guard);

    std::thread::sleep(Duration::from_millis(200));
    engine.shutdown();

    jpeg_bytes
}

// -----------------------------------------------------------------------
// Hamming distance for imagehash
// -----------------------------------------------------------------------

fn hamming_distance(a: &imagehash::Hash, b: &imagehash::Hash) -> usize {
    a.bits
        .iter()
        .zip(b.bits.iter())
        .filter(|(x, y)| x != y)
        .count()
}

// -----------------------------------------------------------------------
// Test
// -----------------------------------------------------------------------

#[test]
fn cpu_gpu_render_similarity() {
    let _ = gstreamer::init();
    let _ = cuda_init(0);
    if !common::has_nvenc() {
        eprintln!("NVENC not available — skipping H264 test");
        return;
    }
    let cpu_jpeg = render_cpu();
    let gpu_canvas_jpeg = render_gpu();
    let gpu_encoded_jpeg = render_gpu_jpeg_encoded();

    let dir = output_dir();
    let cpu_path = dir.join("cpu.jpg");
    let gpu_canvas_path = dir.join("gpu_canvas.jpg");
    let gpu_encoded_path = dir.join("gpu_encoded.jpg");
    std::fs::write(&cpu_path, &cpu_jpeg).unwrap();
    std::fs::write(&gpu_canvas_path, &gpu_canvas_jpeg).unwrap();
    std::fs::write(&gpu_encoded_path, &gpu_encoded_jpeg).unwrap();
    eprintln!("CPU image:         {}", cpu_path.display());
    eprintln!("GPU canvas image:  {}", gpu_canvas_path.display());
    eprintln!("GPU encoded image: {}", gpu_encoded_path.display());

    let cpu_img = image::load_from_memory(&cpu_jpeg).expect("load CPU JPEG");
    let gpu_canvas_img = image::load_from_memory(&gpu_canvas_jpeg).expect("load GPU canvas JPEG");
    let gpu_encoded_img =
        image::load_from_memory(&gpu_encoded_jpeg).expect("load GPU encoded JPEG");

    let cpu_hash = imagehash::perceptual_hash(&cpu_img);
    let gpu_canvas_hash = imagehash::perceptual_hash(&gpu_canvas_img);
    let gpu_encoded_hash = imagehash::perceptual_hash(&gpu_encoded_img);

    let dist_canvas = hamming_distance(&cpu_hash, &gpu_canvas_hash);
    let dist_encoded = hamming_distance(&cpu_hash, &gpu_encoded_hash);
    let dist_canvas_encoded = hamming_distance(&gpu_canvas_hash, &gpu_encoded_hash);

    eprintln!("CPU pHash:         {cpu_hash}");
    eprintln!("GPU canvas pHash:  {gpu_canvas_hash}");
    eprintln!("GPU encoded pHash: {gpu_encoded_hash}");
    eprintln!("CPU ↔ GPU canvas:  {dist_canvas} / 64");
    eprintln!("CPU ↔ GPU encoded: {dist_encoded} / 64");
    eprintln!("canvas ↔ encoded:  {dist_canvas_encoded} / 64");

    assert!(
        dist_canvas <= 10,
        "CPU vs GPU canvas too different: hamming = {dist_canvas}/64. \
         Inspect {cpu_path:?} vs {gpu_canvas_path:?}",
    );
    assert!(
        dist_encoded <= 10,
        "CPU vs GPU JPEG-encoded too different: hamming = {dist_encoded}/64. \
         Inspect {cpu_path:?} vs {gpu_encoded_path:?}",
    );
    assert!(
        dist_canvas_encoded <= 10,
        "GPU canvas vs GPU JPEG-encoded too different: hamming = {dist_canvas_encoded}/64. \
         Inspect {gpu_canvas_path:?} vs {gpu_encoded_path:?}",
    );
}
