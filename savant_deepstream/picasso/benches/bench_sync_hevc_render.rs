//! Criterion benchmark: FullHD HEVC low-latency throughput with object drawing.
//!
//! Measures end-to-end pipeline latency: send frame → Skia render (N objects) →
//! HEVC encode → receive encoded output.
//!
//! Methodology:
//! 1. Engine created once outside the benchmark loop.
//! 2. A few warm-up frames are sent to prime the encoder and Skia.
//! 3. Each criterion iteration: send one frame, block until encoded output arrives.
//!
//! Benchmarks are parameterized by object count: 20, 50, 100.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p picasso --bench bench_hevc_throughput
//! ```

use criterion::{criterion_group, criterion_main, Criterion};
use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::{SurfaceView, TransformConfig};
use picasso::prelude::*;
use savant_core::draw::{
    BoundingBoxDraw, ColorDraw, DotDraw, LabelDraw, LabelPosition, ObjectDraw, PaddingDraw,
};
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::object::{IdCollisionResolutionPolicy, VideoObjectBuilder};
use savant_core::primitives::RBBox;
use std::cell::Cell;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Once;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const FPS: i32 = 30;
const FRAME_DURATION_NS: u64 = 1_000_000_000 / FPS as u64;
const WARMUP_FRAMES: u64 = 5;

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(|| {
        gstreamer::init().expect("GStreamer init failed");
        cuda_init(0).expect("CUDA init failed");
    });
}

/// Callback that signals when a VideoFrame (not EOS) is encoded.
///
/// Uses a bounded (non-rendezvous) channel so the drain thread never blocks
/// inside the callback — critical because the drain thread runs independently.
struct EncodedSignal(mpsc::SyncSender<()>);

impl OnEncodedFrame for EncodedSignal {
    fn call(&self, output: EncodedOutput) {
        if let EncodedOutput::VideoFrame(_) = output {
            let _ = self.0.try_send(());
        }
    }
}

/// Builds HEVC encoder config for FullHD with low-latency tuning.
fn hevc_low_latency_encoder_config() -> EncoderConfig {
    let props = EncoderProperties::HevcDgpu(HevcDgpuProps {
        preset: Some(DgpuPreset::P1),
        tuning_info: Some(TuningPreset::LowLatency),
        ..Default::default()
    });
    EncoderConfig::new(Codec::Hevc, WIDTH, HEIGHT)
        .format(VideoFormat::RGBA)
        .fps(FPS, 1)
        .properties(props)
}

/// Builds ObjectDrawSpec with bbox, dot, and label for the "det"/"obj" class.
fn build_draw_spec() -> ObjectDrawSpec {
    let mut spec = ObjectDrawSpec::new();
    let border = ColorDraw::new(0, 255, 0, 255).unwrap();
    let bg = ColorDraw::new(0, 255, 0, 50).unwrap();
    let bb = BoundingBoxDraw::new(border, bg, 2, PaddingDraw::default_padding()).unwrap();
    let dot_color = ColorDraw::new(0, 255, 0, 255).unwrap();
    let dot = DotDraw::new(dot_color, 4).unwrap();
    let font_color = ColorDraw::new(0, 0, 0, 255).unwrap();
    let label_bg = ColorDraw::new(0, 255, 0, 200).unwrap();
    let label_border = ColorDraw::new(0, 0, 0, 0).unwrap();
    let label = LabelDraw::new(
        font_color,
        label_bg,
        label_border,
        1.4,
        1,
        LabelPosition::default_position().unwrap(),
        PaddingDraw::new(4, 2, 4, 2).unwrap(),
        vec!["{label}".to_string()],
    )
    .unwrap();
    spec.insert(
        "det",
        "obj",
        ObjectDraw::new(Some(bb), Some(dot), Some(label), false),
    );
    spec
}

/// Creates a FullHD frame with `num_objects` detection boxes for Skia rendering.
fn make_frame_with_objects(source_id: &str, frame_idx: i64, num_objects: usize) -> VideoFrameProxy {
    let pts = (frame_idx as u64 * FRAME_DURATION_NS) as i64;
    let frame = VideoFrameProxy::new(
        source_id,
        "30/1",
        WIDTH as i64,
        HEIGHT as i64,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        &None,
        None,
        (1, 1_000_000_000),
        pts,
        None,
        Some(FRAME_DURATION_NS as i64),
    )
    .unwrap();

    let scene_w = WIDTH as f32;
    let scene_h = HEIGHT as f32;
    for i in 0..num_objects {
        let cx = 50.0 + (i as f32 * 37.0) % (scene_w - 100.0);
        let cy = 50.0 + (i as f32 * 23.0) % (scene_h - 100.0);
        let obj = VideoObjectBuilder::default()
            .id(0)
            .namespace("det".to_string())
            .label("obj".to_string())
            .detection_box(RBBox::new(cx, cy, 60.0, 40.0, None))
            .confidence(Some(0.9))
            .build()
            .unwrap();
        let _ = frame.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId);
    }
    frame
}

/// Acquires an RGBA GPU surface from the generator for the given frame index.
fn make_gpu_buffer(gen: &DsNvSurfaceBufferGenerator, frame_idx: i64) -> gstreamer::Buffer {
    gen.acquire_surface(Some(frame_idx)).unwrap()
}

fn bench_sync_hevc_render(c: &mut Criterion) {
    ensure_init();

    let mut group = c.benchmark_group("hevc_fullhd_low_latency");
    group.sample_size(50);
    group.warm_up_time(std::time::Duration::from_secs(3));
    group.measurement_time(std::time::Duration::from_secs(10));

    let (tx, rx) = mpsc::sync_channel::<()>(16);
    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(EncodedSignal(tx))),
        ..Default::default()
    };

    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            ..Default::default()
        },
        callbacks,
    );

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(hevc_low_latency_encoder_config()),
        },
        draw: build_draw_spec(),
        ..Default::default()
    };

    let empty_spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(hevc_low_latency_encoder_config()),
        },
        ..Default::default()
    };

    engine
        .set_source_spec("bench", spec.clone())
        .expect("set_source_spec failed");

    let gen = DsNvSurfaceBufferGenerator::new(
        VideoFormat::RGBA,
        WIDTH,
        HEIGHT,
        FPS,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("DsNvSurfaceBufferGenerator::new failed");

    let frame_counter = Cell::new(WARMUP_FRAMES as i64);

    // Warm-up: send a few frames and wait for them
    for i in 0..WARMUP_FRAMES {
        let frame = make_frame_with_objects("bench", i as i64, 0);
        let buf = make_gpu_buffer(&gen, i as i64);
        let view = SurfaceView::from_buffer(&buf, 0).unwrap();
        engine
            .send_frame("bench", frame, view, None)
            .unwrap_or_else(|e| panic!("send_frame failed: {}", e));
        rx.recv().expect("warm-up recv failed");
    }

    for num_objects in [0_usize, 10, 20, 50, 100, 200] {
        group.bench_function(format!("{num_objects}_objects"), |b| {
            if num_objects > 0 {
                engine
                    .set_source_spec("bench", spec.clone())
                    .expect("set_source_spec failed");
            } else {
                engine
                    .set_source_spec("bench", empty_spec.clone())
                    .expect("set_source_spec failed");
            }

            while rx.try_recv().is_ok() {}

            b.iter(|| {
                let idx = frame_counter.get();
                frame_counter.set(idx + 1);
                let frame = make_frame_with_objects("bench", idx, num_objects);
                let buf = make_gpu_buffer(&gen, idx);
                let view = SurfaceView::from_buffer(&buf, 0).unwrap();
                engine
                    .send_frame("bench", frame, view, None)
                    .expect("send_frame failed");
                rx.recv().expect("encoded recv failed");
            });
        });
    }
    engine.shutdown();
    group.finish();
}

criterion_group!(benches, bench_sync_hevc_render);
criterion_main!(benches);
