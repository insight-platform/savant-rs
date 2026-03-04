//! E2E test: Picasso pipeline with CPU-based PNG encoding.
//!
//! Verifies that the Picasso pipeline correctly encodes frames to PNG using
//! GStreamer's pngenc (gst-plugins-good) via the nvvideoconvert -> pngenc
//! chain. The encoded PNG is captured from OnEncodedFrame and validated.

use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::{Padding, TransformConfig};
use picasso::prelude::*;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::object::{IdCollisionResolutionPolicy, VideoObjectBuilder};
use savant_core::primitives::RBBox;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const SRC_W: u32 = 320;
const SRC_H: u32 = 240;
const DST_W: u32 = 640;
const DST_H: u32 = 480;

/// PNG magic bytes.
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

fn output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("png_e2e");
    std::fs::create_dir_all(&dir).expect("create png_e2e dir");
    dir
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
    // Add a simple detection object for visual verification
    let obj = VideoObjectBuilder::default()
        .id(0)
        .namespace("det".to_string())
        .label("person".to_string())
        .detection_box(RBBox::new(160.0, 120.0, 80.0, 120.0, None))
        .confidence(Some(0.95))
        .build()
        .unwrap();
    let _ = frame.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId);
    frame
}

struct PngCapture {
    data: Mutex<Option<Vec<u8>>>,
    ready: Condvar,
}

impl OnEncodedFrame for PngCapture {
    fn call(&self, output: EncodedOutput) {
        let EncodedOutput::VideoFrame(frame) = output else {
            return;
        };
        let mut guard = self.data.lock().unwrap();
        if guard.is_some() {
            return;
        }
        let content = frame.get_content();
        if let VideoFrameContent::Internal(data) = content.as_ref() {
            *guard = Some(data.clone());
            self.ready.notify_all();
        }
    }
}

fn render_png_encoded() -> Vec<u8> {
    gstreamer::init().unwrap();
    cuda_init(0).unwrap();

    let png_capture = Arc::new(PngCapture {
        data: Mutex::new(None),
        ready: Condvar::new(),
    });

    let callbacks = Callbacks {
        on_encoded_frame: Some(png_capture.clone()),
        ..Default::default()
    };

    let general = GeneralSpec {
        idle_timeout_secs: 300,
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    let enc_config = EncoderConfig::new(Codec::Png, DST_W, DST_H)
        .format(VideoFormat::RGBA)
        .fps(30, 1)
        .properties(EncoderProperties::Png(PngProps {
            compression_level: Some(6),
        }));

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig {
                padding: Padding::Symmetric,
                ..Default::default()
            },
            encoder: Box::new(enc_config),
        },
        draw: ObjectDrawSpec::new(),
        font_family: "sans-serif".to_string(),
        use_on_render: false,
        use_on_gpumat: false,
        ..Default::default()
    };
    engine.set_source_spec("png", spec).unwrap();

    let gen = DsNvSurfaceBufferGenerator::new(
        VideoFormat::RGBA,
        SRC_W,
        SRC_H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let frame = create_frame("png");
    let mut buf = gen.acquire_surface(Some(0)).unwrap();
    {
        let buf_ref = buf.make_mut();
        buf_ref.set_pts(gstreamer::ClockTime::ZERO);
        buf_ref.set_duration(gstreamer::ClockTime::from_nseconds(33_333_333));
    }

    let view = deepstream_nvbufsurface::SurfaceView::from_buffer(&buf, 0).unwrap();
    engine.send_frame("png", frame, view, None).unwrap();
    engine.send_eos("png").unwrap();

    let guard = png_capture.data.lock().unwrap();
    let (guard, timeout) = png_capture
        .ready
        .wait_timeout_while(guard, Duration::from_secs(10), |d| d.is_none())
        .unwrap();
    assert!(
        !timeout.timed_out(),
        "PNG encoder capture timed out after 10 s"
    );
    let png_bytes = guard.as_ref().unwrap().clone();
    drop(guard);

    std::thread::sleep(Duration::from_millis(200));
    engine.shutdown();

    png_bytes
}

#[test]
fn png_encode_produces_valid_image() {
    let png_bytes = render_png_encoded();

    // Verify PNG signature
    assert!(
        png_bytes.len() >= PNG_SIGNATURE.len(),
        "PNG output too short: {} bytes",
        png_bytes.len()
    );
    assert_eq!(
        &png_bytes[..PNG_SIGNATURE.len()],
        PNG_SIGNATURE,
        "Invalid PNG signature"
    );

    // Load and verify dimensions using image crate
    let img = image::load_from_memory(&png_bytes).expect("load PNG from memory");
    assert_eq!(
        img.width(),
        DST_W,
        "PNG width mismatch: expected {}, got {}",
        DST_W,
        img.width()
    );
    assert_eq!(
        img.height(),
        DST_H,
        "PNG height mismatch: expected {}, got {}",
        DST_H,
        img.height()
    );

    // Save for manual inspection
    let path = output_dir().join("picasso_png_e2e.png");
    std::fs::write(&path, &png_bytes).unwrap();
    eprintln!("PNG saved to {}", path.display());
}
