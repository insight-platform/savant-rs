use crate::callbacks::{Callbacks, OnEncodedFrame};
use crate::error::PicassoError;
use crate::message::EncodedOutput;
use crate::pipeline::FrameInput;
use crate::skia::context::DrawContext;
use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::{Padding, Rect, SkiaRenderer, TransformConfig};
use log::{debug, error, warn};
use savant_core::geometry::{CropRect, LetterBoxKind, ScaleSpec};
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTransformation,
};
use savant_core::primitives::object::ObjectOperations;
use savant_core::primitives::rust::VideoFrameTranscodingMethod;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::spec::draw::ObjectDrawSpec;

/// Shared encoder state protected by a mutex so the worker (submit) and
/// drain threads can access it concurrently.
pub(crate) type SharedEncoder = Arc<parking_lot::Mutex<NvEncoder>>;

/// Shared map of frames submitted to the encoder but not yet drained.
pub(crate) type SharedPendingFrames = Arc<parking_lot::Mutex<HashMap<u128, VideoFrameProxy>>>;

/// Handle to the background drain thread that continuously pulls encoded
/// output from the hardware encoder, independent of frame submission.
pub(crate) struct DrainHandle {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl DrainHandle {
    /// Spawn a drain thread for `source_id` that polls the encoder for
    /// encoded output and fires callbacks.
    pub(crate) fn spawn(
        source_id: String,
        encoder: SharedEncoder,
        callbacks: Arc<Callbacks>,
        pending_frames: SharedPendingFrames,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = stop.clone();

        let thread = std::thread::Builder::new()
            .name(format!("picasso-drain-{source_id}"))
            .spawn(move || {
                drain_loop(
                    &source_id,
                    &encoder,
                    &callbacks,
                    &pending_frames,
                    &stop_flag,
                );
            })
            .expect("failed to spawn drain thread");

        Self {
            stop,
            thread: Some(thread),
        }
    }

    /// Signal the drain thread to stop and wait for it to finish.
    pub(crate) fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for DrainHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Polling interval when the encoder has no output ready.
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Process-global lock that serializes Skia EGL rendering and the
/// EGL-to-CUDA copy (`render_to_nvbuf`).  Concurrent `SkiaRenderer` GL
/// contexts on the same GPU corrupt each other's output.
static SKIA_EGL_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Render-specific options and mutable state.  When provided to
/// [`process_encode`], Skia overlays are drawn between the GPU transform
/// and the hardware encode step.
pub(crate) struct RenderOpts<'a> {
    pub(crate) draw_spec: &'a ObjectDrawSpec,
    pub(crate) use_on_render: bool,
    pub(crate) gpu_id: u32,
    pub(crate) renderer: &'a mut Option<SkiaRenderer>,
    pub(crate) draw_ctx: &'a mut DrawContext,
}

/// Unified encode pipeline:
///
/// GPU-transform → (optional Skia render) → optional on_gpumat → encode.
///
/// Draining encoded output is handled by a separate [`DrainHandle`] thread,
/// so this function only submits frames to the encoder.
///
/// When `render` is `Some`, the Skia overlay step is inserted after the GPU
/// transform.  Otherwise the path is pure transform-and-encode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_encode(
    source_id: &str,
    input: FrameInput,
    transform_config: &TransformConfig,
    encoder: &SharedEncoder,
    callbacks: &Arc<Callbacks>,
    use_on_gpumat: bool,
    render: Option<&mut RenderOpts<'_>>,
    pending_frames: &SharedPendingFrames,
    src_rect: Option<&Rect>,
) -> Result<(), PicassoError> {
    let (target_w, target_h, need_ptr);
    let (mut dst_buf, data_ptr, pitch);

    {
        let enc = encoder.lock();
        let generator = enc.generator();
        target_w = generator.width();
        target_h = generator.height();

        let encoder_gpu = generator.gpu_id();
        if let Ok(buf_gpu) = deepstream_nvbufsurface::buffer_gpu_id(input.buffer.as_ref()) {
            if buf_gpu != encoder_gpu {
                return Err(PicassoError::GpuMismatch {
                    source_id: source_id.to_string(),
                    buffer_gpu: buf_gpu,
                    encoder_gpu,
                });
            }
        }

        need_ptr = render.is_some() || (use_on_gpumat && callbacks.on_gpumat.is_some());

        if need_ptr {
            let result = generator
                .transform_with_ptr(
                    &input.buffer,
                    transform_config,
                    Some(input.frame_id as i64),
                    src_rect,
                )
                .map_err(|e| PicassoError::Transform(source_id.to_string(), e.to_string()))?;
            dst_buf = result.0;
            data_ptr = result.1;
            pitch = result.2;
        } else {
            let buf = generator
                .transform(
                    &input.buffer,
                    transform_config,
                    Some(input.frame_id as i64),
                    src_rect,
                )
                .map_err(|e| PicassoError::Transform(source_id.to_string(), e.to_string()))?;
            dst_buf = buf;
            data_ptr = std::ptr::null_mut();
            pitch = 0;
        }
    }

    rewrite_frame_transformations(&input.frame, target_w, target_h, transform_config, src_rect)?;

    if let Some(render) = render {
        let objects = input.frame.get_all_objects();
        let resolved: Vec<_> = objects
            .iter()
            .filter_map(|obj| {
                let ns = obj.get_namespace();
                let label = obj.get_label();
                let static_spec = render.draw_spec.lookup(&ns, &label);
                let cb_draw = callbacks
                    .on_object_draw_spec
                    .as_ref()
                    .and_then(|cb| cb.call(source_id, obj, static_spec));
                let from_callback = cb_draw.is_some();
                let draw = cb_draw.or_else(|| static_spec.cloned());
                draw.map(|d| {
                    let templates = if from_callback {
                        render.draw_ctx.resolve_templates_ephemeral(&ns, &label, &d)
                    } else {
                        render.draw_ctx.resolve_templates(&ns, &label, &d).cloned()
                    };
                    (obj, d, templates)
                })
            })
            .collect();

        let _egl = SKIA_EGL_LOCK.lock();

        let skia = match render.renderer {
            Some(r) => {
                r.load_from_nvbuf(data_ptr, pitch)
                    .map_err(|e| PicassoError::Renderer(source_id.to_string(), e.to_string()))?;
                r
            }
            None => {
                let r =
                    SkiaRenderer::from_nvbuf(target_w, target_h, render.gpu_id, data_ptr, pitch)
                        .map_err(|e| {
                            PicassoError::Renderer(source_id.to_string(), e.to_string())
                        })?;
                *render.renderer = Some(r);
                render.renderer.as_mut().unwrap()
            }
        };

        for (obj, d, templates) in &resolved {
            crate::skia::object::draw_object(
                skia.canvas(),
                obj,
                d,
                templates.as_ref(),
                render.draw_ctx,
            );
        }

        if render.use_on_render {
            if let Some(cb) = &callbacks.on_render {
                cb.call(source_id, skia, &input.frame);
            }
        }

        let buf_ref = dst_buf.make_mut();
        skia.render_to_nvbuf(buf_ref, None)
            .map_err(|e| PicassoError::Renderer(source_id.to_string(), e.to_string()))?;
    }

    if use_on_gpumat && !data_ptr.is_null() {
        if let Some(cb) = &callbacks.on_gpumat {
            cb.call(
                source_id,
                &input.frame,
                data_ptr as usize,
                pitch,
                target_w,
                target_h,
            );
        }
    }

    let pts = input.frame.get_pts().max(0) as u64;
    let duration = input.frame.get_duration().map(|d| d.max(0) as u64);

    let frame_id = input.frame.get_uuid_u128();

    encoder
        .lock()
        .submit_frame(dst_buf, frame_id, pts, duration)
        .map_err(|e| PicassoError::Encoder(source_id.to_string(), e.to_string()))?;

    pending_frames.lock().insert(frame_id, input.frame);

    Ok(())
}

/// Append GPU operations (crop + letterbox) to the frame's transformation
/// chain and call `transform_forward` so that object coordinates end up in
/// the target (encoded) space.
///
/// The chain is expected to contain only `InitialSize(w, h)` whose dimensions
/// match the frame's width/height.  Objects are in the InitialSize coordinate
/// space — no backward transform is needed.
pub fn rewrite_frame_transformations(
    frame: &VideoFrameProxy,
    target_w: u32,
    target_h: u32,
    config: &TransformConfig,
    src_rect: Option<&Rect>,
) -> Result<(), PicassoError> {
    let mut f = frame.clone();

    let chain = f.get_transformations();
    let src_w = f.get_width() as u64;
    let src_h = f.get_height() as u64;

    if chain.len() != 1
        || !matches!(chain[0], VideoFrameTransformation::InitialSize(w, h) if w == src_w && h == src_h)
    {
        return Err(PicassoError::InvalidTransformationChain(format!(
            "expected [InitialSize({src_w}, {src_h})], got {chain:?}"
        )));
    }

    let scale = ScaleSpec {
        source_width: src_w,
        source_height: src_h,
        dest_width: target_w as u64,
        dest_height: target_h as u64,
        letterbox: padding_to_letterbox_kind(config.padding),
        crop: src_rect.map(|r| CropRect {
            left: r.left as u64,
            top: r.top as u64,
            width: r.width as u64,
            height: r.height as u64,
        }),
    };

    let ops = scale
        .to_transformations()
        .map_err(|e| PicassoError::InvalidTransformationChain(e.to_string()))?;

    for t in ops {
        f.add_transformation(t);
    }

    f.transform_forward()
        .map_err(|e| PicassoError::InvalidTransformationChain(e.to_string()))
}

fn padding_to_letterbox_kind(padding: Padding) -> LetterBoxKind {
    match padding {
        Padding::None => LetterBoxKind::Stretch,
        Padding::Symmetric => LetterBoxKind::Symmetric,
        Padding::RightBottom => LetterBoxKind::RightBottom,
    }
}

/// Background loop that continuously pulls encoded frames from the encoder.
///
/// Runs until `stop` is set to `true`.  Each drained frame is matched to its
/// original [`VideoFrameProxy`] via `frame_id` in `pending_frames`.
fn drain_loop(
    source_id: &str,
    encoder: &SharedEncoder,
    callbacks: &Arc<Callbacks>,
    pending_frames: &SharedPendingFrames,
    stop: &AtomicBool,
) {
    debug!("drain thread started: source={source_id}");
    while !stop.load(Ordering::Acquire) {
        let encoded = {
            let mut enc = encoder.lock();
            match enc.pull_encoded() {
                Ok(frame) => frame,
                Err(e) => {
                    error!("drain error: source={source_id}, err={e}");
                    None
                }
            }
        };

        if let Some(encoded) = encoded {
            if let Some(cb) = &callbacks.on_encoded_frame {
                let frame = pending_frames.lock().remove(&encoded.frame_id);
                if let Some(frame) = frame {
                    fill_encoded_frame(frame, encoded, cb);
                } else {
                    warn!(
                        "drain: no pending frame for frame_id={}, source={source_id}",
                        encoded.frame_id
                    );
                }
            }
        } else {
            std::thread::sleep(DRAIN_POLL_INTERVAL);
        }
    }
    debug!("drain thread stopped: source={source_id}");
}

/// Pull all available encoded frames from the encoder and fire callbacks.
///
/// Used during EOS / shutdown after the drain thread has been stopped, so
/// the caller has exclusive access to the encoder.
pub(crate) fn drain_remaining(
    source_id: &str,
    encoder: &mut NvEncoder,
    callbacks: &Arc<Callbacks>,
    pending_frames: &mut HashMap<u128, VideoFrameProxy>,
) {
    loop {
        match encoder.pull_encoded() {
            Ok(Some(encoded)) => {
                if let Some(cb) = &callbacks.on_encoded_frame {
                    if let Some(frame) = pending_frames.remove(&encoded.frame_id) {
                        fill_encoded_frame(frame, encoded, cb);
                    } else {
                        warn!(
                            "drain: no pending frame for frame_id={}, source={source_id}",
                            encoded.frame_id
                        );
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                error!("encode drain error: source={source_id}, err={e}");
                break;
            }
        }
    }
}

/// Update a [`VideoFrameProxy`] with encoded output and fire the callback.
pub(crate) fn fill_encoded_frame(
    mut frame: VideoFrameProxy,
    encoded: EncodedFrame,
    cb: &Arc<dyn OnEncodedFrame>,
) {
    frame.set_content(VideoFrameContent::Internal(encoded.data));
    frame.set_transcoding_method(VideoFrameTranscodingMethod::Encoded);
    frame.set_pts(encoded.pts_ns as i64).ok();
    frame.set_dts(encoded.dts_ns.map(|v| v as i64)).ok();
    frame
        .set_duration(encoded.duration_ns.map(|v| v as i64))
        .ok();
    frame.set_time_base(encoded.time_base).ok();
    frame.set_codec(Some(encoded.codec.name().to_string()));
    frame.set_keyframe(Some(encoded.keyframe));
    cb.call(EncodedOutput::VideoFrame(frame));
}
