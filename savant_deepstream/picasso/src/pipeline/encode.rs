use crate::callbacks::{Callbacks, OnEncodedFrame};
use crate::error::PicassoError;
use crate::message::OutputMessage;
use crate::pipeline::FrameInput;
use crate::skia::context::DrawContext;
use crate::spec::source::CallbackInvocationOrder;
use deepstream_buffers::CudaStream;
use deepstream_buffers::{Padding, Rect, SkiaRenderer, TransformConfig};
use deepstream_encoders::prelude::*;
use log::{debug, error, warn};
use savant_core::geometry::{CropRect, DstInset, LetterBoxKind, ScaleSpec};
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTransformation,
};
use savant_core::primitives::object::ObjectOperations;
use savant_core::primitives::rust::VideoFrameTranscodingMethod;
use savant_core::primitives::video_codec::VideoCodec;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::spec::draw::ObjectDrawSpec;

/// Shared encoder state. The new [`NvEncoder`] uses interior mutability
/// and exposes an `&self` API, so no external lock is required.
pub(crate) type SharedEncoder = Arc<NvEncoder>;

/// Shared map of frames submitted to the encoder but not yet drained.
pub(crate) type SharedPendingFrames = Arc<parking_lot::Mutex<HashMap<u128, VideoFrameProxy>>>;

/// Handle to the background drain thread that blocks on
/// [`NvEncoder::recv_timeout`] and fires callbacks for encoded output.
pub(crate) struct DrainHandle {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Poll interval for the drain thread; controls how quickly the thread
/// reacts to the `stop` flag set by the worker.
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(25);

impl DrainHandle {
    /// Spawn a drain thread for `source_id`.
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

/// Fire the `on_gpumat` callback.
///
/// Stream synchronisation is handled by `drop(view)` — the view's
/// `CudaStream` is synced on release.
fn fire_on_gpumat(
    source_id: &str,
    callbacks: &Callbacks,
    frame: &savant_core::primitives::frame::VideoFrameProxy,
    view: &deepstream_buffers::SurfaceView,
) {
    if let Some(cb) = &callbacks.on_gpumat {
        cb.call(source_id, frame, view);
    }
}

/// Execute the Skia rendering pipeline: resolve draw specs, create/load
/// the SkiaRenderer, draw objects, optionally fire `on_render`, and
/// write back to the destination NvBufSurface.
///
/// The caller supplies the CUDA device pointer and pitch (resolved via
/// [`SurfaceView`]) so Skia can load from and render into the buffer
/// without needing platform-specific pointer resolution.
#[allow(clippy::too_many_arguments)]
fn do_skia_render(
    source_id: &str,
    input: &FrameInput,
    dst_buf: &mut gstreamer::Buffer,
    target_w: u32,
    target_h: u32,
    data_ptr: *mut std::ffi::c_void,
    pitch: usize,
    render: &mut RenderOpts<'_>,
    callbacks: &Callbacks,
    cuda_stream: &CudaStream,
) -> Result<(), PicassoError> {
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

    let skia = match render.renderer {
        Some(r) => {
            r.set_cuda_stream(cuda_stream.clone());
            unsafe {
                r.load_from_nvbuf(data_ptr as *const std::ffi::c_void, pitch)
                    .map_err(|e| PicassoError::Renderer(source_id.to_string(), e.to_string()))?;
            }
            r
        }
        None => {
            let r = unsafe {
                SkiaRenderer::from_nvbuf(
                    target_w,
                    target_h,
                    render.gpu_id,
                    data_ptr as *const std::ffi::c_void,
                    pitch,
                )
                .map_err(|e| PicassoError::Renderer(source_id.to_string(), e.to_string()))?
                .with_cuda_stream(cuda_stream.clone())
            };
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
    unsafe {
        skia.render_to_nvbuf_with_ptr(buf_ref, data_ptr, pitch, None)
            .map_err(|e| PicassoError::Renderer(source_id.to_string(), e.to_string()))?;
    }

    Ok(())
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
    encoder: &NvEncoder,
    callbacks: &Arc<Callbacks>,
    use_on_gpumat: bool,
    render: Option<&mut RenderOpts<'_>>,
    pending_frames: &SharedPendingFrames,
    src_rect: Option<&Rect>,
    callback_order: CallbackInvocationOrder,
    cuda_stream: &CudaStream,
) -> Result<(), PicassoError> {
    let (target_w, target_h);
    let shared: deepstream_buffers::SharedBuffer;

    {
        let generator_arc = encoder.generator();
        let generator = generator_arc.lock();
        target_w = generator.width();
        target_h = generator.height();

        let encoder_gpu = generator.gpu_id();
        {
            let input_guard = input.view.gst_buffer();
            if let Ok(buf_gpu) = deepstream_buffers::buffer_gpu_id(input_guard.as_ref()) {
                if buf_gpu != encoder_gpu {
                    return Err(PicassoError::GpuMismatch {
                        source_id: source_id.to_string(),
                        buffer_gpu: buf_gpu,
                        encoder_gpu,
                    });
                }
            }
            drop(input_guard);

            shared = generator
                .acquire(Some(input.frame_id))
                .map_err(|e| PicassoError::Transform(source_id.to_string(), e.to_string()))?;
            let dst_view = deepstream_buffers::SurfaceView::from_buffer(&shared, 0)
                .map_err(|e| PicassoError::Transform(source_id.to_string(), e.to_string()))?;
            input
                .view
                .transform_into(&dst_view, transform_config, src_rect)
                .map_err(|e| PicassoError::Transform(source_id.to_string(), e.to_string()))?;
        }
    }

    rewrite_frame_transformations(&input.frame, target_w, target_h, transform_config, src_rect)?;

    let gpumat_active = use_on_gpumat && callbacks.on_gpumat.is_some();
    let need_view = gpumat_active || render.is_some();

    // ONE SurfaceView for the entire encode scope — resolves the CUDA
    // pointer once and serves both Skia rendering and on_gpumat callbacks.
    // The SharedBuffer is passed to submit_frame after the view
    // is dropped.
    let view = if need_view {
        Some(
            deepstream_buffers::SurfaceView::from_buffer(&shared, 0)
                .map_err(|e| PicassoError::Transform(source_id.to_string(), e.to_string()))?
                .with_cuda_stream(cuda_stream.clone()),
        )
    } else {
        None
    };

    let (data_ptr, pitch) = view
        .as_ref()
        .map(|v| (v.data_ptr(), v.pitch() as usize))
        .unwrap_or((std::ptr::null_mut(), 0));

    // --- GpuMatSkia / GpuMatSkiaGpuMat: fire on_gpumat BEFORE Skia ---
    if gpumat_active
        && matches!(
            callback_order,
            CallbackInvocationOrder::GpuMatSkia | CallbackInvocationOrder::GpuMatSkiaGpuMat
        )
    {
        fire_on_gpumat(source_id, callbacks, &input.frame, view.as_ref().unwrap());
    }

    // --- Skia rendering ---
    if let Some(render) = render {
        let v = view.as_ref().unwrap();
        let mut buf_guard = v.gst_buffer();
        do_skia_render(
            source_id,
            &input,
            &mut buf_guard,
            target_w,
            target_h,
            data_ptr,
            pitch,
            render,
            callbacks,
            cuda_stream,
        )?;
        drop(buf_guard);
    }

    // --- SkiaGpuMat / GpuMatSkiaGpuMat: fire on_gpumat AFTER Skia ---
    if gpumat_active
        && matches!(
            callback_order,
            CallbackInvocationOrder::SkiaGpuMat | CallbackInvocationOrder::GpuMatSkiaGpuMat
        )
    {
        fire_on_gpumat(source_id, callbacks, &input.frame, view.as_ref().unwrap());
    }

    let pts = input.frame.get_pts().max(0) as u64;
    let duration = input.frame.get_duration().map(|d| d.max(0) as u64);

    let frame_id = input.frame.get_uuid_u128();

    // Drop the view before extracting the buffer for the encoder.
    drop(view);

    let buffer = shared.into_buffer().map_err(|s| {
        PicassoError::Encoder(
            source_id.to_string(),
            format!("cannot extract buffer: strong_count={}", s.strong_count()),
        )
    })?;

    // Register the frame *before* submitting to the encoder so the async
    // drain thread can never pull an encoded buffer whose frame_id is
    // missing from the map (see drain_loop correlation below).
    pending_frames.lock().insert(frame_id, input.frame);

    let submit_result = encoder.submit_frame(buffer, frame_id, pts, duration);

    if let Err(e) = submit_result {
        pending_frames.lock().remove(&frame_id);
        return Err(PicassoError::Encoder(source_id.to_string(), e.to_string()));
    }

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
        dst_inset: config.dst_padding.map(|p| DstInset {
            left: p.left as u64,
            top: p.top as u64,
            right: p.right as u64,
            bottom: p.bottom as u64,
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

/// Background loop that blocks on the encoder's output channel and
/// dispatches [`NvEncoderOutput`] variants to the appropriate callback.
///
/// Runs until:
/// * `stop` is set (worker stopping the encoder for hot-swap / EOS),
/// * the encoder emits [`NvEncoderOutput::Eos`], or
/// * the encoder transitions to a failed state.
fn drain_loop(
    source_id: &str,
    encoder: &NvEncoder,
    callbacks: &Arc<Callbacks>,
    pending_frames: &SharedPendingFrames,
    stop: &AtomicBool,
) {
    debug!("drain thread started: source={source_id}");
    while !stop.load(Ordering::Acquire) {
        match encoder.recv_timeout(DRAIN_POLL_INTERVAL) {
            Ok(Some(NvEncoderOutput::Frame(encoded))) => {
                dispatch_encoded(source_id, encoded, callbacks, pending_frames);
            }
            Ok(Some(NvEncoderOutput::SourceEos { source_id: sid })) => {
                if let Some(cb) = &callbacks.on_encoded_frame {
                    cb.call(OutputMessage::EndOfStream(EndOfStream::new(&sid)));
                }
            }
            Ok(Some(NvEncoderOutput::Event(_))) => {
                // Non-savant events from the encoder are not forwarded;
                // the picasso output protocol only surfaces VideoFrame
                // and EndOfStream.
            }
            Ok(Some(NvEncoderOutput::Eos)) => break,
            Ok(Some(NvEncoderOutput::Error(e))) => {
                error!("encoder drain error: source={source_id}, err={e}");
                break;
            }
            Ok(None) => {}
            Err(e) => {
                error!("encoder recv error: source={source_id}, err={e}");
                break;
            }
        }
    }
    debug!("drain thread stopped: source={source_id}");
}

/// Route an encoded buffer to the [`OnEncodedFrame`] callback and update
/// the pending-frames map.  Shared between the drain thread and the
/// graceful-shutdown path.
pub(crate) fn dispatch_encoded(
    source_id: &str,
    encoded: EncodedFrame,
    callbacks: &Arc<Callbacks>,
    pending_frames: &SharedPendingFrames,
) {
    if let Some(cb) = &callbacks.on_encoded_frame {
        let frame = encoded
            .frame_id
            .and_then(|id| pending_frames.lock().remove(&id));
        if let Some(frame) = frame {
            fill_encoded_frame(frame, encoded, cb);
        } else if !encoded.data.is_empty() {
            // Stream-level codec headers (e.g. AV1 OBU_SEQUENCE_HEADER)
            // are inlined by `NvEncoder` into the next user frame, so
            // reaching this branch means the encoder emitted a
            // non-empty payload that could not be correlated to any
            // pending frame and was not recognised as a codec header.
            warn!(
                "drain: cannot correlate encoded payload ({} bytes), frame_id={:?}, source={source_id}",
                encoded.data.len(),
                encoded.frame_id
            );
        } else {
            warn!(
                "drain: no pending frame for frame_id={:?}, source={source_id} (empty buffer)",
                encoded.frame_id
            );
        }
    }
}

fn gst_codec_to_video_codec(c: Codec) -> VideoCodec {
    match c {
        Codec::H264 => VideoCodec::H264,
        Codec::Hevc => VideoCodec::Hevc,
        Codec::Jpeg => VideoCodec::Jpeg,
        Codec::Av1 => VideoCodec::Av1,
        Codec::Png => VideoCodec::Png,
        Codec::Vp8 => VideoCodec::Vp8,
        Codec::Vp9 => VideoCodec::Vp9,
        Codec::RawRgba => VideoCodec::RawRgba,
        Codec::RawRgb => VideoCodec::RawRgb,
        Codec::RawNv12 => VideoCodec::RawNv12,
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
    frame
        .set_time_base((
            i64::from(encoded.time_base.0),
            i64::from(encoded.time_base.1),
        ))
        .ok();
    frame.set_codec(Some(gst_codec_to_video_codec(encoded.codec)));
    frame.set_keyframe(Some(encoded.keyframe));
    cb.call(OutputMessage::VideoFrame(frame));
}
