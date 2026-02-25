use crate::callbacks::{Callbacks, OnEncodedFrame};
use crate::error::PicassoError;
use crate::message::EncodedOutput;
use crate::pipeline::FrameInput;
use crate::skia::context::DrawContext;
use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::{Padding, SkiaRenderer, TransformConfig};
use gstreamer as gst;
use log::{debug, error, warn};
use savant_core::geometry::{CropRect, LetterBoxKind, ScaleSpec};
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTransformation,
};
use savant_core::primitives::object::ObjectOperations;
use savant_core::primitives::rust::VideoFrameTranscodingMethod;
use std::collections::HashMap;
use std::sync::Arc;

use crate::spec::draw::ObjectDrawSpec;

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
/// GPU-transform → (optional Skia render) → optional on_gpumat → encode → drain.
///
/// When `render` is `Some`, the Skia overlay step is inserted after the GPU
/// transform.  Otherwise the path is pure transform-and-encode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_encode(
    source_id: &str,
    input: FrameInput,
    transform_config: &TransformConfig,
    encoder: &mut NvEncoder,
    callbacks: &Arc<Callbacks>,
    use_on_gpumat: bool,
    render: Option<&mut RenderOpts<'_>>,
    pending_frames: &mut HashMap<u128, VideoFrameProxy>,
) -> Result<(), PicassoError> {
    let generator = encoder.generator();
    let target_w = generator.width();
    let target_h = generator.height();

    let need_ptr = render.is_some() || (use_on_gpumat && callbacks.on_gpumat.is_some());

    let (mut dst_buf, data_ptr, pitch) = if need_ptr {
        generator
            .transform_with_ptr(&input.buffer, transform_config, Some(input.frame_id as i64))
            .map_err(|e| PicassoError::Transform(source_id.to_string(), e.to_string()))?
    } else {
        let buf = generator
            .transform(&input.buffer, transform_config, Some(input.frame_id as i64))
            .map_err(|e| PicassoError::Transform(source_id.to_string(), e.to_string()))?;
        (buf, std::ptr::null_mut(), 0)
    };

    rewrite_frame_transformations(&input.frame, target_w, target_h, transform_config)?;

    if let Some(render) = render {
        let objects = input.frame.get_all_objects();
        let resolved: Vec<_> = objects
            .iter()
            .filter_map(|obj| {
                let ns = obj.get_namespace();
                let label = obj.get_label();
                let static_spec = render.draw_spec.lookup(&ns, &label);
                let draw = callbacks
                    .on_object_draw_spec
                    .as_ref()
                    .and_then(|cb| cb.call(source_id, obj, static_spec))
                    .or_else(|| static_spec.cloned());
                draw.map(|d| {
                    let templates = render.draw_ctx.resolve_templates(&ns, &label, &d).cloned();
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

    let pts = frame_pts(&input.buffer);
    let duration = frame_duration(&input.buffer);

    pending_frames.insert(input.frame_id, input.frame);

    encoder
        .submit_frame(dst_buf, input.frame_id, pts, duration)
        .map_err(|e| PicassoError::Encoder(source_id.to_string(), e.to_string()))?;

    drain_encoded(source_id, encoder, callbacks, pending_frames);

    debug!("encode: source={source_id}, pts={pts}");
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
        crop: config.src_rect.as_ref().map(|r| CropRect {
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

/// Pull all available encoded frames from the encoder and fire callbacks.
///
/// Each drained frame is matched to its original [`VideoFrameProxy`] via
/// `frame_id` in `pending_frames`.  The proxy is updated with the encoded
/// bitstream, pts/dts/duration, and codec, then delivered as
/// [`EncodedOutput::VideoFrame`].
pub(crate) fn drain_encoded(
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

pub(crate) fn frame_pts(buf: &gst::Buffer) -> u64 {
    buf.pts().map(|t| t.nseconds()).unwrap_or(0)
}

pub(crate) fn frame_duration(buf: &gst::Buffer) -> Option<u64> {
    buf.duration().map(|t| t.nseconds())
}
