use crate::callbacks::Callbacks;
use crate::draw_context::DrawContext;
use crate::error::PicassoError;
use crate::message::EncodedOutput;
use crate::pipeline::FrameInput;
use deepstream_encoders::NvEncoder;
use deepstream_nvbufsurface::{Padding, SkiaRenderer, TransformConfig};
use gstreamer as gst;
use log::{debug, error};
use savant_core::geometry::{CropRect, LetterBoxKind, ScaleSpec};
use savant_core::primitives::frame::{VideoFrameProxy, VideoFrameTransformation};
use savant_core::primitives::object::ObjectOperations;
use std::sync::Arc;

use crate::spec::draw::ObjectDrawSpec;

/// Render-specific options and mutable state.  When provided to
/// [`process_encode`], Skia overlays are drawn between the GPU transform
/// and the hardware encode step.
pub struct RenderOpts<'a> {
    pub draw_spec: &'a ObjectDrawSpec,
    pub use_on_render: bool,
    pub gpu_id: u32,
    pub renderer: &'a mut Option<SkiaRenderer>,
    pub draw_ctx: &'a mut DrawContext,
}

/// Unified encode pipeline:
///
/// GPU-transform → (optional Skia render) → optional on_gpumat → encode → drain.
///
/// When `render` is `Some`, the Skia overlay step is inserted after the GPU
/// transform.  Otherwise the path is pure transform-and-encode.
pub fn process_encode(
    source_id: &str,
    input: FrameInput,
    transform_config: &TransformConfig,
    encoder: &mut NvEncoder,
    callbacks: &Arc<Callbacks>,
    use_on_gpumat: bool,
    render: Option<&mut RenderOpts<'_>>,
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

        let objects = input.frame.get_all_objects();
        for obj in &objects {
            let ns = obj.get_namespace();
            let label = obj.get_label();

            let static_spec = render.draw_spec.lookup(&ns, &label);

            let draw = callbacks
                .on_object_draw_spec
                .as_ref()
                .and_then(|cb| cb.call(source_id, obj, static_spec))
                .or_else(|| static_spec.cloned());

            if let Some(ref d) = draw {
                let templates = render.draw_ctx.resolve_templates(&ns, &label, d).cloned();
                crate::render::object::draw_object(
                    skia.canvas(),
                    obj,
                    d,
                    templates.as_ref(),
                    render.draw_ctx,
                );
            }
        }

        if render.use_on_render {
            if let Some(cb) = &callbacks.on_render {
                cb.call(source_id, skia.canvas(), &input.frame);
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

    encoder
        .submit_frame(dst_buf, input.frame_id, pts, duration)
        .map_err(|e| PicassoError::Encoder(source_id.to_string(), e.to_string()))?;

    drain_encoded(source_id, &input.frame, encoder, callbacks);

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
pub fn drain_encoded(
    source_id: &str,
    frame: &VideoFrameProxy,
    encoder: &mut NvEncoder,
    callbacks: &Arc<Callbacks>,
) {
    loop {
        match encoder.pull_encoded() {
            Ok(Some(encoded)) => {
                if let Some(cb) = &callbacks.on_encoded_frame {
                    let mut buf = gst::Buffer::with_size(encoded.data.len()).unwrap();
                    {
                        let buf_ref = buf.get_mut().unwrap();
                        buf_ref.copy_from_slice(0, &encoded.data).unwrap();
                        buf_ref.set_pts(gst::ClockTime::from_nseconds(encoded.pts_ns));
                        if let Some(dur) = encoded.duration_ns {
                            buf_ref.set_duration(gst::ClockTime::from_nseconds(dur));
                        }
                    }
                    cb.call(EncodedOutput {
                        source_id: source_id.to_string(),
                        frame: frame.clone(),
                        buffer: Some(buf),
                        pts: encoded.pts_ns,
                        duration: encoded.duration_ns,
                        is_keyframe: false,
                        is_eos: false,
                    });
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

pub fn frame_pts(buf: &gst::Buffer) -> u64 {
    buf.pts().map(|t| t.nseconds()).unwrap_or(0)
}

pub fn frame_duration(buf: &gst::Buffer) -> Option<u64> {
    buf.duration().map(|t| t.nseconds())
}
