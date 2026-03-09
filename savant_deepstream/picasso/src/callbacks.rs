use crate::message::{BypassOutput, EncodedOutput};
use crate::spec::EvictionDecision;
use deepstream_nvbufsurface::SkiaRenderer;
use savant_core::draw::ObjectDraw;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::object::BorrowedVideoObject;
use std::sync::Arc;

/// Called when an encoded frame (or EOS sentinel) is ready.
pub trait OnEncodedFrame: Send + Sync + 'static {
    fn call(&self, output: EncodedOutput);
}

/// Called in bypass mode with transformed bboxes.
pub trait OnBypassFrame: Send + Sync + 'static {
    fn call(&self, output: BypassOutput);
}

/// Called before Skia flush — allows custom drawing on the canvas.
///
/// The callback receives the [`SkiaRenderer`] that owns the current GPU
/// canvas.  Rust callers can obtain the Skia canvas via
/// [`SkiaRenderer::canvas()`]; Python callers receive the renderer's FBO
/// info and build a `SkiaCanvas` wrapper around it.
pub trait OnRender: Send + Sync + 'static {
    fn call(&self, source_id: &str, renderer: &mut SkiaRenderer, frame: &VideoFrameProxy);
}

/// Per-object callback that can override the static `ObjectDrawSpec`.
///
/// `current_spec` is the draw spec resolved from the static
/// [`ObjectDrawSpec`] table for this object's `(namespace, label)`.  The
/// callback may return `None` to use `current_spec` as-is, or return a
/// replacement `ObjectDraw`.
pub trait OnObjectDrawSpec: Send + Sync + 'static {
    fn call(
        &self,
        source_id: &str,
        object: &BorrowedVideoObject,
        current_spec: Option<&ObjectDraw>,
    ) -> Option<ObjectDraw>;
}

/// Called with the CUDA pointer of the destination buffer after transform.
///
/// `cuda_stream` is the non-blocking CUDA stream handle owned by the
/// Picasso worker (as a `usize` pointer).  Callers may enqueue GPU work
/// on this stream; the worker will synchronise the stream before
/// proceeding to the next pipeline stage.
pub trait OnGpuMat: Send + Sync + 'static {
    #[allow(clippy::too_many_arguments)]
    fn call(
        &self,
        source_id: &str,
        frame: &VideoFrameProxy,
        data_ptr: usize,
        pitch: u32,
        width: u32,
        height: u32,
        cuda_stream: usize,
    );
}

/// Called when a source has been idle longer than its timeout.
pub trait OnEviction: Send + Sync + 'static {
    fn call(&self, source_id: &str) -> EvictionDecision;
}

/// Aggregate holder for all optional callbacks.
#[derive(Default)]
pub struct Callbacks {
    pub on_encoded_frame: Option<Arc<dyn OnEncodedFrame>>,
    pub on_bypass_frame: Option<Arc<dyn OnBypassFrame>>,
    pub on_render: Option<Arc<dyn OnRender>>,
    pub on_object_draw_spec: Option<Arc<dyn OnObjectDrawSpec>>,
    pub on_gpumat: Option<Arc<dyn OnGpuMat>>,
    pub on_eviction: Option<Arc<dyn OnEviction>>,
}
