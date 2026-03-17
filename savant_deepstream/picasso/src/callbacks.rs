use crate::message::OutputMessage;
use crate::spec::EvictionDecision;
use deepstream_nvbufsurface::{SkiaRenderer, SurfaceView};
use savant_core::draw::ObjectDraw;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::object::BorrowedVideoObject;
use std::sync::Arc;

/// Called when an encoded frame (or EOS sentinel) is ready.
pub trait OnEncodedFrame: Send + Sync + 'static {
    fn call(&self, output: OutputMessage);
}

/// Called in bypass mode with transformed bboxes.
pub trait OnBypassFrame: Send + Sync + 'static {
    fn call(&self, output: OutputMessage);
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

/// Called with a [`SurfaceView`] of the destination buffer after transform.
///
/// The `SurfaceView` provides zero-copy access to the CUDA device pointer
/// and surface metadata (pitch, width, height).  It works transparently
/// on both dGPU and Jetson.
///
/// # Pointer validity
///
/// The `data_ptr()` obtained from the `&SurfaceView` is **only valid for
/// the duration of this callback**.  On Jetson the pointer is tied to the
/// EGL-CUDA registration on the buffer; on dGPU it is the NvBufSurface
/// `dataPtr`.  In both cases the buffer may be recycled after the encode
/// pipeline returns.  Storing the raw pointer for later use is undefined
/// behaviour.
///
/// The CUDA stream used by the Picasso worker is available via
/// `view.cuda_stream()`.  Callers may enqueue GPU work on this stream;
/// the worker synchronises the stream when the view is dropped.
pub trait OnGpuMat: Send + Sync + 'static {
    fn call(&self, source_id: &str, frame: &VideoFrameProxy, view: &SurfaceView);
}

/// Called when a source has been idle longer than its timeout.
pub trait OnEviction: Send + Sync + 'static {
    fn call(&self, source_id: &str) -> EvictionDecision;
}

/// Reason the worker's encoder was reset (destroyed and recreated).
#[derive(Debug, Clone)]
pub enum StreamResetReason {
    /// PTS decreased or stayed equal relative to the previous frame.
    PtsDecreased {
        /// PTS of the last successfully accepted frame (nanoseconds).
        last_pts_ns: u64,
        /// PTS of the incoming frame that triggered the reset (nanoseconds).
        new_pts_ns: u64,
    },
}

/// Fired when the worker resets its encoder due to a PTS anomaly.
///
/// The callback receives the `source_id` and the [`StreamResetReason`]
/// describing what triggered the reset.
pub trait OnStreamReset: Send + Sync + 'static {
    fn call(&self, source_id: &str, reason: StreamResetReason);
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
    pub on_stream_reset: Option<Arc<dyn OnStreamReset>>,
}
