//! Safe Rust API for NVIDIA DeepStream NvBufSurface buffer generation.
//!
//! This crate provides Rust wrappers for DeepStream NvBufSurface buffer
//! allocation and GPU surface operations.
//!
//! # Overview
//!
//! [`UniformBatchGenerator`] creates a DeepStream buffer pool and
//! produces [`SharedBuffer`]s.  Access individual slots via
//! [`SurfaceView::from_buffer`](SurfaceView::from_buffer).
//!
//! # Example (Rust)
//!
//! ```rust,no_run
//! use deepstream_buffers::{
//!     BufferGenerator, NvBufSurfaceMemType, SurfaceView, VideoFormat,
//! };
//!
//! gstreamer::init().unwrap();
//!
//! let gen = BufferGenerator::new(
//!     VideoFormat::RGBA, 640, 480, 30, 1, 0, NvBufSurfaceMemType::Default,
//! ).unwrap();
//!
//! let shared = gen.acquire(None).unwrap();
//! let view = SurfaceView::from_buffer(&shared, 0).unwrap();
//! ```

pub mod cuda_stream;
pub mod ffi;
pub mod prelude;
pub mod transform;

pub mod buffers;
pub mod pipeline;
pub mod shared_buffer;
pub mod surface_view;

pub use cuda_stream::CudaStream;
pub use shared_buffer::SharedBuffer;
pub use surface_view::SurfaceView;

#[cfg(target_arch = "aarch64")]
pub mod egl_cuda_meta;

#[cfg(feature = "skia")]
pub mod egl_context;
#[cfg(feature = "skia")]
pub mod skia_renderer;
#[cfg(feature = "skia")]
pub use skia_renderer::SkiaRenderer;

pub use transform::extract_nvbufsurface;
pub use transform::{
    buffer_gpu_id, ComputeMode, DstPadding, Interpolation, Padding, Rect, TransformConfig,
    TransformConfigBuilder, TransformError, MIN_EFFECTIVE_DIM,
};

// Re-export so downstream crates (benches, examples) can use these directly.
pub use savant_gstreamer::id_meta::{SavantIdMeta, SavantIdMetaKind};
pub use savant_gstreamer::VideoFormat;

pub use buffers::*;

use gstreamer as gst;
use gstreamer::prelude::*;
use log::debug;
use lru::LruCache;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Error type for NvBufSurface operations.
#[derive(Debug, thiserror::Error)]
pub enum NvBufSurfaceError {
    #[error("Failed to create NvDS buffer pool")]
    PoolCreationFailed,

    #[error("Failed to get buffer pool configuration")]
    PoolConfigFailed,

    #[error("Failed to set buffer pool configuration: {0}")]
    PoolSetConfigFailed(String),

    #[error("Failed to activate buffer pool: {0}")]
    PoolActivationFailed(String),

    #[error("Failed to acquire buffer from pool: {0}")]
    BufferAcquisitionFailed(String),

    #[error("Failed to copy buffer contents: {0}")]
    BufferCopyFailed(String),

    #[error("Null pointer: {0}")]
    NullPointer(String),

    #[error("CUDA initialization failed with error code {0}")]
    CudaInitFailed(i32),

    #[error("Batch overflow: tried to fill more than {max} slots")]
    BatchOverflow { max: u32 },

    #[error("Slot index {index} out of bounds (max batch size {max})")]
    SlotOutOfBounds { index: u32, max: u32 },

    #[error("Operation requires finalize() to be called first")]
    NotFinalized,

    #[error("Batch has already been finalized; mutation is not allowed")]
    AlreadyFinalized,

    #[error("NvBufSurfaceMap failed (code {0})")]
    SurfaceMapFailed(i32),

    #[error("NvBufSurfaceUnMap failed (code {0})")]
    SurfaceUnmapFailed(i32),

    #[error("NvBufSurfaceSyncForDevice failed (code {0})")]
    SurfaceSyncFailed(i32),

    #[error("CUDA driver API {function} failed (code {code})")]
    CudaDriverError { function: &'static str, code: u32 },

    #[error("{0}")]
    InvalidInput(String),

    #[error("Element missing required pad: {0}")]
    MissingPad(String),

    #[error("Transform error: {0}")]
    Transform(#[from] crate::transform::TransformError),
}

/// NvBufSurface memory types.
///
/// Specifies the type of memory to allocate for NvBufSurface buffers.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvBufSurfaceMemType {
    /// Default memory type (CUDA Device for dGPU, Surface Array for Jetson).
    Default = 0,
    /// CUDA Host (pinned) memory.
    CudaPinned = 1,
    /// CUDA Device memory.
    CudaDevice = 2,
    /// CUDA Unified memory.
    CudaUnified = 3,
    /// NVRM Surface Array (Jetson only).
    SurfaceArray = 4,
    /// NVRM Handle (Jetson only).
    Handle = 5,
    /// System memory (malloc).
    System = 6,
}

impl From<u32> for NvBufSurfaceMemType {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Default,
            1 => Self::CudaPinned,
            2 => Self::CudaDevice,
            3 => Self::CudaUnified,
            4 => Self::SurfaceArray,
            5 => Self::Handle,
            6 => Self::System,
            _ => Self::Default,
        }
    }
}

impl From<NvBufSurfaceMemType> for u32 {
    fn from(value: NvBufSurfaceMemType) -> Self {
        value as u32
    }
}

/// Initialize CUDA context for the given GPU device.
///
/// This must be called before creating a [`UniformBatchGenerator`] when
/// not running inside a DeepStream pipeline (which handles CUDA initialization
/// automatically). This is particularly needed in standalone usage and tests.
///
/// Internally, this calls `cudaSetDevice` followed by `cudaFree(NULL)` to
/// trigger lazy CUDA context creation.
///
/// # Arguments
///
/// * `gpu_id` - GPU device ID to initialize (typically 0).
///
/// # Errors
///
/// Returns an error if CUDA initialization fails (e.g., no GPU available).
pub fn cuda_init(gpu_id: u32) -> Result<(), NvBufSurfaceError> {
    extern "C" {
        fn cudaSetDevice(device: i32) -> i32;
        fn cudaFree(dev_ptr: *mut std::ffi::c_void) -> i32;
        fn cuInit(flags: u32) -> u32;
    }

    unsafe {
        // Initialize the CUDA driver API (needed for cuGraphicsEGL* on Jetson).
        let err = cuInit(0);
        if err != 0 {
            return Err(NvBufSurfaceError::CudaInitFailed(err as i32));
        }

        let err = cudaSetDevice(gpu_id as i32);
        if err != 0 {
            return Err(NvBufSurfaceError::CudaInitFailed(err));
        }
        // cudaFree(NULL) triggers lazy CUDA context creation
        let err = cudaFree(std::ptr::null_mut());
        if err != 0 {
            return Err(NvBufSurfaceError::CudaInitFailed(err));
        }
    }

    debug!("CUDA initialized for GPU {}", gpu_id);
    Ok(())
}

/// Helper: set a uint field on a GstStructure using glib's safe Value API.
///
/// This avoids calling the variadic `gst_structure_set()` which cannot be
/// called safely from Rust.
pub(crate) unsafe fn set_structure_uint(
    structure: *mut gst::ffi::GstStructure,
    field_name: &str,
    value: u32,
) {
    use glib::prelude::ToValue;
    use glib::translate::ToGlibPtr;
    let c_name = std::ffi::CString::new(field_name).expect("field_name must not contain NUL bytes");
    let gvalue = value.to_value();
    gst::ffi::gst_structure_set_value(structure, c_name.as_ptr(), gvalue.to_glib_none().0);
}

// ─── PTS-keyed meta bridge ───────────────────────────────────────────────────

/// Maximum number of in-flight PTS→meta entries before eviction kicks in.
/// If the map exceeds this size, the oldest half of entries (by PTS) are
/// removed.  This guards against slow memory leaks when an encoder drops
/// buffers without producing matching output.
const MAX_BRIDGE_MAP_SIZE: usize = 1024;

/// Maximum number of meta entries stored per PTS key.  If exceeded, the oldest
/// entry is evicted — this indicates the src pad is not draining fast enough.
pub const MAX_ENTRIES_PER_PTS: usize = 32;

/// Install pad probes on `element` to propagate [`SavantIdMeta`] across
/// elements that create new output buffers (e.g. hardware video encoders).
///
/// Hardware encoders like `nvv4l2h265enc` allocate fresh buffers for the
/// compressed bitstream and do **not** copy custom `GstMeta` from input to
/// output.  This function works around that limitation by using PTS-keyed
/// side-channel storage:
///
/// 1. A **sink-pad** probe intercepts each incoming buffer, reads any
///    `SavantIdMeta`, and stores the mapping `PTS → Vec<Vec<SavantIdMetaKind>>`
///    in a shared LRU cache.  Multiple buffers with the same PTS (e.g. in
///    multi-stream scenarios) are appended, not overwritten.
/// 2. A **src-pad** probe intercepts each outgoing buffer, pops the first
///    entry for that PTS from the cache, and re-attaches the `SavantIdMeta`.
///
/// PTS is guaranteed to be preserved by all GStreamer encoder elements.
/// B-frame reordering is handled naturally because lookups are by value,
/// not by order.  The LRU cache evicts the least-recently-used entries
/// when the map exceeds [`MAX_BRIDGE_MAP_SIZE`], which is safe regardless
/// of output ordering.
///
/// # Errors
///
/// Returns [`NvBufSurfaceError::MissingPad`] if `element` does not have
/// both `sink` and `src` static pads.
///
/// # Example
///
/// ```rust,no_run
/// # use deepstream_buffers::bridge_savant_id_meta;
/// # use gstreamer as gst;
/// # use gstreamer::prelude::*;
/// # gstreamer::init().unwrap();
/// let enc = gst::ElementFactory::make("nvv4l2h265enc")
///     .build()
///     .unwrap();
/// bridge_savant_id_meta(&enc).unwrap();
/// // From this point, SavantIdMeta on buffers entering the encoder's
/// // sink pad will automatically appear on the encoder's src pad output.
/// ```
pub fn bridge_savant_id_meta(element: &gst::Element) -> Result<(), NvBufSurfaceError> {
    // Multi-value LRU map: each PTS key holds a FIFO queue of meta vectors,
    // supporting multiple buffers that share the same PTS value.
    let map: Arc<Mutex<LruCache<u64, VecDeque<Vec<SavantIdMetaKind>>>>> = Arc::new(Mutex::new(
        LruCache::new(NonZeroUsize::new(MAX_BRIDGE_MAP_SIZE).unwrap()),
    ));

    // ── Sink pad probe: extract meta, store by PTS ──────────────────────
    let sink_map = map.clone();
    let sink_pad = element
        .static_pad("sink")
        .ok_or_else(|| NvBufSurfaceError::MissingPad("sink".to_string()))?;

    sink_pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        if let Some(buffer) = info.buffer() {
            if let Some(meta) = buffer.meta::<SavantIdMeta>() {
                if let Some(pts) = buffer.pts() {
                    let ids = meta.ids().to_vec();
                    let mut map = sink_map.lock();
                    if map.len() == map.cap().get() {
                        log::error!(
                            "bridge_savant_id_meta: PTS map is at full capacity ({}); \
                             LRU entry will be evicted — src pad is not consuming entries \
                             fast enough or the element is dropping buffers",
                            map.cap(),
                        );
                    }
                    if let Some(existing) = map.get(&pts.nseconds()) {
                        log::warn!(
                            "bridge_savant_id_meta: PTS collision at {} ns; \
                             existing entries={}, new meta={:?}",
                            pts.nseconds(),
                            existing.len(),
                            ids,
                        );
                    }
                    let entries = map.get_or_insert_mut(pts.nseconds(), VecDeque::new);
                    entries.push_back(ids);
                    if entries.len() > MAX_ENTRIES_PER_PTS {
                        let evicted = entries.pop_front();
                        log::error!(
                            "bridge_savant_id_meta: per-PTS limit ({}) exceeded at {} ns; \
                             evicted={:?}, added={:?}",
                            MAX_ENTRIES_PER_PTS,
                            pts.nseconds(),
                            evicted,
                            entries.back(),
                        );
                    }
                }
            }
        }
        gst::PadProbeReturn::Ok
    });

    // ── Src pad probe: look up PTS, re-attach meta ──────────────────────
    let src_map = map;
    let src_pad = element
        .static_pad("src")
        .ok_or_else(|| NvBufSurfaceError::MissingPad("src".to_string()))?;

    src_pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        if let Some(buffer) = info.buffer_mut() {
            if let Some(pts) = buffer.pts() {
                let mut map = src_map.lock();
                let should_remove;
                if let Some(entries) = map.get_mut(&pts.nseconds()) {
                    if let Some(ids) = entries.pop_front() {
                        let buf_ref = buffer.make_mut();
                        SavantIdMeta::replace(buf_ref, ids);
                    }
                    should_remove = entries.is_empty();
                } else {
                    should_remove = false;
                }
                if should_remove {
                    map.pop(&pts.nseconds());
                }
            }
        }
        gst::PadProbeReturn::Ok
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::TransformError;

    #[test]
    fn transform_error_into_nvbufsurface_error() {
        let te = TransformError::TransformFailed(42);
        let nbe: NvBufSurfaceError = te.into();
        assert!(matches!(nbe, NvBufSurfaceError::Transform(_)));
    }

    #[test]
    fn transform_error_via_question_mark() {
        fn inner() -> Result<(), NvBufSurfaceError> {
            Err(TransformError::InvalidBuffer("test"))?
        }
        let err = inner().unwrap_err();
        assert!(matches!(err, NvBufSurfaceError::Transform(_)));
    }
}

// PyO3 Python bindings have been moved to savant_core_py::deepstream.
// Enable the `deepstream` feature on savant_core_py / savant_python to use them.
