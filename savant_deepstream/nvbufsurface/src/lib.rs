//! Safe Rust API for NVIDIA DeepStream NvBufSurface buffer generation.
//!
//! This crate provides Rust wrappers for DeepStream NvBufSurface buffer
//! allocation and GPU surface operations.
//!
//! # Overview
//!
//! [`DsNvUniformSurfaceBufferGenerator`] creates a DeepStream buffer pool and
//! produces [`SharedMutableGstBuffer`]s.  Access individual slots via
//! [`SurfaceView::from_shared`](SurfaceView::from_shared).
//!
//! # Example (Rust)
//!
//! ```rust,no_run
//! use deepstream_nvbufsurface::{
//!     DsNvUniformSurfaceBufferGenerator, NvBufSurfaceMemType, SurfaceView, VideoFormat,
//! };
//!
//! gstreamer::init().unwrap();
//!
//! let gen = DsNvUniformSurfaceBufferGenerator::new(
//!     VideoFormat::RGBA, 640, 480, 1, 2, 0, NvBufSurfaceMemType::Default,
//! ).unwrap();
//!
//! let shared = gen.acquire_buffer(None).unwrap();
//! let view = SurfaceView::from_shared(&shared, 0).unwrap();
//! ```

pub mod cuda_stream;
pub mod ffi;
pub mod surface_ops;
pub mod transform;

pub mod buffers;
pub mod shared_buffer;
pub mod surface_view;

pub use cuda_stream::CudaStream;
pub use shared_buffer::SharedMutableGstBuffer;
pub use surface_view::SurfaceView;

#[cfg(target_arch = "aarch64")]
pub mod egl_cuda_meta;

#[cfg(feature = "skia")]
pub mod egl_context;
#[cfg(feature = "skia")]
pub mod skia_renderer;
#[cfg(feature = "skia")]
pub use skia_renderer::SkiaRenderer;

pub use surface_ops::{fill_surface, memset_surface, upload_to_surface};
pub use transform::extract_nvbufsurface;
pub use transform::{
    buffer_gpu_id, ComputeMode, DstPadding, Interpolation, Padding, Rect, TransformConfig,
    TransformError, MIN_EFFECTIVE_DIM,
};

// Re-export so downstream crates (benches, examples) can use these directly.
pub use savant_gstreamer::id_meta::{SavantIdMeta, SavantIdMetaKind};
pub use savant_gstreamer::VideoFormat;

pub use buffers::*;

use gstreamer as gst;
use gstreamer::prelude::*;
use log::debug;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
/// This must be called before creating a [`DsNvUniformSurfaceBufferGenerator`] when
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
    let c_name = std::ffi::CString::new(field_name).unwrap();
    let gvalue = value.to_value();
    gst::ffi::gst_structure_set_value(structure, c_name.as_ptr(), gvalue.to_glib_none().0);
}

// ─── PTS-keyed meta bridge ───────────────────────────────────────────────────

/// Maximum number of in-flight PTS→meta entries before eviction kicks in.
/// If the map exceeds this size, the oldest half of entries (by PTS) are
/// removed.  This guards against slow memory leaks when an encoder drops
/// buffers without producing matching output.
const MAX_BRIDGE_MAP_SIZE: usize = 256;

/// Install pad probes on `element` to propagate [`SavantIdMeta`] across
/// elements that create new output buffers (e.g. hardware video encoders).
///
/// Hardware encoders like `nvv4l2h265enc` allocate fresh buffers for the
/// compressed bitstream and do **not** copy custom `GstMeta` from input to
/// output.  This function works around that limitation by using PTS-keyed
/// side-channel storage:
///
/// 1. A **sink-pad** probe intercepts each incoming buffer, reads any
///    `SavantIdMeta`, and stores the mapping `PTS → Vec<SavantIdMetaKind>`
///    in a shared `HashMap`.
/// 2. A **src-pad** probe intercepts each outgoing buffer, looks up the PTS
///    in the map, and re-attaches the `SavantIdMeta`.
///
/// PTS is guaranteed to be preserved by all GStreamer encoder elements.
/// B-frame reordering is handled naturally because lookups are by value,
/// not by order.
///
/// # Panics
///
/// Panics if `element` does not have both `sink` and `src` static pads.
///
/// # Example
///
/// ```rust,no_run
/// # use deepstream_nvbufsurface::bridge_savant_id_meta;
/// # use gstreamer as gst;
/// # use gstreamer::prelude::*;
/// # gstreamer::init().unwrap();
/// let enc = gst::ElementFactory::make("nvv4l2h265enc")
///     .build()
///     .unwrap();
/// bridge_savant_id_meta(&enc);
/// // From this point, SavantIdMeta on buffers entering the encoder's
/// // sink pad will automatically appear on the encoder's src pad output.
/// ```
pub fn bridge_savant_id_meta(element: &gst::Element) {
    let map: Arc<Mutex<HashMap<u64, Vec<SavantIdMetaKind>>>> = Arc::new(Mutex::new(HashMap::new()));

    // ── Sink pad probe: extract meta, store by PTS ──────────────────────
    let sink_map = map.clone();
    let sink_pad = element
        .static_pad("sink")
        .expect("bridge_savant_id_meta: element has no 'sink' pad");

    sink_pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        if let Some(buffer) = info.buffer() {
            if let Some(meta) = buffer.meta::<SavantIdMeta>() {
                if let Some(pts) = buffer.pts() {
                    let ids = meta.ids().to_vec();
                    let mut map = sink_map.lock().unwrap();
                    map.insert(pts.nseconds(), ids);
                    if map.len() > MAX_BRIDGE_MAP_SIZE {
                        log::warn!(
                            "bridge_savant_id_meta: PTS map exceeded {} entries, evicting stale entries",
                            MAX_BRIDGE_MAP_SIZE,
                        );
                        let mut keys: Vec<u64> = map.keys().copied().collect();
                        keys.sort_unstable();
                        let cutoff = keys[keys.len() / 2];
                        map.retain(|&pts, _| pts >= cutoff);
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
        .expect("bridge_savant_id_meta: element has no 'src' pad");

    src_pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        if let Some(buffer) = info.buffer_mut() {
            if let Some(pts) = buffer.pts() {
                if let Some(ids) = src_map.lock().unwrap().remove(&pts.nseconds()) {
                    let buf_ref = buffer.make_mut();
                    SavantIdMeta::replace(buf_ref, ids);
                }
            }
        }
        gst::PadProbeReturn::Ok
    });
}

// PyO3 Python bindings have been moved to savant_core_py::deepstream.
// Enable the `deepstream` feature on savant_core_py / savant_python to use them.
