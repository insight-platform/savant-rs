//! GStreamer pipeline integration helpers.
//!
//! This module collects functions and traits that depend on GStreamer types
//! (`gst::Buffer`, `gst::Caps`, `AppSrc`, etc.) and are needed only when
//! operating inside a GStreamer pipeline.  The core public API
//! ([`SharedBuffer`], [`SurfaceView`], generators) is usable without
//! importing anything from this module.
//!
//! [`SharedBuffer`]: crate::SharedBuffer
//! [`SurfaceView`]: crate::SurfaceView

pub use crate::transform::{buffer_gpu_id, extract_nvbufsurface};

use crate::BufferGenerator;

/// Extension trait providing GStreamer-level access to [`BufferGenerator`].
///
/// These methods require GStreamer types and are used for pipeline
/// integration (obtaining `gst::Caps`, etc.).
pub trait BufferGeneratorExt {
    /// Return NVMM caps as a `gst::Caps` object.
    fn nvmm_caps_gst(&self) -> gstreamer::Caps;

    /// Return raw caps as a `gst::Caps` object.
    fn raw_caps_gst(&self) -> gstreamer::Caps;
}

impl BufferGeneratorExt for BufferGenerator {
    fn nvmm_caps_gst(&self) -> gstreamer::Caps {
        self.nvmm_caps_gst()
    }

    fn raw_caps_gst(&self) -> gstreamer::Caps {
        self.raw_caps_gst()
    }
}

/// Extension trait providing GStreamer-level access to [`UniformBatchGenerator`](crate::UniformBatchGenerator).
pub trait UniformBatchGeneratorExt {
    /// Return NVMM caps as a `gst::Caps` object.
    fn nvmm_caps_gst(&self) -> gstreamer::Caps;

    /// Return raw caps as a `gst::Caps` object.
    fn raw_caps_gst(&self) -> gstreamer::Caps;
}

impl UniformBatchGeneratorExt for crate::UniformBatchGenerator {
    fn nvmm_caps_gst(&self) -> gstreamer::Caps {
        self.nvmm_caps_gst()
    }

    fn raw_caps_gst(&self) -> gstreamer::Caps {
        self.raw_caps_gst()
    }
}
