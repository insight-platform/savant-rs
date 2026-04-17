//! Python bindings for [`deepstream_decoders`] decoder configurations.
//!
//! Mirrors the per-codec builder-style Rust API with one PyO3 class per
//! [`DecoderConfig`] variant plus an umbrella [`PyDecoderConfig`] that
//! wraps the Rust enum.
//!
//! Platform gating follows the DeepStream 7.1 Gst-nvvideo4linux2 docs:
//!
//! | Property                  | dGPU | Jetson |
//! |---------------------------|------|--------|
//! | `num-extra-surfaces`      | yes  | yes    |
//! | `drop-frame-interval`     | yes  | yes    |
//! | `cudadec-memtype`         | yes  | no     |
//! | `low-latency-mode`        | yes  | no     |
//! | `enable-max-performance`  | no   | yes    |
//! | `disable-dpb` (low_latency) | no | yes    |
//!
//! Platform-specific methods / getters are compiled in only on the
//! matching target.

use crate::gstreamer::PyCodec;
#[cfg(not(target_arch = "aarch64"))]
use deepstream_decoders::CudadecMemtype;
use deepstream_decoders::{
    Av1DecoderConfig, DecoderConfig, H264DecoderConfig, H264StreamFormat, HevcDecoderConfig,
    HevcStreamFormat, JpegBackend, JpegDecoderConfig, PngDecoderConfig, RawRgbDecoderConfig,
    RawRgbaDecoderConfig, Vp8DecoderConfig, Vp9DecoderConfig,
};
use pyo3::prelude::*;

// ─── H.264 stream format enum ───────────────────────────────────────────

/// H.264 bitstream format carried in the GStreamer caps.
#[pyclass(
    from_py_object,
    name = "H264StreamFormat",
    module = "savant_rs.deepstream",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyH264StreamFormat {
    /// Annex-B byte-stream (start-code prefixed NALUs).
    #[pyo3(name = "BYTE_STREAM")]
    ByteStream = 0,
    /// AVC (length-prefixed NALUs, requires `codec_data`).
    #[pyo3(name = "AVC")]
    Avc = 1,
    /// AVC3 (length-prefixed NALUs with in-band parameter sets).
    #[pyo3(name = "AVC3")]
    Avc3 = 2,
}

#[pymethods]
impl PyH264StreamFormat {
    fn __repr__(&self) -> &'static str {
        match self {
            PyH264StreamFormat::ByteStream => "H264StreamFormat.BYTE_STREAM",
            PyH264StreamFormat::Avc => "H264StreamFormat.AVC",
            PyH264StreamFormat::Avc3 => "H264StreamFormat.AVC3",
        }
    }
}

impl From<PyH264StreamFormat> for H264StreamFormat {
    fn from(v: PyH264StreamFormat) -> Self {
        match v {
            PyH264StreamFormat::ByteStream => H264StreamFormat::ByteStream,
            PyH264StreamFormat::Avc => H264StreamFormat::Avc,
            PyH264StreamFormat::Avc3 => H264StreamFormat::Avc3,
        }
    }
}

impl From<H264StreamFormat> for PyH264StreamFormat {
    fn from(v: H264StreamFormat) -> Self {
        match v {
            H264StreamFormat::ByteStream => PyH264StreamFormat::ByteStream,
            H264StreamFormat::Avc => PyH264StreamFormat::Avc,
            H264StreamFormat::Avc3 => PyH264StreamFormat::Avc3,
        }
    }
}

// ─── HEVC stream format enum ────────────────────────────────────────────

/// HEVC bitstream format carried in the GStreamer caps.
#[pyclass(
    from_py_object,
    name = "HevcStreamFormat",
    module = "savant_rs.deepstream",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyHevcStreamFormat {
    /// Annex-B byte-stream.
    #[pyo3(name = "BYTE_STREAM")]
    ByteStream = 0,
    /// HVC1 (length-prefixed NALUs, out-of-band parameter sets).
    #[pyo3(name = "HVC1")]
    Hvc1 = 1,
    /// HEV1 (length-prefixed NALUs, in-band parameter sets).
    #[pyo3(name = "HEV1")]
    Hev1 = 2,
}

#[pymethods]
impl PyHevcStreamFormat {
    fn __repr__(&self) -> &'static str {
        match self {
            PyHevcStreamFormat::ByteStream => "HevcStreamFormat.BYTE_STREAM",
            PyHevcStreamFormat::Hvc1 => "HevcStreamFormat.HVC1",
            PyHevcStreamFormat::Hev1 => "HevcStreamFormat.HEV1",
        }
    }
}

impl From<PyHevcStreamFormat> for HevcStreamFormat {
    fn from(v: PyHevcStreamFormat) -> Self {
        match v {
            PyHevcStreamFormat::ByteStream => HevcStreamFormat::ByteStream,
            PyHevcStreamFormat::Hvc1 => HevcStreamFormat::Hvc1,
            PyHevcStreamFormat::Hev1 => HevcStreamFormat::Hev1,
        }
    }
}

impl From<HevcStreamFormat> for PyHevcStreamFormat {
    fn from(v: HevcStreamFormat) -> Self {
        match v {
            HevcStreamFormat::ByteStream => PyHevcStreamFormat::ByteStream,
            HevcStreamFormat::Hvc1 => PyHevcStreamFormat::Hvc1,
            HevcStreamFormat::Hev1 => PyHevcStreamFormat::Hev1,
        }
    }
}

// ─── JPEG backend enum ──────────────────────────────────────────────────

/// Backend used by the JPEG decoder.
#[pyclass(
    from_py_object,
    name = "JpegBackend",
    module = "savant_rs.deepstream",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyJpegBackend {
    /// GPU decoder (`nvjpegdec`).
    #[pyo3(name = "GPU")]
    Gpu = 0,
    /// CPU decoder (`jpegparse ! jpegdec`).
    #[pyo3(name = "CPU")]
    Cpu = 1,
}

#[pymethods]
impl PyJpegBackend {
    fn __repr__(&self) -> &'static str {
        match self {
            PyJpegBackend::Gpu => "JpegBackend.GPU",
            PyJpegBackend::Cpu => "JpegBackend.CPU",
        }
    }
}

impl From<PyJpegBackend> for JpegBackend {
    fn from(v: PyJpegBackend) -> Self {
        match v {
            PyJpegBackend::Gpu => JpegBackend::Gpu,
            PyJpegBackend::Cpu => JpegBackend::Cpu,
        }
    }
}

impl From<JpegBackend> for PyJpegBackend {
    fn from(v: JpegBackend) -> Self {
        match v {
            JpegBackend::Gpu => PyJpegBackend::Gpu,
            JpegBackend::Cpu => PyJpegBackend::Cpu,
        }
    }
}

// ─── CUDA decoder memory-type enum (dGPU only) ──────────────────────────

/// CUDA memory type for the ``nvv4l2decoder`` ``cudadec-memtype`` property.
///
/// Only available on desktop GPUs; this class is not exposed on Jetson.
#[cfg(not(target_arch = "aarch64"))]
#[pyclass(
    from_py_object,
    name = "CudadecMemtype",
    module = "savant_rs.deepstream",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyCudadecMemtype {
    /// CUDA device memory (default).
    #[pyo3(name = "DEVICE")]
    Device = 0,
    /// CUDA pinned host memory.
    #[pyo3(name = "PINNED")]
    Pinned = 1,
    /// CUDA unified memory.
    #[pyo3(name = "UNIFIED")]
    Unified = 2,
}

#[cfg(not(target_arch = "aarch64"))]
#[pymethods]
impl PyCudadecMemtype {
    fn __repr__(&self) -> &'static str {
        match self {
            PyCudadecMemtype::Device => "CudadecMemtype.DEVICE",
            PyCudadecMemtype::Pinned => "CudadecMemtype.PINNED",
            PyCudadecMemtype::Unified => "CudadecMemtype.UNIFIED",
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl From<PyCudadecMemtype> for CudadecMemtype {
    fn from(v: PyCudadecMemtype) -> Self {
        match v {
            PyCudadecMemtype::Device => CudadecMemtype::Device,
            PyCudadecMemtype::Pinned => CudadecMemtype::Pinned,
            PyCudadecMemtype::Unified => CudadecMemtype::Unified,
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl From<CudadecMemtype> for PyCudadecMemtype {
    fn from(v: CudadecMemtype) -> Self {
        match v {
            CudadecMemtype::Device => PyCudadecMemtype::Device,
            CudadecMemtype::Pinned => PyCudadecMemtype::Pinned,
            CudadecMemtype::Unified => PyCudadecMemtype::Unified,
        }
    }
}

// ─── H.264 decoder config ───────────────────────────────────────────────

/// H.264 decoder configuration (see :class:`deepstream_decoders::H264DecoderConfig`).
///
/// ``codec_data`` is intentionally not exposed: it is derived by the
/// pipeline from the bitstream's parameter sets and must not be overridden
/// from user code.
#[pyclass(
    from_py_object,
    name = "H264DecoderConfig",
    module = "savant_rs.deepstream"
)]
#[derive(Clone)]
pub struct PyH264DecoderConfig(pub(crate) H264DecoderConfig);

#[pymethods]
impl PyH264DecoderConfig {
    #[new]
    fn new(stream_format: PyH264StreamFormat) -> Self {
        Self(H264DecoderConfig::new(stream_format.into()))
    }

    fn with_num_extra_surfaces(&self, n: u32) -> Self {
        Self(self.0.clone().num_extra_surfaces(n))
    }

    fn with_drop_frame_interval(&self, n: u32) -> Self {
        Self(self.0.clone().drop_frame_interval(n))
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn with_cudadec_memtype(&self, t: PyCudadecMemtype) -> Self {
        Self(self.0.clone().cudadec_memtype(t.into()))
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn with_low_latency_mode(&self, v: bool) -> Self {
        Self(self.0.clone().low_latency_mode(v))
    }

    #[cfg(target_arch = "aarch64")]
    fn with_enable_max_performance(&self, v: bool) -> Self {
        Self(self.0.clone().enable_max_performance(v))
    }

    #[cfg(target_arch = "aarch64")]
    fn with_low_latency(&self, v: bool) -> Self {
        Self(self.0.clone().low_latency(v))
    }

    #[getter]
    fn stream_format(&self) -> PyH264StreamFormat {
        self.0.stream_format.into()
    }

    #[getter]
    fn num_extra_surfaces(&self) -> Option<u32> {
        self.0.num_extra_surfaces
    }

    #[getter]
    fn drop_frame_interval(&self) -> Option<u32> {
        self.0.drop_frame_interval
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[getter]
    fn cudadec_memtype(&self) -> Option<PyCudadecMemtype> {
        self.0.cudadec_memtype.map(Into::into)
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[getter]
    fn low_latency_mode(&self) -> Option<bool> {
        self.0.low_latency_mode
    }

    #[cfg(target_arch = "aarch64")]
    #[getter]
    fn enable_max_performance(&self) -> Option<bool> {
        self.0.enable_max_performance
    }

    #[cfg(target_arch = "aarch64")]
    #[getter]
    fn low_latency(&self) -> Option<bool> {
        self.0.low_latency
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn __repr__(&self) -> String {
        format!(
            "H264DecoderConfig(stream_format={}, num_extra_surfaces={:?}, \
             drop_frame_interval={:?}, cudadec_memtype={:?}, \
             low_latency_mode={:?})",
            self.0.stream_format,
            self.0.num_extra_surfaces,
            self.0.drop_frame_interval,
            self.0.cudadec_memtype,
            self.0.low_latency_mode,
        )
    }

    #[cfg(target_arch = "aarch64")]
    fn __repr__(&self) -> String {
        format!(
            "H264DecoderConfig(stream_format={}, num_extra_surfaces={:?}, \
             drop_frame_interval={:?}, enable_max_performance={:?}, \
             low_latency={:?})",
            self.0.stream_format,
            self.0.num_extra_surfaces,
            self.0.drop_frame_interval,
            self.0.enable_max_performance,
            self.0.low_latency,
        )
    }
}

// ─── HEVC decoder config ────────────────────────────────────────────────

/// HEVC decoder configuration (see :class:`deepstream_decoders::HevcDecoderConfig`).
///
/// ``codec_data`` is intentionally not exposed (derived from the bitstream).
#[pyclass(
    from_py_object,
    name = "HevcDecoderConfig",
    module = "savant_rs.deepstream"
)]
#[derive(Clone)]
pub struct PyHevcDecoderConfig(pub(crate) HevcDecoderConfig);

#[pymethods]
impl PyHevcDecoderConfig {
    #[new]
    fn new(stream_format: PyHevcStreamFormat) -> Self {
        Self(HevcDecoderConfig::new(stream_format.into()))
    }

    fn with_num_extra_surfaces(&self, n: u32) -> Self {
        Self(self.0.clone().num_extra_surfaces(n))
    }

    fn with_drop_frame_interval(&self, n: u32) -> Self {
        Self(self.0.clone().drop_frame_interval(n))
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn with_cudadec_memtype(&self, t: PyCudadecMemtype) -> Self {
        Self(self.0.clone().cudadec_memtype(t.into()))
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn with_low_latency_mode(&self, v: bool) -> Self {
        Self(self.0.clone().low_latency_mode(v))
    }

    #[cfg(target_arch = "aarch64")]
    fn with_enable_max_performance(&self, v: bool) -> Self {
        Self(self.0.clone().enable_max_performance(v))
    }

    #[cfg(target_arch = "aarch64")]
    fn with_low_latency(&self, v: bool) -> Self {
        Self(self.0.clone().low_latency(v))
    }

    #[getter]
    fn stream_format(&self) -> PyHevcStreamFormat {
        self.0.stream_format.into()
    }

    #[getter]
    fn num_extra_surfaces(&self) -> Option<u32> {
        self.0.num_extra_surfaces
    }

    #[getter]
    fn drop_frame_interval(&self) -> Option<u32> {
        self.0.drop_frame_interval
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[getter]
    fn cudadec_memtype(&self) -> Option<PyCudadecMemtype> {
        self.0.cudadec_memtype.map(Into::into)
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[getter]
    fn low_latency_mode(&self) -> Option<bool> {
        self.0.low_latency_mode
    }

    #[cfg(target_arch = "aarch64")]
    #[getter]
    fn enable_max_performance(&self) -> Option<bool> {
        self.0.enable_max_performance
    }

    #[cfg(target_arch = "aarch64")]
    #[getter]
    fn low_latency(&self) -> Option<bool> {
        self.0.low_latency
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn __repr__(&self) -> String {
        format!(
            "HevcDecoderConfig(stream_format={}, num_extra_surfaces={:?}, \
             drop_frame_interval={:?}, cudadec_memtype={:?}, \
             low_latency_mode={:?})",
            self.0.stream_format,
            self.0.num_extra_surfaces,
            self.0.drop_frame_interval,
            self.0.cudadec_memtype,
            self.0.low_latency_mode,
        )
    }

    #[cfg(target_arch = "aarch64")]
    fn __repr__(&self) -> String {
        format!(
            "HevcDecoderConfig(stream_format={}, num_extra_surfaces={:?}, \
             drop_frame_interval={:?}, enable_max_performance={:?}, \
             low_latency={:?})",
            self.0.stream_format,
            self.0.num_extra_surfaces,
            self.0.drop_frame_interval,
            self.0.enable_max_performance,
            self.0.low_latency,
        )
    }
}

// Helper macro for VP8 / VP9 / AV1 (same tunable surface).
macro_rules! nvv4l2_codec_config {
    ($name:ident, $rust_ty:ident, $py_name:literal, $repr_name:literal) => {
        #[pyclass(from_py_object, name = $py_name, module = "savant_rs.deepstream")]
        #[derive(Clone)]
        pub struct $name(pub(crate) $rust_ty);

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> Self {
                Self($rust_ty::default())
            }

            fn with_num_extra_surfaces(&self, n: u32) -> Self {
                Self(self.0.clone().num_extra_surfaces(n))
            }

            fn with_drop_frame_interval(&self, n: u32) -> Self {
                Self(self.0.clone().drop_frame_interval(n))
            }

            #[cfg(not(target_arch = "aarch64"))]
            fn with_cudadec_memtype(&self, t: PyCudadecMemtype) -> Self {
                Self(self.0.clone().cudadec_memtype(t.into()))
            }

            #[cfg(not(target_arch = "aarch64"))]
            fn with_low_latency_mode(&self, v: bool) -> Self {
                Self(self.0.clone().low_latency_mode(v))
            }

            #[cfg(target_arch = "aarch64")]
            fn with_enable_max_performance(&self, v: bool) -> Self {
                Self(self.0.clone().enable_max_performance(v))
            }

            #[cfg(target_arch = "aarch64")]
            fn with_low_latency(&self, v: bool) -> Self {
                Self(self.0.clone().low_latency(v))
            }

            #[getter]
            fn num_extra_surfaces(&self) -> Option<u32> {
                self.0.num_extra_surfaces
            }

            #[getter]
            fn drop_frame_interval(&self) -> Option<u32> {
                self.0.drop_frame_interval
            }

            #[cfg(not(target_arch = "aarch64"))]
            #[getter]
            fn cudadec_memtype(&self) -> Option<PyCudadecMemtype> {
                self.0.cudadec_memtype.map(Into::into)
            }

            #[cfg(not(target_arch = "aarch64"))]
            #[getter]
            fn low_latency_mode(&self) -> Option<bool> {
                self.0.low_latency_mode
            }

            #[cfg(target_arch = "aarch64")]
            #[getter]
            fn enable_max_performance(&self) -> Option<bool> {
                self.0.enable_max_performance
            }

            #[cfg(target_arch = "aarch64")]
            #[getter]
            fn low_latency(&self) -> Option<bool> {
                self.0.low_latency
            }

            #[cfg(not(target_arch = "aarch64"))]
            fn __repr__(&self) -> String {
                format!(
                    concat!(
                        $repr_name,
                        "(num_extra_surfaces={:?}, drop_frame_interval={:?}, ",
                        "cudadec_memtype={:?}, low_latency_mode={:?})"
                    ),
                    self.0.num_extra_surfaces,
                    self.0.drop_frame_interval,
                    self.0.cudadec_memtype,
                    self.0.low_latency_mode,
                )
            }

            #[cfg(target_arch = "aarch64")]
            fn __repr__(&self) -> String {
                format!(
                    concat!(
                        $repr_name,
                        "(num_extra_surfaces={:?}, drop_frame_interval={:?}, ",
                        "enable_max_performance={:?}, low_latency={:?})"
                    ),
                    self.0.num_extra_surfaces,
                    self.0.drop_frame_interval,
                    self.0.enable_max_performance,
                    self.0.low_latency,
                )
            }
        }
    };
}

nvv4l2_codec_config!(
    PyVp8DecoderConfig,
    Vp8DecoderConfig,
    "Vp8DecoderConfig",
    "Vp8DecoderConfig"
);
nvv4l2_codec_config!(
    PyVp9DecoderConfig,
    Vp9DecoderConfig,
    "Vp9DecoderConfig",
    "Vp9DecoderConfig"
);
nvv4l2_codec_config!(
    PyAv1DecoderConfig,
    Av1DecoderConfig,
    "Av1DecoderConfig",
    "Av1DecoderConfig"
);

// ─── JPEG decoder config ────────────────────────────────────────────────

/// JPEG decoder configuration. Use the :meth:`gpu` / :meth:`cpu` factory
/// methods to select a backend.
#[pyclass(
    from_py_object,
    name = "JpegDecoderConfig",
    module = "savant_rs.deepstream"
)]
#[derive(Clone)]
pub struct PyJpegDecoderConfig(pub(crate) JpegDecoderConfig);

#[pymethods]
impl PyJpegDecoderConfig {
    #[staticmethod]
    fn gpu() -> Self {
        Self(JpegDecoderConfig::gpu())
    }

    #[staticmethod]
    fn cpu() -> Self {
        Self(JpegDecoderConfig::cpu())
    }

    #[getter]
    fn backend(&self) -> PyJpegBackend {
        self.0.backend.into()
    }

    fn __repr__(&self) -> String {
        format!("JpegDecoderConfig(backend={:?})", self.backend())
    }
}

// ─── PNG decoder config ─────────────────────────────────────────────────

/// PNG decoder configuration (no tunable parameters).
#[pyclass(
    from_py_object,
    name = "PngDecoderConfig",
    module = "savant_rs.deepstream"
)]
#[derive(Clone)]
pub struct PyPngDecoderConfig(pub(crate) PngDecoderConfig);

#[pymethods]
impl PyPngDecoderConfig {
    #[new]
    fn new() -> Self {
        Self(PngDecoderConfig)
    }

    fn __repr__(&self) -> &'static str {
        "PngDecoderConfig()"
    }
}

// ─── Raw RGBA decoder config ────────────────────────────────────────────

/// Raw RGBA decoder configuration (dimensions carried from the video frame).
#[pyclass(
    from_py_object,
    name = "RawRgbaDecoderConfig",
    module = "savant_rs.deepstream"
)]
#[derive(Clone)]
pub struct PyRawRgbaDecoderConfig(pub(crate) RawRgbaDecoderConfig);

#[pymethods]
impl PyRawRgbaDecoderConfig {
    #[new]
    fn new(width: u32, height: u32) -> Self {
        Self(RawRgbaDecoderConfig::new(width, height))
    }

    #[getter]
    fn width(&self) -> u32 {
        self.0.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.0.height
    }

    fn __repr__(&self) -> String {
        format!(
            "RawRgbaDecoderConfig(width={}, height={})",
            self.0.width, self.0.height
        )
    }
}

// ─── Raw RGB decoder config ─────────────────────────────────────────────

/// Raw RGB decoder configuration (dimensions carried from the video frame).
#[pyclass(
    from_py_object,
    name = "RawRgbDecoderConfig",
    module = "savant_rs.deepstream"
)]
#[derive(Clone)]
pub struct PyRawRgbDecoderConfig(pub(crate) RawRgbDecoderConfig);

#[pymethods]
impl PyRawRgbDecoderConfig {
    #[new]
    fn new(width: u32, height: u32) -> Self {
        Self(RawRgbDecoderConfig::new(width, height))
    }

    #[getter]
    fn width(&self) -> u32 {
        self.0.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.0.height
    }

    fn __repr__(&self) -> String {
        format!(
            "RawRgbDecoderConfig(width={}, height={})",
            self.0.width, self.0.height
        )
    }
}

// ─── Umbrella PyDecoderConfig ───────────────────────────────────────────

/// Umbrella wrapper for the Rust [`DecoderConfig`] enum. Inspect the
/// variant via :meth:`codec` and convert to/from a typed inner class with
/// ``as_*`` / ``with_*`` methods.
#[pyclass(
    from_py_object,
    name = "DecoderConfig",
    module = "savant_rs.deepstream"
)]
#[derive(Clone)]
pub struct PyDecoderConfig(pub(crate) DecoderConfig);

impl PyDecoderConfig {
    pub(crate) fn from_rust(cfg: DecoderConfig) -> Self {
        Self(cfg)
    }

    pub(crate) fn into_rust(self) -> DecoderConfig {
        self.0
    }
}

#[pymethods]
impl PyDecoderConfig {
    // ── variant constructors ─────────────────────────────────────────

    #[staticmethod]
    fn from_h264(cfg: &PyH264DecoderConfig) -> Self {
        Self(DecoderConfig::H264(cfg.0.clone()))
    }

    #[staticmethod]
    fn from_hevc(cfg: &PyHevcDecoderConfig) -> Self {
        Self(DecoderConfig::Hevc(cfg.0.clone()))
    }

    #[staticmethod]
    fn from_vp8(cfg: &PyVp8DecoderConfig) -> Self {
        Self(DecoderConfig::Vp8(cfg.0.clone()))
    }

    #[staticmethod]
    fn from_vp9(cfg: &PyVp9DecoderConfig) -> Self {
        Self(DecoderConfig::Vp9(cfg.0.clone()))
    }

    #[staticmethod]
    fn from_av1(cfg: &PyAv1DecoderConfig) -> Self {
        Self(DecoderConfig::Av1(cfg.0.clone()))
    }

    #[staticmethod]
    fn from_jpeg(cfg: &PyJpegDecoderConfig) -> Self {
        Self(DecoderConfig::Jpeg(cfg.0.clone()))
    }

    #[staticmethod]
    fn from_png(cfg: &PyPngDecoderConfig) -> Self {
        Self(DecoderConfig::Png(cfg.0.clone()))
    }

    #[staticmethod]
    fn from_raw_rgba(cfg: &PyRawRgbaDecoderConfig) -> Self {
        Self(DecoderConfig::RawRgba(cfg.0.clone()))
    }

    #[staticmethod]
    fn from_raw_rgb(cfg: &PyRawRgbDecoderConfig) -> Self {
        Self(DecoderConfig::RawRgb(cfg.0.clone()))
    }

    // ── query ────────────────────────────────────────────────────────

    /// The codec of the wrapped config.
    fn codec(&self) -> PyCodec {
        self.0.codec().into()
    }

    // ── typed accessors ──────────────────────────────────────────────

    fn as_h264(&self) -> Option<PyH264DecoderConfig> {
        match &self.0 {
            DecoderConfig::H264(c) => Some(PyH264DecoderConfig(c.clone())),
            _ => None,
        }
    }

    fn as_hevc(&self) -> Option<PyHevcDecoderConfig> {
        match &self.0 {
            DecoderConfig::Hevc(c) => Some(PyHevcDecoderConfig(c.clone())),
            _ => None,
        }
    }

    fn as_vp8(&self) -> Option<PyVp8DecoderConfig> {
        match &self.0 {
            DecoderConfig::Vp8(c) => Some(PyVp8DecoderConfig(c.clone())),
            _ => None,
        }
    }

    fn as_vp9(&self) -> Option<PyVp9DecoderConfig> {
        match &self.0 {
            DecoderConfig::Vp9(c) => Some(PyVp9DecoderConfig(c.clone())),
            _ => None,
        }
    }

    fn as_av1(&self) -> Option<PyAv1DecoderConfig> {
        match &self.0 {
            DecoderConfig::Av1(c) => Some(PyAv1DecoderConfig(c.clone())),
            _ => None,
        }
    }

    fn as_jpeg(&self) -> Option<PyJpegDecoderConfig> {
        match &self.0 {
            DecoderConfig::Jpeg(c) => Some(PyJpegDecoderConfig(c.clone())),
            _ => None,
        }
    }

    fn as_png(&self) -> Option<PyPngDecoderConfig> {
        match &self.0 {
            DecoderConfig::Png(c) => Some(PyPngDecoderConfig(c.clone())),
            _ => None,
        }
    }

    fn as_raw_rgba(&self) -> Option<PyRawRgbaDecoderConfig> {
        match &self.0 {
            DecoderConfig::RawRgba(c) => Some(PyRawRgbaDecoderConfig(c.clone())),
            _ => None,
        }
    }

    fn as_raw_rgb(&self) -> Option<PyRawRgbDecoderConfig> {
        match &self.0 {
            DecoderConfig::RawRgb(c) => Some(PyRawRgbDecoderConfig(c.clone())),
            _ => None,
        }
    }

    // ── builder-style replace-variant methods ────────────────────────

    /// Return a new :class:`DecoderConfig` whose inner variant is replaced
    /// by the given H.264 config. The codec identity is preserved by the
    /// caller; the method simply constructs ``DecoderConfig.H264(...)``.
    fn with_h264(&self, cfg: &PyH264DecoderConfig) -> Self {
        Self(DecoderConfig::H264(cfg.0.clone()))
    }

    fn with_hevc(&self, cfg: &PyHevcDecoderConfig) -> Self {
        Self(DecoderConfig::Hevc(cfg.0.clone()))
    }

    fn with_vp8(&self, cfg: &PyVp8DecoderConfig) -> Self {
        Self(DecoderConfig::Vp8(cfg.0.clone()))
    }

    fn with_vp9(&self, cfg: &PyVp9DecoderConfig) -> Self {
        Self(DecoderConfig::Vp9(cfg.0.clone()))
    }

    fn with_av1(&self, cfg: &PyAv1DecoderConfig) -> Self {
        Self(DecoderConfig::Av1(cfg.0.clone()))
    }

    fn with_jpeg(&self, cfg: &PyJpegDecoderConfig) -> Self {
        Self(DecoderConfig::Jpeg(cfg.0.clone()))
    }

    fn with_png(&self, cfg: &PyPngDecoderConfig) -> Self {
        Self(DecoderConfig::Png(cfg.0.clone()))
    }

    fn with_raw_rgba(&self, cfg: &PyRawRgbaDecoderConfig) -> Self {
        Self(DecoderConfig::RawRgba(cfg.0.clone()))
    }

    fn with_raw_rgb(&self, cfg: &PyRawRgbDecoderConfig) -> Self {
        Self(DecoderConfig::RawRgb(cfg.0.clone()))
    }

    fn __repr__(&self) -> String {
        format!("DecoderConfig({:?})", self.0)
    }
}

// ─── Module registration ────────────────────────────────────────────────

pub fn register_decoder_config_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyH264StreamFormat>()?;
    m.add_class::<PyHevcStreamFormat>()?;
    m.add_class::<PyJpegBackend>()?;
    #[cfg(not(target_arch = "aarch64"))]
    m.add_class::<PyCudadecMemtype>()?;
    m.add_class::<PyH264DecoderConfig>()?;
    m.add_class::<PyHevcDecoderConfig>()?;
    m.add_class::<PyVp8DecoderConfig>()?;
    m.add_class::<PyVp9DecoderConfig>()?;
    m.add_class::<PyAv1DecoderConfig>()?;
    m.add_class::<PyJpegDecoderConfig>()?;
    m.add_class::<PyPngDecoderConfig>()?;
    m.add_class::<PyRawRgbaDecoderConfig>()?;
    m.add_class::<PyRawRgbDecoderConfig>()?;
    m.add_class::<PyDecoderConfig>()?;
    Ok(())
}
