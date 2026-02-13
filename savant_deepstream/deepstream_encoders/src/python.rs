//! Python bindings for the GPU-accelerated encoder API.
//!
//! Provides a high-level Python interface that hides all GStreamer and
//! DeepStream internals. Users configure an encoder, submit NVMM buffers,
//! and read back encoded frames.
//!
//! # Example
//!
//! ```python
//! from deepstream_encoders import NvEncoder, Codec, EncoderConfig
//! from deepstream_encoders import HevcDgpuProps, HevcProfile
//! from deepstream_nvbufsurface import init_cuda
//!
//! init_cuda()
//!
//! props = HevcDgpuProps(bitrate=8_000_000, profile=HevcProfile.Main)
//! config = EncoderConfig(Codec.HEVC, 1920, 1080, properties=props)
//! encoder = NvEncoder(config)
//!
//! gen = encoder.generator
//! for i in range(100):
//!     buf = gen.acquire_surface(id=i)
//!     encoder.submit_frame(buf, frame_id=i, pts_ns=i * 33_333_333, duration_ns=33_333_333)
//!
//! # Pull encoded frames
//! while True:
//!     frame = encoder.pull_encoded()
//!     if frame is None:
//!         break
//!     print(f"Encoded frame {frame.frame_id}: {len(frame.data)} bytes")
//!
//! remaining = encoder.finish()
//! ```

use crate::properties::{self, EncoderProperties};
use crate::{
    EncodedFrame, EncoderConfig, EncoderError, NvBufSurfaceMemType, NvEncoder, VideoFormat,
};
use gstreamer as gst;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use savant_gstreamer::Codec;

// ═══════════════════════════════════════════════════════════════════════
// Codec / VideoFormat / MemType — same as before
// ═══════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
// PyCodec
// ---------------------------------------------------------------------------

/// Python enum for video codecs.
///
/// - ``H264`` — H.264 / AVC.
/// - ``HEVC`` — H.265 / HEVC.
/// - ``JPEG`` — Motion JPEG.
/// - ``AV1``  — AV1.
#[pyclass(name = "Codec", module = "deepstream_encoders._native", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyCodec {
    #[pyo3(name = "H264")]
    H264 = 0,
    #[pyo3(name = "HEVC")]
    Hevc = 1,
    #[pyo3(name = "JPEG")]
    Jpeg = 2,
    #[pyo3(name = "AV1")]
    Av1 = 3,
}

#[pymethods]
impl PyCodec {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match Codec::from_name(name) {
            Some(c) => Ok(c.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown codec: '{}'. Expected one of: h264, hevc, h265, jpeg, av1",
                name
            ))),
        }
    }

    fn name(&self) -> &'static str {
        let c: Codec = (*self).into();
        c.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "Codec.{}",
            match self {
                PyCodec::H264 => "H264",
                PyCodec::Hevc => "HEVC",
                PyCodec::Jpeg => "JPEG",
                PyCodec::Av1 => "AV1",
            }
        )
    }
}

impl From<PyCodec> for Codec {
    fn from(c: PyCodec) -> Self {
        match c {
            PyCodec::H264 => Codec::H264,
            PyCodec::Hevc => Codec::Hevc,
            PyCodec::Jpeg => Codec::Jpeg,
            PyCodec::Av1 => Codec::Av1,
        }
    }
}

impl From<Codec> for PyCodec {
    fn from(c: Codec) -> Self {
        match c {
            Codec::H264 => PyCodec::H264,
            Codec::Hevc => PyCodec::Hevc,
            Codec::Jpeg => PyCodec::Jpeg,
            Codec::Av1 => PyCodec::Av1,
        }
    }
}

fn extract_codec(ob: &Bound<'_, PyAny>) -> PyResult<Codec> {
    if let Ok(py_codec) = ob.extract::<PyCodec>() {
        return Ok(py_codec.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return Codec::from_name(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown codec: '{s}'. Expected one of: h264, hevc, h265, jpeg, av1"
            ))
        });
    }
    if let Ok(name_val) = ob.call_method0("name") {
        if let Ok(s) = name_val.extract::<String>() {
            return Codec::from_name(&s).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown codec: '{s}'. Expected one of: h264, hevc, h265, jpeg, av1"
                ))
            });
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a Codec enum value or a codec name string (h264, hevc, h265, jpeg, av1)",
    ))
}

// ---------------------------------------------------------------------------
// PyVideoFormat
// ---------------------------------------------------------------------------

/// Video pixel format.
#[pyclass(
    name = "VideoFormat",
    module = "deepstream_encoders._native",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyVideoFormat {
    #[pyo3(name = "RGBA")]
    Rgba = 0,
    #[pyo3(name = "BGRx")]
    Bgrx = 1,
    #[pyo3(name = "NV12")]
    Nv12 = 2,
    #[pyo3(name = "NV21")]
    Nv21 = 3,
    #[pyo3(name = "I420")]
    I420 = 4,
    #[pyo3(name = "UYVY")]
    Uyvy = 5,
    #[pyo3(name = "GRAY8")]
    Gray8 = 6,
}

#[pymethods]
impl PyVideoFormat {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match VideoFormat::from_name(name) {
            Some(f) => Ok(f.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown video format: '{name}'"
            ))),
        }
    }

    fn name(&self) -> &'static str {
        let f: VideoFormat = (*self).into();
        f.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoFormat.{}",
            match self {
                PyVideoFormat::Rgba => "RGBA",
                PyVideoFormat::Bgrx => "BGRx",
                PyVideoFormat::Nv12 => "NV12",
                PyVideoFormat::Nv21 => "NV21",
                PyVideoFormat::I420 => "I420",
                PyVideoFormat::Uyvy => "UYVY",
                PyVideoFormat::Gray8 => "GRAY8",
            }
        )
    }
}

impl From<PyVideoFormat> for VideoFormat {
    fn from(f: PyVideoFormat) -> Self {
        match f {
            PyVideoFormat::Rgba => VideoFormat::RGBA,
            PyVideoFormat::Bgrx => VideoFormat::BGRx,
            PyVideoFormat::Nv12 => VideoFormat::NV12,
            PyVideoFormat::Nv21 => VideoFormat::NV21,
            PyVideoFormat::I420 => VideoFormat::I420,
            PyVideoFormat::Uyvy => VideoFormat::UYVY,
            PyVideoFormat::Gray8 => VideoFormat::GRAY8,
        }
    }
}

impl From<VideoFormat> for PyVideoFormat {
    fn from(f: VideoFormat) -> Self {
        match f {
            VideoFormat::RGBA => PyVideoFormat::Rgba,
            VideoFormat::BGRx => PyVideoFormat::Bgrx,
            VideoFormat::NV12 => PyVideoFormat::Nv12,
            VideoFormat::NV21 => PyVideoFormat::Nv21,
            VideoFormat::I420 => PyVideoFormat::I420,
            VideoFormat::UYVY => PyVideoFormat::Uyvy,
            VideoFormat::GRAY8 => PyVideoFormat::Gray8,
        }
    }
}

fn extract_video_format(ob: &Bound<'_, PyAny>) -> PyResult<VideoFormat> {
    if let Ok(py_fmt) = ob.extract::<PyVideoFormat>() {
        return Ok(py_fmt.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return VideoFormat::from_name(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unknown video format: '{s}'"))
        });
    }
    if let Ok(name_val) = ob.call_method0("name") {
        if let Ok(s) = name_val.extract::<String>() {
            return VideoFormat::from_name(&s).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Unknown video format: '{s}'"))
            });
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a VideoFormat enum value or a format name string",
    ))
}

// ---------------------------------------------------------------------------
// PyMemType
// ---------------------------------------------------------------------------

/// NvBufSurface memory type.
#[pyclass(name = "MemType", module = "deepstream_encoders._native", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyMemType {
    #[pyo3(name = "DEFAULT")]
    Default = 0,
    #[pyo3(name = "CUDA_PINNED")]
    CudaPinned = 1,
    #[pyo3(name = "CUDA_DEVICE")]
    CudaDevice = 2,
    #[pyo3(name = "CUDA_UNIFIED")]
    CudaUnified = 3,
    #[pyo3(name = "SURFACE_ARRAY")]
    SurfaceArray = 4,
    #[pyo3(name = "HANDLE")]
    Handle = 5,
    #[pyo3(name = "SYSTEM")]
    System = 6,
}

#[pymethods]
impl PyMemType {
    fn name(&self) -> &'static str {
        match self {
            PyMemType::Default => "default",
            PyMemType::CudaPinned => "cuda_pinned",
            PyMemType::CudaDevice => "cuda_device",
            PyMemType::CudaUnified => "cuda_unified",
            PyMemType::SurfaceArray => "surface_array",
            PyMemType::Handle => "handle",
            PyMemType::System => "system",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MemType.{}",
            match self {
                PyMemType::Default => "DEFAULT",
                PyMemType::CudaPinned => "CUDA_PINNED",
                PyMemType::CudaDevice => "CUDA_DEVICE",
                PyMemType::CudaUnified => "CUDA_UNIFIED",
                PyMemType::SurfaceArray => "SURFACE_ARRAY",
                PyMemType::Handle => "HANDLE",
                PyMemType::System => "SYSTEM",
            }
        )
    }
}

impl From<PyMemType> for NvBufSurfaceMemType {
    fn from(m: PyMemType) -> Self {
        match m {
            PyMemType::Default => NvBufSurfaceMemType::Default,
            PyMemType::CudaPinned => NvBufSurfaceMemType::CudaPinned,
            PyMemType::CudaDevice => NvBufSurfaceMemType::CudaDevice,
            PyMemType::CudaUnified => NvBufSurfaceMemType::CudaUnified,
            PyMemType::SurfaceArray => NvBufSurfaceMemType::SurfaceArray,
            PyMemType::Handle => NvBufSurfaceMemType::Handle,
            PyMemType::System => NvBufSurfaceMemType::System,
        }
    }
}

impl From<NvBufSurfaceMemType> for PyMemType {
    fn from(m: NvBufSurfaceMemType) -> Self {
        match m {
            NvBufSurfaceMemType::Default => PyMemType::Default,
            NvBufSurfaceMemType::CudaPinned => PyMemType::CudaPinned,
            NvBufSurfaceMemType::CudaDevice => PyMemType::CudaDevice,
            NvBufSurfaceMemType::CudaUnified => PyMemType::CudaUnified,
            NvBufSurfaceMemType::SurfaceArray => PyMemType::SurfaceArray,
            NvBufSurfaceMemType::Handle => PyMemType::Handle,
            NvBufSurfaceMemType::System => PyMemType::System,
        }
    }
}

fn extract_mem_type(ob: &Bound<'_, PyAny>) -> PyResult<NvBufSurfaceMemType> {
    if let Ok(py_mt) = ob.extract::<PyMemType>() {
        return Ok(py_mt.into());
    }
    if let Ok(v) = ob.extract::<u32>() {
        return Ok(NvBufSurfaceMemType::from(v));
    }
    if let Ok(name_val) = ob.call_method0("name") {
        if let Ok(s) = name_val.extract::<String>() {
            let mt = match s.as_str() {
                "default" => NvBufSurfaceMemType::Default,
                "cuda_pinned" => NvBufSurfaceMemType::CudaPinned,
                "cuda_device" => NvBufSurfaceMemType::CudaDevice,
                "cuda_unified" => NvBufSurfaceMemType::CudaUnified,
                "surface_array" => NvBufSurfaceMemType::SurfaceArray,
                "handle" => NvBufSurfaceMemType::Handle,
                "system" => NvBufSurfaceMemType::System,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown memory type: '{s}'"
                    )));
                }
            };
            return Ok(mt);
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a MemType enum value or an int memory type (0-6)",
    ))
}

// ═══════════════════════════════════════════════════════════════════════
// Encoder property enums
// ═══════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
// PyPlatform
// ---------------------------------------------------------------------------

/// Target hardware platform.
///
/// - ``DGPU``   — Discrete GPU (desktop/server).
/// - ``JETSON`` — NVIDIA Jetson embedded platform.
#[pyclass(name = "Platform", module = "deepstream_encoders._native", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyPlatform {
    #[pyo3(name = "DGPU")]
    Dgpu = 0,
    #[pyo3(name = "JETSON")]
    Jetson = 1,
}

#[pymethods]
impl PyPlatform {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match properties::Platform::from_name(name) {
            Some(p) => Ok(p.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown platform: '{name}'. Expected 'dgpu' or 'jetson'"
            ))),
        }
    }

    fn name(&self) -> &'static str {
        let p: properties::Platform = (*self).into();
        p.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "Platform.{}",
            match self {
                PyPlatform::Dgpu => "DGPU",
                PyPlatform::Jetson => "JETSON",
            }
        )
    }
}

impl From<PyPlatform> for properties::Platform {
    fn from(p: PyPlatform) -> Self {
        match p {
            PyPlatform::Dgpu => properties::Platform::Dgpu,
            PyPlatform::Jetson => properties::Platform::Jetson,
        }
    }
}

impl From<properties::Platform> for PyPlatform {
    fn from(p: properties::Platform) -> Self {
        match p {
            properties::Platform::Dgpu => PyPlatform::Dgpu,
            properties::Platform::Jetson => PyPlatform::Jetson,
        }
    }
}

// ---------------------------------------------------------------------------
// PyRateControl
// ---------------------------------------------------------------------------

/// Rate-control mode.
///
/// - ``VBR``  — Variable bitrate.
/// - ``CBR``  — Constant bitrate.
/// - ``CQP``  — Constant QP.
#[pyclass(
    name = "RateControl",
    module = "deepstream_encoders._native",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyRateControl {
    #[pyo3(name = "VBR")]
    Vbr = 0,
    #[pyo3(name = "CBR")]
    Cbr = 1,
    #[pyo3(name = "CQP")]
    Cqp = 2,
}

#[pymethods]
impl PyRateControl {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match properties::RateControl::from_name(name) {
            Some(r) => Ok(r.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown rate control: '{name}'"
            ))),
        }
    }
    fn name(&self) -> &'static str {
        let r: properties::RateControl = (*self).into();
        r.name()
    }
    fn __repr__(&self) -> String {
        format!(
            "RateControl.{}",
            match self {
                PyRateControl::Vbr => "VBR",
                PyRateControl::Cbr => "CBR",
                PyRateControl::Cqp => "CQP",
            }
        )
    }
}

impl From<PyRateControl> for properties::RateControl {
    fn from(r: PyRateControl) -> Self {
        match r {
            PyRateControl::Vbr => properties::RateControl::VariableBitrate,
            PyRateControl::Cbr => properties::RateControl::ConstantBitrate,
            PyRateControl::Cqp => properties::RateControl::ConstantQP,
        }
    }
}

impl From<properties::RateControl> for PyRateControl {
    fn from(r: properties::RateControl) -> Self {
        match r {
            properties::RateControl::VariableBitrate => PyRateControl::Vbr,
            properties::RateControl::ConstantBitrate => PyRateControl::Cbr,
            properties::RateControl::ConstantQP => PyRateControl::Cqp,
        }
    }
}

fn extract_rate_control(ob: &Bound<'_, PyAny>) -> PyResult<properties::RateControl> {
    if let Ok(v) = ob.extract::<PyRateControl>() {
        return Ok(v.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return properties::RateControl::from_name(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unknown rate control: '{s}'"))
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a RateControl enum or string",
    ))
}

// ---------------------------------------------------------------------------
// PyH264Profile
// ---------------------------------------------------------------------------

/// H.264 encoding profile.
///
/// - ``BASELINE`` — Baseline profile.
/// - ``MAIN``     — Main profile.
/// - ``HIGH``     — High profile.
/// - ``HIGH444``  — High 4:4:4 Predictive (dGPU only).
#[pyclass(
    name = "H264Profile",
    module = "deepstream_encoders._native",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyH264Profile {
    #[pyo3(name = "BASELINE")]
    Baseline = 0,
    #[pyo3(name = "MAIN")]
    Main = 2,
    #[pyo3(name = "HIGH")]
    High = 4,
    #[pyo3(name = "HIGH444")]
    High444 = 7,
}

#[pymethods]
impl PyH264Profile {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match properties::H264Profile::from_name(name) {
            Some(p) => Ok(p.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown H.264 profile: '{name}'"
            ))),
        }
    }
    fn name(&self) -> &'static str {
        let p: properties::H264Profile = (*self).into();
        p.name()
    }
    fn __repr__(&self) -> String {
        format!(
            "H264Profile.{}",
            match self {
                PyH264Profile::Baseline => "BASELINE",
                PyH264Profile::Main => "MAIN",
                PyH264Profile::High => "HIGH",
                PyH264Profile::High444 => "HIGH444",
            }
        )
    }
}

impl From<PyH264Profile> for properties::H264Profile {
    fn from(p: PyH264Profile) -> Self {
        match p {
            PyH264Profile::Baseline => properties::H264Profile::Baseline,
            PyH264Profile::Main => properties::H264Profile::Main,
            PyH264Profile::High => properties::H264Profile::High,
            PyH264Profile::High444 => properties::H264Profile::High444,
        }
    }
}

impl From<properties::H264Profile> for PyH264Profile {
    fn from(p: properties::H264Profile) -> Self {
        match p {
            properties::H264Profile::Baseline => PyH264Profile::Baseline,
            properties::H264Profile::Main => PyH264Profile::Main,
            properties::H264Profile::High => PyH264Profile::High,
            properties::H264Profile::High444 => PyH264Profile::High444,
        }
    }
}

fn extract_h264_profile(ob: &Bound<'_, PyAny>) -> PyResult<properties::H264Profile> {
    if let Ok(v) = ob.extract::<PyH264Profile>() {
        return Ok(v.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return properties::H264Profile::from_name(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unknown H.264 profile: '{s}'"))
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected an H264Profile enum or string",
    ))
}

// ---------------------------------------------------------------------------
// PyHevcProfile
// ---------------------------------------------------------------------------

/// HEVC (H.265) encoding profile.
///
/// - ``MAIN``   — Main profile.
/// - ``MAIN10`` — Main 10-bit profile.
/// - ``FREXT``  — Format Range Extensions.
#[pyclass(
    name = "HevcProfile",
    module = "deepstream_encoders._native",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyHevcProfile {
    #[pyo3(name = "MAIN")]
    Main = 0,
    #[pyo3(name = "MAIN10")]
    Main10 = 1,
    #[pyo3(name = "FREXT")]
    Frext = 3,
}

#[pymethods]
impl PyHevcProfile {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match properties::HevcProfile::from_name(name) {
            Some(p) => Ok(p.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown HEVC profile: '{name}'"
            ))),
        }
    }
    fn name(&self) -> &'static str {
        let p: properties::HevcProfile = (*self).into();
        p.name()
    }
    fn __repr__(&self) -> String {
        format!(
            "HevcProfile.{}",
            match self {
                PyHevcProfile::Main => "MAIN",
                PyHevcProfile::Main10 => "MAIN10",
                PyHevcProfile::Frext => "FREXT",
            }
        )
    }
}

impl From<PyHevcProfile> for properties::HevcProfile {
    fn from(p: PyHevcProfile) -> Self {
        match p {
            PyHevcProfile::Main => properties::HevcProfile::Main,
            PyHevcProfile::Main10 => properties::HevcProfile::Main10,
            PyHevcProfile::Frext => properties::HevcProfile::Frext,
        }
    }
}

impl From<properties::HevcProfile> for PyHevcProfile {
    fn from(p: properties::HevcProfile) -> Self {
        match p {
            properties::HevcProfile::Main => PyHevcProfile::Main,
            properties::HevcProfile::Main10 => PyHevcProfile::Main10,
            properties::HevcProfile::Frext => PyHevcProfile::Frext,
        }
    }
}

fn extract_hevc_profile(ob: &Bound<'_, PyAny>) -> PyResult<properties::HevcProfile> {
    if let Ok(v) = ob.extract::<PyHevcProfile>() {
        return Ok(v.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return properties::HevcProfile::from_name(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unknown HEVC profile: '{s}'"))
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a HevcProfile enum or string",
    ))
}

// ---------------------------------------------------------------------------
// PyDgpuPreset
// ---------------------------------------------------------------------------

/// dGPU NVENC preset (P1–P7).
///
/// Lower values are faster; higher values yield better quality.
#[pyclass(
    name = "DgpuPreset",
    module = "deepstream_encoders._native",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyDgpuPreset {
    P1 = 1,
    P2 = 2,
    P3 = 3,
    P4 = 4,
    P5 = 5,
    P6 = 6,
    P7 = 7,
}

#[pymethods]
impl PyDgpuPreset {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match properties::DgpuPreset::from_name(name) {
            Some(p) => Ok(p.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown dGPU preset: '{name}'"
            ))),
        }
    }
    fn name(&self) -> &'static str {
        let p: properties::DgpuPreset = (*self).into();
        p.name()
    }
    fn __repr__(&self) -> String {
        format!("DgpuPreset.{}", self.name())
    }
}

impl From<PyDgpuPreset> for properties::DgpuPreset {
    fn from(p: PyDgpuPreset) -> Self {
        match p {
            PyDgpuPreset::P1 => properties::DgpuPreset::P1,
            PyDgpuPreset::P2 => properties::DgpuPreset::P2,
            PyDgpuPreset::P3 => properties::DgpuPreset::P3,
            PyDgpuPreset::P4 => properties::DgpuPreset::P4,
            PyDgpuPreset::P5 => properties::DgpuPreset::P5,
            PyDgpuPreset::P6 => properties::DgpuPreset::P6,
            PyDgpuPreset::P7 => properties::DgpuPreset::P7,
        }
    }
}

impl From<properties::DgpuPreset> for PyDgpuPreset {
    fn from(p: properties::DgpuPreset) -> Self {
        match p {
            properties::DgpuPreset::P1 => PyDgpuPreset::P1,
            properties::DgpuPreset::P2 => PyDgpuPreset::P2,
            properties::DgpuPreset::P3 => PyDgpuPreset::P3,
            properties::DgpuPreset::P4 => PyDgpuPreset::P4,
            properties::DgpuPreset::P5 => PyDgpuPreset::P5,
            properties::DgpuPreset::P6 => PyDgpuPreset::P6,
            properties::DgpuPreset::P7 => PyDgpuPreset::P7,
        }
    }
}

fn extract_dgpu_preset(ob: &Bound<'_, PyAny>) -> PyResult<properties::DgpuPreset> {
    if let Ok(v) = ob.extract::<PyDgpuPreset>() {
        return Ok(v.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return properties::DgpuPreset::from_name(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unknown dGPU preset: '{s}'"))
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a DgpuPreset enum or string",
    ))
}

// ---------------------------------------------------------------------------
// PyTuningPreset
// ---------------------------------------------------------------------------

/// dGPU tuning-info preset.
///
/// - ``HIGH_QUALITY``      — Optimize for quality.
/// - ``LOW_LATENCY``       — Low latency (default).
/// - ``ULTRA_LOW_LATENCY`` — Ultra-low latency.
/// - ``LOSSLESS``          — Lossless encoding.
#[pyclass(
    name = "TuningPreset",
    module = "deepstream_encoders._native",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyTuningPreset {
    #[pyo3(name = "HIGH_QUALITY")]
    HighQuality = 1,
    #[pyo3(name = "LOW_LATENCY")]
    LowLatency = 2,
    #[pyo3(name = "ULTRA_LOW_LATENCY")]
    UltraLowLatency = 3,
    #[pyo3(name = "LOSSLESS")]
    Lossless = 4,
}

#[pymethods]
impl PyTuningPreset {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match properties::TuningPreset::from_name(name) {
            Some(t) => Ok(t.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown tuning preset: '{name}'"
            ))),
        }
    }
    fn name(&self) -> &'static str {
        let t: properties::TuningPreset = (*self).into();
        t.name()
    }
    fn __repr__(&self) -> String {
        format!(
            "TuningPreset.{}",
            match self {
                PyTuningPreset::HighQuality => "HIGH_QUALITY",
                PyTuningPreset::LowLatency => "LOW_LATENCY",
                PyTuningPreset::UltraLowLatency => "ULTRA_LOW_LATENCY",
                PyTuningPreset::Lossless => "LOSSLESS",
            }
        )
    }
}

impl From<PyTuningPreset> for properties::TuningPreset {
    fn from(t: PyTuningPreset) -> Self {
        match t {
            PyTuningPreset::HighQuality => properties::TuningPreset::HighQuality,
            PyTuningPreset::LowLatency => properties::TuningPreset::LowLatency,
            PyTuningPreset::UltraLowLatency => properties::TuningPreset::UltraLowLatency,
            PyTuningPreset::Lossless => properties::TuningPreset::Lossless,
        }
    }
}

impl From<properties::TuningPreset> for PyTuningPreset {
    fn from(t: properties::TuningPreset) -> Self {
        match t {
            properties::TuningPreset::HighQuality => PyTuningPreset::HighQuality,
            properties::TuningPreset::LowLatency => PyTuningPreset::LowLatency,
            properties::TuningPreset::UltraLowLatency => PyTuningPreset::UltraLowLatency,
            properties::TuningPreset::Lossless => PyTuningPreset::Lossless,
        }
    }
}

fn extract_tuning_preset(ob: &Bound<'_, PyAny>) -> PyResult<properties::TuningPreset> {
    if let Ok(v) = ob.extract::<PyTuningPreset>() {
        return Ok(v.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return properties::TuningPreset::from_name(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unknown tuning preset: '{s}'"))
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a TuningPreset enum or string",
    ))
}

// ---------------------------------------------------------------------------
// PyJetsonPresetLevel
// ---------------------------------------------------------------------------

/// Jetson HW encoder preset level.
///
/// - ``DISABLED``   — Disable HW preset.
/// - ``ULTRA_FAST`` — Ultra-fast (default).
/// - ``FAST``       — Fast.
/// - ``MEDIUM``     — Medium.
/// - ``SLOW``       — Slow (highest quality).
#[pyclass(
    name = "JetsonPresetLevel",
    module = "deepstream_encoders._native",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyJetsonPresetLevel {
    #[pyo3(name = "DISABLED")]
    Disabled = 0,
    #[pyo3(name = "ULTRA_FAST")]
    UltraFast = 1,
    #[pyo3(name = "FAST")]
    Fast = 2,
    #[pyo3(name = "MEDIUM")]
    Medium = 3,
    #[pyo3(name = "SLOW")]
    Slow = 4,
}

#[pymethods]
impl PyJetsonPresetLevel {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match properties::JetsonPresetLevel::from_name(name) {
            Some(p) => Ok(p.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown Jetson preset level: '{name}'"
            ))),
        }
    }
    fn name(&self) -> &'static str {
        let p: properties::JetsonPresetLevel = (*self).into();
        p.name()
    }
    fn __repr__(&self) -> String {
        format!(
            "JetsonPresetLevel.{}",
            match self {
                PyJetsonPresetLevel::Disabled => "DISABLED",
                PyJetsonPresetLevel::UltraFast => "ULTRA_FAST",
                PyJetsonPresetLevel::Fast => "FAST",
                PyJetsonPresetLevel::Medium => "MEDIUM",
                PyJetsonPresetLevel::Slow => "SLOW",
            }
        )
    }
}

impl From<PyJetsonPresetLevel> for properties::JetsonPresetLevel {
    fn from(p: PyJetsonPresetLevel) -> Self {
        match p {
            PyJetsonPresetLevel::Disabled => properties::JetsonPresetLevel::Disabled,
            PyJetsonPresetLevel::UltraFast => properties::JetsonPresetLevel::UltraFast,
            PyJetsonPresetLevel::Fast => properties::JetsonPresetLevel::Fast,
            PyJetsonPresetLevel::Medium => properties::JetsonPresetLevel::Medium,
            PyJetsonPresetLevel::Slow => properties::JetsonPresetLevel::Slow,
        }
    }
}

impl From<properties::JetsonPresetLevel> for PyJetsonPresetLevel {
    fn from(p: properties::JetsonPresetLevel) -> Self {
        match p {
            properties::JetsonPresetLevel::Disabled => PyJetsonPresetLevel::Disabled,
            properties::JetsonPresetLevel::UltraFast => PyJetsonPresetLevel::UltraFast,
            properties::JetsonPresetLevel::Fast => PyJetsonPresetLevel::Fast,
            properties::JetsonPresetLevel::Medium => PyJetsonPresetLevel::Medium,
            properties::JetsonPresetLevel::Slow => PyJetsonPresetLevel::Slow,
        }
    }
}

fn extract_jetson_preset(ob: &Bound<'_, PyAny>) -> PyResult<properties::JetsonPresetLevel> {
    if let Ok(v) = ob.extract::<PyJetsonPresetLevel>() {
        return Ok(v.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return properties::JetsonPresetLevel::from_name(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unknown Jetson preset level: '{s}'"))
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a JetsonPresetLevel enum or string",
    ))
}

// ═══════════════════════════════════════════════════════════════════════
// Python wrappers for per-codec/platform property structs
// ═══════════════════════════════════════════════════════════════════════

macro_rules! opt_getter {
    ($self:ident . $field:ident -> u32) => {
        $self.inner.$field
    };
    ($self:ident . $field:ident -> bool) => {
        $self.inner.$field
    };
    ($self:ident . $field:ident -> String) => {
        $self.inner.$field.clone()
    };
}

// ---------------------------------------------------------------------------
// PyH264DgpuProps
// ---------------------------------------------------------------------------

/// H.264 encoder properties for dGPU (``nvv4l2h264enc``).
///
/// All parameters are optional.  ``None`` means "use encoder default."
#[pyclass(name = "H264DgpuProps", module = "deepstream_encoders._native")]
#[derive(Debug, Clone)]
pub struct PyH264DgpuProps {
    inner: properties::H264DgpuProps,
}

#[pymethods]
impl PyH264DgpuProps {
    #[new]
    #[pyo3(signature = (
        bitrate = None,
        control_rate = None,
        profile = None,
        iframeinterval = None,
        idrinterval = None,
        preset = None,
        tuning_info = None,
        qp_range = None,
        const_qp = None,
        init_qp = None,
        max_bitrate = None,
        vbv_buf_size = None,
        vbv_init = None,
        cq = None,
        aq = None,
        temporal_aq = None,
        extended_colorformat = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bitrate: Option<u32>,
        control_rate: Option<&Bound<'_, PyAny>>,
        profile: Option<&Bound<'_, PyAny>>,
        iframeinterval: Option<u32>,
        idrinterval: Option<u32>,
        preset: Option<&Bound<'_, PyAny>>,
        tuning_info: Option<&Bound<'_, PyAny>>,
        qp_range: Option<String>,
        const_qp: Option<String>,
        init_qp: Option<String>,
        max_bitrate: Option<u32>,
        vbv_buf_size: Option<u32>,
        vbv_init: Option<u32>,
        cq: Option<u32>,
        aq: Option<u32>,
        temporal_aq: Option<bool>,
        extended_colorformat: Option<bool>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: properties::H264DgpuProps {
                bitrate,
                control_rate: control_rate.map(extract_rate_control).transpose()?,
                profile: profile.map(extract_h264_profile).transpose()?,
                iframeinterval,
                idrinterval,
                preset: preset.map(extract_dgpu_preset).transpose()?,
                tuning_info: tuning_info.map(extract_tuning_preset).transpose()?,
                qp_range,
                const_qp,
                init_qp,
                max_bitrate,
                vbv_buf_size,
                vbv_init,
                cq,
                aq,
                temporal_aq,
                extended_colorformat,
            },
        })
    }

    #[getter]
    fn bitrate(&self) -> Option<u32> {
        opt_getter!(self.bitrate -> u32)
    }
    #[getter]
    fn control_rate(&self) -> Option<PyRateControl> {
        self.inner.control_rate.map(Into::into)
    }
    #[getter]
    fn profile(&self) -> Option<PyH264Profile> {
        self.inner.profile.map(Into::into)
    }
    #[getter]
    fn iframeinterval(&self) -> Option<u32> {
        opt_getter!(self.iframeinterval -> u32)
    }
    #[getter]
    fn idrinterval(&self) -> Option<u32> {
        opt_getter!(self.idrinterval -> u32)
    }
    #[getter]
    fn preset(&self) -> Option<PyDgpuPreset> {
        self.inner.preset.map(Into::into)
    }
    #[getter]
    fn tuning_info(&self) -> Option<PyTuningPreset> {
        self.inner.tuning_info.map(Into::into)
    }
    #[getter]
    fn qp_range(&self) -> Option<String> {
        opt_getter!(self.qp_range -> String)
    }
    #[getter]
    fn const_qp(&self) -> Option<String> {
        opt_getter!(self.const_qp -> String)
    }
    #[getter]
    fn init_qp(&self) -> Option<String> {
        opt_getter!(self.init_qp -> String)
    }
    #[getter]
    fn max_bitrate(&self) -> Option<u32> {
        opt_getter!(self.max_bitrate -> u32)
    }
    #[getter]
    fn vbv_buf_size(&self) -> Option<u32> {
        opt_getter!(self.vbv_buf_size -> u32)
    }
    #[getter]
    fn vbv_init(&self) -> Option<u32> {
        opt_getter!(self.vbv_init -> u32)
    }
    #[getter]
    fn cq(&self) -> Option<u32> {
        opt_getter!(self.cq -> u32)
    }
    #[getter]
    fn aq(&self) -> Option<u32> {
        opt_getter!(self.aq -> u32)
    }
    #[getter]
    fn temporal_aq(&self) -> Option<bool> {
        opt_getter!(self.temporal_aq -> bool)
    }
    #[getter]
    fn extended_colorformat(&self) -> Option<bool> {
        opt_getter!(self.extended_colorformat -> bool)
    }

    #[staticmethod]
    fn from_pairs(pairs: std::collections::HashMap<String, String>) -> PyResult<Self> {
        let inner = properties::H264DgpuProps::from_pairs(&pairs).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "H264DgpuProps(bitrate={:?}, profile={:?})",
            self.inner.bitrate,
            self.inner.profile.map(|p| p.name())
        )
    }
}

// ---------------------------------------------------------------------------
// PyHevcDgpuProps
// ---------------------------------------------------------------------------

/// HEVC encoder properties for dGPU (``nvv4l2h265enc``).
#[pyclass(name = "HevcDgpuProps", module = "deepstream_encoders._native")]
#[derive(Debug, Clone)]
pub struct PyHevcDgpuProps {
    inner: properties::HevcDgpuProps,
}

#[pymethods]
impl PyHevcDgpuProps {
    #[new]
    #[pyo3(signature = (
        bitrate = None,
        control_rate = None,
        profile = None,
        iframeinterval = None,
        idrinterval = None,
        preset = None,
        tuning_info = None,
        qp_range = None,
        const_qp = None,
        init_qp = None,
        max_bitrate = None,
        vbv_buf_size = None,
        vbv_init = None,
        cq = None,
        aq = None,
        temporal_aq = None,
        extended_colorformat = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bitrate: Option<u32>,
        control_rate: Option<&Bound<'_, PyAny>>,
        profile: Option<&Bound<'_, PyAny>>,
        iframeinterval: Option<u32>,
        idrinterval: Option<u32>,
        preset: Option<&Bound<'_, PyAny>>,
        tuning_info: Option<&Bound<'_, PyAny>>,
        qp_range: Option<String>,
        const_qp: Option<String>,
        init_qp: Option<String>,
        max_bitrate: Option<u32>,
        vbv_buf_size: Option<u32>,
        vbv_init: Option<u32>,
        cq: Option<u32>,
        aq: Option<u32>,
        temporal_aq: Option<bool>,
        extended_colorformat: Option<bool>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: properties::HevcDgpuProps {
                bitrate,
                control_rate: control_rate.map(extract_rate_control).transpose()?,
                profile: profile.map(extract_hevc_profile).transpose()?,
                iframeinterval,
                idrinterval,
                preset: preset.map(extract_dgpu_preset).transpose()?,
                tuning_info: tuning_info.map(extract_tuning_preset).transpose()?,
                qp_range,
                const_qp,
                init_qp,
                max_bitrate,
                vbv_buf_size,
                vbv_init,
                cq,
                aq,
                temporal_aq,
                extended_colorformat,
            },
        })
    }

    #[getter]
    fn bitrate(&self) -> Option<u32> {
        opt_getter!(self.bitrate -> u32)
    }
    #[getter]
    fn control_rate(&self) -> Option<PyRateControl> {
        self.inner.control_rate.map(Into::into)
    }
    #[getter]
    fn profile(&self) -> Option<PyHevcProfile> {
        self.inner.profile.map(Into::into)
    }
    #[getter]
    fn iframeinterval(&self) -> Option<u32> {
        opt_getter!(self.iframeinterval -> u32)
    }
    #[getter]
    fn idrinterval(&self) -> Option<u32> {
        opt_getter!(self.idrinterval -> u32)
    }
    #[getter]
    fn preset(&self) -> Option<PyDgpuPreset> {
        self.inner.preset.map(Into::into)
    }
    #[getter]
    fn tuning_info(&self) -> Option<PyTuningPreset> {
        self.inner.tuning_info.map(Into::into)
    }
    #[getter]
    fn qp_range(&self) -> Option<String> {
        opt_getter!(self.qp_range -> String)
    }
    #[getter]
    fn const_qp(&self) -> Option<String> {
        opt_getter!(self.const_qp -> String)
    }
    #[getter]
    fn init_qp(&self) -> Option<String> {
        opt_getter!(self.init_qp -> String)
    }
    #[getter]
    fn max_bitrate(&self) -> Option<u32> {
        opt_getter!(self.max_bitrate -> u32)
    }
    #[getter]
    fn vbv_buf_size(&self) -> Option<u32> {
        opt_getter!(self.vbv_buf_size -> u32)
    }
    #[getter]
    fn vbv_init(&self) -> Option<u32> {
        opt_getter!(self.vbv_init -> u32)
    }
    #[getter]
    fn cq(&self) -> Option<u32> {
        opt_getter!(self.cq -> u32)
    }
    #[getter]
    fn aq(&self) -> Option<u32> {
        opt_getter!(self.aq -> u32)
    }
    #[getter]
    fn temporal_aq(&self) -> Option<bool> {
        opt_getter!(self.temporal_aq -> bool)
    }
    #[getter]
    fn extended_colorformat(&self) -> Option<bool> {
        opt_getter!(self.extended_colorformat -> bool)
    }

    #[staticmethod]
    fn from_pairs(pairs: std::collections::HashMap<String, String>) -> PyResult<Self> {
        let inner = properties::HevcDgpuProps::from_pairs(&pairs).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "HevcDgpuProps(bitrate={:?}, profile={:?})",
            self.inner.bitrate,
            self.inner.profile.map(|p| p.name())
        )
    }
}

// ---------------------------------------------------------------------------
// PyH264JetsonProps
// ---------------------------------------------------------------------------

/// H.264 encoder properties for Jetson (``nvv4l2h264enc``).
#[pyclass(name = "H264JetsonProps", module = "deepstream_encoders._native")]
#[derive(Debug, Clone)]
pub struct PyH264JetsonProps {
    inner: properties::H264JetsonProps,
}

#[pymethods]
impl PyH264JetsonProps {
    #[new]
    #[pyo3(signature = (
        bitrate = None,
        control_rate = None,
        profile = None,
        iframeinterval = None,
        idrinterval = None,
        preset_level = None,
        peak_bitrate = None,
        vbv_size = None,
        qp_range = None,
        quant_i_frames = None,
        quant_p_frames = None,
        ratecontrol_enable = None,
        maxperf_enable = None,
        two_pass_cbr = None,
        num_ref_frames = None,
        insert_sps_pps = None,
        insert_aud = None,
        insert_vui = None,
        disable_cabac = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bitrate: Option<u32>,
        control_rate: Option<&Bound<'_, PyAny>>,
        profile: Option<&Bound<'_, PyAny>>,
        iframeinterval: Option<u32>,
        idrinterval: Option<u32>,
        preset_level: Option<&Bound<'_, PyAny>>,
        peak_bitrate: Option<u32>,
        vbv_size: Option<u32>,
        qp_range: Option<String>,
        quant_i_frames: Option<u32>,
        quant_p_frames: Option<u32>,
        ratecontrol_enable: Option<bool>,
        maxperf_enable: Option<bool>,
        two_pass_cbr: Option<bool>,
        num_ref_frames: Option<u32>,
        insert_sps_pps: Option<bool>,
        insert_aud: Option<bool>,
        insert_vui: Option<bool>,
        disable_cabac: Option<bool>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: properties::H264JetsonProps {
                bitrate,
                control_rate: control_rate.map(extract_rate_control).transpose()?,
                profile: profile.map(extract_h264_profile).transpose()?,
                iframeinterval,
                idrinterval,
                preset_level: preset_level.map(extract_jetson_preset).transpose()?,
                peak_bitrate,
                vbv_size,
                qp_range,
                quant_i_frames,
                quant_p_frames,
                ratecontrol_enable,
                maxperf_enable,
                two_pass_cbr,
                num_ref_frames,
                insert_sps_pps,
                insert_aud,
                insert_vui,
                disable_cabac,
            },
        })
    }

    #[getter]
    fn bitrate(&self) -> Option<u32> {
        opt_getter!(self.bitrate -> u32)
    }
    #[getter]
    fn control_rate(&self) -> Option<PyRateControl> {
        self.inner.control_rate.map(Into::into)
    }
    #[getter]
    fn profile(&self) -> Option<PyH264Profile> {
        self.inner.profile.map(Into::into)
    }
    #[getter]
    fn iframeinterval(&self) -> Option<u32> {
        opt_getter!(self.iframeinterval -> u32)
    }
    #[getter]
    fn idrinterval(&self) -> Option<u32> {
        opt_getter!(self.idrinterval -> u32)
    }
    #[getter]
    fn preset_level(&self) -> Option<PyJetsonPresetLevel> {
        self.inner.preset_level.map(Into::into)
    }
    #[getter]
    fn peak_bitrate(&self) -> Option<u32> {
        opt_getter!(self.peak_bitrate -> u32)
    }
    #[getter]
    fn vbv_size(&self) -> Option<u32> {
        opt_getter!(self.vbv_size -> u32)
    }
    #[getter]
    fn qp_range(&self) -> Option<String> {
        opt_getter!(self.qp_range -> String)
    }
    #[getter]
    fn quant_i_frames(&self) -> Option<u32> {
        opt_getter!(self.quant_i_frames -> u32)
    }
    #[getter]
    fn quant_p_frames(&self) -> Option<u32> {
        opt_getter!(self.quant_p_frames -> u32)
    }
    #[getter]
    fn ratecontrol_enable(&self) -> Option<bool> {
        opt_getter!(self.ratecontrol_enable -> bool)
    }
    #[getter]
    fn maxperf_enable(&self) -> Option<bool> {
        opt_getter!(self.maxperf_enable -> bool)
    }
    #[getter]
    fn two_pass_cbr(&self) -> Option<bool> {
        opt_getter!(self.two_pass_cbr -> bool)
    }
    #[getter]
    fn num_ref_frames(&self) -> Option<u32> {
        opt_getter!(self.num_ref_frames -> u32)
    }
    #[getter]
    fn insert_sps_pps(&self) -> Option<bool> {
        opt_getter!(self.insert_sps_pps -> bool)
    }
    #[getter]
    fn insert_aud(&self) -> Option<bool> {
        opt_getter!(self.insert_aud -> bool)
    }
    #[getter]
    fn insert_vui(&self) -> Option<bool> {
        opt_getter!(self.insert_vui -> bool)
    }
    #[getter]
    fn disable_cabac(&self) -> Option<bool> {
        opt_getter!(self.disable_cabac -> bool)
    }

    #[staticmethod]
    fn from_pairs(pairs: std::collections::HashMap<String, String>) -> PyResult<Self> {
        let inner = properties::H264JetsonProps::from_pairs(&pairs).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "H264JetsonProps(bitrate={:?}, profile={:?})",
            self.inner.bitrate,
            self.inner.profile.map(|p| p.name())
        )
    }
}

// ---------------------------------------------------------------------------
// PyHevcJetsonProps
// ---------------------------------------------------------------------------

/// HEVC encoder properties for Jetson (``nvv4l2h265enc``).
#[pyclass(name = "HevcJetsonProps", module = "deepstream_encoders._native")]
#[derive(Debug, Clone)]
pub struct PyHevcJetsonProps {
    inner: properties::HevcJetsonProps,
}

#[pymethods]
impl PyHevcJetsonProps {
    #[new]
    #[pyo3(signature = (
        bitrate = None,
        control_rate = None,
        profile = None,
        iframeinterval = None,
        idrinterval = None,
        preset_level = None,
        peak_bitrate = None,
        vbv_size = None,
        qp_range = None,
        quant_i_frames = None,
        quant_p_frames = None,
        ratecontrol_enable = None,
        maxperf_enable = None,
        two_pass_cbr = None,
        num_ref_frames = None,
        enable_lossless = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bitrate: Option<u32>,
        control_rate: Option<&Bound<'_, PyAny>>,
        profile: Option<&Bound<'_, PyAny>>,
        iframeinterval: Option<u32>,
        idrinterval: Option<u32>,
        preset_level: Option<&Bound<'_, PyAny>>,
        peak_bitrate: Option<u32>,
        vbv_size: Option<u32>,
        qp_range: Option<String>,
        quant_i_frames: Option<u32>,
        quant_p_frames: Option<u32>,
        ratecontrol_enable: Option<bool>,
        maxperf_enable: Option<bool>,
        two_pass_cbr: Option<bool>,
        num_ref_frames: Option<u32>,
        enable_lossless: Option<bool>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: properties::HevcJetsonProps {
                bitrate,
                control_rate: control_rate.map(extract_rate_control).transpose()?,
                profile: profile.map(extract_hevc_profile).transpose()?,
                iframeinterval,
                idrinterval,
                preset_level: preset_level.map(extract_jetson_preset).transpose()?,
                peak_bitrate,
                vbv_size,
                qp_range,
                quant_i_frames,
                quant_p_frames,
                ratecontrol_enable,
                maxperf_enable,
                two_pass_cbr,
                num_ref_frames,
                enable_lossless,
            },
        })
    }

    #[getter]
    fn bitrate(&self) -> Option<u32> {
        opt_getter!(self.bitrate -> u32)
    }
    #[getter]
    fn control_rate(&self) -> Option<PyRateControl> {
        self.inner.control_rate.map(Into::into)
    }
    #[getter]
    fn profile(&self) -> Option<PyHevcProfile> {
        self.inner.profile.map(Into::into)
    }
    #[getter]
    fn iframeinterval(&self) -> Option<u32> {
        opt_getter!(self.iframeinterval -> u32)
    }
    #[getter]
    fn idrinterval(&self) -> Option<u32> {
        opt_getter!(self.idrinterval -> u32)
    }
    #[getter]
    fn preset_level(&self) -> Option<PyJetsonPresetLevel> {
        self.inner.preset_level.map(Into::into)
    }
    #[getter]
    fn peak_bitrate(&self) -> Option<u32> {
        opt_getter!(self.peak_bitrate -> u32)
    }
    #[getter]
    fn vbv_size(&self) -> Option<u32> {
        opt_getter!(self.vbv_size -> u32)
    }
    #[getter]
    fn qp_range(&self) -> Option<String> {
        opt_getter!(self.qp_range -> String)
    }
    #[getter]
    fn quant_i_frames(&self) -> Option<u32> {
        opt_getter!(self.quant_i_frames -> u32)
    }
    #[getter]
    fn quant_p_frames(&self) -> Option<u32> {
        opt_getter!(self.quant_p_frames -> u32)
    }
    #[getter]
    fn ratecontrol_enable(&self) -> Option<bool> {
        opt_getter!(self.ratecontrol_enable -> bool)
    }
    #[getter]
    fn maxperf_enable(&self) -> Option<bool> {
        opt_getter!(self.maxperf_enable -> bool)
    }
    #[getter]
    fn two_pass_cbr(&self) -> Option<bool> {
        opt_getter!(self.two_pass_cbr -> bool)
    }
    #[getter]
    fn num_ref_frames(&self) -> Option<u32> {
        opt_getter!(self.num_ref_frames -> u32)
    }
    #[getter]
    fn enable_lossless(&self) -> Option<bool> {
        opt_getter!(self.enable_lossless -> bool)
    }

    #[staticmethod]
    fn from_pairs(pairs: std::collections::HashMap<String, String>) -> PyResult<Self> {
        let inner = properties::HevcJetsonProps::from_pairs(&pairs).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "HevcJetsonProps(bitrate={:?}, profile={:?})",
            self.inner.bitrate,
            self.inner.profile.map(|p| p.name())
        )
    }
}

// ---------------------------------------------------------------------------
// PyJpegProps
// ---------------------------------------------------------------------------

/// JPEG encoder properties (``nvjpegenc``).
#[pyclass(name = "JpegProps", module = "deepstream_encoders._native")]
#[derive(Debug, Clone)]
pub struct PyJpegProps {
    inner: properties::JpegProps,
}

#[pymethods]
impl PyJpegProps {
    #[new]
    #[pyo3(signature = (quality = None))]
    fn new(quality: Option<u32>) -> Self {
        Self {
            inner: properties::JpegProps { quality },
        }
    }

    #[getter]
    fn quality(&self) -> Option<u32> {
        self.inner.quality
    }

    #[staticmethod]
    fn from_pairs(pairs: std::collections::HashMap<String, String>) -> PyResult<Self> {
        let inner = properties::JpegProps::from_pairs(&pairs).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("JpegProps(quality={:?})", self.inner.quality)
    }
}

// ---------------------------------------------------------------------------
// PyAv1DgpuProps
// ---------------------------------------------------------------------------

/// AV1 encoder properties for dGPU (``nvv4l2av1enc``).
#[pyclass(name = "Av1DgpuProps", module = "deepstream_encoders._native")]
#[derive(Debug, Clone)]
pub struct PyAv1DgpuProps {
    inner: properties::Av1DgpuProps,
}

#[pymethods]
impl PyAv1DgpuProps {
    #[new]
    #[pyo3(signature = (
        bitrate = None,
        control_rate = None,
        iframeinterval = None,
        idrinterval = None,
        preset = None,
        tuning_info = None,
        qp_range = None,
        max_bitrate = None,
        vbv_buf_size = None,
        vbv_init = None,
        cq = None,
        aq = None,
        temporal_aq = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bitrate: Option<u32>,
        control_rate: Option<&Bound<'_, PyAny>>,
        iframeinterval: Option<u32>,
        idrinterval: Option<u32>,
        preset: Option<&Bound<'_, PyAny>>,
        tuning_info: Option<&Bound<'_, PyAny>>,
        qp_range: Option<String>,
        max_bitrate: Option<u32>,
        vbv_buf_size: Option<u32>,
        vbv_init: Option<u32>,
        cq: Option<u32>,
        aq: Option<u32>,
        temporal_aq: Option<bool>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: properties::Av1DgpuProps {
                bitrate,
                control_rate: control_rate.map(extract_rate_control).transpose()?,
                iframeinterval,
                idrinterval,
                preset: preset.map(extract_dgpu_preset).transpose()?,
                tuning_info: tuning_info.map(extract_tuning_preset).transpose()?,
                qp_range,
                max_bitrate,
                vbv_buf_size,
                vbv_init,
                cq,
                aq,
                temporal_aq,
            },
        })
    }

    #[getter]
    fn bitrate(&self) -> Option<u32> {
        opt_getter!(self.bitrate -> u32)
    }
    #[getter]
    fn control_rate(&self) -> Option<PyRateControl> {
        self.inner.control_rate.map(Into::into)
    }
    #[getter]
    fn iframeinterval(&self) -> Option<u32> {
        opt_getter!(self.iframeinterval -> u32)
    }
    #[getter]
    fn idrinterval(&self) -> Option<u32> {
        opt_getter!(self.idrinterval -> u32)
    }
    #[getter]
    fn preset(&self) -> Option<PyDgpuPreset> {
        self.inner.preset.map(Into::into)
    }
    #[getter]
    fn tuning_info(&self) -> Option<PyTuningPreset> {
        self.inner.tuning_info.map(Into::into)
    }
    #[getter]
    fn qp_range(&self) -> Option<String> {
        opt_getter!(self.qp_range -> String)
    }
    #[getter]
    fn max_bitrate(&self) -> Option<u32> {
        opt_getter!(self.max_bitrate -> u32)
    }
    #[getter]
    fn vbv_buf_size(&self) -> Option<u32> {
        opt_getter!(self.vbv_buf_size -> u32)
    }
    #[getter]
    fn vbv_init(&self) -> Option<u32> {
        opt_getter!(self.vbv_init -> u32)
    }
    #[getter]
    fn cq(&self) -> Option<u32> {
        opt_getter!(self.cq -> u32)
    }
    #[getter]
    fn aq(&self) -> Option<u32> {
        opt_getter!(self.aq -> u32)
    }
    #[getter]
    fn temporal_aq(&self) -> Option<bool> {
        opt_getter!(self.temporal_aq -> bool)
    }

    #[staticmethod]
    fn from_pairs(pairs: std::collections::HashMap<String, String>) -> PyResult<Self> {
        let inner = properties::Av1DgpuProps::from_pairs(&pairs).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("Av1DgpuProps(bitrate={:?})", self.inner.bitrate)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Helper: convert PyProps into Rust EncoderProperties
// ═══════════════════════════════════════════════════════════════════════

fn extract_encoder_properties(ob: &Bound<'_, PyAny>) -> PyResult<EncoderProperties> {
    if let Ok(p) = ob.extract::<PyH264DgpuProps>() {
        return Ok(EncoderProperties::H264Dgpu(p.inner));
    }
    if let Ok(p) = ob.extract::<PyHevcDgpuProps>() {
        return Ok(EncoderProperties::HevcDgpu(p.inner));
    }
    if let Ok(p) = ob.extract::<PyH264JetsonProps>() {
        return Ok(EncoderProperties::H264Jetson(p.inner));
    }
    if let Ok(p) = ob.extract::<PyHevcJetsonProps>() {
        return Ok(EncoderProperties::HevcJetson(p.inner));
    }
    if let Ok(p) = ob.extract::<PyJpegProps>() {
        return Ok(EncoderProperties::Jpeg(p.inner));
    }
    if let Ok(p) = ob.extract::<PyAv1DgpuProps>() {
        return Ok(EncoderProperties::Av1Dgpu(p.inner));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected one of: H264DgpuProps, HevcDgpuProps, H264JetsonProps, HevcJetsonProps, JpegProps, Av1DgpuProps",
    ))
}

// ═══════════════════════════════════════════════════════════════════════
// PyEncoderConfig
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for creating an :class:`NvEncoder`.
///
/// Args:
///     codec (Codec | str): Video codec.
///     width (int): Frame width in pixels.
///     height (int): Frame height in pixels.
///     format (VideoFormat | str | None): Video format (default ``NV12``).
///     fps_num (int): Framerate numerator (default 30).
///     fps_den (int): Framerate denominator (default 1).
///     gpu_id (int): GPU device ID (default 0).
///     mem_type (MemType | int | None): Memory type (default ``DEFAULT``).
///     properties: Typed encoder properties — one of
///         :class:`H264DgpuProps`, :class:`HevcDgpuProps`,
///         :class:`H264JetsonProps`, :class:`HevcJetsonProps`,
///         :class:`JpegProps`, :class:`Av1DgpuProps`.
///
/// Example::
///
///     props = HevcDgpuProps(bitrate=8_000_000, profile=HevcProfile.MAIN)
///     config = EncoderConfig(Codec.HEVC, 1920, 1080, properties=props)
#[pyclass(name = "EncoderConfig", module = "deepstream_encoders._native")]
#[derive(Debug, Clone)]
pub struct PyEncoderConfig {
    inner: EncoderConfig,
}

#[pymethods]
impl PyEncoderConfig {
    #[new]
    #[pyo3(signature = (
        codec,
        width,
        height,
        format = None,
        fps_num = 30,
        fps_den = 1,
        gpu_id = 0,
        mem_type = None,
        properties = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        codec: &Bound<'_, PyAny>,
        width: u32,
        height: u32,
        format: Option<&Bound<'_, PyAny>>,
        fps_num: i32,
        fps_den: i32,
        gpu_id: u32,
        mem_type: Option<&Bound<'_, PyAny>>,
        properties: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let codec = extract_codec(codec)?;
        let format = match format {
            Some(f) => extract_video_format(f)?,
            None => VideoFormat::NV12,
        };
        let mem_type = match mem_type {
            Some(m) => extract_mem_type(m)?,
            None => NvBufSurfaceMemType::Default,
        };
        let encoder_params = match properties {
            Some(p) => Some(extract_encoder_properties(p)?),
            None => None,
        };

        let config = EncoderConfig::new(codec, width, height)
            .format(format)
            .fps(fps_num, fps_den)
            .gpu_id(gpu_id)
            .mem_type(mem_type);

        let config = match encoder_params {
            Some(ep) => config.properties(ep),
            None => config,
        };

        Ok(Self { inner: config })
    }

    #[getter]
    fn codec(&self) -> PyCodec {
        self.inner.codec.into()
    }
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }
    #[getter]
    fn format(&self) -> PyVideoFormat {
        self.inner.format.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "EncoderConfig(codec={:?}, {}x{}, format={}, fps={}/{})",
            self.inner.codec,
            self.inner.width,
            self.inner.height,
            self.inner.format,
            self.inner.fps_num,
            self.inner.fps_den,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PyEncodedFrame / PyNvEncoder — unchanged from before
// ═══════════════════════════════════════════════════════════════════════

/// A single encoded frame returned by :meth:`NvEncoder.pull_encoded`.
#[pyclass(name = "EncodedFrame", module = "deepstream_encoders._native")]
pub struct PyEncodedFrame {
    inner: EncodedFrame,
}

#[pymethods]
impl PyEncodedFrame {
    #[getter]
    fn frame_id(&self) -> u128 {
        self.inner.frame_id
    }
    #[getter]
    fn pts_ns(&self) -> u64 {
        self.inner.pts_ns
    }
    #[getter]
    fn dts_ns(&self) -> Option<u64> {
        self.inner.dts_ns
    }
    #[getter]
    fn duration_ns(&self) -> Option<u64> {
        self.inner.duration_ns
    }
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.data)
    }
    #[getter]
    fn codec(&self) -> PyCodec {
        self.inner.codec.into()
    }
    #[getter]
    fn size(&self) -> usize {
        self.inner.data.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "EncodedFrame(frame_id={}, pts_ns={}, size={} bytes, codec={:?})",
            self.inner.frame_id,
            self.inner.pts_ns,
            self.inner.data.len(),
            self.inner.codec,
        )
    }
}

/// GPU-accelerated video encoder.
#[pyclass(name = "NvEncoder", module = "deepstream_encoders._native", unsendable)]
pub struct PyNvEncoder {
    inner: NvEncoder,
}

#[pymethods]
impl PyNvEncoder {
    #[new]
    fn new(py: Python<'_>, config: &PyEncoderConfig) -> PyResult<Self> {
        let _ = gst::init();
        // Release the GIL while building the pipeline — NvEncoder::new
        // allocates GPU buffer pools, creates CUDA streams, and sets the
        // GStreamer pipeline to Playing, all of which may block.
        let config_inner = config.inner.clone();
        py.detach(|| {
            let inner = NvEncoder::new(&config_inner).map_err(to_py_err)?;
            Ok(Self { inner })
        })
    }

    #[getter]
    fn codec(&self) -> PyCodec {
        self.inner.codec().into()
    }

    fn nvmm_caps_str(&self) -> String {
        self.inner.generator().nvmm_caps().to_string()
    }

    #[pyo3(signature = (id=None))]
    fn acquire_surface(&self, py: Python<'_>, id: Option<i64>) -> PyResult<usize> {
        py.detach(|| {
            let buffer = self
                .inner
                .generator()
                .acquire_surface(id)
                .map_err(|e| to_py_err(EncoderError::from(e)))?;
            let raw = unsafe {
                use glib::translate::IntoGlibPtr;
                buffer.into_glib_ptr() as usize
            };
            Ok(raw)
        })
    }

    #[pyo3(signature = (buffer_ptr, frame_id, pts_ns, duration_ns=None))]
    fn submit_frame(
        &mut self,
        py: Python<'_>,
        buffer_ptr: usize,
        frame_id: u128,
        pts_ns: u64,
        duration_ns: Option<u64>,
    ) -> PyResult<()> {
        if buffer_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "buffer_ptr is null",
            ));
        }
        let buffer =
            unsafe { glib::translate::from_glib_full(buffer_ptr as *mut gst::ffi::GstBuffer) };
        py.detach(|| {
            self.inner
                .submit_frame(buffer, frame_id, pts_ns, duration_ns)
                .map_err(to_py_err)
        })
    }

    fn pull_encoded(&mut self, py: Python<'_>) -> PyResult<Option<PyEncodedFrame>> {
        py.detach(|| {
            self.inner
                .pull_encoded()
                .map(|opt| opt.map(|f| PyEncodedFrame { inner: f }))
                .map_err(to_py_err)
        })
    }

    #[pyo3(signature = (timeout_ms=100))]
    fn pull_encoded_timeout(
        &mut self,
        py: Python<'_>,
        timeout_ms: u64,
    ) -> PyResult<Option<PyEncodedFrame>> {
        py.detach(|| {
            self.inner
                .pull_encoded_timeout(timeout_ms)
                .map(|opt| opt.map(|f| PyEncodedFrame { inner: f }))
                .map_err(to_py_err)
        })
    }

    #[pyo3(signature = (drain_timeout_ms=None))]
    fn finish(
        &mut self,
        py: Python<'_>,
        drain_timeout_ms: Option<u64>,
    ) -> PyResult<Vec<PyEncodedFrame>> {
        py.detach(|| {
            self.inner
                .finish(drain_timeout_ms)
                .map(|frames| {
                    frames
                        .into_iter()
                        .map(|f| PyEncodedFrame { inner: f })
                        .collect()
                })
                .map_err(to_py_err)
        })
    }

    fn check_error(&self) -> PyResult<()> {
        self.inner.check_error().map_err(to_py_err)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Error conversion + module init
// ═══════════════════════════════════════════════════════════════════════

fn to_py_err(e: EncoderError) -> PyErr {
    match &e {
        EncoderError::InvalidProperty { .. } => {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        }
        EncoderError::PtsReordered { .. }
        | EncoderError::OutputPtsReordered { .. }
        | EncoderError::OutputDtsExceedsPts { .. } => {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        }
        EncoderError::AlreadyFinalized => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
        _ => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    }
}

/// Register the Python module.
#[pymodule]
#[pyo3(name = "_native")]
pub fn deepstream_encoders(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core enums
    m.add_class::<PyCodec>()?;
    m.add_class::<PyVideoFormat>()?;
    m.add_class::<PyMemType>()?;
    // Property enums
    m.add_class::<PyPlatform>()?;
    m.add_class::<PyRateControl>()?;
    m.add_class::<PyH264Profile>()?;
    m.add_class::<PyHevcProfile>()?;
    m.add_class::<PyDgpuPreset>()?;
    m.add_class::<PyTuningPreset>()?;
    m.add_class::<PyJetsonPresetLevel>()?;
    // Property structs
    m.add_class::<PyH264DgpuProps>()?;
    m.add_class::<PyHevcDgpuProps>()?;
    m.add_class::<PyH264JetsonProps>()?;
    m.add_class::<PyHevcJetsonProps>()?;
    m.add_class::<PyJpegProps>()?;
    m.add_class::<PyAv1DgpuProps>()?;
    // Main API
    m.add_class::<PyEncoderConfig>()?;
    m.add_class::<PyEncodedFrame>()?;
    m.add_class::<PyNvEncoder>()?;
    Ok(())
}
