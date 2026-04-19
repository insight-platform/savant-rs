use crate::deepstream::{PyMemType, PyVideoFormat};
use crate::gstreamer::PyCodec;
use deepstream_encoders::prelude::*;
use pyo3::prelude::*;

// ─── PyPlatform ────────────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "Platform",
    module = "savant_rs.deepstream",
    eq,
    eq_int
)]
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
        match Platform::from_name(name) {
            Some(p) => Ok(p.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown platform: '{name}'"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Platform.{}",
            match self {
                Self::Dgpu => "DGPU",
                Self::Jetson => "JETSON",
            }
        )
    }
}

impl From<PyPlatform> for Platform {
    fn from(p: PyPlatform) -> Self {
        match p {
            PyPlatform::Dgpu => Platform::Dgpu,
            PyPlatform::Jetson => Platform::Jetson,
        }
    }
}

impl From<Platform> for PyPlatform {
    fn from(p: Platform) -> Self {
        match p {
            Platform::Dgpu => PyPlatform::Dgpu,
            Platform::Jetson => PyPlatform::Jetson,
        }
    }
}

// ─── PyRateControl ─────────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "RateControl",
    module = "savant_rs.deepstream",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyRateControl {
    #[pyo3(name = "VARIABLE_BITRATE")]
    VariableBitrate = 0,
    #[pyo3(name = "CONSTANT_BITRATE")]
    ConstantBitrate = 1,
    #[pyo3(name = "CONSTANT_QP")]
    ConstantQP = 2,
}

#[pymethods]
impl PyRateControl {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match RateControl::from_name(name) {
            Some(rc) => Ok(rc.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown rate control: '{name}'"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RateControl.{}",
            match self {
                Self::VariableBitrate => "VARIABLE_BITRATE",
                Self::ConstantBitrate => "CONSTANT_BITRATE",
                Self::ConstantQP => "CONSTANT_QP",
            }
        )
    }
}

impl From<PyRateControl> for RateControl {
    fn from(rc: PyRateControl) -> Self {
        match rc {
            PyRateControl::VariableBitrate => RateControl::VariableBitrate,
            PyRateControl::ConstantBitrate => RateControl::ConstantBitrate,
            PyRateControl::ConstantQP => RateControl::ConstantQP,
        }
    }
}

impl From<RateControl> for PyRateControl {
    fn from(rc: RateControl) -> Self {
        match rc {
            RateControl::VariableBitrate => PyRateControl::VariableBitrate,
            RateControl::ConstantBitrate => PyRateControl::ConstantBitrate,
            RateControl::ConstantQP => PyRateControl::ConstantQP,
        }
    }
}

// ─── PyH264Profile ─────────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "H264Profile",
    module = "savant_rs.deepstream",
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
        match H264Profile::from_name(name) {
            Some(p) => Ok(p.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown H264 profile: '{name}'"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "H264Profile.{}",
            match self {
                Self::Baseline => "BASELINE",
                Self::Main => "MAIN",
                Self::High => "HIGH",
                Self::High444 => "HIGH444",
            }
        )
    }
}

impl From<PyH264Profile> for H264Profile {
    fn from(p: PyH264Profile) -> Self {
        match p {
            PyH264Profile::Baseline => H264Profile::Baseline,
            PyH264Profile::Main => H264Profile::Main,
            PyH264Profile::High => H264Profile::High,
            PyH264Profile::High444 => H264Profile::High444,
        }
    }
}

impl From<H264Profile> for PyH264Profile {
    fn from(p: H264Profile) -> Self {
        match p {
            H264Profile::Baseline => PyH264Profile::Baseline,
            H264Profile::Main => PyH264Profile::Main,
            H264Profile::High => PyH264Profile::High,
            H264Profile::High444 => PyH264Profile::High444,
        }
    }
}

// ─── PyHevcProfile ─────────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "HevcProfile",
    module = "savant_rs.deepstream",
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
        match HevcProfile::from_name(name) {
            Some(p) => Ok(p.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown HEVC profile: '{name}'"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HevcProfile.{}",
            match self {
                Self::Main => "MAIN",
                Self::Main10 => "MAIN10",
                Self::Frext => "FREXT",
            }
        )
    }
}

impl From<PyHevcProfile> for HevcProfile {
    fn from(p: PyHevcProfile) -> Self {
        match p {
            PyHevcProfile::Main => HevcProfile::Main,
            PyHevcProfile::Main10 => HevcProfile::Main10,
            PyHevcProfile::Frext => HevcProfile::Frext,
        }
    }
}

impl From<HevcProfile> for PyHevcProfile {
    fn from(p: HevcProfile) -> Self {
        match p {
            HevcProfile::Main => PyHevcProfile::Main,
            HevcProfile::Main10 => PyHevcProfile::Main10,
            HevcProfile::Frext => PyHevcProfile::Frext,
        }
    }
}

// ─── PyDgpuPreset ──────────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "DgpuPreset",
    module = "savant_rs.deepstream",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyDgpuPreset {
    #[pyo3(name = "P1")]
    P1 = 1,
    #[pyo3(name = "P2")]
    P2 = 2,
    #[pyo3(name = "P3")]
    P3 = 3,
    #[pyo3(name = "P4")]
    P4 = 4,
    #[pyo3(name = "P5")]
    P5 = 5,
    #[pyo3(name = "P6")]
    P6 = 6,
    #[pyo3(name = "P7")]
    P7 = 7,
}

#[pymethods]
impl PyDgpuPreset {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match DgpuPreset::from_name(name) {
            Some(p) => Ok(p.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown dGPU preset: '{name}'"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DgpuPreset.{}",
            match self {
                Self::P1 => "P1",
                Self::P2 => "P2",
                Self::P3 => "P3",
                Self::P4 => "P4",
                Self::P5 => "P5",
                Self::P6 => "P6",
                Self::P7 => "P7",
            }
        )
    }
}

impl From<PyDgpuPreset> for DgpuPreset {
    fn from(p: PyDgpuPreset) -> Self {
        match p {
            PyDgpuPreset::P1 => DgpuPreset::P1,
            PyDgpuPreset::P2 => DgpuPreset::P2,
            PyDgpuPreset::P3 => DgpuPreset::P3,
            PyDgpuPreset::P4 => DgpuPreset::P4,
            PyDgpuPreset::P5 => DgpuPreset::P5,
            PyDgpuPreset::P6 => DgpuPreset::P6,
            PyDgpuPreset::P7 => DgpuPreset::P7,
        }
    }
}

impl From<DgpuPreset> for PyDgpuPreset {
    fn from(p: DgpuPreset) -> Self {
        match p {
            DgpuPreset::P1 => PyDgpuPreset::P1,
            DgpuPreset::P2 => PyDgpuPreset::P2,
            DgpuPreset::P3 => PyDgpuPreset::P3,
            DgpuPreset::P4 => PyDgpuPreset::P4,
            DgpuPreset::P5 => PyDgpuPreset::P5,
            DgpuPreset::P6 => PyDgpuPreset::P6,
            DgpuPreset::P7 => PyDgpuPreset::P7,
        }
    }
}

// ─── PyTuningPreset ────────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "TuningPreset",
    module = "savant_rs.deepstream",
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
        match TuningPreset::from_name(name) {
            Some(t) => Ok(t.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown tuning preset: '{name}'"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TuningPreset.{}",
            match self {
                Self::HighQuality => "HIGH_QUALITY",
                Self::LowLatency => "LOW_LATENCY",
                Self::UltraLowLatency => "ULTRA_LOW_LATENCY",
                Self::Lossless => "LOSSLESS",
            }
        )
    }
}

impl From<PyTuningPreset> for TuningPreset {
    fn from(t: PyTuningPreset) -> Self {
        match t {
            PyTuningPreset::HighQuality => TuningPreset::HighQuality,
            PyTuningPreset::LowLatency => TuningPreset::LowLatency,
            PyTuningPreset::UltraLowLatency => TuningPreset::UltraLowLatency,
            PyTuningPreset::Lossless => TuningPreset::Lossless,
        }
    }
}

impl From<TuningPreset> for PyTuningPreset {
    fn from(t: TuningPreset) -> Self {
        match t {
            TuningPreset::HighQuality => PyTuningPreset::HighQuality,
            TuningPreset::LowLatency => PyTuningPreset::LowLatency,
            TuningPreset::UltraLowLatency => PyTuningPreset::UltraLowLatency,
            TuningPreset::Lossless => PyTuningPreset::Lossless,
        }
    }
}

// ─── PyJetsonPresetLevel ───────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "JetsonPresetLevel",
    module = "savant_rs.deepstream",
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
        match JetsonPresetLevel::from_name(name) {
            Some(l) => Ok(l.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown Jetson preset level: '{name}'"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "JetsonPresetLevel.{}",
            match self {
                Self::Disabled => "DISABLED",
                Self::UltraFast => "ULTRA_FAST",
                Self::Fast => "FAST",
                Self::Medium => "MEDIUM",
                Self::Slow => "SLOW",
            }
        )
    }
}

impl From<PyJetsonPresetLevel> for JetsonPresetLevel {
    fn from(l: PyJetsonPresetLevel) -> Self {
        match l {
            PyJetsonPresetLevel::Disabled => JetsonPresetLevel::Disabled,
            PyJetsonPresetLevel::UltraFast => JetsonPresetLevel::UltraFast,
            PyJetsonPresetLevel::Fast => JetsonPresetLevel::Fast,
            PyJetsonPresetLevel::Medium => JetsonPresetLevel::Medium,
            PyJetsonPresetLevel::Slow => JetsonPresetLevel::Slow,
        }
    }
}

impl From<JetsonPresetLevel> for PyJetsonPresetLevel {
    fn from(l: JetsonPresetLevel) -> Self {
        match l {
            JetsonPresetLevel::Disabled => PyJetsonPresetLevel::Disabled,
            JetsonPresetLevel::UltraFast => PyJetsonPresetLevel::UltraFast,
            JetsonPresetLevel::Fast => PyJetsonPresetLevel::Fast,
            JetsonPresetLevel::Medium => PyJetsonPresetLevel::Medium,
            JetsonPresetLevel::Slow => PyJetsonPresetLevel::Slow,
        }
    }
}

// ─── PyH264DgpuProps ───────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "H264DgpuProps",
    module = "savant_rs.deepstream"
)]
#[derive(Debug, Clone)]
pub struct PyH264DgpuProps {
    #[pyo3(get, set)]
    pub bitrate: Option<u32>,
    #[pyo3(get, set)]
    pub control_rate: Option<PyRateControl>,
    #[pyo3(get, set)]
    pub profile: Option<PyH264Profile>,
    #[pyo3(get, set)]
    pub iframeinterval: Option<u32>,
    #[pyo3(get, set)]
    pub idrinterval: Option<u32>,
    #[pyo3(get, set)]
    pub preset: Option<PyDgpuPreset>,
    #[pyo3(get, set)]
    pub tuning_info: Option<PyTuningPreset>,
    #[pyo3(get, set)]
    pub qp_range: Option<String>,
    #[pyo3(get, set)]
    pub const_qp: Option<String>,
    #[pyo3(get, set)]
    pub init_qp: Option<String>,
    #[pyo3(get, set)]
    pub max_bitrate: Option<u32>,
    #[pyo3(get, set)]
    pub vbv_buf_size: Option<u32>,
    #[pyo3(get, set)]
    pub vbv_init: Option<u32>,
    #[pyo3(get, set)]
    pub cq: Option<u32>,
    #[pyo3(get, set)]
    pub aq: Option<u32>,
    #[pyo3(get, set)]
    pub temporal_aq: Option<bool>,
    #[pyo3(get, set)]
    pub extended_colorformat: Option<bool>,
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
        control_rate: Option<PyRateControl>,
        profile: Option<PyH264Profile>,
        iframeinterval: Option<u32>,
        idrinterval: Option<u32>,
        preset: Option<PyDgpuPreset>,
        tuning_info: Option<PyTuningPreset>,
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
    ) -> Self {
        Self {
            bitrate,
            control_rate,
            profile,
            iframeinterval,
            idrinterval,
            preset,
            tuning_info,
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
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl PyH264DgpuProps {
    pub(crate) fn to_rust(&self) -> H264DgpuProps {
        H264DgpuProps {
            bitrate: self.bitrate,
            control_rate: self.control_rate.map(Into::into),
            profile: self.profile.map(Into::into),
            iframeinterval: self.iframeinterval,
            idrinterval: self.idrinterval,
            preset: self.preset.map(Into::into),
            tuning_info: self.tuning_info.map(Into::into),
            qp_range: self.qp_range.clone(),
            const_qp: self.const_qp.clone(),
            init_qp: self.init_qp.clone(),
            max_bitrate: self.max_bitrate,
            vbv_buf_size: self.vbv_buf_size,
            vbv_init: self.vbv_init,
            cq: self.cq,
            aq: self.aq,
            temporal_aq: self.temporal_aq,
            extended_colorformat: self.extended_colorformat,
        }
    }
}

// ─── PyHevcDgpuProps ───────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "HevcDgpuProps",
    module = "savant_rs.deepstream"
)]
#[derive(Debug, Clone)]
pub struct PyHevcDgpuProps {
    #[pyo3(get, set)]
    pub bitrate: Option<u32>,
    #[pyo3(get, set)]
    pub control_rate: Option<PyRateControl>,
    #[pyo3(get, set)]
    pub profile: Option<PyHevcProfile>,
    #[pyo3(get, set)]
    pub iframeinterval: Option<u32>,
    #[pyo3(get, set)]
    pub idrinterval: Option<u32>,
    #[pyo3(get, set)]
    pub preset: Option<PyDgpuPreset>,
    #[pyo3(get, set)]
    pub tuning_info: Option<PyTuningPreset>,
    #[pyo3(get, set)]
    pub qp_range: Option<String>,
    #[pyo3(get, set)]
    pub const_qp: Option<String>,
    #[pyo3(get, set)]
    pub init_qp: Option<String>,
    #[pyo3(get, set)]
    pub max_bitrate: Option<u32>,
    #[pyo3(get, set)]
    pub vbv_buf_size: Option<u32>,
    #[pyo3(get, set)]
    pub vbv_init: Option<u32>,
    #[pyo3(get, set)]
    pub cq: Option<u32>,
    #[pyo3(get, set)]
    pub aq: Option<u32>,
    #[pyo3(get, set)]
    pub temporal_aq: Option<bool>,
    #[pyo3(get, set)]
    pub extended_colorformat: Option<bool>,
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
        control_rate: Option<PyRateControl>,
        profile: Option<PyHevcProfile>,
        iframeinterval: Option<u32>,
        idrinterval: Option<u32>,
        preset: Option<PyDgpuPreset>,
        tuning_info: Option<PyTuningPreset>,
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
    ) -> Self {
        Self {
            bitrate,
            control_rate,
            profile,
            iframeinterval,
            idrinterval,
            preset,
            tuning_info,
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
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl PyHevcDgpuProps {
    pub(crate) fn to_rust(&self) -> HevcDgpuProps {
        HevcDgpuProps {
            bitrate: self.bitrate,
            control_rate: self.control_rate.map(Into::into),
            profile: self.profile.map(Into::into),
            iframeinterval: self.iframeinterval,
            idrinterval: self.idrinterval,
            preset: self.preset.map(Into::into),
            tuning_info: self.tuning_info.map(Into::into),
            qp_range: self.qp_range.clone(),
            const_qp: self.const_qp.clone(),
            init_qp: self.init_qp.clone(),
            max_bitrate: self.max_bitrate,
            vbv_buf_size: self.vbv_buf_size,
            vbv_init: self.vbv_init,
            cq: self.cq,
            aq: self.aq,
            temporal_aq: self.temporal_aq,
            extended_colorformat: self.extended_colorformat,
        }
    }
}

// ─── PyH264JetsonProps ─────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "H264JetsonProps",
    module = "savant_rs.deepstream"
)]
#[derive(Debug, Clone)]
pub struct PyH264JetsonProps {
    #[pyo3(get, set)]
    pub bitrate: Option<u32>,
    #[pyo3(get, set)]
    pub control_rate: Option<PyRateControl>,
    #[pyo3(get, set)]
    pub profile: Option<PyH264Profile>,
    #[pyo3(get, set)]
    pub iframeinterval: Option<u32>,
    #[pyo3(get, set)]
    pub idrinterval: Option<u32>,
    #[pyo3(get, set)]
    pub preset_level: Option<PyJetsonPresetLevel>,
    #[pyo3(get, set)]
    pub peak_bitrate: Option<u32>,
    #[pyo3(get, set)]
    pub vbv_size: Option<u32>,
    #[pyo3(get, set)]
    pub qp_range: Option<String>,
    #[pyo3(get, set)]
    pub quant_i_frames: Option<u32>,
    #[pyo3(get, set)]
    pub quant_p_frames: Option<u32>,
    #[pyo3(get, set)]
    pub ratecontrol_enable: Option<bool>,
    #[pyo3(get, set)]
    pub maxperf_enable: Option<bool>,
    #[pyo3(get, set)]
    pub two_pass_cbr: Option<bool>,
    #[pyo3(get, set)]
    pub num_ref_frames: Option<u32>,
    #[pyo3(get, set)]
    pub insert_sps_pps: Option<bool>,
    #[pyo3(get, set)]
    pub insert_aud: Option<bool>,
    #[pyo3(get, set)]
    pub insert_vui: Option<bool>,
    #[pyo3(get, set)]
    pub disable_cabac: Option<bool>,
}

#[pymethods]
impl PyH264JetsonProps {
    #[new]
    #[pyo3(signature = (
        bitrate = None, control_rate = None, profile = None,
        iframeinterval = None, idrinterval = None, preset_level = None,
        peak_bitrate = None, vbv_size = None, qp_range = None,
        quant_i_frames = None, quant_p_frames = None,
        ratecontrol_enable = None, maxperf_enable = None,
        two_pass_cbr = None, num_ref_frames = None,
        insert_sps_pps = None, insert_aud = None, insert_vui = None,
        disable_cabac = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bitrate: Option<u32>,
        control_rate: Option<PyRateControl>,
        profile: Option<PyH264Profile>,
        iframeinterval: Option<u32>,
        idrinterval: Option<u32>,
        preset_level: Option<PyJetsonPresetLevel>,
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
    ) -> Self {
        Self {
            bitrate,
            control_rate,
            profile,
            iframeinterval,
            idrinterval,
            preset_level,
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
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl PyH264JetsonProps {
    pub(crate) fn to_rust(&self) -> H264JetsonProps {
        H264JetsonProps {
            bitrate: self.bitrate,
            control_rate: self.control_rate.map(Into::into),
            profile: self.profile.map(Into::into),
            iframeinterval: self.iframeinterval,
            idrinterval: self.idrinterval,
            preset_level: self.preset_level.map(Into::into),
            peak_bitrate: self.peak_bitrate,
            vbv_size: self.vbv_size,
            qp_range: self.qp_range.clone(),
            quant_i_frames: self.quant_i_frames,
            quant_p_frames: self.quant_p_frames,
            ratecontrol_enable: self.ratecontrol_enable,
            maxperf_enable: self.maxperf_enable,
            two_pass_cbr: self.two_pass_cbr,
            num_ref_frames: self.num_ref_frames,
            insert_sps_pps: self.insert_sps_pps,
            insert_aud: self.insert_aud,
            insert_vui: self.insert_vui,
            disable_cabac: self.disable_cabac,
        }
    }
}

// ─── PyHevcJetsonProps ─────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "HevcJetsonProps",
    module = "savant_rs.deepstream"
)]
#[derive(Debug, Clone)]
pub struct PyHevcJetsonProps {
    #[pyo3(get, set)]
    pub bitrate: Option<u32>,
    #[pyo3(get, set)]
    pub control_rate: Option<PyRateControl>,
    #[pyo3(get, set)]
    pub profile: Option<PyHevcProfile>,
    #[pyo3(get, set)]
    pub iframeinterval: Option<u32>,
    #[pyo3(get, set)]
    pub idrinterval: Option<u32>,
    #[pyo3(get, set)]
    pub preset_level: Option<PyJetsonPresetLevel>,
    #[pyo3(get, set)]
    pub peak_bitrate: Option<u32>,
    #[pyo3(get, set)]
    pub vbv_size: Option<u32>,
    #[pyo3(get, set)]
    pub qp_range: Option<String>,
    #[pyo3(get, set)]
    pub quant_i_frames: Option<u32>,
    #[pyo3(get, set)]
    pub quant_p_frames: Option<u32>,
    #[pyo3(get, set)]
    pub ratecontrol_enable: Option<bool>,
    #[pyo3(get, set)]
    pub maxperf_enable: Option<bool>,
    #[pyo3(get, set)]
    pub two_pass_cbr: Option<bool>,
    #[pyo3(get, set)]
    pub num_ref_frames: Option<u32>,
    #[pyo3(get, set)]
    pub enable_lossless: Option<bool>,
}

#[pymethods]
impl PyHevcJetsonProps {
    #[new]
    #[pyo3(signature = (
        bitrate = None, control_rate = None, profile = None,
        iframeinterval = None, idrinterval = None, preset_level = None,
        peak_bitrate = None, vbv_size = None, qp_range = None,
        quant_i_frames = None, quant_p_frames = None,
        ratecontrol_enable = None, maxperf_enable = None,
        two_pass_cbr = None, num_ref_frames = None,
        enable_lossless = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bitrate: Option<u32>,
        control_rate: Option<PyRateControl>,
        profile: Option<PyHevcProfile>,
        iframeinterval: Option<u32>,
        idrinterval: Option<u32>,
        preset_level: Option<PyJetsonPresetLevel>,
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
    ) -> Self {
        Self {
            bitrate,
            control_rate,
            profile,
            iframeinterval,
            idrinterval,
            preset_level,
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
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl PyHevcJetsonProps {
    pub(crate) fn to_rust(&self) -> HevcJetsonProps {
        HevcJetsonProps {
            bitrate: self.bitrate,
            control_rate: self.control_rate.map(Into::into),
            profile: self.profile.map(Into::into),
            iframeinterval: self.iframeinterval,
            idrinterval: self.idrinterval,
            preset_level: self.preset_level.map(Into::into),
            peak_bitrate: self.peak_bitrate,
            vbv_size: self.vbv_size,
            qp_range: self.qp_range.clone(),
            quant_i_frames: self.quant_i_frames,
            quant_p_frames: self.quant_p_frames,
            ratecontrol_enable: self.ratecontrol_enable,
            maxperf_enable: self.maxperf_enable,
            two_pass_cbr: self.two_pass_cbr,
            num_ref_frames: self.num_ref_frames,
            enable_lossless: self.enable_lossless,
        }
    }
}

// ─── PyJpegProps ───────────────────────────────────────────────────────

#[pyclass(from_py_object, name = "JpegProps", module = "savant_rs.deepstream")]
#[derive(Debug, Clone)]
pub struct PyJpegProps {
    #[pyo3(get, set)]
    pub quality: Option<u32>,
}

#[pymethods]
impl PyJpegProps {
    #[new]
    #[pyo3(signature = (quality = None))]
    fn new(quality: Option<u32>) -> Self {
        Self { quality }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl PyJpegProps {
    pub(crate) fn to_rust(&self) -> JpegProps {
        JpegProps {
            quality: self.quality,
        }
    }
}

// ─── PyPngProps ───────────────────────────────────────────────────────

/// PNG encoder properties (`pngenc`, CPU-based, gst-plugins-good).
#[pyclass(from_py_object, name = "PngProps", module = "savant_rs.deepstream")]
#[derive(Debug, Clone)]
pub struct PyPngProps {
    /// PNG compression level (0–9, default: 6). Higher = smaller file, slower.
    #[pyo3(get, set)]
    pub compression_level: Option<u32>,
}

#[pymethods]
impl PyPngProps {
    #[new]
    #[pyo3(signature = (compression_level = None))]
    fn new(compression_level: Option<u32>) -> Self {
        Self { compression_level }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl PyPngProps {
    pub(crate) fn to_rust(&self) -> PngProps {
        PngProps {
            compression_level: self.compression_level,
        }
    }
}

// ─── PyAv1DgpuProps ────────────────────────────────────────────────────

#[pyclass(from_py_object, name = "Av1DgpuProps", module = "savant_rs.deepstream")]
#[derive(Debug, Clone)]
pub struct PyAv1DgpuProps {
    #[pyo3(get, set)]
    pub bitrate: Option<u32>,
    #[pyo3(get, set)]
    pub control_rate: Option<PyRateControl>,
    #[pyo3(get, set)]
    pub iframeinterval: Option<u32>,
    #[pyo3(get, set)]
    pub idrinterval: Option<u32>,
    #[pyo3(get, set)]
    pub preset: Option<PyDgpuPreset>,
    #[pyo3(get, set)]
    pub tuning_info: Option<PyTuningPreset>,
    #[pyo3(get, set)]
    pub qp_range: Option<String>,
    #[pyo3(get, set)]
    pub max_bitrate: Option<u32>,
    #[pyo3(get, set)]
    pub vbv_buf_size: Option<u32>,
    #[pyo3(get, set)]
    pub vbv_init: Option<u32>,
    #[pyo3(get, set)]
    pub cq: Option<u32>,
    #[pyo3(get, set)]
    pub aq: Option<u32>,
    #[pyo3(get, set)]
    pub temporal_aq: Option<bool>,
}

#[pymethods]
impl PyAv1DgpuProps {
    #[new]
    #[pyo3(signature = (
        bitrate = None, control_rate = None, iframeinterval = None,
        idrinterval = None, preset = None, tuning_info = None,
        qp_range = None, max_bitrate = None, vbv_buf_size = None,
        vbv_init = None, cq = None, aq = None, temporal_aq = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bitrate: Option<u32>,
        control_rate: Option<PyRateControl>,
        iframeinterval: Option<u32>,
        idrinterval: Option<u32>,
        preset: Option<PyDgpuPreset>,
        tuning_info: Option<PyTuningPreset>,
        qp_range: Option<String>,
        max_bitrate: Option<u32>,
        vbv_buf_size: Option<u32>,
        vbv_init: Option<u32>,
        cq: Option<u32>,
        aq: Option<u32>,
        temporal_aq: Option<bool>,
    ) -> Self {
        Self {
            bitrate,
            control_rate,
            iframeinterval,
            idrinterval,
            preset,
            tuning_info,
            qp_range,
            max_bitrate,
            vbv_buf_size,
            vbv_init,
            cq,
            aq,
            temporal_aq,
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl PyAv1DgpuProps {
    pub(crate) fn to_rust(&self) -> Av1DgpuProps {
        Av1DgpuProps {
            bitrate: self.bitrate,
            control_rate: self.control_rate.map(Into::into),
            iframeinterval: self.iframeinterval,
            idrinterval: self.idrinterval,
            preset: self.preset.map(Into::into),
            tuning_info: self.tuning_info.map(Into::into),
            qp_range: self.qp_range.clone(),
            max_bitrate: self.max_bitrate,
            vbv_buf_size: self.vbv_buf_size,
            vbv_init: self.vbv_init,
            cq: self.cq,
            aq: self.aq,
            temporal_aq: self.temporal_aq,
        }
    }
}

// ─── PyAv1JetsonProps ──────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "Av1JetsonProps",
    module = "savant_rs.deepstream"
)]
#[derive(Debug, Clone)]
pub struct PyAv1JetsonProps {
    #[pyo3(get, set)]
    pub bitrate: Option<u32>,
    #[pyo3(get, set)]
    pub control_rate: Option<PyRateControl>,
    #[pyo3(get, set)]
    pub iframeinterval: Option<u32>,
    #[pyo3(get, set)]
    pub idrinterval: Option<u32>,
    #[pyo3(get, set)]
    pub preset_level: Option<PyJetsonPresetLevel>,
    #[pyo3(get, set)]
    pub peak_bitrate: Option<u32>,
    #[pyo3(get, set)]
    pub vbv_size: Option<u32>,
    #[pyo3(get, set)]
    pub qp_range: Option<String>,
    #[pyo3(get, set)]
    pub quant_i_frames: Option<u32>,
    #[pyo3(get, set)]
    pub quant_p_frames: Option<u32>,
    #[pyo3(get, set)]
    pub quant_b_frames: Option<u32>,
    #[pyo3(get, set)]
    pub ratecontrol_enable: Option<bool>,
    #[pyo3(get, set)]
    pub maxperf_enable: Option<bool>,
    #[pyo3(get, set)]
    pub two_pass_cbr: Option<bool>,
    #[pyo3(get, set)]
    pub num_ref_frames: Option<u32>,
    #[pyo3(get, set)]
    pub insert_seq_hdr: Option<bool>,
    #[pyo3(get, set)]
    pub tiles: Option<String>,
}

#[pymethods]
impl PyAv1JetsonProps {
    #[new]
    #[pyo3(signature = (
        bitrate = None, control_rate = None, iframeinterval = None,
        idrinterval = None, preset_level = None, peak_bitrate = None,
        vbv_size = None, qp_range = None, quant_i_frames = None,
        quant_p_frames = None, quant_b_frames = None,
        ratecontrol_enable = None, maxperf_enable = None,
        two_pass_cbr = None, num_ref_frames = None,
        insert_seq_hdr = None, tiles = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bitrate: Option<u32>,
        control_rate: Option<PyRateControl>,
        iframeinterval: Option<u32>,
        idrinterval: Option<u32>,
        preset_level: Option<PyJetsonPresetLevel>,
        peak_bitrate: Option<u32>,
        vbv_size: Option<u32>,
        qp_range: Option<String>,
        quant_i_frames: Option<u32>,
        quant_p_frames: Option<u32>,
        quant_b_frames: Option<u32>,
        ratecontrol_enable: Option<bool>,
        maxperf_enable: Option<bool>,
        two_pass_cbr: Option<bool>,
        num_ref_frames: Option<u32>,
        insert_seq_hdr: Option<bool>,
        tiles: Option<String>,
    ) -> Self {
        Self {
            bitrate,
            control_rate,
            iframeinterval,
            idrinterval,
            preset_level,
            peak_bitrate,
            vbv_size,
            qp_range,
            quant_i_frames,
            quant_p_frames,
            quant_b_frames,
            ratecontrol_enable,
            maxperf_enable,
            two_pass_cbr,
            num_ref_frames,
            insert_seq_hdr,
            tiles,
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl PyAv1JetsonProps {
    pub(crate) fn to_rust(&self) -> Av1JetsonProps {
        Av1JetsonProps {
            bitrate: self.bitrate,
            control_rate: self.control_rate.map(Into::into),
            iframeinterval: self.iframeinterval,
            idrinterval: self.idrinterval,
            preset_level: self.preset_level.map(Into::into),
            peak_bitrate: self.peak_bitrate,
            vbv_size: self.vbv_size,
            qp_range: self.qp_range.clone(),
            quant_i_frames: self.quant_i_frames,
            quant_p_frames: self.quant_p_frames,
            quant_b_frames: self.quant_b_frames,
            ratecontrol_enable: self.ratecontrol_enable,
            maxperf_enable: self.maxperf_enable,
            two_pass_cbr: self.two_pass_cbr,
            num_ref_frames: self.num_ref_frames,
            insert_seq_hdr: self.insert_seq_hdr,
            tiles: self.tiles.clone(),
        }
    }
}

// ─── PyEncoderProperties ───────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "EncoderProperties",
    module = "savant_rs.deepstream"
)]
#[derive(Debug, Clone)]
pub struct PyEncoderProperties {
    pub(crate) inner: EncoderProperties,
}

#[pymethods]
impl PyEncoderProperties {
    #[staticmethod]
    fn h264_dgpu(props: PyH264DgpuProps) -> Self {
        Self {
            inner: EncoderProperties::H264Dgpu(props.to_rust()),
        }
    }

    #[staticmethod]
    fn h264_jetson(props: PyH264JetsonProps) -> Self {
        Self {
            inner: EncoderProperties::H264Jetson(props.to_rust()),
        }
    }

    #[staticmethod]
    fn hevc_dgpu(props: PyHevcDgpuProps) -> Self {
        Self {
            inner: EncoderProperties::HevcDgpu(props.to_rust()),
        }
    }

    #[staticmethod]
    fn hevc_jetson(props: PyHevcJetsonProps) -> Self {
        Self {
            inner: EncoderProperties::HevcJetson(props.to_rust()),
        }
    }

    #[staticmethod]
    fn jpeg(props: PyJpegProps) -> Self {
        Self {
            inner: EncoderProperties::Jpeg(props.to_rust()),
        }
    }

    #[staticmethod]
    fn av1_dgpu(props: PyAv1DgpuProps) -> Self {
        Self {
            inner: EncoderProperties::Av1Dgpu(props.to_rust()),
        }
    }

    #[staticmethod]
    fn av1_jetson(props: PyAv1JetsonProps) -> Self {
        Self {
            inner: EncoderProperties::Av1Jetson(props.to_rust()),
        }
    }

    /// Create PNG encoder properties (CPU-based, gst-plugins-good).
    #[staticmethod]
    fn png(props: PyPngProps) -> Self {
        Self {
            inner: EncoderProperties::Png(props.to_rust()),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

// ─── PyEncoderConfig ───────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "EncoderConfig",
    module = "savant_rs.deepstream"
)]
#[derive(Debug, Clone)]
pub struct PyEncoderConfig {
    codec: PyCodec,
    width: u32,
    height: u32,
    format: PyVideoFormat,
    fps_num: i32,
    fps_den: i32,
    gpu_id: u32,
    mem_type: PyMemType,
    encoder_params: Option<PyEncoderProperties>,
}

#[pymethods]
impl PyEncoderConfig {
    #[new]
    fn new(codec: PyCodec, width: u32, height: u32) -> Self {
        Self {
            codec,
            width,
            height,
            format: PyVideoFormat::Nv12,
            fps_num: 30,
            fps_den: 1,
            gpu_id: 0,
            mem_type: PyMemType::Default,
            encoder_params: None,
        }
    }

    #[getter]
    fn get_format(&self) -> PyVideoFormat {
        self.format
    }

    #[setter]
    fn set_format(&mut self, f: PyVideoFormat) {
        self.format = f;
    }

    #[getter]
    fn get_fps_num(&self) -> i32 {
        self.fps_num
    }

    #[setter]
    fn set_fps_num(&mut self, v: i32) {
        self.fps_num = v;
    }

    #[getter]
    fn get_fps_den(&self) -> i32 {
        self.fps_den
    }

    #[setter]
    fn set_fps_den(&mut self, v: i32) {
        self.fps_den = v;
    }

    #[getter]
    fn get_gpu_id(&self) -> u32 {
        self.gpu_id
    }

    #[setter]
    fn set_gpu_id(&mut self, v: u32) {
        self.gpu_id = v;
    }

    #[getter]
    fn get_mem_type(&self) -> PyMemType {
        self.mem_type
    }

    #[setter]
    fn set_mem_type(&mut self, mt: PyMemType) {
        self.mem_type = mt;
    }

    #[getter]
    fn get_encoder_params(&self) -> Option<PyEncoderProperties> {
        self.encoder_params.clone()
    }

    #[setter]
    fn set_encoder_params(&mut self, props: Option<PyEncoderProperties>) {
        self.encoder_params = props;
    }

    /// Builder method: set the video format.
    #[pyo3(name = "format")]
    fn with_format(mut slf: PyRefMut<'_, Self>, f: PyVideoFormat) -> PyRefMut<'_, Self> {
        slf.format = f;
        slf
    }

    /// Builder method: set the framerate.
    fn fps(mut slf: PyRefMut<'_, Self>, num: i32, den: i32) -> PyRefMut<'_, Self> {
        slf.fps_num = num;
        slf.fps_den = den;
        slf
    }

    /// Builder method: set the GPU device ID.
    #[pyo3(name = "gpu_id")]
    fn with_gpu_id(mut slf: PyRefMut<'_, Self>, id: u32) -> PyRefMut<'_, Self> {
        slf.gpu_id = id;
        slf
    }

    /// Builder method: set the memory type.
    #[pyo3(name = "mem_type")]
    fn with_mem_type(mut slf: PyRefMut<'_, Self>, mt: PyMemType) -> PyRefMut<'_, Self> {
        slf.mem_type = mt;
        slf
    }

    /// Builder method: set encoder properties.
    fn properties(mut slf: PyRefMut<'_, Self>, props: PyEncoderProperties) -> PyRefMut<'_, Self> {
        slf.encoder_params = Some(props);
        slf
    }

    fn __repr__(&self) -> String {
        format!(
            "EncoderConfig(codec={:?}, {}x{}, format={:?}, fps={}/{}, gpu_id={}, mem_type={:?}, encoder_params={:?})",
            self.codec, self.width, self.height, self.format,
            self.fps_num, self.fps_den, self.gpu_id, self.mem_type,
            self.encoder_params.as_ref().map(|p| format!("{:?}", p.inner)).unwrap_or_else(|| "None".to_string()),
        )
    }
}

impl PyEncoderConfig {
    /// Materialise the Python-side builder state into a runtime
    /// [`NvEncoderConfig`] that wraps a codec-specific [`EncoderConfig`]
    /// enum variant with platform-appropriate properties.
    ///
    /// Codec dispatch rules:
    /// * `Codec::H264` → [`EncoderConfig::H264`] with
    ///   [`H264DgpuProps`] / [`H264JetsonProps`] when `encoder_params`
    ///   matches the host platform.
    /// * `Codec::Hevc` → [`EncoderConfig::Hevc`] with
    ///   [`HevcDgpuProps`] / [`HevcJetsonProps`].
    /// * `Codec::Av1` → [`EncoderConfig::Av1`] with
    ///   [`Av1DgpuProps`] / [`Av1JetsonProps`].
    /// * `Codec::Jpeg` → [`EncoderConfig::Jpeg`] with [`JpegProps`].
    /// * `Codec::Png` → [`EncoderConfig::Png`] with [`PngProps`].
    /// * Raw codecs (`RawRgba`/`RawRgb`/`RawNv12`) → [`EncoderConfig`]
    ///   raw variants with default [`RawProps`].
    ///
    /// `encoder_params` whose codec or target platform does not match the
    /// configured [`Codec`] and the current build target are rejected with a
    /// Python `ValueError` so that mistakes such as pairing
    /// [`HevcDgpuProps`] with an `H264` encoder (or Jetson props on a dGPU
    /// build) fail loudly instead of silently reverting to defaults.
    /// Callers are expected to use the helper factories on
    /// [`PyEncoderProperties`] that correspond to both the target codec and
    /// the target platform.
    pub(crate) fn to_rust(&self) -> PyResult<deepstream_encoders::NvEncoderConfig> {
        use deepstream_encoders::{EncoderConfig as E, NvEncoderConfig};
        let codec: Codec = self.codec.into();
        let format = self.format.into();
        let fps_num = self.fps_num;
        let fps_den = self.fps_den;
        let w = self.width;
        let h = self.height;

        let params = self.encoder_params.as_ref().map(|p| p.inner.clone());

        // Validate that, if the user supplied encoder_params, they match both
        // the configured codec and the platform this binary was built for.
        // Mirrors the old NvEncoder::new behaviour (EncoderError::InvalidProperty).
        if let Some(ref props) = params {
            let props_codec = props.codec();
            if props_codec != codec {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "EncoderConfig codec mismatch: config.codec = {}, \
                     but encoder_params is for codec {} \
                     (variant {}). Use EncoderProperties::{}_* helpers that match \
                     the configured codec.",
                    codec.name(),
                    props_codec.name(),
                    encoder_props_variant_name(props),
                    codec.name(),
                )));
            }
            if let Some(props_platform) = props.platform() {
                #[cfg(not(target_arch = "aarch64"))]
                let build_platform = Platform::Dgpu;
                #[cfg(target_arch = "aarch64")]
                let build_platform = Platform::Jetson;
                if props_platform != build_platform {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "EncoderConfig platform mismatch: this build targets {}, \
                         but encoder_params is for {} (variant {}). \
                         Use the EncoderProperties::*_{} helper that matches \
                         the build target.",
                        build_platform.name(),
                        props_platform.name(),
                        encoder_props_variant_name(props),
                        build_platform.name(),
                    )));
                }
            }
        }

        let encoder_cfg = match codec {
            Codec::H264 => {
                let mut cfg = H264EncoderConfig::new(w, h)
                    .format(format)
                    .fps(fps_num, fps_den);
                #[cfg(not(target_arch = "aarch64"))]
                if let Some(EncoderProperties::H264Dgpu(p)) = params {
                    cfg = cfg.props(p);
                }
                #[cfg(target_arch = "aarch64")]
                if let Some(EncoderProperties::H264Jetson(p)) = params {
                    cfg = cfg.props(p);
                }
                E::H264(cfg)
            }
            Codec::Hevc => {
                let mut cfg = HevcEncoderConfig::new(w, h)
                    .format(format)
                    .fps(fps_num, fps_den);
                #[cfg(not(target_arch = "aarch64"))]
                if let Some(EncoderProperties::HevcDgpu(p)) = params {
                    cfg = cfg.props(p);
                }
                #[cfg(target_arch = "aarch64")]
                if let Some(EncoderProperties::HevcJetson(p)) = params {
                    cfg = cfg.props(p);
                }
                E::Hevc(cfg)
            }
            Codec::Av1 => {
                let mut cfg = Av1EncoderConfig::new(w, h)
                    .format(format)
                    .fps(fps_num, fps_den);
                #[cfg(not(target_arch = "aarch64"))]
                if let Some(EncoderProperties::Av1Dgpu(p)) = params {
                    cfg = cfg.props(p);
                }
                #[cfg(target_arch = "aarch64")]
                if let Some(EncoderProperties::Av1Jetson(p)) = params {
                    cfg = cfg.props(p);
                }
                E::Av1(cfg)
            }
            Codec::Jpeg => {
                let mut cfg = JpegEncoderConfig::new(w, h)
                    .format(format)
                    .fps(fps_num, fps_den);
                if let Some(EncoderProperties::Jpeg(p)) = params {
                    cfg = cfg.props(p);
                }
                E::Jpeg(cfg)
            }
            Codec::Png => {
                let mut cfg = PngEncoderConfig::new(w, h)
                    .format(format)
                    .fps(fps_num, fps_den);
                if let Some(EncoderProperties::Png(p)) = params {
                    cfg = cfg.props(p);
                }
                E::Png(cfg)
            }
            Codec::RawRgba => {
                E::RawRgba(RawEncoderConfig::new(w, h, VideoFormat::RGBA).fps(fps_num, fps_den))
            }
            Codec::RawRgb => {
                // Raw RGB pseudoencoder uses RGBA-shaped surfaces internally; the
                // GPU→CPU download layer drops the alpha byte before packaging the
                // payload. See `savant_deepstream/encoders/src/pipeline.rs`
                // (`Codec::RawRgb => VideoFormat::RGBA`) for the surface layout.
                E::RawRgb(RawEncoderConfig::new(w, h, VideoFormat::RGBA).fps(fps_num, fps_den))
            }
            Codec::RawNv12 => {
                E::RawNv12(RawEncoderConfig::new(w, h, VideoFormat::NV12).fps(fps_num, fps_den))
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "EncoderConfig: unsupported codec {other:?} for NvEncoder \
                     (Vp8/Vp9 are decode-only)"
                )));
            }
        };

        Ok(NvEncoderConfig::new(self.gpu_id, encoder_cfg).mem_type(self.mem_type.into()))
    }
}

/// Human-readable name of an [`EncoderProperties`] variant, used for
/// diagnostic error messages when codec/platform validation fails.
fn encoder_props_variant_name(p: &EncoderProperties) -> &'static str {
    match p {
        EncoderProperties::H264Dgpu(_) => "H264Dgpu",
        EncoderProperties::H264Jetson(_) => "H264Jetson",
        EncoderProperties::HevcDgpu(_) => "HevcDgpu",
        EncoderProperties::HevcJetson(_) => "HevcJetson",
        EncoderProperties::Jpeg(_) => "Jpeg",
        EncoderProperties::Av1Dgpu(_) => "Av1Dgpu",
        EncoderProperties::Av1Jetson(_) => "Av1Jetson",
        EncoderProperties::Png(_) => "Png",
        EncoderProperties::RawRgba(_) => "RawRgba",
        EncoderProperties::RawRgb(_) => "RawRgb",
        EncoderProperties::RawNv12(_) => "RawNv12",
    }
}

// ─── Module registration ───────────────────────────────────────────────

pub fn register_encoder_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPlatform>()?;
    m.add_class::<PyRateControl>()?;
    m.add_class::<PyH264Profile>()?;
    m.add_class::<PyHevcProfile>()?;
    m.add_class::<PyDgpuPreset>()?;
    m.add_class::<PyTuningPreset>()?;
    m.add_class::<PyJetsonPresetLevel>()?;
    m.add_class::<PyH264DgpuProps>()?;
    m.add_class::<PyHevcDgpuProps>()?;
    m.add_class::<PyH264JetsonProps>()?;
    m.add_class::<PyHevcJetsonProps>()?;
    m.add_class::<PyJpegProps>()?;
    m.add_class::<PyPngProps>()?;
    m.add_class::<PyAv1DgpuProps>()?;
    m.add_class::<PyAv1JetsonProps>()?;
    m.add_class::<PyEncoderProperties>()?;
    m.add_class::<PyEncoderConfig>()?;
    Ok(())
}
