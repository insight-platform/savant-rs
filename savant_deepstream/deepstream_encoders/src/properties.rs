//! Typed encoder properties for NVIDIA hardware encoders (DeepStream 7.1).
//!
//! Properties are organized by codec and platform (dGPU vs Jetson) to provide
//! compile-time type safety.  Every field is `Option<T>`: `None` means "use the
//! encoder's built-in default."
//!
//! # Variants
//!
//! | Variant | Codec | Platform | GStreamer element |
//! |---|---|---|---|
//! | [`H264Dgpu`](H264DgpuProps) | H.264 | dGPU | `nvv4l2h264enc` |
//! | [`H264Jetson`](H264JetsonProps) | H.264 | Jetson | `nvv4l2h264enc` |
//! | [`HevcDgpu`](HevcDgpuProps) | HEVC | dGPU | `nvv4l2h265enc` |
//! | [`HevcJetson`](HevcJetsonProps) | HEVC | Jetson | `nvv4l2h265enc` |
//! | [`Jpeg`](JpegProps) | JPEG | Both | `nvjpegenc` |
//! | [`Av1Dgpu`](Av1DgpuProps) | AV1 | dGPU | `nvv4l2av1enc` |

use crate::error::EncoderError;
use savant_gstreamer::Codec;
use std::collections::HashMap;

// ─── Key normalization ─────────────────────────────────────────────────

/// Normalize a property key: lowercase, dashes to underscores.
fn normalize_key(key: &str) -> String {
    key.to_lowercase().replace('-', "_")
}

// ─── Parse helpers ─────────────────────────────────────────────────────

fn parse_u32(key: &str, value: &str) -> Result<u32, EncoderError> {
    value
        .parse::<u32>()
        .map_err(|_| EncoderError::InvalidProperty {
            name: key.to_string(),
            reason: format!("expected unsigned integer, got '{value}'"),
        })
}

fn parse_bool(key: &str, value: &str) -> Result<bool, EncoderError> {
    match value.to_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(EncoderError::InvalidProperty {
            name: key.to_string(),
            reason: format!("expected boolean (true/false/1/0), got '{value}'"),
        }),
    }
}

fn push_u32(pairs: &mut Vec<(&'static str, String)>, key: &'static str, val: &Option<u32>) {
    if let Some(v) = val {
        pairs.push((key, v.to_string()));
    }
}

fn push_bool(pairs: &mut Vec<(&'static str, String)>, key: &'static str, val: &Option<bool>) {
    if let Some(v) = val {
        pairs.push((key, if *v { "true" } else { "false" }.to_string()));
    }
}

fn push_str(pairs: &mut Vec<(&'static str, String)>, key: &'static str, val: &Option<String>) {
    if let Some(v) = val {
        pairs.push((key, v.clone()));
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Enums
// ═══════════════════════════════════════════════════════════════════════

// ─── Platform ──────────────────────────────────────────────────────────

/// Target hardware platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Platform {
    /// Discrete GPU (desktop/server with NVIDIA dGPU).
    Dgpu,
    /// NVIDIA Jetson embedded platform.
    Jetson,
}

impl Platform {
    /// Parse a platform from a string name.
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "dgpu" | "gpu" | "discrete" => Some(Platform::Dgpu),
            "jetson" | "tegra" => Some(Platform::Jetson),
            _ => None,
        }
    }

    /// Canonical name.
    pub fn name(&self) -> &'static str {
        match self {
            Platform::Dgpu => "dgpu",
            Platform::Jetson => "jetson",
        }
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─── RateControl ───────────────────────────────────────────────────────

/// Rate-control mode for video encoders.
///
/// Maps to the `control-rate` GStreamer property on both dGPU and Jetson.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RateControl {
    /// Variable bitrate (VBR).
    VariableBitrate = 0,
    /// Constant bitrate (CBR).
    ConstantBitrate = 1,
    /// Constant QP.
    ConstantQP = 2,
}

impl RateControl {
    /// GStreamer element property value.
    pub fn gst_value(&self) -> &'static str {
        match self {
            RateControl::VariableBitrate => "0",
            RateControl::ConstantBitrate => "1",
            RateControl::ConstantQP => "2",
        }
    }

    /// Parse from a name or numeric string.
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "variable_bitrate" | "vbr" | "0" => Some(Self::VariableBitrate),
            "constant_bitrate" | "cbr" | "1" => Some(Self::ConstantBitrate),
            "constant_qp" | "cqp" | "constantqp" | "2" => Some(Self::ConstantQP),
            _ => None,
        }
    }

    /// Canonical name.
    pub fn name(&self) -> &'static str {
        match self {
            RateControl::VariableBitrate => "variable_bitrate",
            RateControl::ConstantBitrate => "constant_bitrate",
            RateControl::ConstantQP => "constant_qp",
        }
    }
}

impl std::fmt::Display for RateControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─── H264Profile ───────────────────────────────────────────────────────

/// H.264 encoding profile.
///
/// Numeric values match the `profile` GStreamer property on `nvv4l2h264enc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum H264Profile {
    /// Baseline profile.
    Baseline = 0,
    /// Main profile.
    Main = 2,
    /// High profile.
    High = 4,
    /// High 4:4:4 Predictive profile (dGPU only).
    High444 = 7,
}

impl H264Profile {
    pub fn gst_value(&self) -> &'static str {
        match self {
            Self::Baseline => "0",
            Self::Main => "2",
            Self::High => "4",
            Self::High444 => "7",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "baseline" | "0" => Some(Self::Baseline),
            "main" | "2" => Some(Self::Main),
            "high" | "4" => Some(Self::High),
            "high444" | "high444predictive" | "7" => Some(Self::High444),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Main => "main",
            Self::High => "high",
            Self::High444 => "high444",
        }
    }
}

impl std::fmt::Display for H264Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─── HevcProfile ───────────────────────────────────────────────────────

/// HEVC (H.265) encoding profile.
///
/// Numeric values match the `profile` GStreamer property on `nvv4l2h265enc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HevcProfile {
    /// Main profile.
    Main = 0,
    /// Main 10-bit profile.
    Main10 = 1,
    /// Format Range Extensions (FREXT) profile.
    Frext = 3,
}

impl HevcProfile {
    pub fn gst_value(&self) -> &'static str {
        match self {
            Self::Main => "0",
            Self::Main10 => "1",
            Self::Frext => "3",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "main" | "0" => Some(Self::Main),
            "main10" | "1" => Some(Self::Main10),
            "frext" | "3" => Some(Self::Frext),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Main10 => "main10",
            Self::Frext => "frext",
        }
    }
}

impl std::fmt::Display for HevcProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─── DgpuPreset ────────────────────────────────────────────────────────

/// NVENC preset for dGPU (maps to `preset-id`, values 1–7).
///
/// Lower values are faster; higher values produce better quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DgpuPreset {
    P1 = 1,
    P2 = 2,
    P3 = 3,
    P4 = 4,
    P5 = 5,
    P6 = 6,
    P7 = 7,
}

impl DgpuPreset {
    pub fn gst_value(&self) -> &'static str {
        match self {
            Self::P1 => "1",
            Self::P2 => "2",
            Self::P3 => "3",
            Self::P4 => "4",
            Self::P5 => "5",
            Self::P6 => "6",
            Self::P7 => "7",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "p1" | "1" => Some(Self::P1),
            "p2" | "2" => Some(Self::P2),
            "p3" | "3" => Some(Self::P3),
            "p4" | "4" => Some(Self::P4),
            "p5" | "5" => Some(Self::P5),
            "p6" | "6" => Some(Self::P6),
            "p7" | "7" => Some(Self::P7),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::P1 => "P1",
            Self::P2 => "P2",
            Self::P3 => "P3",
            Self::P4 => "P4",
            Self::P5 => "P5",
            Self::P6 => "P6",
            Self::P7 => "P7",
        }
    }
}

impl std::fmt::Display for DgpuPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─── TuningPreset ──────────────────────────────────────────────────────

/// dGPU tuning-info preset (maps to `tuning-info-id`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TuningPreset {
    /// High quality.
    HighQuality = 1,
    /// Low latency (default on dGPU).
    LowLatency = 2,
    /// Ultra-low latency.
    UltraLowLatency = 3,
    /// Lossless.
    Lossless = 4,
}

impl TuningPreset {
    pub fn gst_value(&self) -> &'static str {
        match self {
            Self::HighQuality => "1",
            Self::LowLatency => "2",
            Self::UltraLowLatency => "3",
            Self::Lossless => "4",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_lowercase().replace('-', "_").as_str() {
            "high_quality" | "highquality" | "1" => Some(Self::HighQuality),
            "low_latency" | "lowlatency" | "2" => Some(Self::LowLatency),
            "ultra_low_latency" | "ultralowlatency" | "3" => Some(Self::UltraLowLatency),
            "lossless" | "4" => Some(Self::Lossless),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::HighQuality => "high_quality",
            Self::LowLatency => "low_latency",
            Self::UltraLowLatency => "ultra_low_latency",
            Self::Lossless => "lossless",
        }
    }
}

impl std::fmt::Display for TuningPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─── JetsonPresetLevel ─────────────────────────────────────────────────

/// Jetson HW encoder preset level (maps to `preset-level`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JetsonPresetLevel {
    /// Disable HW preset.
    Disabled = 0,
    /// Ultra-fast preset (default on Jetson).
    UltraFast = 1,
    /// Fast preset.
    Fast = 2,
    /// Medium preset.
    Medium = 3,
    /// Slow preset (highest quality).
    Slow = 4,
}

impl JetsonPresetLevel {
    pub fn gst_value(&self) -> &'static str {
        match self {
            Self::Disabled => "0",
            Self::UltraFast => "1",
            Self::Fast => "2",
            Self::Medium => "3",
            Self::Slow => "4",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_lowercase().replace('-', "_").as_str() {
            "disabled" | "disable" | "0" => Some(Self::Disabled),
            "ultrafast" | "ultra_fast" | "1" => Some(Self::UltraFast),
            "fast" | "2" => Some(Self::Fast),
            "medium" | "3" => Some(Self::Medium),
            "slow" | "4" => Some(Self::Slow),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::UltraFast => "ultra_fast",
            Self::Fast => "fast",
            Self::Medium => "medium",
            Self::Slow => "slow",
        }
    }
}

impl std::fmt::Display for JetsonPresetLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Per-codec / per-platform property structs
// ═══════════════════════════════════════════════════════════════════════

// ─── H.264 dGPU ────────────────────────────────────────────────────────

/// H.264 encoder properties for dGPU (`nvv4l2h264enc`).
#[derive(Debug, Clone, Default)]
pub struct H264DgpuProps {
    /// Bitrate in bits/sec (default: 4 000 000).
    pub bitrate: Option<u32>,
    /// Rate-control mode (default: constant bitrate).
    pub control_rate: Option<RateControl>,
    /// H.264 profile (default: Baseline).
    pub profile: Option<H264Profile>,
    /// I-frame interval (default: 30).
    pub iframeinterval: Option<u32>,
    /// IDR-frame interval (default: 256).
    pub idrinterval: Option<u32>,
    /// NVENC preset P1–P7 (default: P1).
    pub preset: Option<DgpuPreset>,
    /// Tuning-info preset (default: LowLatency).
    pub tuning_info: Option<TuningPreset>,
    /// QP range, format `"minQP:maxQP"`.
    pub qp_range: Option<String>,
    /// Constant QP values, format `"I:P:B"`.
    pub const_qp: Option<String>,
    /// Initial QP hint, format `"I:P:B"`.
    pub init_qp: Option<String>,
    /// Maximum bitrate for VBR mode (bits/sec).
    pub max_bitrate: Option<u32>,
    /// VBV buffer size in bits.
    pub vbv_buf_size: Option<u32>,
    /// VBV initial delay in bits.
    pub vbv_init: Option<u32>,
    /// Target constant-quality level for VBR (0–51).
    pub cq: Option<u32>,
    /// Spatial adaptive-quantization strength (0–15, 0 = auto).
    pub aq: Option<u32>,
    /// Enable temporal adaptive quantization.
    pub temporal_aq: Option<bool>,
    /// Extended colour format (YUV 0–255 range in VUI info).
    pub extended_colorformat: Option<bool>,
}

impl H264DgpuProps {
    /// Convert set properties to GStreamer key-value pairs.
    pub fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        let mut p = Vec::new();
        push_u32(&mut p, "bitrate", &self.bitrate);
        if let Some(v) = &self.control_rate {
            p.push(("control-rate", v.gst_value().to_string()));
        }
        if let Some(v) = &self.profile {
            p.push(("profile", v.gst_value().to_string()));
        }
        push_u32(&mut p, "iframeinterval", &self.iframeinterval);
        push_u32(&mut p, "idrinterval", &self.idrinterval);
        if let Some(v) = &self.preset {
            p.push(("preset-id", v.gst_value().to_string()));
        }
        if let Some(v) = &self.tuning_info {
            p.push(("tuning-info-id", v.gst_value().to_string()));
        }
        push_str(&mut p, "qp-range", &self.qp_range);
        push_str(&mut p, "constqp", &self.const_qp);
        push_str(&mut p, "initqp", &self.init_qp);
        push_u32(&mut p, "maxbitrate", &self.max_bitrate);
        push_u32(&mut p, "vbvbufsize", &self.vbv_buf_size);
        push_u32(&mut p, "vbvinit", &self.vbv_init);
        push_u32(&mut p, "cq", &self.cq);
        push_u32(&mut p, "aq", &self.aq);
        push_bool(&mut p, "temporalaq", &self.temporal_aq);
        push_bool(&mut p, "extended-colorformat", &self.extended_colorformat);
        p
    }

    /// Parse from GStreamer-style key-value pairs.
    pub fn from_pairs(pairs: &HashMap<String, String>) -> Result<Self, EncoderError> {
        let mut props = Self::default();
        for (key, value) in pairs {
            match normalize_key(key).as_str() {
                "bitrate" => props.bitrate = Some(parse_u32(key, value)?),
                "control_rate" => {
                    props.control_rate = Some(RateControl::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown rate control '{value}'"),
                        }
                    })?)
                }
                "profile" => {
                    props.profile = Some(H264Profile::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown H.264 profile '{value}'"),
                        }
                    })?)
                }
                "iframeinterval" => props.iframeinterval = Some(parse_u32(key, value)?),
                "idrinterval" => props.idrinterval = Some(parse_u32(key, value)?),
                "preset_id" | "preset" => {
                    props.preset = Some(DgpuPreset::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown dGPU preset '{value}'"),
                        }
                    })?)
                }
                "tuning_info_id" | "tuning_info" => {
                    props.tuning_info = Some(TuningPreset::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown tuning preset '{value}'"),
                        }
                    })?)
                }
                "qp_range" => props.qp_range = Some(value.clone()),
                "constqp" | "const_qp" => props.const_qp = Some(value.clone()),
                "initqp" | "init_qp" => props.init_qp = Some(value.clone()),
                "maxbitrate" | "max_bitrate" => props.max_bitrate = Some(parse_u32(key, value)?),
                "vbvbufsize" | "vbv_buf_size" => props.vbv_buf_size = Some(parse_u32(key, value)?),
                "vbvinit" | "vbv_init" => props.vbv_init = Some(parse_u32(key, value)?),
                "cq" => props.cq = Some(parse_u32(key, value)?),
                "aq" => props.aq = Some(parse_u32(key, value)?),
                "temporalaq" | "temporal_aq" => props.temporal_aq = Some(parse_bool(key, value)?),
                "extended_colorformat" => {
                    props.extended_colorformat = Some(parse_bool(key, value)?)
                }
                _ => {
                    return Err(EncoderError::InvalidProperty {
                        name: key.clone(),
                        reason: format!("unknown H.264 dGPU property '{key}'"),
                    })
                }
            }
        }
        Ok(props)
    }
}

// ─── HEVC dGPU ─────────────────────────────────────────────────────────

/// HEVC encoder properties for dGPU (`nvv4l2h265enc`).
#[derive(Debug, Clone, Default)]
pub struct HevcDgpuProps {
    /// Bitrate in bits/sec (default: 4 000 000).
    pub bitrate: Option<u32>,
    /// Rate-control mode (default: constant bitrate).
    pub control_rate: Option<RateControl>,
    /// HEVC profile (default: Main).
    pub profile: Option<HevcProfile>,
    /// I-frame interval (default: 30).
    pub iframeinterval: Option<u32>,
    /// IDR-frame interval (default: 256).
    pub idrinterval: Option<u32>,
    /// NVENC preset P1–P7 (default: P1).
    pub preset: Option<DgpuPreset>,
    /// Tuning-info preset (default: LowLatency).
    pub tuning_info: Option<TuningPreset>,
    /// QP range, format `"minQP:maxQP"`.
    pub qp_range: Option<String>,
    /// Constant QP values, format `"I:P:B"`.
    pub const_qp: Option<String>,
    /// Initial QP hint, format `"I:P:B"`.
    pub init_qp: Option<String>,
    /// Maximum bitrate for VBR mode (bits/sec).
    pub max_bitrate: Option<u32>,
    /// VBV buffer size in bits.
    pub vbv_buf_size: Option<u32>,
    /// VBV initial delay in bits.
    pub vbv_init: Option<u32>,
    /// Target constant-quality level for VBR (0–51).
    pub cq: Option<u32>,
    /// Spatial adaptive-quantization strength (0–15, 0 = auto).
    pub aq: Option<u32>,
    /// Enable temporal adaptive quantization.
    pub temporal_aq: Option<bool>,
    /// Extended colour format (YUV 0–255 range in VUI info).
    pub extended_colorformat: Option<bool>,
}

impl HevcDgpuProps {
    pub fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        let mut p = Vec::new();
        push_u32(&mut p, "bitrate", &self.bitrate);
        if let Some(v) = &self.control_rate {
            p.push(("control-rate", v.gst_value().to_string()));
        }
        if let Some(v) = &self.profile {
            p.push(("profile", v.gst_value().to_string()));
        }
        push_u32(&mut p, "iframeinterval", &self.iframeinterval);
        push_u32(&mut p, "idrinterval", &self.idrinterval);
        if let Some(v) = &self.preset {
            p.push(("preset-id", v.gst_value().to_string()));
        }
        if let Some(v) = &self.tuning_info {
            p.push(("tuning-info-id", v.gst_value().to_string()));
        }
        push_str(&mut p, "qp-range", &self.qp_range);
        push_str(&mut p, "constqp", &self.const_qp);
        push_str(&mut p, "initqp", &self.init_qp);
        push_u32(&mut p, "maxbitrate", &self.max_bitrate);
        push_u32(&mut p, "vbvbufsize", &self.vbv_buf_size);
        push_u32(&mut p, "vbvinit", &self.vbv_init);
        push_u32(&mut p, "cq", &self.cq);
        push_u32(&mut p, "aq", &self.aq);
        push_bool(&mut p, "temporalaq", &self.temporal_aq);
        push_bool(&mut p, "extended-colorformat", &self.extended_colorformat);
        p
    }

    pub fn from_pairs(pairs: &HashMap<String, String>) -> Result<Self, EncoderError> {
        let mut props = Self::default();
        for (key, value) in pairs {
            match normalize_key(key).as_str() {
                "bitrate" => props.bitrate = Some(parse_u32(key, value)?),
                "control_rate" => {
                    props.control_rate = Some(RateControl::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown rate control '{value}'"),
                        }
                    })?)
                }
                "profile" => {
                    props.profile = Some(HevcProfile::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown HEVC profile '{value}'"),
                        }
                    })?)
                }
                "iframeinterval" => props.iframeinterval = Some(parse_u32(key, value)?),
                "idrinterval" => props.idrinterval = Some(parse_u32(key, value)?),
                "preset_id" | "preset" => {
                    props.preset = Some(DgpuPreset::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown dGPU preset '{value}'"),
                        }
                    })?)
                }
                "tuning_info_id" | "tuning_info" => {
                    props.tuning_info = Some(TuningPreset::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown tuning preset '{value}'"),
                        }
                    })?)
                }
                "qp_range" => props.qp_range = Some(value.clone()),
                "constqp" | "const_qp" => props.const_qp = Some(value.clone()),
                "initqp" | "init_qp" => props.init_qp = Some(value.clone()),
                "maxbitrate" | "max_bitrate" => props.max_bitrate = Some(parse_u32(key, value)?),
                "vbvbufsize" | "vbv_buf_size" => props.vbv_buf_size = Some(parse_u32(key, value)?),
                "vbvinit" | "vbv_init" => props.vbv_init = Some(parse_u32(key, value)?),
                "cq" => props.cq = Some(parse_u32(key, value)?),
                "aq" => props.aq = Some(parse_u32(key, value)?),
                "temporalaq" | "temporal_aq" => props.temporal_aq = Some(parse_bool(key, value)?),
                "extended_colorformat" => {
                    props.extended_colorformat = Some(parse_bool(key, value)?)
                }
                _ => {
                    return Err(EncoderError::InvalidProperty {
                        name: key.clone(),
                        reason: format!("unknown HEVC dGPU property '{key}'"),
                    })
                }
            }
        }
        Ok(props)
    }
}

// ─── H.264 Jetson ──────────────────────────────────────────────────────

/// H.264 encoder properties for Jetson (`nvv4l2h264enc`).
#[derive(Debug, Clone, Default)]
pub struct H264JetsonProps {
    /// Bitrate in bits/sec (default: 4 000 000).
    pub bitrate: Option<u32>,
    /// Rate-control mode (default: constant bitrate).
    pub control_rate: Option<RateControl>,
    /// H.264 profile (default: Baseline).
    pub profile: Option<H264Profile>,
    /// I-frame interval (default: 30).
    pub iframeinterval: Option<u32>,
    /// IDR-frame interval (default: 256).
    pub idrinterval: Option<u32>,
    /// HW preset level (default: UltraFast).
    pub preset_level: Option<JetsonPresetLevel>,
    /// Peak bitrate for VBR (bits/sec).
    pub peak_bitrate: Option<u32>,
    /// VBV buffer size (bits).
    pub vbv_size: Option<u32>,
    /// QP range, format `"minQP:maxQP"`.
    pub qp_range: Option<String>,
    /// Quantization parameter for I-frames (with ratecontrol off).
    pub quant_i_frames: Option<u32>,
    /// Quantization parameter for P-frames (with ratecontrol off).
    pub quant_p_frames: Option<u32>,
    /// Enable rate control (default: true).
    pub ratecontrol_enable: Option<bool>,
    /// Enable maximum performance mode.
    pub maxperf_enable: Option<bool>,
    /// Enable two-pass CBR.
    pub two_pass_cbr: Option<bool>,
    /// Number of reference frames (0–8, default: 1).
    pub num_ref_frames: Option<u32>,
    /// Insert SPS/PPS at every IDR frame.
    pub insert_sps_pps: Option<bool>,
    /// Insert H.264 Access Unit Delimiter (AUD).
    pub insert_aud: Option<bool>,
    /// Insert VUI (Video Usability Information) in SPS.
    pub insert_vui: Option<bool>,
    /// Use CAVLC instead of CABAC entropy coding.
    pub disable_cabac: Option<bool>,
}

impl H264JetsonProps {
    pub fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        let mut p = Vec::new();
        push_u32(&mut p, "bitrate", &self.bitrate);
        if let Some(v) = &self.control_rate {
            p.push(("control-rate", v.gst_value().to_string()));
        }
        if let Some(v) = &self.profile {
            p.push(("profile", v.gst_value().to_string()));
        }
        push_u32(&mut p, "iframeinterval", &self.iframeinterval);
        push_u32(&mut p, "idrinterval", &self.idrinterval);
        if let Some(v) = &self.preset_level {
            p.push(("preset-level", v.gst_value().to_string()));
        }
        push_u32(&mut p, "peak-bitrate", &self.peak_bitrate);
        push_u32(&mut p, "vbv-size", &self.vbv_size);
        push_str(&mut p, "qp-range", &self.qp_range);
        push_u32(&mut p, "quant-i-frames", &self.quant_i_frames);
        push_u32(&mut p, "quant-p-frames", &self.quant_p_frames);
        push_bool(&mut p, "ratecontrol-enable", &self.ratecontrol_enable);
        push_bool(&mut p, "maxperf-enable", &self.maxperf_enable);
        push_bool(&mut p, "EnableTwopassCBR", &self.two_pass_cbr);
        push_u32(&mut p, "num-Ref-Frames", &self.num_ref_frames);
        push_bool(&mut p, "insert-sps-pps", &self.insert_sps_pps);
        push_bool(&mut p, "insert-aud", &self.insert_aud);
        push_bool(&mut p, "insert-vui", &self.insert_vui);
        push_bool(&mut p, "disable-cabac", &self.disable_cabac);
        p
    }

    pub fn from_pairs(pairs: &HashMap<String, String>) -> Result<Self, EncoderError> {
        let mut props = Self::default();
        for (key, value) in pairs {
            match normalize_key(key).as_str() {
                "bitrate" => props.bitrate = Some(parse_u32(key, value)?),
                "control_rate" => {
                    props.control_rate = Some(RateControl::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown rate control '{value}'"),
                        }
                    })?)
                }
                "profile" => {
                    props.profile = Some(H264Profile::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown H.264 profile '{value}'"),
                        }
                    })?)
                }
                "iframeinterval" => props.iframeinterval = Some(parse_u32(key, value)?),
                "idrinterval" => props.idrinterval = Some(parse_u32(key, value)?),
                "preset_level" => {
                    props.preset_level =
                        Some(JetsonPresetLevel::from_name(value).ok_or_else(|| {
                            EncoderError::InvalidProperty {
                                name: key.clone(),
                                reason: format!("unknown Jetson preset level '{value}'"),
                            }
                        })?)
                }
                "peak_bitrate" => props.peak_bitrate = Some(parse_u32(key, value)?),
                "vbv_size" => props.vbv_size = Some(parse_u32(key, value)?),
                "qp_range" => props.qp_range = Some(value.clone()),
                "quant_i_frames" => props.quant_i_frames = Some(parse_u32(key, value)?),
                "quant_p_frames" => props.quant_p_frames = Some(parse_u32(key, value)?),
                "ratecontrol_enable" => props.ratecontrol_enable = Some(parse_bool(key, value)?),
                "maxperf_enable" => props.maxperf_enable = Some(parse_bool(key, value)?),
                "enabletwopasscbr" | "two_pass_cbr" => {
                    props.two_pass_cbr = Some(parse_bool(key, value)?)
                }
                "num_ref_frames" => props.num_ref_frames = Some(parse_u32(key, value)?),
                "insert_sps_pps" => props.insert_sps_pps = Some(parse_bool(key, value)?),
                "insert_aud" => props.insert_aud = Some(parse_bool(key, value)?),
                "insert_vui" => props.insert_vui = Some(parse_bool(key, value)?),
                "disable_cabac" => props.disable_cabac = Some(parse_bool(key, value)?),
                _ => {
                    return Err(EncoderError::InvalidProperty {
                        name: key.clone(),
                        reason: format!("unknown H.264 Jetson property '{key}'"),
                    })
                }
            }
        }
        Ok(props)
    }
}

// ─── HEVC Jetson ───────────────────────────────────────────────────────

/// HEVC encoder properties for Jetson (`nvv4l2h265enc`).
#[derive(Debug, Clone, Default)]
pub struct HevcJetsonProps {
    /// Bitrate in bits/sec (default: 4 000 000).
    pub bitrate: Option<u32>,
    /// Rate-control mode (default: constant bitrate).
    pub control_rate: Option<RateControl>,
    /// HEVC profile (default: Main).
    pub profile: Option<HevcProfile>,
    /// I-frame interval (default: 30).
    pub iframeinterval: Option<u32>,
    /// IDR-frame interval (default: 256).
    pub idrinterval: Option<u32>,
    /// HW preset level (default: UltraFast).
    pub preset_level: Option<JetsonPresetLevel>,
    /// Peak bitrate for VBR (bits/sec).
    pub peak_bitrate: Option<u32>,
    /// VBV buffer size (bits).
    pub vbv_size: Option<u32>,
    /// QP range, format `"minQP:maxQP"`.
    pub qp_range: Option<String>,
    /// Quantization parameter for I-frames (with ratecontrol off).
    pub quant_i_frames: Option<u32>,
    /// Quantization parameter for P-frames (with ratecontrol off).
    pub quant_p_frames: Option<u32>,
    /// Enable rate control (default: true).
    pub ratecontrol_enable: Option<bool>,
    /// Enable maximum performance mode.
    pub maxperf_enable: Option<bool>,
    /// Enable two-pass CBR.
    pub two_pass_cbr: Option<bool>,
    /// Number of reference frames (0–8, default: 1).
    pub num_ref_frames: Option<u32>,
    /// Enable lossless encoding (requires YUV444 input).
    pub enable_lossless: Option<bool>,
}

impl HevcJetsonProps {
    pub fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        let mut p = Vec::new();
        push_u32(&mut p, "bitrate", &self.bitrate);
        if let Some(v) = &self.control_rate {
            p.push(("control-rate", v.gst_value().to_string()));
        }
        if let Some(v) = &self.profile {
            p.push(("profile", v.gst_value().to_string()));
        }
        push_u32(&mut p, "iframeinterval", &self.iframeinterval);
        push_u32(&mut p, "idrinterval", &self.idrinterval);
        if let Some(v) = &self.preset_level {
            p.push(("preset-level", v.gst_value().to_string()));
        }
        push_u32(&mut p, "peak-bitrate", &self.peak_bitrate);
        push_u32(&mut p, "vbv-size", &self.vbv_size);
        push_str(&mut p, "qp-range", &self.qp_range);
        push_u32(&mut p, "quant-i-frames", &self.quant_i_frames);
        push_u32(&mut p, "quant-p-frames", &self.quant_p_frames);
        push_bool(&mut p, "ratecontrol-enable", &self.ratecontrol_enable);
        push_bool(&mut p, "maxperf-enable", &self.maxperf_enable);
        push_bool(&mut p, "EnableTwopassCBR", &self.two_pass_cbr);
        push_u32(&mut p, "num-Ref-Frames", &self.num_ref_frames);
        push_bool(&mut p, "enable-lossless", &self.enable_lossless);
        p
    }

    pub fn from_pairs(pairs: &HashMap<String, String>) -> Result<Self, EncoderError> {
        let mut props = Self::default();
        for (key, value) in pairs {
            match normalize_key(key).as_str() {
                "bitrate" => props.bitrate = Some(parse_u32(key, value)?),
                "control_rate" => {
                    props.control_rate = Some(RateControl::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown rate control '{value}'"),
                        }
                    })?)
                }
                "profile" => {
                    props.profile = Some(HevcProfile::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown HEVC profile '{value}'"),
                        }
                    })?)
                }
                "iframeinterval" => props.iframeinterval = Some(parse_u32(key, value)?),
                "idrinterval" => props.idrinterval = Some(parse_u32(key, value)?),
                "preset_level" => {
                    props.preset_level =
                        Some(JetsonPresetLevel::from_name(value).ok_or_else(|| {
                            EncoderError::InvalidProperty {
                                name: key.clone(),
                                reason: format!("unknown Jetson preset level '{value}'"),
                            }
                        })?)
                }
                "peak_bitrate" => props.peak_bitrate = Some(parse_u32(key, value)?),
                "vbv_size" => props.vbv_size = Some(parse_u32(key, value)?),
                "qp_range" => props.qp_range = Some(value.clone()),
                "quant_i_frames" => props.quant_i_frames = Some(parse_u32(key, value)?),
                "quant_p_frames" => props.quant_p_frames = Some(parse_u32(key, value)?),
                "ratecontrol_enable" => props.ratecontrol_enable = Some(parse_bool(key, value)?),
                "maxperf_enable" => props.maxperf_enable = Some(parse_bool(key, value)?),
                "enabletwopasscbr" | "two_pass_cbr" => {
                    props.two_pass_cbr = Some(parse_bool(key, value)?)
                }
                "num_ref_frames" => props.num_ref_frames = Some(parse_u32(key, value)?),
                "enable_lossless" => props.enable_lossless = Some(parse_bool(key, value)?),
                _ => {
                    return Err(EncoderError::InvalidProperty {
                        name: key.clone(),
                        reason: format!("unknown HEVC Jetson property '{key}'"),
                    })
                }
            }
        }
        Ok(props)
    }
}

// ─── JPEG ──────────────────────────────────────────────────────────────

/// JPEG encoder properties (`nvjpegenc`, both platforms).
#[derive(Debug, Clone, Default)]
pub struct JpegProps {
    /// JPEG quality (0–100, default: 85).
    pub quality: Option<u32>,
}

impl JpegProps {
    pub fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        let mut p = Vec::new();
        push_u32(&mut p, "quality", &self.quality);
        p
    }

    pub fn from_pairs(pairs: &HashMap<String, String>) -> Result<Self, EncoderError> {
        let mut props = Self::default();
        for (key, value) in pairs {
            match normalize_key(key).as_str() {
                "quality" => props.quality = Some(parse_u32(key, value)?),
                _ => {
                    return Err(EncoderError::InvalidProperty {
                        name: key.clone(),
                        reason: format!("unknown JPEG property '{key}'"),
                    })
                }
            }
        }
        Ok(props)
    }
}

// ─── AV1 dGPU ──────────────────────────────────────────────────────────

/// AV1 encoder properties for dGPU (`nvv4l2av1enc`).
#[derive(Debug, Clone, Default)]
pub struct Av1DgpuProps {
    /// Bitrate in bits/sec (default: 4 000 000).
    pub bitrate: Option<u32>,
    /// Rate-control mode (default: constant bitrate).
    pub control_rate: Option<RateControl>,
    /// I-frame interval (default: 30).
    pub iframeinterval: Option<u32>,
    /// IDR-frame interval (default: 256).
    pub idrinterval: Option<u32>,
    /// NVENC preset P1–P7 (default: P1).
    pub preset: Option<DgpuPreset>,
    /// Tuning-info preset (default: LowLatency).
    pub tuning_info: Option<TuningPreset>,
    /// QP range, format `"minQP:maxQP"`.
    pub qp_range: Option<String>,
    /// Maximum bitrate for VBR mode (bits/sec).
    pub max_bitrate: Option<u32>,
    /// VBV buffer size in bits.
    pub vbv_buf_size: Option<u32>,
    /// VBV initial delay in bits.
    pub vbv_init: Option<u32>,
    /// Target constant-quality level for VBR (0–51).
    pub cq: Option<u32>,
    /// Spatial adaptive-quantization strength (0–15, 0 = auto).
    pub aq: Option<u32>,
    /// Enable temporal adaptive quantization.
    pub temporal_aq: Option<bool>,
}

impl Av1DgpuProps {
    pub fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        let mut p = Vec::new();
        push_u32(&mut p, "bitrate", &self.bitrate);
        if let Some(v) = &self.control_rate {
            p.push(("control-rate", v.gst_value().to_string()));
        }
        push_u32(&mut p, "iframeinterval", &self.iframeinterval);
        push_u32(&mut p, "idrinterval", &self.idrinterval);
        if let Some(v) = &self.preset {
            p.push(("preset-id", v.gst_value().to_string()));
        }
        if let Some(v) = &self.tuning_info {
            p.push(("tuning-info-id", v.gst_value().to_string()));
        }
        push_str(&mut p, "qp-range", &self.qp_range);
        push_u32(&mut p, "maxbitrate", &self.max_bitrate);
        push_u32(&mut p, "vbvbufsize", &self.vbv_buf_size);
        push_u32(&mut p, "vbvinit", &self.vbv_init);
        push_u32(&mut p, "cq", &self.cq);
        push_u32(&mut p, "aq", &self.aq);
        push_bool(&mut p, "temporalaq", &self.temporal_aq);
        p
    }

    pub fn from_pairs(pairs: &HashMap<String, String>) -> Result<Self, EncoderError> {
        let mut props = Self::default();
        for (key, value) in pairs {
            match normalize_key(key).as_str() {
                "bitrate" => props.bitrate = Some(parse_u32(key, value)?),
                "control_rate" => {
                    props.control_rate = Some(RateControl::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown rate control '{value}'"),
                        }
                    })?)
                }
                "iframeinterval" => props.iframeinterval = Some(parse_u32(key, value)?),
                "idrinterval" => props.idrinterval = Some(parse_u32(key, value)?),
                "preset_id" | "preset" => {
                    props.preset = Some(DgpuPreset::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown dGPU preset '{value}'"),
                        }
                    })?)
                }
                "tuning_info_id" | "tuning_info" => {
                    props.tuning_info = Some(TuningPreset::from_name(value).ok_or_else(|| {
                        EncoderError::InvalidProperty {
                            name: key.clone(),
                            reason: format!("unknown tuning preset '{value}'"),
                        }
                    })?)
                }
                "qp_range" => props.qp_range = Some(value.clone()),
                "maxbitrate" | "max_bitrate" => props.max_bitrate = Some(parse_u32(key, value)?),
                "vbvbufsize" | "vbv_buf_size" => props.vbv_buf_size = Some(parse_u32(key, value)?),
                "vbvinit" | "vbv_init" => props.vbv_init = Some(parse_u32(key, value)?),
                "cq" => props.cq = Some(parse_u32(key, value)?),
                "aq" => props.aq = Some(parse_u32(key, value)?),
                "temporalaq" | "temporal_aq" => props.temporal_aq = Some(parse_bool(key, value)?),
                _ => {
                    return Err(EncoderError::InvalidProperty {
                        name: key.clone(),
                        reason: format!("unknown AV1 dGPU property '{key}'"),
                    })
                }
            }
        }
        Ok(props)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// EncoderProperties — top-level enum
// ═══════════════════════════════════════════════════════════════════════

/// Typed encoder properties.
///
/// Each variant matches a specific codec+platform combination and carries
/// only the properties that are valid for that encoder element.
///
/// # Example (Rust)
///
/// ```rust
/// use deepstream_encoders::properties::*;
///
/// let props = EncoderProperties::HevcDgpu(HevcDgpuProps {
///     bitrate: Some(8_000_000),
///     profile: Some(HevcProfile::Main10),
///     preset: Some(DgpuPreset::P5),
///     ..Default::default()
/// });
///
/// assert_eq!(props.codec(), savant_gstreamer::Codec::Hevc);
/// ```
#[derive(Debug, Clone)]
pub enum EncoderProperties {
    /// H.264 encoder on dGPU (`nvv4l2h264enc`).
    H264Dgpu(H264DgpuProps),
    /// H.264 encoder on Jetson (`nvv4l2h264enc`).
    H264Jetson(H264JetsonProps),
    /// HEVC encoder on dGPU (`nvv4l2h265enc`).
    HevcDgpu(HevcDgpuProps),
    /// HEVC encoder on Jetson (`nvv4l2h265enc`).
    HevcJetson(HevcJetsonProps),
    /// JPEG encoder (`nvjpegenc`, works on both platforms).
    Jpeg(JpegProps),
    /// AV1 encoder on dGPU (`nvv4l2av1enc`).
    Av1Dgpu(Av1DgpuProps),
}

impl EncoderProperties {
    /// The codec this variant is for.
    pub fn codec(&self) -> Codec {
        match self {
            Self::H264Dgpu(_) | Self::H264Jetson(_) => Codec::H264,
            Self::HevcDgpu(_) | Self::HevcJetson(_) => Codec::Hevc,
            Self::Jpeg(_) => Codec::Jpeg,
            Self::Av1Dgpu(_) => Codec::Av1,
        }
    }

    /// The target platform, if platform-specific (`None` for JPEG).
    pub fn platform(&self) -> Option<Platform> {
        match self {
            Self::H264Dgpu(_) | Self::HevcDgpu(_) | Self::Av1Dgpu(_) => Some(Platform::Dgpu),
            Self::H264Jetson(_) | Self::HevcJetson(_) => Some(Platform::Jetson),
            Self::Jpeg(_) => None,
        }
    }

    /// Convert to GStreamer element key-value pairs (only set fields).
    pub fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::H264Dgpu(p) => p.to_gst_pairs(),
            Self::H264Jetson(p) => p.to_gst_pairs(),
            Self::HevcDgpu(p) => p.to_gst_pairs(),
            Self::HevcJetson(p) => p.to_gst_pairs(),
            Self::Jpeg(p) => p.to_gst_pairs(),
            Self::Av1Dgpu(p) => p.to_gst_pairs(),
        }
    }

    /// Create encoder properties from GStreamer-style key-value pairs.
    ///
    /// Selects the correct variant based on `codec` and `platform`, then
    /// parses each pair into the typed fields.  Unknown keys and B-frame
    /// properties cause an error.
    ///
    /// For JPEG, `platform` is ignored.
    pub fn from_pairs(
        codec: Codec,
        platform: Platform,
        pairs: &HashMap<String, String>,
    ) -> Result<Self, EncoderError> {
        match (codec, platform) {
            (Codec::H264, Platform::Dgpu) => Ok(Self::H264Dgpu(H264DgpuProps::from_pairs(pairs)?)),
            (Codec::H264, Platform::Jetson) => {
                Ok(Self::H264Jetson(H264JetsonProps::from_pairs(pairs)?))
            }
            (Codec::Hevc, Platform::Dgpu) => Ok(Self::HevcDgpu(HevcDgpuProps::from_pairs(pairs)?)),
            (Codec::Hevc, Platform::Jetson) => {
                Ok(Self::HevcJetson(HevcJetsonProps::from_pairs(pairs)?))
            }
            (Codec::Jpeg, _) => Ok(Self::Jpeg(JpegProps::from_pairs(pairs)?)),
            (Codec::Av1, Platform::Dgpu) => Ok(Self::Av1Dgpu(Av1DgpuProps::from_pairs(pairs)?)),
            (Codec::Av1, Platform::Jetson) => Err(EncoderError::UnsupportedCodec(
                "AV1 encoding is not supported on Jetson".to_string(),
            )),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ─── Enum round-trips ──────────────────────────────────────────────

    #[test]
    fn test_rate_control_round_trip() {
        for rc in [
            RateControl::VariableBitrate,
            RateControl::ConstantBitrate,
            RateControl::ConstantQP,
        ] {
            assert_eq!(RateControl::from_name(rc.name()), Some(rc));
            assert_eq!(RateControl::from_name(rc.gst_value()), Some(rc));
        }
    }

    #[test]
    fn test_h264_profile_round_trip() {
        for p in [
            H264Profile::Baseline,
            H264Profile::Main,
            H264Profile::High,
            H264Profile::High444,
        ] {
            assert_eq!(H264Profile::from_name(p.name()), Some(p));
            assert_eq!(H264Profile::from_name(p.gst_value()), Some(p));
        }
    }

    #[test]
    fn test_hevc_profile_round_trip() {
        for p in [HevcProfile::Main, HevcProfile::Main10, HevcProfile::Frext] {
            assert_eq!(HevcProfile::from_name(p.name()), Some(p));
            assert_eq!(HevcProfile::from_name(p.gst_value()), Some(p));
        }
    }

    #[test]
    fn test_dgpu_preset_round_trip() {
        for p in [
            DgpuPreset::P1,
            DgpuPreset::P2,
            DgpuPreset::P3,
            DgpuPreset::P4,
            DgpuPreset::P5,
            DgpuPreset::P6,
            DgpuPreset::P7,
        ] {
            assert_eq!(DgpuPreset::from_name(p.name()), Some(p));
            assert_eq!(DgpuPreset::from_name(p.gst_value()), Some(p));
        }
    }

    #[test]
    fn test_tuning_preset_round_trip() {
        for p in [
            TuningPreset::HighQuality,
            TuningPreset::LowLatency,
            TuningPreset::UltraLowLatency,
            TuningPreset::Lossless,
        ] {
            assert_eq!(TuningPreset::from_name(p.name()), Some(p));
            assert_eq!(TuningPreset::from_name(p.gst_value()), Some(p));
        }
    }

    #[test]
    fn test_jetson_preset_round_trip() {
        for p in [
            JetsonPresetLevel::Disabled,
            JetsonPresetLevel::UltraFast,
            JetsonPresetLevel::Fast,
            JetsonPresetLevel::Medium,
            JetsonPresetLevel::Slow,
        ] {
            assert_eq!(JetsonPresetLevel::from_name(p.name()), Some(p));
            assert_eq!(JetsonPresetLevel::from_name(p.gst_value()), Some(p));
        }
    }

    #[test]
    fn test_platform_round_trip() {
        assert_eq!(Platform::from_name("dgpu"), Some(Platform::Dgpu));
        assert_eq!(Platform::from_name("jetson"), Some(Platform::Jetson));
        assert_eq!(Platform::from_name("nope"), None);
    }

    // ─── to_gst_pairs ─────────────────────────────────────────────────

    #[test]
    fn test_default_produces_no_pairs() {
        let props = H264DgpuProps::default();
        assert!(props.to_gst_pairs().is_empty());
    }

    #[test]
    fn test_h264_dgpu_pairs() {
        let props = H264DgpuProps {
            bitrate: Some(8_000_000),
            profile: Some(H264Profile::High),
            preset: Some(DgpuPreset::P5),
            ..Default::default()
        };
        let pairs = props.to_gst_pairs();
        assert!(pairs.contains(&("bitrate", "8000000".to_string())));
        assert!(pairs.contains(&("profile", "4".to_string())));
        assert!(pairs.contains(&("preset-id", "5".to_string())));
    }

    #[test]
    fn test_jpeg_pairs() {
        let props = JpegProps { quality: Some(95) };
        let pairs = props.to_gst_pairs();
        assert_eq!(pairs, vec![("quality", "95".to_string())]);
    }

    // ─── from_pairs ────────────────────────────────────────────────────

    #[test]
    fn test_from_pairs_h264_dgpu() {
        let mut m = HashMap::new();
        m.insert("bitrate".into(), "6000000".into());
        m.insert("profile".into(), "high".into());
        m.insert("control-rate".into(), "vbr".into());

        let props = H264DgpuProps::from_pairs(&m).unwrap();
        assert_eq!(props.bitrate, Some(6_000_000));
        assert_eq!(props.profile, Some(H264Profile::High));
        assert_eq!(props.control_rate, Some(RateControl::VariableBitrate));
    }

    #[test]
    fn test_from_pairs_rejects_unknown_key() {
        let mut m = HashMap::new();
        m.insert("magic-beans".into(), "42".into());

        let result = H264DgpuProps::from_pairs(&m);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_pairs_via_enum() {
        let mut m = HashMap::new();
        m.insert("bitrate".into(), "5000000".into());

        let props = EncoderProperties::from_pairs(Codec::Hevc, Platform::Dgpu, &m).unwrap();
        assert_eq!(props.codec(), Codec::Hevc);
        assert_eq!(props.platform(), Some(Platform::Dgpu));
    }

    #[test]
    fn test_av1_jetson_unsupported() {
        let m = HashMap::new();
        let result = EncoderProperties::from_pairs(Codec::Av1, Platform::Jetson, &m);
        assert!(result.is_err());
    }

    // ─── EncoderProperties codec/platform ──────────────────────────────

    #[test]
    fn test_encoder_properties_codec() {
        let p = EncoderProperties::H264Dgpu(H264DgpuProps::default());
        assert_eq!(p.codec(), Codec::H264);
        assert_eq!(p.platform(), Some(Platform::Dgpu));

        let p = EncoderProperties::HevcJetson(HevcJetsonProps::default());
        assert_eq!(p.codec(), Codec::Hevc);
        assert_eq!(p.platform(), Some(Platform::Jetson));

        let p = EncoderProperties::Jpeg(JpegProps::default());
        assert_eq!(p.codec(), Codec::Jpeg);
        assert_eq!(p.platform(), None);
    }
}
