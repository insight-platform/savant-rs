use savant_core::primitives::video_codec::VideoCodec;
use std::time::Duration;

/// CUDA memory type exposed by the `nvv4l2decoder` element via its
/// `cudadec-memtype` GObject property. Only meaningful on desktop GPUs;
/// the property is absent on Jetson, so the type itself is compiled
/// in only on non-`aarch64` targets.
///
/// Numeric mapping mirrors the GStreamer property enum:
///
/// - `Device`  = 0 — `memtype_device` (CUDA device memory, default).
/// - `Pinned`  = 1 — `memtype_pinned` (CUDA pinned host memory).
/// - `Unified` = 2 — `memtype_unified` (CUDA unified memory).
#[cfg(not(target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CudadecMemtype {
    Device = 0,
    Pinned = 1,
    Unified = 2,
}

#[cfg(not(target_arch = "aarch64"))]
impl CudadecMemtype {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Pinned => "pinned",
            Self::Unified => "unified",
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl std::fmt::Display for CudadecMemtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum H264StreamFormat {
    ByteStream,
    Avc,
    Avc3,
}

impl H264StreamFormat {
    pub fn gst_name(&self) -> &'static str {
        match self {
            Self::ByteStream => "byte-stream",
            Self::Avc => "avc",
            Self::Avc3 => "avc3",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "byte-stream" | "bytestream" => Some(Self::ByteStream),
            "avc" => Some(Self::Avc),
            "avc3" => Some(Self::Avc3),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        self.gst_name()
    }
}

impl std::fmt::Display for H264StreamFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HevcStreamFormat {
    ByteStream,
    Hvc1,
    Hev1,
}

impl HevcStreamFormat {
    pub fn gst_name(&self) -> &'static str {
        match self {
            Self::ByteStream => "byte-stream",
            Self::Hvc1 => "hvc1",
            Self::Hev1 => "hev1",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "byte-stream" | "bytestream" => Some(Self::ByteStream),
            "hvc1" => Some(Self::Hvc1),
            "hev1" => Some(Self::Hev1),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        self.gst_name()
    }
}

impl std::fmt::Display for HevcStreamFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─── H.264 ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct H264DecoderConfig {
    pub stream_format: H264StreamFormat,
    /// AVCDecoderConfigurationRecord bytes; required for `Avc` stream format.
    pub codec_data: Option<Vec<u8>>,
    pub num_extra_surfaces: Option<u32>,
    pub drop_frame_interval: Option<u32>,
    #[cfg(not(target_arch = "aarch64"))]
    pub cudadec_memtype: Option<CudadecMemtype>,
    /// Enable dGPU `low-latency-mode` (no frame reordering; requires
    /// IDR-only or low-delay-encoded bitstreams).
    #[cfg(not(target_arch = "aarch64"))]
    pub low_latency_mode: Option<bool>,
    #[cfg(target_arch = "aarch64")]
    pub enable_max_performance: Option<bool>,
    /// Enable Jetson `disable-dpb` (low-latency mode for IDR-only or
    /// IPPP bitstreams).
    #[cfg(target_arch = "aarch64")]
    pub low_latency: Option<bool>,
}

impl H264DecoderConfig {
    pub fn new(stream_format: H264StreamFormat) -> Self {
        Self {
            stream_format,
            codec_data: None,
            num_extra_surfaces: None,
            drop_frame_interval: None,
            #[cfg(not(target_arch = "aarch64"))]
            cudadec_memtype: None,
            #[cfg(not(target_arch = "aarch64"))]
            low_latency_mode: None,
            #[cfg(target_arch = "aarch64")]
            enable_max_performance: None,
            #[cfg(target_arch = "aarch64")]
            low_latency: None,
        }
    }

    pub fn codec_data(mut self, data: Vec<u8>) -> Self {
        self.codec_data = Some(data);
        self
    }

    pub fn num_extra_surfaces(mut self, n: u32) -> Self {
        self.num_extra_surfaces = Some(n);
        self
    }

    pub fn drop_frame_interval(mut self, n: u32) -> Self {
        self.drop_frame_interval = Some(n);
        self
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn cudadec_memtype(mut self, t: CudadecMemtype) -> Self {
        self.cudadec_memtype = Some(t);
        self
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn low_latency_mode(mut self, v: bool) -> Self {
        self.low_latency_mode = Some(v);
        self
    }

    #[cfg(target_arch = "aarch64")]
    pub fn enable_max_performance(mut self, v: bool) -> Self {
        self.enable_max_performance = Some(v);
        self
    }

    #[cfg(target_arch = "aarch64")]
    pub fn low_latency(mut self, v: bool) -> Self {
        self.low_latency = Some(v);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        #[cfg(not(target_arch = "aarch64"))]
        {
            collect_v4l2_pairs(
                self.num_extra_surfaces,
                self.drop_frame_interval,
                self.cudadec_memtype,
                self.low_latency_mode,
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            collect_v4l2_pairs(
                self.num_extra_surfaces,
                self.drop_frame_interval,
                self.enable_max_performance,
                self.low_latency,
            )
        }
    }
}

// ─── HEVC ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HevcDecoderConfig {
    pub stream_format: HevcStreamFormat,
    /// HEVCDecoderConfigurationRecord bytes; required for `Hvc1` stream format.
    pub codec_data: Option<Vec<u8>>,
    pub num_extra_surfaces: Option<u32>,
    pub drop_frame_interval: Option<u32>,
    #[cfg(not(target_arch = "aarch64"))]
    pub cudadec_memtype: Option<CudadecMemtype>,
    /// Enable dGPU `low-latency-mode` (no frame reordering; requires
    /// IDR-only or low-delay-encoded bitstreams).
    #[cfg(not(target_arch = "aarch64"))]
    pub low_latency_mode: Option<bool>,
    #[cfg(target_arch = "aarch64")]
    pub enable_max_performance: Option<bool>,
    /// Enable Jetson `disable-dpb` (low-latency mode for IDR-only or
    /// IPPP bitstreams).
    #[cfg(target_arch = "aarch64")]
    pub low_latency: Option<bool>,
}

impl HevcDecoderConfig {
    pub fn new(stream_format: HevcStreamFormat) -> Self {
        Self {
            stream_format,
            codec_data: None,
            num_extra_surfaces: None,
            drop_frame_interval: None,
            #[cfg(not(target_arch = "aarch64"))]
            cudadec_memtype: None,
            #[cfg(not(target_arch = "aarch64"))]
            low_latency_mode: None,
            #[cfg(target_arch = "aarch64")]
            enable_max_performance: None,
            #[cfg(target_arch = "aarch64")]
            low_latency: None,
        }
    }

    pub fn codec_data(mut self, data: Vec<u8>) -> Self {
        self.codec_data = Some(data);
        self
    }

    pub fn num_extra_surfaces(mut self, n: u32) -> Self {
        self.num_extra_surfaces = Some(n);
        self
    }

    pub fn drop_frame_interval(mut self, n: u32) -> Self {
        self.drop_frame_interval = Some(n);
        self
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn cudadec_memtype(mut self, t: CudadecMemtype) -> Self {
        self.cudadec_memtype = Some(t);
        self
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn low_latency_mode(mut self, v: bool) -> Self {
        self.low_latency_mode = Some(v);
        self
    }

    #[cfg(target_arch = "aarch64")]
    pub fn enable_max_performance(mut self, v: bool) -> Self {
        self.enable_max_performance = Some(v);
        self
    }

    #[cfg(target_arch = "aarch64")]
    pub fn low_latency(mut self, v: bool) -> Self {
        self.low_latency = Some(v);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        #[cfg(not(target_arch = "aarch64"))]
        {
            collect_v4l2_pairs(
                self.num_extra_surfaces,
                self.drop_frame_interval,
                self.cudadec_memtype,
                self.low_latency_mode,
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            collect_v4l2_pairs(
                self.num_extra_surfaces,
                self.drop_frame_interval,
                self.enable_max_performance,
                self.low_latency,
            )
        }
    }
}

// ─── VP8 / VP9 / AV1 (identical tunable surface) ────────────────────────

macro_rules! nvv4l2_codec_struct {
    ($name:ident) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            pub num_extra_surfaces: Option<u32>,
            pub drop_frame_interval: Option<u32>,
            #[cfg(not(target_arch = "aarch64"))]
            pub cudadec_memtype: Option<CudadecMemtype>,
            #[cfg(not(target_arch = "aarch64"))]
            pub low_latency_mode: Option<bool>,
            #[cfg(target_arch = "aarch64")]
            pub enable_max_performance: Option<bool>,
            #[cfg(target_arch = "aarch64")]
            pub low_latency: Option<bool>,
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    num_extra_surfaces: None,
                    drop_frame_interval: None,
                    #[cfg(not(target_arch = "aarch64"))]
                    cudadec_memtype: None,
                    #[cfg(not(target_arch = "aarch64"))]
                    low_latency_mode: None,
                    #[cfg(target_arch = "aarch64")]
                    enable_max_performance: None,
                    #[cfg(target_arch = "aarch64")]
                    low_latency: None,
                }
            }

            pub fn num_extra_surfaces(mut self, n: u32) -> Self {
                self.num_extra_surfaces = Some(n);
                self
            }

            pub fn drop_frame_interval(mut self, n: u32) -> Self {
                self.drop_frame_interval = Some(n);
                self
            }

            #[cfg(not(target_arch = "aarch64"))]
            pub fn cudadec_memtype(mut self, t: CudadecMemtype) -> Self {
                self.cudadec_memtype = Some(t);
                self
            }

            #[cfg(not(target_arch = "aarch64"))]
            pub fn low_latency_mode(mut self, v: bool) -> Self {
                self.low_latency_mode = Some(v);
                self
            }

            #[cfg(target_arch = "aarch64")]
            pub fn enable_max_performance(mut self, v: bool) -> Self {
                self.enable_max_performance = Some(v);
                self
            }

            #[cfg(target_arch = "aarch64")]
            pub fn low_latency(mut self, v: bool) -> Self {
                self.low_latency = Some(v);
                self
            }

            pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
                #[cfg(not(target_arch = "aarch64"))]
                {
                    collect_v4l2_pairs(
                        self.num_extra_surfaces,
                        self.drop_frame_interval,
                        self.cudadec_memtype,
                        self.low_latency_mode,
                    )
                }
                #[cfg(target_arch = "aarch64")]
                {
                    collect_v4l2_pairs(
                        self.num_extra_surfaces,
                        self.drop_frame_interval,
                        self.enable_max_performance,
                        self.low_latency,
                    )
                }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

nvv4l2_codec_struct!(Vp8DecoderConfig);
nvv4l2_codec_struct!(Vp9DecoderConfig);
nvv4l2_codec_struct!(Av1DecoderConfig);

// ─── JPEG / PNG / Raw ───────────────────────────────────────────────────

/// Selects GPU-accelerated or CPU-only JPEG decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JpegBackend {
    /// Use `nvjpegdec` (NVIDIA GPU-accelerated JPEG decoder).
    #[default]
    Gpu,
    /// Use `jpegparse ! jpegdec` (software CPU decoder, same pattern as
    /// PNG).  No GPU decoder dependency.
    Cpu,
}

/// Configuration for the JPEG decoder.
#[derive(Debug, Clone)]
pub struct JpegDecoderConfig {
    /// Decoder backend: GPU (`nvjpegdec`) or CPU (`jpegdec`).
    pub backend: JpegBackend,
}

impl JpegDecoderConfig {
    /// Create a config targeting the GPU JPEG decoder (`nvjpegdec`).
    pub fn gpu() -> Self {
        Self {
            backend: JpegBackend::Gpu,
        }
    }

    /// Create a config targeting the CPU JPEG decoder (`jpegdec`).
    pub fn cpu() -> Self {
        Self {
            backend: JpegBackend::Cpu,
        }
    }
}

impl Default for JpegDecoderConfig {
    fn default() -> Self {
        Self::gpu()
    }
}

#[derive(Debug, Clone, Default)]
pub struct PngDecoderConfig;

#[derive(Debug, Clone)]
pub struct RawRgbaDecoderConfig {
    pub width: u32,
    pub height: u32,
}

impl RawRgbaDecoderConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

#[derive(Debug, Clone)]
pub struct RawRgbDecoderConfig {
    pub width: u32,
    pub height: u32,
}

impl RawRgbDecoderConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

#[derive(Debug, Clone)]
pub enum DecoderConfig {
    H264(H264DecoderConfig),
    Hevc(HevcDecoderConfig),
    Vp8(Vp8DecoderConfig),
    Vp9(Vp9DecoderConfig),
    Av1(Av1DecoderConfig),
    Jpeg(JpegDecoderConfig),
    Png(PngDecoderConfig),
    RawRgba(RawRgbaDecoderConfig),
    RawRgb(RawRgbDecoderConfig),
}

impl DecoderConfig {
    pub fn codec(&self) -> VideoCodec {
        match self {
            Self::H264(_) => VideoCodec::H264,
            Self::Hevc(_) => VideoCodec::Hevc,
            Self::Vp8(_) => VideoCodec::Vp8,
            Self::Vp9(_) => VideoCodec::Vp9,
            Self::Av1(_) => VideoCodec::Av1,
            Self::Jpeg(cfg) => match cfg.backend {
                JpegBackend::Gpu => VideoCodec::Jpeg,
                JpegBackend::Cpu => VideoCodec::SwJpeg,
            },
            Self::Png(_) => VideoCodec::Png,
            Self::RawRgba(_) => VideoCodec::RawRgba,
            Self::RawRgb(_) => VideoCodec::RawRgb,
        }
    }
}

/// Configuration for the channel-based `NvDecoder` wrapper API.
#[derive(Debug, Clone)]
pub struct NvDecoderConfig {
    pub name: String,
    pub gpu_id: u32,
    pub decoder: DecoderConfig,
    pub input_channel_capacity: usize,
    pub output_channel_capacity: usize,
    pub operation_timeout: Duration,
    pub drain_poll_interval: Duration,
}

impl NvDecoderConfig {
    pub fn new(gpu_id: u32, decoder: DecoderConfig) -> Self {
        Self {
            name: String::new(),
            gpu_id,
            decoder,
            input_channel_capacity: 16,
            output_channel_capacity: 16,
            operation_timeout: Duration::from_secs(30),
            drain_poll_interval: Duration::from_millis(100),
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn input_channel_capacity(mut self, capacity: usize) -> Self {
        self.input_channel_capacity = capacity;
        self
    }

    pub fn output_channel_capacity(mut self, capacity: usize) -> Self {
        self.output_channel_capacity = capacity;
        self
    }

    pub fn operation_timeout(mut self, timeout: Duration) -> Self {
        self.operation_timeout = timeout;
        self
    }

    pub fn drain_poll_interval(mut self, interval: Duration) -> Self {
        self.drain_poll_interval = interval;
        self
    }
}

// ─── GStreamer property emission ────────────────────────────────────────
//
// The nvv4l2decoder element exposes a different set of tunable properties
// depending on the platform (dGPU vs Jetson), per the DeepStream 7.1
// Gst-nvvideo4linux2 docs. We split `collect_v4l2_pairs` into two
// per-arch versions with platform-specific argument lists so each codec
// struct passes only fields that actually exist at compile time.

#[cfg(not(target_arch = "aarch64"))]
fn collect_v4l2_pairs(
    num_extra_surfaces: Option<u32>,
    drop_frame_interval: Option<u32>,
    cudadec_memtype: Option<CudadecMemtype>,
    low_latency_mode: Option<bool>,
) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    if let Some(v) = num_extra_surfaces {
        out.push(("num-extra-surfaces", v.to_string()));
    }
    if let Some(v) = drop_frame_interval {
        out.push(("drop-frame-interval", v.to_string()));
    }
    if let Some(v) = cudadec_memtype {
        out.push(("cudadec-memtype", v.as_u32().to_string()));
    }
    if let Some(v) = low_latency_mode {
        out.push(("low-latency-mode", if v { "1" } else { "0" }.to_string()));
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn collect_v4l2_pairs(
    num_extra_surfaces: Option<u32>,
    drop_frame_interval: Option<u32>,
    enable_max_performance: Option<bool>,
    low_latency: Option<bool>,
) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    if let Some(v) = num_extra_surfaces {
        out.push(("num-extra-surfaces", v.to_string()));
    }
    if let Some(v) = drop_frame_interval {
        out.push(("drop-frame-interval", v.to_string()));
    }
    if let Some(v) = enable_max_performance {
        out.push((
            "enable-max-performance",
            if v { "1" } else { "0" }.to_string(),
        ));
    }
    if let Some(v) = low_latency {
        out.push(("disable-dpb", if v { "1" } else { "0" }.to_string()));
    }
    out
}
