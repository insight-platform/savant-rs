use log::warn;
use savant_gstreamer::Codec;

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

#[derive(Debug, Clone)]
pub struct H264DecoderConfig {
    pub stream_format: H264StreamFormat,
    /// AVCDecoderConfigurationRecord bytes; required for `Avc` stream format.
    pub codec_data: Option<Vec<u8>>,
    pub num_extra_surfaces: Option<u32>,
    pub drop_frame_interval: Option<u32>,
    pub cudadec_memtype: Option<u32>,
    pub enable_max_performance: Option<bool>,
    pub low_latency: Option<bool>,
}

impl H264DecoderConfig {
    pub fn new(stream_format: H264StreamFormat) -> Self {
        Self {
            stream_format,
            codec_data: None,
            num_extra_surfaces: None,
            drop_frame_interval: None,
            cudadec_memtype: None,
            enable_max_performance: None,
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

    pub fn cudadec_memtype(mut self, t: u32) -> Self {
        self.cudadec_memtype = Some(t);
        self
    }

    pub fn enable_max_performance(mut self, v: bool) -> Self {
        self.enable_max_performance = Some(v);
        self
    }

    pub fn low_latency(mut self, v: bool) -> Self {
        self.low_latency = Some(v);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        collect_v4l2_pairs(
            self.num_extra_surfaces,
            self.drop_frame_interval,
            self.cudadec_memtype,
            self.enable_max_performance,
            self.low_latency,
        )
    }
}

#[derive(Debug, Clone)]
pub struct HevcDecoderConfig {
    pub stream_format: HevcStreamFormat,
    /// HEVCDecoderConfigurationRecord bytes; required for `Hvc1` stream format.
    pub codec_data: Option<Vec<u8>>,
    pub num_extra_surfaces: Option<u32>,
    pub drop_frame_interval: Option<u32>,
    pub cudadec_memtype: Option<u32>,
    pub enable_max_performance: Option<bool>,
    pub low_latency: Option<bool>,
}

impl HevcDecoderConfig {
    pub fn new(stream_format: HevcStreamFormat) -> Self {
        Self {
            stream_format,
            codec_data: None,
            num_extra_surfaces: None,
            drop_frame_interval: None,
            cudadec_memtype: None,
            enable_max_performance: None,
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

    pub fn cudadec_memtype(mut self, t: u32) -> Self {
        self.cudadec_memtype = Some(t);
        self
    }

    pub fn enable_max_performance(mut self, v: bool) -> Self {
        self.enable_max_performance = Some(v);
        self
    }

    pub fn low_latency(mut self, v: bool) -> Self {
        self.low_latency = Some(v);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        collect_v4l2_pairs(
            self.num_extra_surfaces,
            self.drop_frame_interval,
            self.cudadec_memtype,
            self.enable_max_performance,
            self.low_latency,
        )
    }
}

#[derive(Debug, Clone)]
pub struct Vp8DecoderConfig {
    pub num_extra_surfaces: Option<u32>,
    pub drop_frame_interval: Option<u32>,
    pub cudadec_memtype: Option<u32>,
    pub enable_max_performance: Option<bool>,
    pub low_latency: Option<bool>,
}

impl Vp8DecoderConfig {
    pub fn new() -> Self {
        Self {
            num_extra_surfaces: None,
            drop_frame_interval: None,
            cudadec_memtype: None,
            enable_max_performance: None,
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

    pub fn cudadec_memtype(mut self, t: u32) -> Self {
        self.cudadec_memtype = Some(t);
        self
    }

    pub fn enable_max_performance(mut self, v: bool) -> Self {
        self.enable_max_performance = Some(v);
        self
    }

    pub fn low_latency(mut self, v: bool) -> Self {
        self.low_latency = Some(v);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        collect_v4l2_pairs(
            self.num_extra_surfaces,
            self.drop_frame_interval,
            self.cudadec_memtype,
            self.enable_max_performance,
            self.low_latency,
        )
    }
}

impl Default for Vp8DecoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Vp9DecoderConfig {
    pub num_extra_surfaces: Option<u32>,
    pub drop_frame_interval: Option<u32>,
    pub cudadec_memtype: Option<u32>,
    pub enable_max_performance: Option<bool>,
    pub low_latency: Option<bool>,
}

impl Vp9DecoderConfig {
    pub fn new() -> Self {
        Self {
            num_extra_surfaces: None,
            drop_frame_interval: None,
            cudadec_memtype: None,
            enable_max_performance: None,
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

    pub fn cudadec_memtype(mut self, t: u32) -> Self {
        self.cudadec_memtype = Some(t);
        self
    }

    pub fn enable_max_performance(mut self, v: bool) -> Self {
        self.enable_max_performance = Some(v);
        self
    }

    pub fn low_latency(mut self, v: bool) -> Self {
        self.low_latency = Some(v);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        collect_v4l2_pairs(
            self.num_extra_surfaces,
            self.drop_frame_interval,
            self.cudadec_memtype,
            self.enable_max_performance,
            self.low_latency,
        )
    }
}

impl Default for Vp9DecoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Av1DecoderConfig {
    pub num_extra_surfaces: Option<u32>,
    pub drop_frame_interval: Option<u32>,
    pub cudadec_memtype: Option<u32>,
    pub enable_max_performance: Option<bool>,
    pub low_latency: Option<bool>,
}

impl Av1DecoderConfig {
    pub fn new() -> Self {
        Self {
            num_extra_surfaces: None,
            drop_frame_interval: None,
            cudadec_memtype: None,
            enable_max_performance: None,
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

    pub fn cudadec_memtype(mut self, t: u32) -> Self {
        self.cudadec_memtype = Some(t);
        self
    }

    pub fn enable_max_performance(mut self, v: bool) -> Self {
        self.enable_max_performance = Some(v);
        self
    }

    pub fn low_latency(mut self, v: bool) -> Self {
        self.low_latency = Some(v);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        collect_v4l2_pairs(
            self.num_extra_surfaces,
            self.drop_frame_interval,
            self.cudadec_memtype,
            self.enable_max_performance,
            self.low_latency,
        )
    }
}

impl Default for Av1DecoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

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
    pub fn codec(&self) -> Codec {
        match self {
            Self::H264(_) => Codec::H264,
            Self::Hevc(_) => Codec::Hevc,
            Self::Vp8(_) => Codec::Vp8,
            Self::Vp9(_) => Codec::Vp9,
            Self::Av1(_) => Codec::Av1,
            Self::Jpeg(_) => Codec::Jpeg,
            Self::Png(_) => Codec::Png,
            Self::RawRgba(_) => Codec::RawRgba,
            Self::RawRgb(_) => Codec::RawRgb,
        }
    }
}

fn collect_v4l2_pairs(
    num_extra_surfaces: Option<u32>,
    drop_frame_interval: Option<u32>,
    cudadec_memtype: Option<u32>,
    enable_max_performance: Option<bool>,
    low_latency: Option<bool>,
) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    if let Some(v) = num_extra_surfaces {
        out.push(("num-extra-surfaces", v.to_string()));
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        if let Some(v) = drop_frame_interval {
            out.push(("drop-frame-interval", v.to_string()));
        }
        if let Some(v) = cudadec_memtype {
            out.push(("cudadec-memtype", v.to_string()));
        }
        if enable_max_performance.is_some() {
            warn!("enable_max_performance is only supported on aarch64 (Jetson); ignored on this platform");
        }
        if low_latency.is_some() {
            warn!("low_latency is only supported on aarch64 (Jetson); ignored on this platform");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(v) = enable_max_performance {
            out.push((
                "enable-max-performance",
                if v { "1" } else { "0" }.to_string(),
            ));
        }
        if let Some(v) = low_latency {
            out.push(("disable-dpb", if v { "1" } else { "0" }.to_string()));
        }
        if drop_frame_interval.is_some() {
            warn!("drop_frame_interval is only supported on desktop GPUs; ignored on Jetson");
        }
        if cudadec_memtype.is_some() {
            warn!("cudadec_memtype is only supported on desktop GPUs; ignored on Jetson");
        }
    }
    out
}
