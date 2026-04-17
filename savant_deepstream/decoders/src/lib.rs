pub mod config;
pub(crate) mod error;
pub mod pipeline;
pub mod prelude;
pub mod stream_detect;

pub use config::{
    Av1DecoderConfig, DecoderConfig, H264DecoderConfig, H264StreamFormat, HevcDecoderConfig,
    HevcStreamFormat, JpegBackend, JpegDecoderConfig, NvDecoderConfig, PngDecoderConfig,
    RawRgbDecoderConfig, RawRgbaDecoderConfig, Vp8DecoderConfig, Vp9DecoderConfig,
};

#[cfg(not(target_arch = "aarch64"))]
pub use config::CudadecMemtype;
pub use error::DecoderError;
pub use pipeline::{NvDecoder, NvDecoderExt, NvDecoderOutput};
pub use stream_detect::{detect_stream_config, is_random_access_point};

pub use deepstream_buffers::{
    cuda_init, BufferGenerator, SharedBuffer, SurfaceView, TransformConfig, VideoFormat,
};
pub use savant_gstreamer::Codec;

#[derive(Debug)]
pub struct DecodedFrame {
    pub frame_id: Option<u128>,
    pub pts_ns: u64,
    pub dts_ns: Option<u64>,
    pub duration_ns: Option<u64>,
    /// Decoded RGBA buffer.  Wrapped in `Option` so downstream can `.take()`
    /// the buffer for delivery without consuming the entire frame.
    pub buffer: Option<SharedBuffer>,
    pub codec: Codec,
    pub format: VideoFormat,
}
