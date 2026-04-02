pub mod config;
pub(crate) mod decoder;
pub(crate) mod error;
pub mod prelude;
pub mod stream_detect;

pub use config::{
    Av1DecoderConfig, DecoderConfig, H264DecoderConfig, H264StreamFormat, HevcDecoderConfig,
    HevcStreamFormat, JpegBackend, JpegDecoderConfig, PngDecoderConfig, RawRgbDecoderConfig,
    RawRgbaDecoderConfig, Vp8DecoderConfig, Vp9DecoderConfig,
};
pub use decoder::NvDecoder;
pub use error::DecoderError;
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
    pub buffer: SharedBuffer,
    pub codec: Codec,
    pub format: VideoFormat,
}

/// Events delivered via the decoder callback.
pub enum DecoderEvent {
    /// A decoded frame is ready.
    Frame(DecodedFrame),
    /// End-of-stream reached — all frames have been delivered.
    Eos,
    /// The internal pipeline encountered an error.
    Error(DecoderError),
    /// The pipeline was restarted after a fatal error.
    PipelineRestarted {
        reason: String,
        lost_frame_count: usize,
    },
}
