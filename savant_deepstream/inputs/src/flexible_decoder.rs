//! Single-stream adaptive GPU decoder: one [`NvDecoder`] per codec/resolution
//! session, automatically replaced on parameter changes.
//!
//! Call [`FlexibleDecoder::submit`] with a [`VideoFrameProxy`] reference and an
//! optional byte slice; the decoder detects codec and resolution from frame
//! metadata, creates / drains / replaces the underlying hardware decoder as
//! needed, and delivers all results through a single callback.
//!
//! [`VideoFrameProxy`]: savant_core::primitives::frame::VideoFrameProxy

mod config;
mod decoder;
mod error;
mod output;

pub use config::FlexibleDecoderConfig;
pub use decoder::FlexibleDecoder;
pub use error::FlexibleDecoderError;
pub use output::{
    DecoderParameters, FlexibleDecoderOutput, SkipReason, DEFAULT_DETECT_BUFFER_LIMIT,
    DEFAULT_IDLE_TIMEOUT_SECS,
};
