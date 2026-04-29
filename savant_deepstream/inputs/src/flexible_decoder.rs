//! Single-stream adaptive GPU decoder: one `NvDecoder` per codec/resolution
//! session, automatically replaced on parameter changes.
//!
//! Call [`FlexibleDecoder::submit`] with a [`VideoFrame`] reference and an
//! optional byte slice; the decoder detects codec and resolution from frame
//! metadata, creates / drains / replaces the underlying hardware decoder as
//! needed, and delivers all results through a single callback.
//!
//! [`VideoFrame`]: savant_core::primitives::frame::VideoFrame

mod config;
mod decoder;
mod error;
mod handle_active;
mod handle_detecting;
mod handle_idle;
mod output;
mod payload_validate;
mod pool_cache;
mod state;

pub use config::{DecoderConfigCallback, FlexibleDecoderConfig};
pub use decoder::FlexibleDecoder;
pub use deepstream_decoders::DecodedFrame;
pub use error::FlexibleDecoderError;
pub use output::{DecoderParameters, FlexibleDecoderOutput, SealedDelivery, SkipReason};
pub use pool_cache::PoolCacheRegistry;
