//! Multi-stream GPU decoder: one [`NvDecoder`] per `source_id`, shared output callback.

mod codec_resolve;
mod config;
mod decoder;
mod error;
mod frame_tracker;
mod stream_slot;
mod watchdog;

pub use config::{MultiStreamDecoderConfig, SessionBoundaryEosPolicy};
pub use decoder::{MultiStreamDecoder, SubmitResult};
pub use deepstream_buffers::Interpolation;
pub use error::{DecoderOutput, EvictionVerdict, MultiStreamError, StopReason, UndecodedReason};
