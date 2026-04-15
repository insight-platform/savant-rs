//! Multi-stream pool of [`FlexibleDecoder`](crate::flexible_decoder::FlexibleDecoder)
//! instances with per-source routing and TTL-based eviction.
//!
//! See [`FlexibleDecoderPool`] for the public API.

mod config;
mod decoder;

pub use config::{EvictionDecision, FlexibleDecoderPoolConfig};
pub use decoder::FlexibleDecoderPool;
