//! Batched NvBufSurface buffers.
//!
//! - [`uniform`]: Homogeneous batched buffers (`BatchedNvBufSurfaceGenerator`, `BatchedSurface`)
//! - [`non_uniform`]: Zero-copy heterogeneous batches (`HeterogeneousBatch`)

mod non_uniform;
mod uniform;

pub use non_uniform::*;
pub use uniform::*;
