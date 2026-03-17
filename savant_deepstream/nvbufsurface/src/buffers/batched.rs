//! Batched NvBufSurface buffers.
//!
//! - [`uniform`]: Homogeneous batched buffers (`DsNvUniformSurfaceBufferGenerator`, `DsNvUniformSurfaceBuffer`)
//! - [`non_uniform`]: Zero-copy heterogeneous batches (`DsNvNonUniformSurfaceBuffer`)

mod non_uniform;
mod uniform;

pub use non_uniform::*;
pub use uniform::*;
