//! Batched NvBufSurface buffers.
//!
//! - [`uniform`]: Homogeneous batched buffers (`UniformBatchGenerator`, `SurfaceBatch`)
//! - [`non_uniform`]: Zero-copy heterogeneous batches (`NonUniformBatch`)

mod non_uniform;
mod uniform;

pub use non_uniform::*;
pub use uniform::*;
