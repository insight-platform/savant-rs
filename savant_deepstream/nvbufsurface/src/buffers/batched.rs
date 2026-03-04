//! Batched NvBufSurface buffers.
//!
//! - [`uniform`]: Homogeneous batched buffers (`DsNvUniformSurfaceBufferGenerator`, `DsNvUniformSurfaceBuffer`)
//! - [`non_uniform`]: Zero-copy heterogeneous batches (`DsNvNonUniformSurfaceBuffer`)
//! - [`slot_view`]: Zero-copy single-frame extraction from a batch ([`extract_slot_view`])

mod non_uniform;
mod slot_view;
mod uniform;

pub use non_uniform::*;
pub use slot_view::*;
pub use uniform::*;
