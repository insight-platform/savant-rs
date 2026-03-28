//! NvBufSurface buffer types and generators.
//!
//! - [`single`]: Single-frame buffers (`BufferGenerator`)
//! - [`batched`]: Batched buffers (uniform and non-uniform)

mod batched;
mod single;

pub use batched::*;
pub use single::*;
