//! NvBufSurface buffer types and generators.
//!
//! - [`single`]: Single-frame buffers (`BufferGenerator`)
//! - [`batched`]: Batched buffers (uniform and non-uniform)

mod batch_state;
mod batched;
mod single;

pub use batch_state::BatchState;
pub use batched::*;
pub use single::*;
