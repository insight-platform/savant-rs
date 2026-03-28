//! Convenience re-exports for all public types needed to drive the Picasso
//! pipeline.
//!
//! ```ignore
//! use picasso::prelude::*;
//! ```

// Callback traits and aggregate holder
pub use crate::callbacks::{
    Callbacks, OnBypassFrame, OnEncodedFrame, OnEviction, OnGpuMat, OnObjectDrawSpec, OnRender,
    OnStreamReset, StreamResetReason,
};

// Engine
pub use crate::engine::PicassoEngine;

// Errors
pub use crate::error::PicassoError;

// Messages / output types
pub use crate::message::OutputMessage;

// Specifications
pub use crate::spec::{
    CallbackInvocationOrder, CodecSpec, ConditionalSpec, EvictionDecision, GeneralSpec,
    ObjectDrawSpec, PtsResetPolicy, SourceSpec,
};
