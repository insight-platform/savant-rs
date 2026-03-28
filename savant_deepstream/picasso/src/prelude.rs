//! Convenience re-exports for all public types needed to drive the Picasso
//! pipeline.
//!
//! ```ignore
//! use picasso::prelude::*;
//! ```

// Callback traits, aggregate holder, and builder
pub use crate::callbacks::{
    Callbacks, CallbacksBuilder, OnBypassFrame, OnEncodedFrame, OnEviction, OnGpuMat,
    OnObjectDrawSpec, OnRender, OnStreamReset, StreamResetReason,
};

// Engine
pub use crate::engine::PicassoEngine;

// Errors
pub use crate::error::PicassoError;

// Messages / output types
pub use crate::message::OutputMessage;

// Specifications and builders
pub use crate::spec::{
    CallbackInvocationOrder, CodecSpec, ConditionalSpec, EvictionDecision, GeneralSpec,
    GeneralSpecBuilder, ObjectDrawSpec, PtsResetPolicy, SourceSpec, SourceSpecBuilder,
};

pub use crate::transform::LetterboxParams;

pub use deepstream_buffers::{
    DstPadding, Rect, SharedBuffer, SurfaceView, TransformConfig, TransformConfigBuilder,
    VideoFormat,
};
