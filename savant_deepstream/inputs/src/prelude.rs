//! Prelude for `deepstream_inputs`.

pub use crate::multistream_decoder::{DecoderOutput, MultiStreamError};
pub use crate::multistream_decoder::{
    EvictionVerdict, MultiStreamDecoder, MultiStreamDecoderConfig, SessionBoundaryEosPolicy,
    StopReason, SubmitResult, UndecodedReason,
};
