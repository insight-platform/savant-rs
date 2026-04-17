//! Prelude for `deepstream_inputs`.

pub use crate::decoder_pool::{EvictionDecision, FlexibleDecoderPool, FlexibleDecoderPoolConfig};
pub use crate::flexible_decoder::{
    DecoderParameters, FlexibleDecoder, FlexibleDecoderConfig, FlexibleDecoderError,
    FlexibleDecoderOutput, SealedDelivery, SkipReason,
};
