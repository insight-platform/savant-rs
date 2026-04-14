//! Errors returned by [`super::FlexibleDecoder`] methods.

use deepstream_decoders::DecoderError;

/// Errors returned by [`super::FlexibleDecoder`] methods.
#[derive(Debug, thiserror::Error)]
pub enum FlexibleDecoderError {
    #[error("decoder is shut down")]
    ShutDown,
    #[error("worker thread exited unexpectedly")]
    WorkerDied,
    #[error("decoder error: {0}")]
    Decoder(#[from] DecoderError),
}
