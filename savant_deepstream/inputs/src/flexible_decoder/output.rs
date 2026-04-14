//! Callback output types for [`super::FlexibleDecoder`].

use deepstream_buffers::SharedBuffer;
use deepstream_decoders::DecoderError;
use gstreamer as gst;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::utils::release_seal::ReleaseSeal;
use savant_gstreamer::Codec;
use std::sync::Arc;

/// Codec, width and height snapshot for a decoder session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderParameters {
    pub codec: Codec,
    pub width: i64,
    pub height: i64,
}

/// Callback payload emitted by [`super::FlexibleDecoder`].
///
/// # Delivery flow (Frame variant)
///
/// 1. Callback receives `FlexibleDecoderOutput::Frame` — reads metadata on
///    [`VideoFrameProxy`] and scalar fields of [`DecodedFrame`].
/// 2. Callback calls [`take_delivery`](Self::take_delivery) to get a
///    [`SealedDelivery`] containing the `(frame, buffer)` pair.
/// 3. Callback (or its scope end) drops this struct.  [`Drop`] releases the
///    seal.
/// 4. Downstream calls [`SealedDelivery::unseal`] (or `try_unseal`) to
///    obtain the pair — the call blocks until step 3 completes.
#[derive(Debug)]
pub enum FlexibleDecoderOutput {
    /// Decoded RGBA frame from the underlying `NvDecoder`, paired with the
    /// original [`VideoFrameProxy`] that was submitted.
    ///
    /// The [`SharedBuffer`] inside `decoded.buffer` can be extracted via
    /// [`take_delivery`](Self::take_delivery).
    Frame {
        frame: VideoFrameProxy,
        decoded: deepstream_decoders::DecodedFrame,
        #[doc(hidden)]
        seal: Arc<ReleaseSeal>,
    },
    /// Codec or resolution changed.  Old decoder has been drained; new one is
    /// active.
    ParameterChange {
        old: DecoderParameters,
        new: DecoderParameters,
    },
    /// Frame was rejected (not submitted to the decoder).
    ///
    /// `data` is the extracted payload when available (`None` for
    /// `SourceIdMismatch` and `NoPayload` where no extraction happened).
    Skipped {
        frame: VideoFrameProxy,
        data: Option<Vec<u8>>,
        reason: SkipReason,
    },
    /// A decoded frame whose `frame_id` had no matching [`VideoFrameProxy`] in
    /// the frame map.  The GPU buffer is still live so the callback can reclaim
    /// or log it.
    OrphanFrame {
        decoded: deepstream_decoders::DecodedFrame,
    },
    /// Logical per-source EOS, ordered with decoded frames.
    SourceEos { source_id: String },
    /// A downstream GStreamer event captured at the pipeline output.
    Event(gst::Event),
    /// Error from the underlying `NvDecoder`.
    Error(DecoderError),
}

impl FlexibleDecoderOutput {
    /// Extract the `(VideoFrameProxy, SharedBuffer)` delivery from a `Frame`
    /// variant as a [`SealedDelivery`].
    ///
    /// Returns `None` for non-`Frame` variants, or if the buffer has already
    /// been taken by a prior call.
    ///
    /// The [`VideoFrameProxy`] is **cloned** (cheap Arc bump) and the
    /// [`SharedBuffer`] is **taken** from the `DecodedFrame`.  Scalar fields
    /// on `decoded` (`pts_ns`, `codec`, `format`, …) remain readable.
    ///
    /// This mirrors the `take_deliveries()` pattern on the nvinfer / nvtracker
    /// batching operators.  Because `FlexibleDecoder` is single-stream, at most
    /// one delivery is produced per output.
    pub fn take_delivery(&mut self) -> Option<SealedDelivery> {
        match self {
            Self::Frame {
                frame,
                decoded,
                seal,
            } => {
                let buffer = decoded.buffer.take()?;
                Some(SealedDelivery {
                    delivery: Some((frame.clone(), buffer)),
                    seal: Arc::clone(seal),
                })
            }
            _ => None,
        }
    }
}

impl Drop for FlexibleDecoderOutput {
    fn drop(&mut self) {
        if let Self::Frame { seal, .. } = self {
            seal.release();
        }
    }
}

// ---------------------------------------------------------------------------
// SealedDelivery
// ---------------------------------------------------------------------------

/// A single `(VideoFrameProxy, SharedBuffer)` pair sealed until the
/// associated [`FlexibleDecoderOutput`] is dropped.
///
/// # Drop safety
///
/// Dropping `SealedDelivery` without calling `unseal()` is safe — the
/// contained `SharedBuffer` is freed and the `Condvar::notify_all` in
/// [`FlexibleDecoderOutput::drop`] runs against zero waiters (a no-op).
pub struct SealedDelivery {
    delivery: Option<(VideoFrameProxy, SharedBuffer)>,
    seal: Arc<ReleaseSeal>,
}

unsafe impl Send for SealedDelivery {}

impl SealedDelivery {
    /// Whether the seal has been released (non-blocking check).
    pub fn is_released(&self) -> bool {
        self.seal.is_released()
    }

    /// Block until the [`FlexibleDecoderOutput`] is dropped, then return
    /// the `(frame, buffer)` pair.
    pub fn unseal(self) -> Option<(VideoFrameProxy, SharedBuffer)> {
        self.seal.wait();
        self.delivery
    }

    /// Block until the seal is released, with a timeout.
    ///
    /// Returns the delivery if released within `timeout`, or returns
    /// `Err(self)` if the timeout expires (so the caller can retry or drop).
    pub fn unseal_timeout(
        self,
        timeout: std::time::Duration,
    ) -> Result<Option<(VideoFrameProxy, SharedBuffer)>, Self> {
        if self.seal.wait_timeout(timeout) {
            Ok(self.delivery)
        } else {
            Err(self)
        }
    }

    /// Non-blocking attempt to unseal.  Returns `Err(self)` if the seal
    /// has not yet been released.
    pub fn try_unseal(self) -> Result<Option<(VideoFrameProxy, SharedBuffer)>, Self> {
        if self.seal.is_released() {
            Ok(self.delivery)
        } else {
            Err(self)
        }
    }
}

impl std::fmt::Debug for SealedDelivery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SealedDelivery")
            .field("has_delivery", &self.delivery.is_some())
            .field("released", &self.seal.is_released())
            .finish()
    }
}

/// Why a frame was not decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// The frame's `source_id` does not match the decoder's bound `source_id`.
    SourceIdMismatch { expected: String, actual: String },
    /// `VideoCodec` not supported or absent.
    UnsupportedCodec(Option<String>),
    /// Frame arrived before the first keyframe (random access point) during
    /// H.264/HEVC stream detection.  Non-decodable without a preceding IDR.
    WaitingForKeyframe,
    /// Too many frames buffered without finding a random access point;
    /// the detection buffer limit was exceeded.
    DetectionBufferOverflow,
    /// Neither the `data` argument nor `frame.get_content()` contained bytes.
    NoPayload,
    /// Payload failed structural validation for the declared codec
    /// (e.g. garbage data labelled as JPEG/PNG).
    InvalidPayload(String),
    /// `NvDecoder::new()` failed when activating.
    DecoderCreationFailed(String),
}
