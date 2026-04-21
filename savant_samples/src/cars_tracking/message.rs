//! Unified ingress message type shared by the infer, tracker, and
//! picasso stages.
//!
//! # Rationale
//!
//! The three middle stages all have structurally-identical ingress
//! channels — they each receive some deliverable, a per-source
//! [`SourceEos`](PipelineMsg::SourceEos) sentinel, and a cooperative
//! [`Shutdown`](PipelineMsg::Shutdown) sentinel broadcast by the
//! orchestrator.  The only thing that varies is the shape of the
//! deliverable itself:
//!
//! * decoder pool → infer: a **single** `(VideoFrameProxy, SharedBuffer)`
//!   pair ([`PipelineMsg::Delivery`]).
//! * nvinfer → tracker: a **batched** set of pairs, sealed by
//!   `deepstream_nvinfer::SealedDeliveries`.
//! * nvtracker → picasso: a **batched** set of pairs, sealed by
//!   `deepstream_nvtracker::SealedDeliveries`.
//!
//! To share a single message type across all three ingress channels
//! we keep the singular and batched deliveries as two distinct
//! variants (`Delivery` and `Deliveries`) and type-erase the two
//! batched sealed-delivery structs behind the [`Deliveries`] trait.
//! Both batched structs already expose the exact same public surface
//! (`len`, `is_empty`, `is_released`, `unseal`), so the trait is a
//! mechanical, behaviour-preserving mapping.
//!
//! Every inter-stage channel in `cars_tracking` carries
//! [`PipelineMsg`] directly; only the typed `Sender`/`Receiver`
//! aliases (e.g. [`DecodedSender`](super::pipeline::decoder::DecodedSender),
//! [`InferResultSender`](super::pipeline::infer::output::InferResultSender),
//! [`TrackerResultSender`](super::tracker::TrackerResultSender))
//! survive per-stage, as self-documenting names at each channel
//! boundary.

use std::borrow::Cow;
use std::fmt;
use std::time::Duration;

use deepstream_buffers::SharedBuffer;
use deepstream_inputs::prelude::SealedDelivery;
use savant_core::primitives::frame::VideoFrameProxy;

/// Type-erasure trait for batched sealed-delivery payloads.
///
/// Implemented for both `deepstream_nvinfer::SealedDeliveries` and
/// `deepstream_nvtracker::SealedDeliveries`, which are structurally
/// identical but nominally distinct types.  The trait mirrors the
/// subset of their public surface that the sample actually uses — the
/// batched consumers (`tracker`, `picasso`) only ever call
/// [`Deliveries::unseal`] on the payload.
pub trait Deliveries: Send {
    /// Number of `(frame, buffer)` pairs in the batch.
    fn len(&self) -> usize;

    /// Whether the batch is empty.
    fn is_empty(&self) -> bool;

    /// Non-blocking check: whether the underlying seal has been
    /// released by the producing operator.
    fn is_released(&self) -> bool;

    /// Block until the seal is released, then return all pairs.
    ///
    /// Consumes the box; the returned vector is the sole owner of
    /// every `SharedBuffer` it contains (matching the semantics of
    /// the underlying `SealedDeliveries::unseal`).
    fn unseal(self: Box<Self>) -> Vec<(VideoFrameProxy, SharedBuffer)>;
}

impl Deliveries for deepstream_nvinfer::SealedDeliveries {
    #[inline]
    fn len(&self) -> usize {
        Self::len(self)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        Self::is_empty(self)
    }

    #[inline]
    fn is_released(&self) -> bool {
        Self::is_released(self)
    }

    #[inline]
    fn unseal(self: Box<Self>) -> Vec<(VideoFrameProxy, SharedBuffer)> {
        (*self).unseal()
    }
}

impl Deliveries for deepstream_nvtracker::SealedDeliveries {
    #[inline]
    fn len(&self) -> usize {
        Self::len(self)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        Self::is_empty(self)
    }

    #[inline]
    fn is_released(&self) -> bool {
        Self::is_released(self)
    }

    #[inline]
    fn unseal(self: Box<Self>) -> Vec<(VideoFrameProxy, SharedBuffer)> {
        (*self).unseal()
    }
}

/// Boxed, type-erased batched deliveries.  Alias kept for brevity at
/// producer/consumer sites.
pub type BoxedDeliveries = Box<dyn Deliveries>;

/// Unified ingress message type used by the infer, tracker, and
/// picasso stages.  See the module docs for the full rationale.
pub enum PipelineMsg {
    /// A **single** `(frame, buffer)` pair produced by the decoder
    /// pool callback.  Consumed by the infer stage.
    Delivery(SealedDelivery),
    /// A **batched** set of `(frame, buffer)` pairs produced by a
    /// batching operator (nvinfer / nvtracker).  Consumed by the
    /// tracker and picasso stages respectively.
    Deliveries(BoxedDeliveries),
    /// In-band end-of-source sentinel carrying the source id.
    ///
    /// Upstream emits this *after* its operator has drained; the
    /// consumer runs its own local `send_eos` + drain on receipt and
    /// then forwards an equivalent sentinel downstream.
    SourceEos {
        /// Source id this EOS belongs to.
        source_id: String,
    },
    /// Cooperative shutdown sentinel broadcast by the orchestrator
    /// after the terminus has joined (or on Ctrl+C).
    ///
    /// * `grace = None` — break after the current message is
    ///   handled.
    /// * `grace = Some(d)` — set a deadline; break the next time
    ///   the receive-loop observes it has expired.
    Shutdown {
        /// Grace period (see variant docs).
        grace: Option<Duration>,
        /// Human-readable reason, logged on receipt.
        reason: Cow<'static, str>,
    },
}

impl PipelineMsg {
    /// Normalize a delivery-carrying message into a flat
    /// `Vec<(VideoFrameProxy, SharedBuffer)>`.
    ///
    /// * [`PipelineMsg::Delivery`]   → 0 or 1 pairs (depending on
    ///   whether the underlying `SealedDelivery` carried a payload).
    /// * [`PipelineMsg::Deliveries`] → the full batch.
    /// * [`PipelineMsg::SourceEos`] / [`PipelineMsg::Shutdown`] → empty
    ///   vec (these variants are expected to be filtered out by the
    ///   caller first; returning empty is a safe no-op).
    ///
    /// This is the shared normalization step that lets every
    /// ingress consumer (`infer`, `tracker`, `picasso`, `drain`)
    /// handle either delivery shape with a single code path —
    /// iterate the vec and feed each pair downstream.
    pub fn into_pairs(self) -> Vec<(VideoFrameProxy, SharedBuffer)> {
        match self {
            PipelineMsg::Delivery(d) => d.unseal().into_iter().collect(),
            PipelineMsg::Deliveries(d) => d.unseal(),
            PipelineMsg::SourceEos { .. } | PipelineMsg::Shutdown { .. } => Vec::new(),
        }
    }

    /// `true` for [`PipelineMsg::Delivery`] and
    /// [`PipelineMsg::Deliveries`]; `false` for the sentinel variants.
    pub fn is_delivery(&self) -> bool {
        matches!(self, PipelineMsg::Delivery(_) | PipelineMsg::Deliveries(_))
    }
}

impl fmt::Debug for PipelineMsg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineMsg::Delivery(d) => f.debug_tuple("Delivery").field(d).finish(),
            PipelineMsg::Deliveries(d) => f
                .debug_struct("Deliveries")
                .field("len", &d.len())
                .field("released", &d.is_released())
                .finish(),
            PipelineMsg::SourceEos { source_id } => f
                .debug_struct("SourceEos")
                .field("source_id", source_id)
                .finish(),
            PipelineMsg::Shutdown { grace, reason } => f
                .debug_struct("Shutdown")
                .field("grace", grace)
                .field("reason", reason)
                .finish(),
        }
    }
}
