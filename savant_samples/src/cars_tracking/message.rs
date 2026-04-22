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
//! The three middle-stage channels all carry [`PipelineMsg`] —
//! function signatures, struct fields, and the orchestrator use bare
//! `Sender<PipelineMsg>` / `Receiver<PipelineMsg>`.  The two *edge*
//! channels (demux → decoder and picasso → mux) both carry encoded
//! bitstream payloads, and therefore share a single [`EncodedMsg`]
//! type defined in this module — one canonical shape for a
//! source-tagged access unit + the same per-source `SourceEos` and
//! pipeline-wide `Shutdown` sentinels that appear on every
//! inter-actor channel.

use std::borrow::Cow;
use std::fmt;
use std::time::{Duration, Instant};

use deepstream_buffers::SharedBuffer;
use deepstream_inputs::prelude::SealedDelivery;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_gstreamer::mp4_demuxer::{DemuxedPacket, VideoInfo};

use super::supervisor::StageName;

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

/// Encoded-bitstream message type carried on two of the pipeline's
/// inter-actor channels:
///
/// * **demux → decoder** — `Mp4Demuxer` emits
///   [`EncodedMsg::StreamInfo`] followed by a stream of
///   [`EncodedMsg::Packet`] (and/or [`EncodedMsg::Frame`] for the
///   pre-built-frame entrypoint), terminated by
///   [`EncodedMsg::SourceEos`].
/// * **picasso → mux (and any other downstream consumer)** — the
///   `OnEncodedFrame` callback emits [`EncodedMsg::Packet`] with the
///   freshly encoded access unit and forwards the picasso-side
///   [`OutputMessage::EndOfStream`](savant_deepstream_picasso::prelude::OutputMessage::EndOfStream)
///   as [`EncodedMsg::SourceEos`] so the terminus stays oblivious to
///   the source multiplexing strategy.
///
/// Reusing one type across both edges keeps the "upstream → downstream"
/// protocol uniform: every consumer handles the same variants, and
/// source id propagation is carried in-band by the message itself
/// (the producer never has to know what the consumer will do with
/// the id).
#[derive(Debug)]
pub enum EncodedMsg {
    /// Stream-level metadata (width, height, framerate, codec)
    /// tagged with the `source_id` it describes.  Emitted exactly
    /// once per source by the demuxer, *before* any
    /// [`EncodedMsg::Packet`] for that source.  Consumers that do
    /// not need stream metadata (e.g. the muxer) simply ignore it.
    StreamInfo {
        /// Source id this stream belongs to.
        source_id: String,
        /// Stream-level metadata (width, height, framerate, codec).
        info: VideoInfo,
    },
    /// A single encoded access unit tagged with its `source_id`.
    /// Produced by:
    ///
    /// * the demuxer — raw container packets for the decoder to
    ///   consume;
    /// * the picasso encoder callback — freshly encoded frames for
    ///   the muxer (or any other downstream consumer) to consume.
    ///
    /// Tagging every packet (rather than relying on a preceding
    /// [`EncodedMsg::StreamInfo`]) keeps the protocol
    /// self-describing: a multiplexed consumer can look up the
    /// matching [`VideoInfo`] for each packet without assuming a
    /// single-source pipeline.
    Packet {
        /// Source id this packet belongs to.
        source_id: String,
        /// The encoded access unit.
        packet: DemuxedPacket,
    },
    /// A pre-built [`VideoFrameProxy`] delivered with its encoded
    /// payload, ready for direct submission to the decoder.
    ///
    /// Unlike [`EncodedMsg::Packet`] — which is raw container output
    /// that the decode actor wraps into a fresh frame — this variant
    /// lets an upstream producer hand off a fully-populated frame
    /// (source id, dimensions, codec, fps, uuid, keyframe,
    /// timestamps) and side-step the `StreamInfo` path entirely.
    ///
    /// Payload resolution follows the `FlexibleDecoder::submit`
    /// contract:
    ///
    /// * `payload = Some(bytes)` — the decoder uses `bytes` directly.
    /// * `payload = None` — the decoder extracts the bitstream from
    ///   the frame's internal content
    ///   ([`VideoFrameContent::Internal`](savant_core::primitives::frame::VideoFrameContent)).
    ///   If the frame's content is `External` or `None`, the decoder
    ///   emits a `Skipped { NoPayload }` callback rather than
    ///   erroring.
    ///
    /// This is the entrypoint for producers that already own the
    /// frame (e.g. reading previously-captured `savant_core` frames
    /// back from storage, or bridging from another message-bus
    /// format) without paying the cost of re-deriving frame
    /// metadata from demuxer caps.  The muxer ignores this variant.
    Frame {
        /// Pre-built frame.  Its `source_id`, `codec`, `width`,
        /// `height`, `fps`, `uuid`, and `keyframe` fields must
        /// already be set.
        frame: VideoFrameProxy,
        /// Encoded bitstream for this frame, or `None` to have the
        /// decoder extract it from `frame.get_content()`.
        payload: Option<Vec<u8>>,
    },
    /// End-of-stream sentinel carrying the source id.  Downstream
    /// consumers react per their role:
    ///
    /// * decoder — runs `decoder.source_eos(source_id)` + drain,
    ///   then forwards an equivalent in-band
    ///   [`PipelineMsg::SourceEos`] sentinel;
    /// * muxer   — finalises the `moov` atom and exits (in a
    ///   single-source pipeline the first `SourceEos` is also the
    ///   last).
    SourceEos {
        /// Source id this EOS belongs to.
        source_id: String,
    },
    /// Cooperative shutdown sentinel, broadcast by the orchestrator
    /// after the supervisor back-channel observes a shutdown-worthy
    /// exit (or when Ctrl+C fires).  Consumers exit via
    /// [`apply_shutdown_signal`] — either immediately (when `grace`
    /// is `None`) or after the grace deadline elapses.
    Shutdown {
        /// `None` — break after the current message is handled.
        /// `Some(d)` — keep processing, break when `recv_timeout`
        /// next fires at or past `now + d`.
        grace: Option<Duration>,
        /// Human-readable reason (logged on receipt).
        reason: Cow<'static, str>,
    },
}

/// Apply a received [`PipelineMsg::Shutdown`] to the caller's
/// receive-loop shutdown state — shared by every actor so shutdown
/// semantics (deadline bookkeeping, log wording) stay uniform across
/// stages.
///
/// This does **not** perform the shutdown itself.  It only translates
/// the `Shutdown { grace, reason }` sentinel into mutations on the
/// caller's `deadline` / `break_now` locals; the receive loop
/// observes them on its next iteration and decides when to exit.
///
/// Invoked with the destructured `grace` and `reason` fields of
/// [`PipelineMsg::Shutdown`] plus the caller's local receive-loop
/// state:
///
/// * `grace = None` — sets `*break_now = true`; the caller exits
///   after finishing the current message.
/// * `grace = Some(d)` — sets `*deadline = min(*deadline,
///   Instant::now() + d)`; the caller keeps running until the
///   deadline is observed expired on a subsequent iteration.  The
///   earliest deadline always wins — a later, longer grace never
///   pushes an already-recorded deadline into the future.
///
/// `stage` is the actor's [`StageName`] set at construction time
/// (kind + optional instance tag).  It's used verbatim as the log
/// prefix so concurrent instances of the same kind (e.g.
/// `infer[yolo11n]` vs `infer[person_attr]`) remain distinguishable
/// in the pipeline's logs.
pub fn apply_shutdown_signal(
    stage: &StageName,
    grace: Option<Duration>,
    reason: &str,
    deadline: &mut Option<Instant>,
    break_now: &mut bool,
) {
    match grace {
        None => {
            log::info!(
                "[{stage}] Shutdown (reason={reason}, grace=none); exiting after current message"
            );
            *break_now = true;
        }
        Some(d) => {
            let new_deadline = Instant::now() + d;
            *deadline = Some(match *deadline {
                Some(existing) if existing < new_deadline => existing,
                _ => new_deadline,
            });
            log::info!(
                "[{stage}] Shutdown (reason={reason}, grace={d:?}); deadline set, continuing"
            );
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cars_tracking::supervisor::StageKind;

    /// `apply_shutdown_signal` records a deadline on graceful
    /// shutdown and sets `break_now` on instant shutdown.  A later,
    /// longer grace must not push an already-recorded deadline
    /// further into the future — earliest wins.
    #[test]
    fn apply_shutdown_signal_semantics() {
        let stage = StageName::unnamed(StageKind::Infer);
        let mut deadline: Option<Instant> = None;
        let mut break_now = false;

        apply_shutdown_signal(
            &stage,
            Some(Duration::from_secs(1)),
            "graceful",
            &mut deadline,
            &mut break_now,
        );
        assert!(deadline.is_some());
        assert!(!break_now);

        apply_shutdown_signal(&stage, None, "instant", &mut deadline, &mut break_now);
        assert!(break_now);

        let earlier = deadline;
        apply_shutdown_signal(
            &stage,
            Some(Duration::from_secs(5)),
            "later-graceful",
            &mut deadline,
            &mut break_now,
        );
        assert_eq!(deadline, earlier, "earliest deadline must win");
    }
}
