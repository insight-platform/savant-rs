//! Framework-level **pipeline envelopes**: the concrete message
//! enum types carried on every inter-actor / source-to-actor
//! channel in the system, together with their per-variant payload
//! wrappers and their [`Envelope`]/[`Dispatch`] integration.
//!
//! This module pairs with [`crate::messages`], which
//! hosts the orthogonal set of **envelope-agnostic per-instance
//! action payloads** ([`ResetStreamPayload`](super::messages::ResetStreamPayload),
//! [`RemoveSourcePayload`](super::messages::RemoveSourcePayload),
//! [`UpdateSourceSpecPayload`](super::messages::UpdateSourceSpecPayload),
//! plus the cross-envelope sentinels [`SourceEosPayload`] and
//! [`ShutdownPayload`]).  The split is:
//!
//! * `messages.rs` — envelope-agnostic per-instance action payloads
//!   and cross-envelope sentinels.  Consumers opt in by adding a
//!   variant to their envelope and routing the payload via
//!   [`Dispatch`].
//! * `envelopes.rs` — the concrete envelope enums themselves
//!   ([`PipelineMsg`], [`EncodedMsg`]) and the per-variant payload
//!   wrappers ([`SingleDelivery`], [`BatchDelivery`],
//!   [`StreamInfoPayload`], [`PacketPayload`], [`FramePayload`])
//!   they destructure into.
//!
//! # Rationale
//!
//! The three middle stages all have structurally-identical ingress
//! channels — each one receives a deliverable, a per-source
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
use std::time::Duration;

use deepstream_buffers::{SealedDeliveries, SharedBuffer};
use deepstream_inputs::prelude::SealedDelivery;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_gstreamer::mp4_demuxer::{DemuxedPacket, VideoInfo};

use super::{Dispatch, Envelope, Flow, Handler, ShutdownHint};

#[doc(inline)]
pub use super::{ShutdownPayload, SourceEosPayload};

/// Unified ingress message type used by the infer, tracker, and
/// picasso stages.  See the module docs for the full rationale.
pub enum PipelineMsg {
    /// A **single** `(frame, buffer)` pair produced by the decoder
    /// pool callback.  Consumed by the infer stage.
    Delivery(SealedDelivery),
    /// A **batched** set of `(frame, buffer)` pairs produced by a
    /// batching operator (nvinfer / nvtracker).  Consumed by the
    /// tracker and picasso stages respectively.
    Deliveries(SealedDeliveries),
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
    /// * [`PipelineMsg::Delivery`]   → exactly 1 pair.
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
            PipelineMsg::Delivery(d) => vec![d.unseal()],
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
    /// A single encoded access unit tagged with its `source_id`
    /// **and the stream-level [`VideoInfo`]** describing the stream
    /// it belongs to.  Produced by:
    ///
    /// * the demuxer — raw container packets for the decoder to
    ///   consume, paired with the `VideoInfo` observed on the most
    ///   recent [`EncodedMsg::StreamInfo`] for the same source.
    /// * the picasso encoder callback — freshly encoded frames for
    ///   the muxer (or any other downstream consumer) to consume.
    ///
    /// Carrying the [`VideoInfo`] in-band makes this envelope
    /// self-describing: downstream consumers (e.g. the decoder)
    /// never have to cache a per-`source_id` map of stream
    /// parameters — every packet arrives with the metadata it needs
    /// to be decoded.  A standalone [`EncodedMsg::StreamInfo`] is
    /// still emitted once per source for observers that want to
    /// react to the stream header itself.
    Packet {
        /// Source id this packet belongs to.
        source_id: String,
        /// Stream-level metadata (width, height, framerate, codec)
        /// for this packet's source, as observed on the preceding
        /// [`EncodedMsg::StreamInfo`].
        info: VideoInfo,
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
    /// exit (or when Ctrl+C fires).  The framework's
    /// [`Context`](crate::Context) translates the sentinel
    /// into receive-loop break flags — either immediately (when
    /// `grace` is `None`) or after the grace deadline elapses.
    Shutdown {
        /// `None` — break after the current message is handled.
        /// `Some(d)` — keep processing, break when `recv_timeout`
        /// next fires at or past `now + d`.
        grace: Option<Duration>,
        /// Human-readable reason (logged on receipt).
        reason: Cow<'static, str>,
    },
}

// ---------------------------------------------------------------------------
// Framework integration — Envelope, Dispatch, per-variant payload wrappers.
// ---------------------------------------------------------------------------
//
// These types bridge the pipeline's envelope enums (`PipelineMsg`,
// `EncodedMsg`) to the generic actor framework in this crate.
//
// * [`Envelope`] tells the receive-loop driver whether an envelope is a
//   cooperative-shutdown sentinel, and how to *build* one for the
//   supervisor's broadcast step.  Both envelopes carry a first-class
//   `Shutdown` variant so `build_shutdown` always returns `Some(_)`.
// * [`Dispatch`] routes each variant to the actor's per-variant
//   [`Handler<V>`] impls.  Actors opt in to a variant by writing an
//   `impl Handler<VariantPayload> for MyActor { … }`; unimplemented
//   variants fall back to the framework's default no-op `handle`
//   returning [`Flow::Cont`].
//
// The per-variant payload structs (`SingleDelivery`, `BatchDelivery`,
// `StreamInfoPayload`, `PacketPayload`, `FramePayload`,
// `SourceEosPayload`, `ShutdownPayload`) are thin wrappers around the
// enum variants' fields.  They keep `impl Handler<…>` callsites
// self-explanatory and mean an actor's signature for each handler
// reflects the variant's exact payload — no destructuring boilerplate.

/// Per-variant payload: a single sealed `(frame, buffer)` pair
/// (from [`PipelineMsg::Delivery`]).
pub struct SingleDelivery(pub SealedDelivery);

impl fmt::Debug for SingleDelivery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SingleDelivery").field(&self.0).finish()
    }
}

/// Per-variant payload: a batched set of `(frame, buffer)` pairs
/// (from [`PipelineMsg::Deliveries`]).
pub struct BatchDelivery(pub SealedDeliveries);

impl fmt::Debug for BatchDelivery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchDelivery")
            .field("len", &self.0.len())
            .field("released", &self.0.is_released())
            .finish()
    }
}

/// Per-variant payload: stream-level metadata tagged with its
/// `source_id` (from [`EncodedMsg::StreamInfo`]).
#[derive(Debug)]
pub struct StreamInfoPayload {
    /// Source id this stream belongs to.
    pub source_id: String,
    /// Stream-level metadata (width, height, framerate, codec).
    pub info: VideoInfo,
}

/// Per-variant payload: a single encoded access unit tagged with its
/// `source_id` and the stream-level [`VideoInfo`] describing its
/// source (from [`EncodedMsg::Packet`]).
#[derive(Debug)]
pub struct PacketPayload {
    /// Source id this packet belongs to.
    pub source_id: String,
    /// Stream-level metadata for this packet's source.  Lets
    /// downstream consumers (decoder, muxer) construct
    /// [`VideoFrameProxy`] values or route by codec without
    /// maintaining a per-source cache of stream parameters.
    pub info: VideoInfo,
    /// The encoded access unit.
    pub packet: DemuxedPacket,
}

/// Per-variant payload: a pre-built [`VideoFrameProxy`] with its
/// encoded bitstream (from [`EncodedMsg::Frame`]).
pub struct FramePayload {
    /// Pre-built frame (source_id, codec, dims, fps, uuid, keyframe
    /// set).
    pub frame: VideoFrameProxy,
    /// Encoded payload, or `None` to have the decoder extract it
    /// from the frame.
    pub payload: Option<Vec<u8>>,
}

impl fmt::Debug for FramePayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FramePayload")
            .field("payload_len", &self.payload.as_ref().map(|p| p.len()))
            .finish()
    }
}

impl Envelope for PipelineMsg {
    fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
        match self {
            PipelineMsg::Shutdown { grace, reason } => Some(ShutdownHint::Graceful {
                grace: *grace,
                reason,
            }),
            _ => None,
        }
    }

    fn build_shutdown(grace: Option<Duration>, reason: Cow<'static, str>) -> Option<Self> {
        Some(PipelineMsg::Shutdown { grace, reason })
    }
}

impl<A> Dispatch<A> for PipelineMsg
where
    A: super::Actor<Msg = PipelineMsg>
        + Handler<SingleDelivery>
        + Handler<BatchDelivery>
        + Handler<SourceEosPayload>
        + Handler<ShutdownPayload>,
{
    fn dispatch(self, actor: &mut A, ctx: &mut super::Context<A>) -> anyhow::Result<Flow> {
        match self {
            PipelineMsg::Delivery(d) => {
                <A as Handler<SingleDelivery>>::handle(actor, SingleDelivery(d), ctx)
            }
            PipelineMsg::Deliveries(b) => {
                <A as Handler<BatchDelivery>>::handle(actor, BatchDelivery(b), ctx)
            }
            PipelineMsg::SourceEos { source_id } => {
                <A as Handler<SourceEosPayload>>::handle(actor, SourceEosPayload { source_id }, ctx)
            }
            PipelineMsg::Shutdown { grace, reason } => <A as Handler<ShutdownPayload>>::handle(
                actor,
                ShutdownPayload { grace, reason },
                ctx,
            ),
        }
    }
}

impl Envelope for EncodedMsg {
    fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
        match self {
            EncodedMsg::Shutdown { grace, reason } => Some(ShutdownHint::Graceful {
                grace: *grace,
                reason,
            }),
            _ => None,
        }
    }

    fn build_shutdown(grace: Option<Duration>, reason: Cow<'static, str>) -> Option<Self> {
        Some(EncodedMsg::Shutdown { grace, reason })
    }
}

impl<A> Dispatch<A> for EncodedMsg
where
    A: super::Actor<Msg = EncodedMsg>
        + Handler<StreamInfoPayload>
        + Handler<PacketPayload>
        + Handler<FramePayload>
        + Handler<SourceEosPayload>
        + Handler<ShutdownPayload>,
{
    fn dispatch(self, actor: &mut A, ctx: &mut super::Context<A>) -> anyhow::Result<Flow> {
        match self {
            EncodedMsg::StreamInfo { source_id, info } => {
                <A as Handler<StreamInfoPayload>>::handle(
                    actor,
                    StreamInfoPayload { source_id, info },
                    ctx,
                )
            }
            EncodedMsg::Packet {
                source_id,
                info,
                packet,
            } => <A as Handler<PacketPayload>>::handle(
                actor,
                PacketPayload {
                    source_id,
                    info,
                    packet,
                },
                ctx,
            ),
            EncodedMsg::Frame { frame, payload } => {
                <A as Handler<FramePayload>>::handle(actor, FramePayload { frame, payload }, ctx)
            }
            EncodedMsg::SourceEos { source_id } => {
                <A as Handler<SourceEosPayload>>::handle(actor, SourceEosPayload { source_id }, ctx)
            }
            EncodedMsg::Shutdown { grace, reason } => <A as Handler<ShutdownPayload>>::handle(
                actor,
                ShutdownPayload { grace, reason },
                ctx,
            ),
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
    /// [`PipelineMsg::Shutdown`] surfaces a graceful shutdown hint;
    /// every other variant is non-shutdown.
    #[test]
    fn pipeline_msg_envelope_shutdown_hint() {
        let sd = PipelineMsg::Shutdown {
            grace: Some(Duration::from_millis(10)),
            reason: Cow::Borrowed("bye"),
        };
        let hint = sd.as_shutdown().expect("shutdown must hint");
        match hint {
            ShutdownHint::Graceful { grace, reason } => {
                assert_eq!(grace, Some(Duration::from_millis(10)));
                assert_eq!(reason, "bye");
            }
            other => panic!("unexpected hint: {other:?}"),
        }
        let eos = PipelineMsg::SourceEos {
            source_id: "s".into(),
        };
        assert!(eos.as_shutdown().is_none());
    }

    /// [`EncodedMsg::Shutdown`] surfaces a graceful shutdown hint;
    /// every other variant is non-shutdown.
    #[test]
    fn encoded_msg_envelope_shutdown_hint() {
        let sd = EncodedMsg::Shutdown {
            grace: None,
            reason: Cow::Borrowed("stop"),
        };
        let hint = sd.as_shutdown().expect("shutdown must hint");
        match hint {
            ShutdownHint::Graceful { grace, reason } => {
                assert_eq!(grace, None);
                assert_eq!(reason, "stop");
            }
            other => panic!("unexpected hint: {other:?}"),
        }
        let eos = EncodedMsg::SourceEos {
            source_id: "s".into(),
        };
        assert!(eos.as_shutdown().is_none());
    }

    /// `Envelope::build_shutdown` round-trips into a shutdown
    /// variant that `as_shutdown` classifies as graceful.
    #[test]
    fn envelope_build_shutdown_round_trips() {
        let m = <PipelineMsg as Envelope>::build_shutdown(
            Some(Duration::from_millis(7)),
            Cow::Borrowed("r1"),
        )
        .expect("must construct");
        let hint = m.as_shutdown().expect("must be shutdown");
        assert!(matches!(
            hint,
            ShutdownHint::Graceful { grace: Some(_), .. }
        ));

        let m = <EncodedMsg as Envelope>::build_shutdown(None, Cow::Borrowed("r2"))
            .expect("must construct");
        let hint = m.as_shutdown().expect("must be shutdown");
        assert!(matches!(hint, ShutdownHint::Graceful { grace: None, .. }));
    }
}
