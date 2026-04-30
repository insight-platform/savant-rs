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
//! Pipelines built on this framework typically have two distinct
//! groups of inter-actor channels:
//!
//! * **Decoded-data channels** carry deliverables of decoded
//!   frames plus their backing GPU buffers — either as a single
//!   `(VideoFrame, SharedBuffer)` pair
//!   ([`PipelineMsg::Delivery`]) or as a batched set of such
//!   pairs ([`PipelineMsg::Deliveries`]).  Both variants share a
//!   single envelope ([`PipelineMsg`]) so any actor that
//!   consumes decoded data can accept either shape uniformly.
//!   The two batched sealed-delivery shapes used in practice
//!   (nvinfer's and nvtracker's outputs) are type-erased behind
//!   the [`Deliveries`](deepstream_buffers::SealedDeliveries)
//!   abstraction so [`PipelineMsg::Deliveries`] is independent
//!   of any specific operator.
//! * **Encoded-data channels** carry encoded access units — raw
//!   container packets from a demuxer, fully-encoded frames from
//!   an encoder, or pre-built [`VideoFrame`] values handed
//!   in by a custom producer.  These all share a single envelope
//!   ([`EncodedMsg`]) so the same actor stage (e.g. a muxer
//!   terminus, a ZeroMQ sink) can consume output from any
//!   producer.
//!
//! Both envelopes also carry a per-source `SourceEos` sentinel
//! and a pipeline-wide `Shutdown` sentinel.  The two sentinels
//! are sized to the same role on every channel, so framework
//! supervision (in-band drain on `SourceEos`, cooperative
//! `Shutdown` broadcast) is uniform across the whole pipeline.

use std::borrow::Cow;
use std::fmt;
use std::time::Duration;

#[cfg(feature = "deepstream")]
use deepstream_buffers::{SealedDeliveries, SharedBuffer};
#[cfg(feature = "deepstream")]
use deepstream_inputs::prelude::SealedDelivery;
use savant_core::primitives::frame::VideoFrame;
use savant_gstreamer::mp4_demuxer::{DemuxedPacket, VideoInfo};

use super::{Dispatch, Envelope, Flow, Handler, ShutdownHint};

#[doc(inline)]
pub use super::{ShutdownPayload, SourceEosPayload};

/// Decoded-data envelope, used wherever an inter-actor channel
/// carries decoded frames plus their backing GPU buffers.  See
/// the module docs for the surrounding rationale.
///
/// Available only with the `deepstream` Cargo feature; the
/// envelope's payload types ([`SealedDelivery`],
/// [`SealedDeliveries`], [`SharedBuffer`]) come from the
/// DeepStream-side crates.
#[cfg(feature = "deepstream")]
pub enum PipelineMsg {
    /// A **single** `(frame, buffer)` pair.  Typical producer:
    /// a per-frame decoder callback that does not batch.
    Delivery(SealedDelivery),
    /// A **batched** set of `(frame, buffer)` pairs.  Typical
    /// producers: batching operators such as nvinfer or
    /// nvtracker.
    Deliveries(SealedDeliveries),
    /// In-band end-of-source sentinel carrying the source id.
    ///
    /// Upstream is expected to emit this *after* its own per-source
    /// drain; consumers run their own local drain on receipt and
    /// then forward an equivalent sentinel downstream so the
    /// per-source drain sequence reaches every stage in order.
    SourceEos {
        /// Source id this EOS belongs to.
        source_id: String,
    },
    /// Cooperative shutdown sentinel.  Sent by the supervisor
    /// when its [`ShutdownHandler`](crate::shutdown::ShutdownHandler)
    /// returns a broadcast action.
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
    /// Type-erased extensibility variant.  Carries an arbitrary
    /// receiver-specific payload through the standard envelope
    /// without forcing the envelope type to be parametric.
    /// See [`MessageExPayload`](crate::message_ex::MessageExPayload)
    /// for the full contract; the [`Dispatch`] impl routes this
    /// variant through
    /// [`Actor::handle_message_ex`](super::actor::Actor::handle_message_ex)
    /// rather than the per-variant [`Handler<V>`] chain so existing
    /// actors compile unchanged.
    MessageEx(super::message_ex::MessageExPayload),
}

#[cfg(feature = "deepstream")]
impl PipelineMsg {
    /// Sender-side sugar for the [`PipelineMsg::MessageEx`] variant.
    /// Equivalent to
    /// `PipelineMsg::MessageEx(MessageExPayload::new(value))`.
    pub fn message_ex<T: std::any::Any + Send>(value: T) -> Self {
        PipelineMsg::MessageEx(super::message_ex::MessageExPayload::new(value))
    }

    /// Normalize a delivery-carrying message into a flat
    /// `Vec<(VideoFrame, SharedBuffer)>`.
    ///
    /// * [`PipelineMsg::Delivery`]   → exactly 1 pair.
    /// * [`PipelineMsg::Deliveries`] → the full batch.
    /// * [`PipelineMsg::SourceEos`] / [`PipelineMsg::Shutdown`] /
    ///   [`PipelineMsg::MessageEx`] → empty vec (these variants
    ///   are expected to be filtered out by the caller first;
    ///   returning empty is a safe no-op).
    ///
    /// This is the shared normalisation step that lets any
    /// consumer of [`PipelineMsg`] handle either delivery shape
    /// with a single code path — iterate the vec and feed each
    /// pair downstream.
    pub fn into_pairs(self) -> Vec<(VideoFrame, SharedBuffer)> {
        match self {
            PipelineMsg::Delivery(d) => vec![d.unseal()],
            PipelineMsg::Deliveries(d) => d.unseal(),
            PipelineMsg::SourceEos { .. }
            | PipelineMsg::Shutdown { .. }
            | PipelineMsg::MessageEx(_) => Vec::new(),
        }
    }

    /// `true` for [`PipelineMsg::Delivery`] and
    /// [`PipelineMsg::Deliveries`]; `false` for the sentinel /
    /// extensibility variants.
    pub fn is_delivery(&self) -> bool {
        matches!(self, PipelineMsg::Delivery(_) | PipelineMsg::Deliveries(_))
    }
}

/// Encoded-bitstream envelope, used wherever an inter-actor
/// channel carries encoded access units instead of decoded
/// frames.  Typical producers and consumers:
///
/// * **producer side** — a demuxer stage (e.g. `Mp4Demuxer`)
///   emitting [`EncodedMsg::StreamInfo`] followed by a stream of
///   [`EncodedMsg::Packet`] terminated by
///   [`EncodedMsg::SourceEos`]; an encoder stage (e.g. the
///   `Picasso` egress) emitting freshly encoded
///   [`EncodedMsg::Packet`] / [`EncodedMsg::SourceEos`] downstream;
///   or a custom producer handing off pre-built
///   [`EncodedMsg::Frame`] values.
/// * **consumer side** — a decoder stage that submits packets
///   to a `FlexibleDecoder`, a muxer terminus that finalises a
///   container, a ZeroMQ sink, etc.
///
/// Reusing one envelope across every encoded-data channel means
/// any consumer stage can be paired with any compatible
/// producer stage: every consumer handles the same variants,
/// and source id is carried in-band so producers do not have to
/// know what their downstream will do with it.
#[derive(Debug)]
pub enum EncodedMsg {
    /// Stream-level metadata (width, height, framerate, codec)
    /// tagged with the `source_id` it describes.  Producers that
    /// emit raw container packets (typically a demuxer) emit
    /// this once per source, *before* any
    /// [`EncodedMsg::Packet`] for that source, so consumers that
    /// observe stream headers (e.g. a logger) can react.
    /// Consumers that do not need the header (e.g. a muxer that
    /// derives caps from each packet's [`VideoInfo`]) simply
    /// ignore it.
    StreamInfo {
        /// Source id this stream belongs to.
        source_id: String,
        /// Stream-level metadata (width, height, framerate, codec).
        info: VideoInfo,
    },
    /// A single encoded access unit tagged with its `source_id`
    /// **and the stream-level [`VideoInfo`]** describing the
    /// stream it belongs to.  Examples of producers:
    ///
    /// * a demuxer stage — raw container packets, paired with
    ///   the `VideoInfo` observed on the most recent
    ///   [`EncodedMsg::StreamInfo`] for the same source.
    /// * an encoder stage — freshly encoded frames headed for
    ///   a muxer terminus or any other downstream consumer.
    ///
    /// Carrying the [`VideoInfo`] in-band makes this envelope
    /// self-describing: downstream consumers do not have to
    /// cache a per-`source_id` map of stream parameters — every
    /// packet arrives with the metadata it needs to be decoded
    /// or muxed.  A standalone [`EncodedMsg::StreamInfo`] is
    /// still emitted by typical demuxers for observers that
    /// want the stream header on its own.
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
    /// A pre-built [`VideoFrame`] delivered with its encoded
    /// payload, ready for direct submission to a decoder.
    ///
    /// Unlike [`EncodedMsg::Packet`] — which is raw container
    /// output that a decoder stage wraps into a fresh frame —
    /// this variant lets an upstream producer hand off a
    /// fully-populated frame (source id, dimensions, codec, fps,
    /// uuid, keyframe, timestamps) and bypass the `StreamInfo`
    /// path entirely.
    ///
    /// Payload resolution follows the `FlexibleDecoder::submit`
    /// contract:
    ///
    /// * `payload = Some(bytes)` — the decoder uses `bytes`
    ///   directly.
    /// * `payload = None` — the decoder extracts the bitstream
    ///   from the frame's internal content
    ///   ([`VideoFrameContent::Internal`](savant_core::primitives::frame::VideoFrameContent)).
    ///   If the frame's content is `External` or `None`, the
    ///   decoder emits a `Skipped { NoPayload }` callback rather
    ///   than erroring.
    ///
    /// Use this variant for producers that already own the
    /// frame — e.g. reading previously-captured `savant_core`
    /// frames back from storage, or bridging from another
    /// message-bus format — without paying the cost of
    /// re-deriving frame metadata from demuxer caps.  Consumers
    /// that don't decode (e.g. a muxer terminus) ignore this
    /// variant.
    Frame {
        /// Pre-built frame.  Its `source_id`, `codec`, `width`,
        /// `height`, `fps`, `uuid`, and `keyframe` fields must
        /// already be set.
        frame: VideoFrame,
        /// Encoded bitstream for this frame, or `None` to have the
        /// decoder extract it from `frame.get_content()`.
        payload: Option<Vec<u8>>,
    },
    /// End-of-stream sentinel carrying the source id.
    /// Consumers react per their role — for example a decoder
    /// stage flushes its operator and forwards a downstream
    /// EOS, a muxer terminus finalises the container, and a
    /// ZeroMQ sink writes a wire-level `EndOfStream` message.
    SourceEos {
        /// Source id this EOS belongs to.
        source_id: String,
    },
    /// Cooperative shutdown sentinel, sent by the supervisor
    /// when its [`ShutdownHandler`](crate::shutdown::ShutdownHandler)
    /// returns a broadcast action (e.g. on Ctrl+C or a stage
    /// exit).  The receive-loop driver observes it via
    /// [`Envelope::as_shutdown`]
    /// and breaks out of the loop — either immediately (when
    /// `grace` is `None`) or once the grace deadline expires.
    Shutdown {
        /// `None` — break after the current message is handled.
        /// `Some(d)` — keep processing, break when `recv_timeout`
        /// next fires at or past `now + d`.
        grace: Option<Duration>,
        /// Human-readable reason (logged on receipt).
        reason: Cow<'static, str>,
    },
    /// Type-erased extensibility variant.  Carries an arbitrary
    /// receiver-specific payload through the standard envelope
    /// without forcing the envelope type to be parametric.
    /// See [`MessageExPayload`](crate::message_ex::MessageExPayload)
    /// for the full contract; the [`Dispatch`] impl routes this
    /// variant through
    /// [`Actor::handle_message_ex`](super::actor::Actor::handle_message_ex)
    /// rather than the per-variant [`Handler<V>`] chain so existing
    /// actors compile unchanged.
    MessageEx(super::message_ex::MessageExPayload),
}

impl EncodedMsg {
    /// Sender-side sugar for the [`EncodedMsg::MessageEx`] variant.
    /// Equivalent to
    /// `EncodedMsg::MessageEx(MessageExPayload::new(value))`.
    pub fn message_ex<T: std::any::Any + Send>(value: T) -> Self {
        EncodedMsg::MessageEx(super::message_ex::MessageExPayload::new(value))
    }
}

// ---------------------------------------------------------------------------
// Framework integration — Envelope, Dispatch, per-variant payload wrappers.
// ---------------------------------------------------------------------------
//
// These types bridge the pipeline's envelope enums (`PipelineMsg`,
// `EncodedMsg`) to the generic actor framework in this crate.
//
// * [`Envelope`] tells the receive-loop driver whether an envelope
//   is a cooperative-shutdown sentinel, and how to *build* one for
//   the supervisor's broadcast step.  Both envelopes carry a
//   first-class `Shutdown` variant so `build_shutdown` always
//   returns `Some(_)`.
// * [`Dispatch`] routes each variant to one of the actor's
//   [`Handler<V>`] impls.  Actors opt in to every variant of the
//   envelope they consume — even with a zero-body
//   `impl Handler<VariantPayload> for MyActor {}`, which inherits
//   the trait's default "drop the payload, return [`Flow::Cont`]"
//   body.  Variants the actor wants to act on override `handle`
//   with domain logic.
//
// The per-variant payload structs (`SingleDelivery`, `BatchDelivery`,
// `StreamInfoPayload`, `PacketPayload`, `FramePayload`,
// `SourceEosPayload`, `ShutdownPayload`) are thin wrappers around
// the enum variants' fields.  They keep `impl Handler<…>` call-sites
// self-explanatory: each handler's signature reflects the variant's
// exact payload, with no destructuring boilerplate at the call-site.

/// Per-variant payload: a single sealed `(frame, buffer)` pair
/// (from [`PipelineMsg::Delivery`]).
///
/// Available only with the `deepstream` Cargo feature.
#[cfg(feature = "deepstream")]
pub struct SingleDelivery(pub SealedDelivery);

#[cfg(feature = "deepstream")]
impl fmt::Debug for SingleDelivery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SingleDelivery").field(&self.0).finish()
    }
}

/// Per-variant payload: a batched set of `(frame, buffer)` pairs
/// (from [`PipelineMsg::Deliveries`]).
///
/// Available only with the `deepstream` Cargo feature.
#[cfg(feature = "deepstream")]
pub struct BatchDelivery(pub SealedDeliveries);

#[cfg(feature = "deepstream")]
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
    /// [`VideoFrame`] values or route by codec without
    /// maintaining a per-source cache of stream parameters.
    pub info: VideoInfo,
    /// The encoded access unit.
    pub packet: DemuxedPacket,
}

/// Per-variant payload: a pre-built [`VideoFrame`] with its
/// encoded bitstream (from [`EncodedMsg::Frame`]).
pub struct FramePayload {
    /// Pre-built frame (source_id, codec, dims, fps, uuid, keyframe
    /// set).
    pub frame: VideoFrame,
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

#[cfg(feature = "deepstream")]
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

#[cfg(feature = "deepstream")]
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
            PipelineMsg::MessageEx(payload) => {
                <A as super::Actor>::handle_message_ex(actor, payload, ctx)
            }
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
            EncodedMsg::MessageEx(payload) => {
                <A as super::Actor>::handle_message_ex(actor, payload, ctx)
            }
        }
    }
}

#[cfg(feature = "deepstream")]
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
            PipelineMsg::MessageEx(payload) => f.debug_tuple("MessageEx").field(payload).finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// [`PipelineMsg::Shutdown`] surfaces a graceful shutdown hint;
    /// every other variant is non-shutdown.
    #[cfg(feature = "deepstream")]
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

    /// `EncodedMsg::message_ex(...)` flows through `Dispatch` into
    /// the actor's `handle_message_ex` override, which can recover
    /// the concrete payload type via `downcast`.
    #[test]
    fn encoded_msg_message_ex_round_trips_through_dispatch() {
        use crate::message_ex::MessageExPayload;
        use crate::registry::Registry;
        use crate::shared::SharedStore;
        use crate::supervisor::{StageKind, StageName};
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        #[derive(Debug, PartialEq)]
        struct MyPayload {
            counter: u32,
        }

        struct CaptureActor {
            received: Option<u32>,
        }

        impl crate::Actor for CaptureActor {
            type Msg = EncodedMsg;
            fn handle(
                &mut self,
                msg: EncodedMsg,
                ctx: &mut crate::Context<Self>,
            ) -> anyhow::Result<crate::Flow> {
                msg.dispatch(self, ctx)
            }
            fn handle_message_ex(
                &mut self,
                msg: MessageExPayload,
                _ctx: &mut crate::Context<Self>,
            ) -> anyhow::Result<crate::Flow> {
                let recovered = msg
                    .downcast::<MyPayload>()
                    .expect("payload must downcast to MyPayload");
                self.received = Some(recovered.counter);
                Ok(crate::Flow::Cont)
            }
        }
        impl Handler<StreamInfoPayload> for CaptureActor {}
        impl Handler<PacketPayload> for CaptureActor {}
        impl Handler<FramePayload> for CaptureActor {}
        impl Handler<SourceEosPayload> for CaptureActor {}
        impl Handler<ShutdownPayload> for CaptureActor {}

        let mut actor = CaptureActor { received: None };
        let mut ctx = crate::Context::<CaptureActor>::new(
            StageName::unnamed(StageKind::DeepStreamFunction),
            Arc::new(Registry::new()),
            Arc::new(SharedStore::new()),
            Arc::new(AtomicBool::new(false)),
        );
        let msg = EncodedMsg::message_ex(MyPayload { counter: 7 });
        msg.dispatch(&mut actor, &mut ctx).unwrap();
        assert_eq!(actor.received, Some(7));
    }

    /// PipelineMsg mirror of the EncodedMsg round-trip test.
    /// Only available under the `deepstream` feature because
    /// `PipelineMsg` itself is feature-gated.
    #[cfg(feature = "deepstream")]
    #[test]
    fn pipeline_msg_message_ex_round_trips_through_dispatch() {
        use crate::message_ex::MessageExPayload;
        use crate::registry::Registry;
        use crate::shared::SharedStore;
        use crate::supervisor::{StageKind, StageName};
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        #[derive(Debug, PartialEq)]
        struct MyPayload {
            tag: &'static str,
        }

        struct CaptureActor {
            received: Option<&'static str>,
        }

        impl crate::Actor for CaptureActor {
            type Msg = PipelineMsg;
            fn handle(
                &mut self,
                msg: PipelineMsg,
                ctx: &mut crate::Context<Self>,
            ) -> anyhow::Result<crate::Flow> {
                msg.dispatch(self, ctx)
            }
            fn handle_message_ex(
                &mut self,
                msg: MessageExPayload,
                _ctx: &mut crate::Context<Self>,
            ) -> anyhow::Result<crate::Flow> {
                let recovered = msg
                    .downcast::<MyPayload>()
                    .expect("payload must downcast to MyPayload");
                self.received = Some(recovered.tag);
                Ok(crate::Flow::Cont)
            }
        }
        impl Handler<SingleDelivery> for CaptureActor {}
        impl Handler<BatchDelivery> for CaptureActor {}
        impl Handler<SourceEosPayload> for CaptureActor {}
        impl Handler<ShutdownPayload> for CaptureActor {}

        let mut actor = CaptureActor { received: None };
        let mut ctx = crate::Context::<CaptureActor>::new(
            StageName::unnamed(StageKind::DeepStreamFunction),
            Arc::new(Registry::new()),
            Arc::new(SharedStore::new()),
            Arc::new(AtomicBool::new(false)),
        );
        let msg = PipelineMsg::message_ex(MyPayload { tag: "via-dispatch" });
        msg.dispatch(&mut actor, &mut ctx).unwrap();
        assert_eq!(actor.received, Some("via-dispatch"));
    }

    /// `Envelope::build_shutdown` round-trips into a shutdown
    /// variant that `as_shutdown` classifies as graceful.
    #[test]
    fn envelope_build_shutdown_round_trips() {
        #[cfg(feature = "deepstream")]
        {
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
        }

        let m = <EncodedMsg as Envelope>::build_shutdown(None, Cow::Borrowed("r2"))
            .expect("must construct");
        let hint = m.as_shutdown().expect("must be shutdown");
        assert!(matches!(hint, ShutdownHint::Graceful { grace: None, .. }));
    }
}
