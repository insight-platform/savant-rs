//! Framework-level envelope-agnostic payload types for
//! *operator-instance actions* — things a peer needs to ask an actor
//! to do on its underlying operator, not ingress events that flow
//! through the pipeline.
//!
//! These payloads are **not** baked into any particular envelope.
//! The Layer-B templates that own an operator provide
//! [`Handler<T>`](crate::Handler) impls for the relevant
//! payload types; sample envelopes opt in by adding a variant that
//! carries the payload and by routing it through their
//! [`Dispatch`](crate::Dispatch) impl.
//!
//! # Pattern
//!
//! A sample that wants to trigger an NvDCF stream reset from one of
//! its actors would:
//!
//! 1. Add a `ResetStream { source_id: String }` variant to its
//!    envelope enum.
//! 2. Destructure that variant in the envelope's
//!    [`Dispatch`](crate::Dispatch) impl to a
//!    [`ResetStreamPayload`] and dispatch via
//!    [`Handler<ResetStreamPayload>`](crate::Handler).
//! 3. Send the variant from any actor that has an
//!    [`Addr`](crate::Addr) of the tracker.
//!
//! The tracker's [`NvTracker`](crate::templates::NvTracker)
//! implements [`Handler<ResetStreamPayload>`] out of the box — so all
//! the sample has to do is wire the variant through its envelope.
//!
//! Equivalent payloads for Picasso source-lifecycle control are
//! [`RemoveSourcePayload`] (call `remove_source_spec`) and
//! [`UpdateSourceSpecPayload`] (re-call `set_source_spec` to hot-swap
//! or reconfigure a running source).
//!
//! # Cross-envelope sentinels
//!
//! The [`SourceEosPayload`] and [`ShutdownPayload`] types are the
//! framework-wide, envelope-agnostic **sentinel payloads** every
//! streaming sample needs.  They ship here (not behind a sample-owned
//! envelope) so any envelope can route its EOS / shutdown variant to
//! the same `Handler<SourceEosPayload>` / `Handler<ShutdownPayload>`
//! implementations the Layer-B templates already provide.
//! [`ShutdownPayload`] mirrors the shape of
//! [`ShutdownHint`](crate::ShutdownHint), which classifies
//! the sentinel at the envelope level; a sample's `Envelope::build_shutdown`
//! constructs the variant, the loop driver tests it via
//! `Envelope::as_shutdown`, and the actor's `Handler<ShutdownPayload>`
//! observes the same `{ grace, reason }` tuple as the payload.

use std::borrow::Cow;
use std::time::Duration;

use picasso::prelude::SourceSpec;

/// Request an NvDCF tracker to reset its per-stream state for
/// `source_id`.
///
/// Handled by [`NvTracker`](crate::templates::NvTracker)
/// via [`NvTrackerBatchingOperator::reset_stream`](deepstream_nvtracker::NvTrackerBatchingOperator::reset_stream).
/// Typical use: a downstream stage detects a discontinuity (scene
/// change, seek, re-keyed stream) and asks the tracker to discard
/// accumulated feature history for that source.
#[derive(Debug, Clone)]
pub struct ResetStreamPayload {
    /// Source identifier whose tracker state should be reset.
    pub source_id: String,
}

/// Request a [`Picasso`](crate::templates::Picasso)
/// to remove the registered `SourceSpec` for `source_id`.
///
/// Translates to
/// [`PicassoEngine::remove_source_spec`](picasso::PicassoEngine::remove_source_spec),
/// which shuts the per-source worker down.  Once removed, any
/// subsequent `Delivery` for the same `source_id` will cause the
/// actor to re-register a fresh spec on its next first-frame
/// codepath, via the builder's `source_spec_factory`.
#[derive(Debug, Clone)]
pub struct RemoveSourcePayload {
    /// Source identifier whose Picasso worker should be torn down.
    pub source_id: String,
}

/// Replace (or install) the [`SourceSpec`] bound to `source_id` on a
/// [`Picasso`](crate::templates::Picasso).
///
/// Calls [`PicassoEngine::set_source_spec`](picasso::PicassoEngine::set_source_spec),
/// which either spawns a new worker (first time) or hot-swaps the
/// spec on an existing worker.  Use this to change encoder bitrate,
/// rewrite the draw spec, toggle bypass mode, etc. without tearing
/// the worker down and back up.
///
/// The `spec` is boxed so envelopes carrying this payload stay
/// cheap to move when it is not populated.
pub struct UpdateSourceSpecPayload {
    /// Source identifier whose spec should be installed / updated.
    pub source_id: String,
    /// New [`SourceSpec`] to hand to the Picasso engine.
    pub spec: Box<SourceSpec>,
}

impl std::fmt::Debug for UpdateSourceSpecPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpdateSourceSpecPayload")
            .field("source_id", &self.source_id)
            .field("spec", &"SourceSpec { .. }")
            .finish()
    }
}

/// End-of-source sentinel payload carrying the `source_id` whose
/// stream has ended.
///
/// Envelope-agnostic: any sample envelope that wants to signal
/// per-source EOS destructures its `SourceEos` variant into this
/// payload and dispatches via [`Handler<SourceEosPayload>`](crate::Handler).
/// Every Layer-B template ships a default `Handler<SourceEosPayload>`
/// impl wired to the operator's `source_eos` / drain step, so a
/// sample only has to route the variant through its
/// [`Dispatch`](crate::Dispatch) impl.
#[derive(Debug, Clone)]
pub struct SourceEosPayload {
    /// Source id this EOS belongs to.
    pub source_id: String,
}

/// Cooperative shutdown sentinel payload.
///
/// Mirrors the shape of
/// [`ShutdownHint::Graceful`](crate::ShutdownHint::Graceful):
/// a sample's `Envelope::build_shutdown` constructs the envelope
/// variant from `(grace, reason)`, the loop driver reads the same
/// tuple via `Envelope::as_shutdown`, and the actor's
/// `Handler<ShutdownPayload>` observes the payload form.
///
/// * `grace = None` — break after the current message is handled.
/// * `grace = Some(d)` — keep processing until the deadline elapses,
///   then break the next time `recv_timeout` fires at or past
///   `now + d`.
#[derive(Debug, Clone)]
pub struct ShutdownPayload {
    /// Grace period (or `None` to break after this message).
    pub grace: Option<Duration>,
    /// Human-readable reason, logged on receipt.
    pub reason: Cow<'static, str>,
}
