//! OpenTelemetry instrumentation primitives shared across the
//! framework.
//!
//! Two layers:
//!
//! * [`open_child`] — explicit-parent, sampling-aware span
//!   constructor.  Builds a child of `parent` (typically
//!   `frame.otel_ctx_clone().as_ref()`) and returns the resulting
//!   [`opentelemetry::Context`].  Returns `Context::default()` (a
//!   no-op context) when `parent` is `None` or its span is invalid;
//!   downstream descendants inherit the no-op and drop out cheaply.
//!   Mirrors savant_core's `Pipeline::get_nested_span` contract.
//! * [`SpanGuard`] — RAII wrapper that enters the span's context as
//!   the thread-local current context for the lifetime of the
//!   guard and ends the span on drop.  Use it for *local* spans
//!   that should serve as `Context::current()` for code running on
//!   the same thread; do *not* rely on a SpanGuard pushed by one
//!   thread to be visible from another (callback worker, source
//!   thread).  Cross-thread parent propagation flows through the
//!   frame's span stack —
//!   [`VideoFrame::push_otel_ctx`](savant_core::primitives::frame::VideoFrame::push_otel_ctx)
//!   /
//!   [`pop_otel_ctx`](savant_core::primitives::frame::VideoFrame::pop_otel_ctx)
//!   /
//!   [`otel_ctx_clone`](savant_core::primitives::frame::VideoFrame::otel_ctx_clone)
//!   — not through thread-local.
//!
//! **Deliberately removed**: an `open_child_of_current` shortcut.
//! Reading `Context::current()` from framework code is fragile
//! under interleaved execution — every span the framework opens is
//! parented explicitly off a frame (or `None` for non-frame
//! events).
//!
//! The tracer is fetched via [`savant_core::get_tracer`], so a host
//! application that already wires a `TracerProvider` for
//! `savant_core` (the standard pattern in this workspace) gets
//! framework spans for free without any extra SDK setup.

use std::borrow::Cow;

use opentelemetry::trace::{SpanBuilder, TraceContextExt, Tracer};
use opentelemetry::{Context as OtelContext, KeyValue};
use savant_core::primitives::frame::VideoFrame;

/// Build an OTel span as a child of `parent` and return the
/// resulting [`opentelemetry::Context`].
///
/// Always calls into the tracer — when `parent` is `None` or
/// carries an invalid span context (e.g. the producing stage's
/// sampling logic chose not to trace this frame), the SDK
/// produces a noop span that won't reach the collector.  Call
/// sites can therefore push, stamp attributes, end and pop
/// without `is_valid()` ceremony: every operation on a noop span
/// is itself a noop.
pub fn open_child(
    name: impl Into<Cow<'static, str>>,
    parent: Option<&OtelContext>,
) -> OtelContext {
    let tracer = savant_core::get_tracer();
    let builder = SpanBuilder::from_name(name);
    let span = match parent {
        Some(p) => tracer.build_with_context(builder, p),
        None => tracer.build_with_context(builder, &OtelContext::default()),
    };
    OtelContext::current_with_span(span)
}

/// RAII handle that:
///
/// 1. Attaches its [`OtelContext`] as the thread-local current
///    context for the duration of its lifetime.  This is a
///    *user-facing* convenience — code running on the same
///    thread as the guard can call
///    [`OtelContext::current`] to find the carried span.  The
///    framework itself does **not** rely on thread-local current
///    for its span hierarchy; cross-thread propagation flows
///    through the frame's span stack
///    ([`VideoFrame::push_otel_ctx`](savant_core::primitives::frame::VideoFrame::push_otel_ctx)
///    / [`otel_ctx_clone`](savant_core::primitives::frame::VideoFrame::otel_ctx_clone)).
/// 2. Ends the span on [`Drop`].
///
/// Both `attach` and `span().end()` are noop on noop spans, so
/// guards built around the default (sampled-out) context cost
/// only a stack allocation on the unsampled trace path.
///
/// Construct via [`SpanGuard::enter`].
pub struct SpanGuard {
    ctx: OtelContext,
    _attach: Option<opentelemetry::ContextGuard>,
}

impl SpanGuard {
    /// Enter `ctx` as the current OTel context.  When the returned
    /// guard drops, the context detaches and the span ends.  Noop
    /// contexts pass through harmlessly — `attach` / `end` are
    /// noops on noop spans, so callers don't need to gate on
    /// `is_valid()`.
    pub fn enter(ctx: OtelContext) -> Self {
        let attach = Some(ctx.clone().attach());
        Self {
            ctx,
            _attach: attach,
        }
    }

    /// Borrow the [`OtelContext`] this guard is keeping alive.
    /// Useful for stamping attributes on the carried span without
    /// needing to thread the original context around.
    pub fn ctx(&self) -> &OtelContext {
        &self.ctx
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        // Detach happens automatically via `_attach`'s Drop.
        // `span().end()` is a noop on noop spans, so we don't need
        // an `is_valid()` gate here either.
        self.ctx.span().end();
    }
}

/// Open a callback span as a child of `frame`'s current OTel
/// context, stamp the standard framework attributes
/// (`pipeline.name`, `stage`), push the new span onto the frame's
/// stack, and return a guard whose [`Drop`] pops the stack again.
///
/// One-call replacement for the open + set_attribute + push +
/// (later) pop dance the framework runs around every per-frame
/// user-hook invocation in async batching paths
/// (`nvinfer`/`nvtracker`/`decoder`/`picasso`-encoder) and in
/// sync handler arms (`deepstream_function`/`bitstream_function`/
/// `mp4_muxer`/`zmq_sink`/`sorter`/`picasso`-engine).  When the
/// frame isn't traced, the spawned span is a noop and every
/// operation through this guard is a noop too — no `is_valid()`
/// ceremony needed.
///
/// The guard holds a cheap [`VideoFrame`] clone (Arc clone,
/// shared inner) so callers don't have to retain their own
/// reference for the pop site — and so the span is popped even
/// when the user hook consumes the original frame (e.g.
/// `picasso`'s encoder callback) or the surrounding scope's
/// borrow of the frame ends earlier.
pub fn enter_callback_span(
    frame: &VideoFrame,
    name: impl Into<Cow<'static, str>>,
    pipeline_name: &str,
    stage_name: &str,
) -> CallbackSpanGuard {
    let span_ctx = open_child(name, frame.otel_ctx_clone().as_ref());
    span_ctx
        .span()
        .set_attribute(KeyValue::new("pipeline.name", pipeline_name.to_string()));
    span_ctx
        .span()
        .set_attribute(KeyValue::new("stage", stage_name.to_string()));
    frame.push_otel_ctx(span_ctx);
    CallbackSpanGuard {
        frame: frame.clone(),
    }
}

/// RAII guard returned by [`enter_callback_span`].  Pops the
/// frame's [`OtelSpanGuard`] on [`Drop`] — same scope as the
/// `enter_callback_span` call by construction, so the unwind
/// can't outlive a missed pop or pop the wrong layer.
pub struct CallbackSpanGuard {
    frame: VideoFrame,
}

impl CallbackSpanGuard {
    /// Construct a guard that only pops on [`Drop`] without
    /// pushing anything — used when the push happened in a
    /// different scope (typically a different thread, e.g. an
    /// async batching stage that pushes its **stage span** in
    /// `handle()` and pops it after the operator callback fires).
    pub fn pop_at_drop(frame: VideoFrame) -> Self {
        Self { frame }
    }
}

impl Drop for CallbackSpanGuard {
    fn drop(&mut self) {
        self.frame.pop_otel_ctx();
    }
}

/// Push a stage-level span onto `frame`'s span stack.  Companion
/// to [`enter_callback_span`] but without RAII: the caller is
/// responsible for popping (typically in a sibling scope or on a
/// different thread, after the stage's work for `frame` has
/// completed).  Used by async batching stages
/// (`nvinfer`/`nvtracker`) where the stage span must live from
/// `handle()` entry through the operator callback's hook return,
/// straddling a thread boundary.
///
/// On the error path — operator drops `frame` without ever firing
/// the callback — the unpopped stage span is bounded by frame
/// lifetime: when the last clone of `frame` drops, the stack is
/// dropped with the inner and every remaining
/// [`OtelSpanGuard`]'s [`Drop`] ends its span.
pub fn push_stage_span(
    frame: &VideoFrame,
    name: impl Into<Cow<'static, str>>,
    pipeline_name: &str,
    stage_name: &str,
) {
    let span_ctx = open_child(name, frame.otel_ctx_clone().as_ref());
    span_ctx
        .span()
        .set_attribute(KeyValue::new("pipeline.name", pipeline_name.to_string()));
    span_ctx
        .span()
        .set_attribute(KeyValue::new("stage", stage_name.to_string()));
    frame.push_otel_ctx(span_ctx);
}
