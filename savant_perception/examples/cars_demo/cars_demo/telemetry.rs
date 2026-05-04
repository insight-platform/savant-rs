//! `cars-demo-zmq`-local OpenTelemetry initialisation helpers.
//!
//! Provides a tiny [`init_stdout_tracer`] entry point that:
//!
//! * Builds an [`opentelemetry_sdk::trace::TracerProvider`] with a
//!   simple stdout span exporter (one line per finished span).
//! * Installs it as the OTel global so `savant_core::get_tracer()`
//!   (which the framework's `instrument` module uses) returns a
//!   tracer that exports spans to stdout.
//! * Installs the W3C [`TraceContextPropagator`] as the global
//!   text-map propagator so `PropagatedContext::inject` /
//!   `extract` round-trips traceparent / tracestate headers
//!   across the ZMQ wire — the cross-process continuation primitive
//!   used by [`ZmqSink`](savant_perception::stages::ZmqSink) and
//!   [`ZmqSource`](savant_perception::stages::ZmqSource).
//!
//! Idempotent: a no-op when [`init_stdout_tracer`] has already been
//! called in the same process.
//!
//! Standalone wrapper for the demo so the sample stays
//! self-contained — savant_core's `telemetry::init` is a heavier
//! tokio-runtime-backed batch exporter setup; this helper just
//! prints to stdout.

use std::sync::Once;

use opentelemetry::global;
use opentelemetry::trace::{SpanBuilder, TraceContextExt, Tracer};
use opentelemetry::Context;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::TracerProvider;

static INIT: Once = Once::new();

/// Initialise OpenTelemetry with a stdout span exporter and the
/// W3C trace-context propagator.  Safe to call from any thread; the
/// first call wins, subsequent calls are no-ops.
///
/// Call **once per process** before [`System::run`] starts so the
/// framework's per-stage / per-callback spans flow through the
/// global tracer.
pub fn init_stdout_tracer(service_name: &'static str) {
    INIT.call_once(|| {
        let exporter = opentelemetry_stdout::SpanExporter::default();
        let provider = TracerProvider::builder()
            .with_simple_exporter(exporter)
            .with_config(
                opentelemetry_sdk::trace::Config::default().with_resource(
                    opentelemetry_sdk::Resource::new(vec![opentelemetry::KeyValue::new(
                        "service.name",
                        service_name,
                    )]),
                ),
            )
            .build();
        global::set_tracer_provider(provider);
        global::set_text_map_propagator(TraceContextPropagator::new());
    });
}

/// Build a fresh **root** OpenTelemetry context carrying a span
/// named `name`.  Bypasses [`Context::current`] entirely so the
/// returned context is unaffected by any per-callback span the
/// framework has installed at the call site (e.g. the producer's
/// `on_packet` span).  Pass into
/// [`VideoFrame::set_otel_ctx`](savant_core::primitives::frame::VideoFrame::set_otel_ctx)
/// to bind the frame's lifetime to this span — every framework
/// callback span downstream (`on_frame`, `on_inference`,
/// `on_tracking`, `on_encoded_frame`, …) auto-parents under it
/// via the frame's span stack.
///
/// Used by the producer's per-packet hook to implement
/// `--trace-frequency N`-style sampling: every `N`-th frame gets a
/// fresh root, the rest are untraced.
pub fn fresh_root_context(name: &'static str) -> Context {
    let tracer = savant_core::get_tracer();
    // Build the span with an explicitly empty parent context so it
    // is rooted in the trace tree regardless of what's currently
    // attached on this thread.
    let span = tracer.build_with_context(SpanBuilder::from_name(name), &Context::default());
    // Return a context that carries only this span (no inheritance
    // from the current context's values).
    Context::default().with_span(span)
}
