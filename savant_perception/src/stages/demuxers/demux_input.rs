//! Shared input-requester vocabulary for the looping demuxer
//! sources ([`Mp4DemuxerSource`](super::super::mp4_demuxer::Mp4DemuxerSource)
//! and [`UriDemuxerSource`](super::super::uri_demuxer::UriDemuxerSource)).
//!
//! Instead of being parameterised by a single `(input, source_id)`
//! pair at construction time, both demuxer sources are driven by a
//! [`InputRequester`] callback the framework asks **before each
//! input run** (and again after every underlying demuxer exits).
//! The callback returns a [`DemuxInputRequest`] telling the source
//! what to do next:
//!
//! * [`DemuxInputRequest::Run`] — process the supplied
//!   `(input, source_id)` pair, then ask again.
//! * [`DemuxInputRequest::Stop`] — terminate the source cleanly.
//! * [`DemuxInputRequest::Idle`] — sleep for the supplied
//!   [`Duration`] (still cooperatively observing the shared stop
//!   flag), then ask again.
//!
//! For the common single-input case, both demuxer builders ship a
//! [`one_shot`](super::super::mp4_demuxer::Mp4DemuxerBuilder::one_shot)
//! helper that emits exactly one
//! [`Run`](DemuxInputRequest::Run) followed by
//! [`Stop`](DemuxInputRequest::Stop) on every subsequent call.

use std::time::Duration;

use crate::HookCtx;

/// Type-tagged wrapper around the demuxer-level input string (path /
/// URI of the run currently in progress).
///
/// The on-packet hook needs to surface both this and a `source_id`
/// `&str`; wrapping the input in `DemuxInput` makes them
/// type-distinct so callers can't swap them by accident.  Both
/// `Deref` and `AsRef<str>` are implemented so the inner string is
/// usable in the closure body without explicit unwrapping.
#[derive(Debug, Clone, Copy)]
pub struct DemuxInput<'a>(pub &'a str);

impl<'a> std::ops::Deref for DemuxInput<'a> {
    type Target = str;
    fn deref(&self) -> &str {
        self.0
    }
}

impl<'a> AsRef<str> for DemuxInput<'a> {
    fn as_ref(&self) -> &str {
        self.0
    }
}

impl<'a> std::fmt::Display for DemuxInput<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// What the framework should do next on a looping demuxer source.
///
/// Returned by an [`InputRequester`] each time the source asks for
/// the next input — once at startup, and again every time the
/// underlying demuxer finishes (clean EOS, idle window, or
/// requester-driven retry loop).
#[derive(Debug, Clone)]
pub enum DemuxInputRequest {
    /// Process the supplied input next.
    Run {
        /// File path or URI handed to the underlying demuxer.
        ///
        /// Forwarded verbatim — for the MP4 demuxer this is a
        /// filesystem path, for the URI demuxer this is the full
        /// URI string (`file://`, `http://`, `rtsp://`, …).
        input: String,
        /// Source identifier stamped on every emitted
        /// [`EncodedMsg`](crate::envelopes::EncodedMsg) variant
        /// for this run.  May change between runs to drive a
        /// multi-stream pipeline through the same source actor.
        source_id: String,
    },
    /// Terminate the source cleanly — the framework runs the
    /// stage's `stopping` hook and returns `Ok(())` from
    /// [`Source::run`](crate::Source::run).
    Stop,
    /// Sleep the supplied [`Duration`] before re-invoking the
    /// requester.  The sleep is poll-sliced so cooperative
    /// shutdown signals (Ctrl+C, supervisor broadcast,
    /// [`HookCtx::request_stop`]) are still observed promptly.
    Idle(Duration),
}

/// Callback type implementing the input-requester contract.
///
/// Invoked on the source's worker thread, once at startup and
/// again every time the underlying demuxer exits cleanly.  The
/// closure receives an off-loop [`HookCtx`] so it can read shared
/// state, resolve peer addresses, or check
/// [`HookCtx::stop_requested`] before deciding what to do next.
///
/// `FnMut` so requesters can keep internal state across calls
/// (e.g. an iterator of paths, a retry counter, an in-memory
/// queue).
pub type InputRequester = Box<dyn FnMut(&HookCtx) -> DemuxInputRequest + Send + 'static>;

/// Build a one-shot [`InputRequester`] that emits a single
/// [`DemuxInputRequest::Run`] for `(input, source_id)` and
/// [`DemuxInputRequest::Stop`] on every subsequent invocation.
///
/// Both demuxer builders expose a `.one_shot(input, source_id)`
/// convenience that wraps this helper, so most call sites do not
/// need to use it directly.  Reach for it when constructing an
/// [`InputRequester`] separately from the builder (e.g. for
/// custom composition with another requester source).
pub fn one_shot_requester(
    input: impl Into<String>,
    source_id: impl Into<String>,
) -> InputRequester {
    let mut once: Option<(String, String)> = Some((input.into(), source_id.into()));
    Box::new(move |_ctx: &HookCtx| match once.take() {
        Some((input, source_id)) => DemuxInputRequest::Run { input, source_id },
        None => DemuxInputRequest::Stop,
    })
}

/// Build an infinitely-looping [`InputRequester`] that emits
/// [`DemuxInputRequest::Run`] for the same `(input, source_id)`
/// pair on every invocation.
///
/// The source only exits when the shared `stop_flag` flips
/// (Ctrl+C, supervisor broadcast, or
/// [`HookCtx::request_stop`] from a hook) — natural EOS at the
/// end of each input simply triggers another run.  Both demuxer
/// builders expose a `.looped(input, source_id)` convenience that
/// wraps this helper.
pub fn looped_requester(
    input: impl Into<String>,
    source_id: impl Into<String>,
) -> InputRequester {
    let input = input.into();
    let source_id = source_id.into();
    Box::new(move |_ctx: &HookCtx| DemuxInputRequest::Run {
        input: input.clone(),
        source_id: source_id.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Registry;
    use crate::shared::SharedStore;
    use crate::supervisor::{StageKind, StageName};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    fn hook_ctx() -> HookCtx {
        HookCtx::new(
            StageName::unnamed(StageKind::BitstreamSource),
            Arc::from("test"),
            Arc::new(Registry::new()),
            Arc::new(SharedStore::new()),
            Arc::new(AtomicBool::new(false)),
            crate::stage_metrics::StageMetrics::new("demux-test"),
        )
    }

    #[test]
    fn one_shot_emits_run_then_stop() {
        let mut req = one_shot_requester("/tmp/x.mp4", "cam1");
        let ctx = hook_ctx();
        match req(&ctx) {
            DemuxInputRequest::Run { input, source_id } => {
                assert_eq!(input, "/tmp/x.mp4");
                assert_eq!(source_id, "cam1");
            }
            other => panic!("expected Run, got {other:?}"),
        }
        assert!(matches!(req(&ctx), DemuxInputRequest::Stop));
        assert!(matches!(req(&ctx), DemuxInputRequest::Stop));
    }

    #[test]
    fn looped_emits_run_indefinitely() {
        let mut req = looped_requester("rtsp://cam/stream", "cam");
        let ctx = hook_ctx();
        for _ in 0..5 {
            match req(&ctx) {
                DemuxInputRequest::Run { input, source_id } => {
                    assert_eq!(input, "rtsp://cam/stream");
                    assert_eq!(source_id, "cam");
                }
                other => panic!("expected Run, got {other:?}"),
            }
        }
    }

    /// Idle holds a Duration that the source loop is expected to
    /// honour by sleeping (cooperatively).  Just round-trip it
    /// through Debug to keep the `Debug` derive honest.
    #[test]
    fn idle_round_trips_duration() {
        let r = DemuxInputRequest::Idle(Duration::from_millis(250));
        let s = format!("{r:?}");
        assert!(s.contains("Idle"));
        assert!(s.contains("250"));
    }
}
