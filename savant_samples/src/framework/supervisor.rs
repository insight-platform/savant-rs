//! Pipeline supervisor — back-channel for stage-exit signalling.
//!
//! Every stage thread owns a [`StageExitGuard`] that pushes a
//! [`StageExit`] onto the supervisor's back-channel when it drops —
//! i.e. on **every** thread exit path, whether the body returned
//! `Ok(_)`, returned `Err(_)`, or is unwinding from a panic.  The
//! orchestrator blocks on [`ExitReceiver::recv`] and treats the first
//! *shutdown-worthy* signal as the trigger to broadcast a cooperative
//! [`PipelineMsg::Shutdown`](super::message::PipelineMsg::Shutdown)
//! sentinel onto every inter-actor channel and then join every
//! thread.
//!
//! # Why Drop?
//!
//! Using Drop rather than an explicit `exit_tx.send(...)` call at the
//! bottom of each stage body guarantees panic-safety for free —
//! unwinding invokes `Drop::drop` on locals, so a panicked stage
//! still unblocks the supervisor.  It also removes the
//! "forgot-to-signal" footgun: you can't accidentally `return Err(_)`
//! without notifying the supervisor.
//!
//! # Why unbounded?
//!
//! Every live stage holds its own guard, so at "broadcast Shutdown +
//! join all" time there are up to `N_stages + 1` senders pushing
//! notifications at once (plus the Ctrl+C handler).  A bounded
//! channel would risk a guard blocking on `send` during unwind,
//! which is precisely the hang we are trying to avoid — an
//! unbounded channel keeps `StageExitGuard::drop` infallibly
//! non-blocking for all future supervisor shapes, at the cost of a
//! handful of queued 16-byte signals that the orchestrator drains
//! anyway.
//!
//! # Multi-instance stages
//!
//! A [`StageKind`] identifies *what* a stage is (demux, decode,
//! infer, tracker, …), while [`StageName`] pairs it with a free-form
//! `instance` tag — so two concurrent nvinfer actors can be
//! distinguished as e.g. `infer[yolo11n]` vs. `infer[person_attr]`.
//! Conditional supervisor policies match on [`StageKind`]; logs use
//! the full [`StageName`].

use crossbeam::channel::{unbounded, Receiver, Sender};
use std::borrow::Cow;
use std::fmt;

/// Discriminator for a pipeline stage — the *role* a thread plays,
/// independent of any instance tag.
///
/// Kept as a plain enum (rather than `&'static str`) so the
/// supervisor — and any future monitoring layer — can react
/// conditionally on *what kind* of stage stopped with exhaustive
/// `match` coverage.  Pair with an instance string via
/// [`StageName`] when a single kind can have multiple live
/// instances (e.g. several nvinfer models in one pipeline).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageKind {
    /// [`mp4_demux`](super::pipeline::mp4_demux) actor — `Mp4Demuxer` wrapper.
    Mp4Demux,
    /// [`decoder`](super::pipeline::decoder) actor — `FlexibleDecoderPool` wrapper.
    Decoder,
    /// [`infer`](super::pipeline::infer) actor — nvinfer batching operator.
    Infer,
    /// [`tracker`](super::pipeline::tracker) actor — NvDCF batching operator.
    Tracker,
    /// [`picasso`](super::pipeline::picasso) actor — Picasso draw + encode.
    Picasso,
    /// [`blackhole`](super::pipeline::blackhole) actor — pipeline tail when
    /// `--no-picasso` disables picasso + mux.
    Blackhole,
    /// [`mp4_mux`](super::pipeline::mp4_mux) actor — `Mp4Muxer` terminus.
    Mp4Mux,
    /// Synthetic signal posted by the Ctrl+C handler so user-cancel
    /// flows through the same single recv-point as natural exits.
    CtrlC,
}

impl StageKind {
    /// Lower-case static identifier used in log lines, e.g.
    /// `"demux"`, `"ctrl-c"`.
    pub const fn as_str(&self) -> &'static str {
        match self {
            StageKind::Mp4Demux => "mp4_demux",
            StageKind::Decoder => "decoder",
            StageKind::Infer => "infer",
            StageKind::Tracker => "tracker",
            StageKind::Picasso => "picasso",
            StageKind::Blackhole => "blackhole",
            StageKind::Mp4Mux => "mp4_mux",
            StageKind::CtrlC => "ctrl-c",
        }
    }
}

impl fmt::Display for StageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Fully-qualified stage identifier — a [`StageKind`] plus an
/// optional instance tag.
///
/// The instance tag disambiguates concurrent actors of the same
/// [`StageKind`] — e.g. a pipeline running three nvinfer models
/// simultaneously can name them `infer[yolo11n]`, `infer[person_attr]`,
/// `infer[plate_ocr]` and the supervisor's logs will reflect exactly
/// which one signalled.
///
/// Conditional supervisor policy (e.g. "ignore `Demux` natural
/// exit") matches on the [`StageKind`] only; the instance is purely
/// for operator visibility.
///
/// Construct with:
///
/// * [`StageName::unnamed`] — single-instance stage, instance
///   field is empty, display is just the kind (e.g. `"demux"`).
/// * [`StageName::new`] — named instance, display is
///   `"{kind}[{instance}]"` (e.g. `"infer[yolo11n]"`).
///
/// `instance` is a [`Cow<'static, str>`] so callers can pass either
/// a static literal (zero-copy) or an owned `String` at no API
/// cost.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StageName {
    /// Stage role — used for conditional supervisor policies.
    pub kind: StageKind,
    /// Free-form instance tag.  Empty means "only one actor of this
    /// kind in the pipeline"; non-empty disambiguates concurrent
    /// instances.
    pub instance: Cow<'static, str>,
}

impl StageName {
    /// Construct a named instance, e.g.
    /// `StageName::new(StageKind::Infer, "yolo11n")`.  Accepts any
    /// `impl Into<Cow<'static, str>>` — `&'static str` and owned
    /// `String` both work at no extra cost.
    pub fn new(kind: StageKind, instance: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind,
            instance: instance.into(),
        }
    }

    /// Construct an unnamed stage — use when the pipeline has a
    /// single actor of this kind and no disambiguation is needed.
    /// Display is just the kind's identifier, e.g. `"demux"`.
    pub const fn unnamed(kind: StageKind) -> Self {
        Self {
            kind,
            instance: Cow::Borrowed(""),
        }
    }

    /// `true` when no instance tag was attached (i.e. constructed
    /// via [`StageName::unnamed`] or [`StageName::new`] with an
    /// empty string).
    pub fn is_unnamed(&self) -> bool {
        self.instance.is_empty()
    }
}

impl fmt::Display for StageName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.instance.is_empty() {
            f.write_str(self.kind.as_str())
        } else {
            write!(f, "{}[{}]", self.kind.as_str(), self.instance)
        }
    }
}

/// Back-channel notification describing which stage exited.
///
/// The supervisor conditionally reads a single *shutdown-worthy*
/// signal today and broadcasts
/// [`PipelineMsg::Shutdown`](super::message::PipelineMsg::Shutdown),
/// but the richer `stage` field is deliberately kept here so
/// future supervisors can react conditionally (e.g. "ignore
/// `Demux` when the terminus has already drained", "treat a
/// `Tracker` panic as pipeline fatal", …).  Error information
/// itself still propagates via each thread's
/// `JoinHandle<Result<()>>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageExit {
    /// Which stage (kind + instance) fired this signal.
    pub stage: StageName,
}

/// Sender half of the supervisor back-channel.
pub type ExitSender = Sender<StageExit>;
/// Receiver half of the supervisor back-channel.
pub type ExitReceiver = Receiver<StageExit>;

/// RAII notification guard.  Construct one at the top of every stage
/// thread body; the `Drop` implementation pushes a [`StageExit`] onto
/// the back-channel exactly once, on every possible exit path
/// (normal return, `?` propagation, and panic unwind).
///
/// Bind with `let _guard = ...` (named binding) — `let _ = ...`
/// would drop immediately and the guard would fire at the wrong
/// time.
#[must_use = "StageExitGuard must be bound to a named variable so it lives for the whole stage body"]
pub struct StageExitGuard {
    stage: StageName,
    tx: ExitSender,
}

impl StageExitGuard {
    /// Create a new guard.  `stage` (kind + optional instance tag)
    /// identifies the thread that will eventually fire this guard;
    /// the value is forwarded onto the back-channel on `Drop` so
    /// the supervisor knows who stopped.
    pub fn new(stage: StageName, tx: ExitSender) -> Self {
        Self { stage, tx }
    }
}

impl Drop for StageExitGuard {
    fn drop(&mut self) {
        // Best-effort send.  By the time we drop the supervisor may
        // already have received a previous stage's exit, broadcast
        // Shutdown, and torn down — in that case the channel is
        // closed and the send returns `Err`, which we silently
        // ignore (the point of signalling has passed).
        //
        // `clone` is cheap: `StageName` holds a `Cow<'static, str>`
        // which clones as a trivial refcount-free borrow for the
        // common static-literal case.
        let _ = self.tx.send(StageExit {
            stage: self.stage.clone(),
        });
    }
}

/// Build an unbounded supervisor back-channel.
///
/// Unbounded is deliberate: `StageExitGuard::drop` must *never*
/// block — a blocked guard during unwind would deadlock the whole
/// join step.  A handful of queued 16-byte `StageExit` structs
/// (one per live stage plus Ctrl+C) is negligible and the
/// orchestrator drains them all.
pub fn exit_channel() -> (ExitSender, ExitReceiver) {
    unbounded()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guard fires exactly once on normal drop, carrying the
    /// stage tag through to the receiver.
    #[test]
    fn guard_sends_on_drop() {
        let (tx, rx) = exit_channel();
        {
            let _guard = StageExitGuard::new(StageName::unnamed(StageKind::Mp4Demux), tx);
        }
        let got = rx.recv().expect("guard must push one signal");
        assert_eq!(got.stage.kind, StageKind::Mp4Demux);
        assert!(got.stage.is_unnamed());
        assert!(rx.try_recv().is_err(), "guard must not fire twice");
    }

    /// A panic in the guard's scope still fires the signal.  This is
    /// the whole point of the Drop-based design.
    #[test]
    fn guard_sends_on_panic() {
        let (tx, rx) = exit_channel();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = StageExitGuard::new(StageName::unnamed(StageKind::Infer), tx);
            panic!("oops");
        }));
        assert!(result.is_err(), "the closure must panic");
        let got = rx.recv().expect("guard must push during unwind");
        assert_eq!(got.stage.kind, StageKind::Infer);
    }

    /// Multiple guards fan into the same receiver; the first signal
    /// is available immediately even if other guards are still
    /// live.  This mirrors the orchestrator's "block on first exit"
    /// contract.
    #[test]
    fn first_exit_is_observable() {
        let (tx, rx) = exit_channel();
        let g1 = StageExitGuard::new(StageName::unnamed(StageKind::Decoder), tx.clone());
        let _g2 = StageExitGuard::new(StageName::unnamed(StageKind::Tracker), tx.clone());
        drop(g1);
        let first = rx.recv().expect("first guard drop must be readable");
        assert_eq!(first.stage.kind, StageKind::Decoder);
    }

    /// Two concurrent stages of the same kind are distinguishable
    /// via their instance tag — the whole point of moving from a
    /// bare enum to a `kind + instance` struct.
    #[test]
    fn concurrent_same_kind_distinguished_by_instance() {
        let (tx, rx) = exit_channel();
        let a = StageExitGuard::new(StageName::new(StageKind::Infer, "yolo11n"), tx.clone());
        let b = StageExitGuard::new(StageName::new(StageKind::Infer, "person_attr"), tx.clone());
        drop(a);
        drop(b);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2 {
            let exit = rx.recv().expect("both guards must fire");
            assert_eq!(exit.stage.kind, StageKind::Infer);
            assert!(!exit.stage.is_unnamed());
            seen.insert(exit.stage.instance.into_owned());
        }
        assert!(seen.contains("yolo11n"));
        assert!(seen.contains("person_attr"));
    }

    /// `StageKind` knows its human-readable identifier; covers
    /// every variant so a new variant added without a matching
    /// `as_str` arm would fail to compile.
    #[test]
    fn stage_kind_as_str_covers_every_variant() {
        for (kind, expected) in [
            (StageKind::Mp4Demux, "mp4_demux"),
            (StageKind::Decoder, "decoder"),
            (StageKind::Infer, "infer"),
            (StageKind::Tracker, "tracker"),
            (StageKind::Picasso, "picasso"),
            (StageKind::Blackhole, "blackhole"),
            (StageKind::Mp4Mux, "mp4_mux"),
            (StageKind::CtrlC, "ctrl-c"),
        ] {
            assert_eq!(kind.as_str(), expected);
            assert_eq!(kind.to_string(), expected);
        }
    }

    /// `StageName` display distinguishes unnamed (`"demux"`) from
    /// named (`"infer[yolo11n]"`) instances.
    #[test]
    fn stage_name_display_formats_instance() {
        let unnamed = StageName::unnamed(StageKind::Mp4Demux);
        assert_eq!(unnamed.to_string(), "mp4_demux");
        assert!(unnamed.is_unnamed());

        let named = StageName::new(StageKind::Infer, "yolo11n");
        assert_eq!(named.to_string(), "infer[yolo11n]");
        assert!(!named.is_unnamed());

        let empty = StageName::new(StageKind::Mp4Mux, "");
        assert_eq!(empty.to_string(), "mp4_mux");
        assert!(empty.is_unnamed());

        let owned = StageName::new(StageKind::Tracker, String::from("lane_a"));
        assert_eq!(owned.to_string(), "tracker[lane_a]");
    }
}
