//! Per-stage metric instruments and the framework's periodic
//! console reporter.
//!
//! Every registered stage — actors *and* sources — automatically
//! gets a [`StageMetrics`] handle that:
//!
//! * Records into OpenTelemetry instruments — [`Counter<u64>`]s
//!   for frames / objects / batches, [`Histogram<f64>`]s for
//!   end-to-end frame and per-call handler latency plus the
//!   per-phase breakdown (`preproc` / `inference` / `postproc`)
//!   contributed by batching stages, and an
//!   [`ObservableGauge<i64>`] for inbox queue length — so an
//!   externally-configured `MeterProvider` (OTLP exporter, stdout
//!   exporter, …) sees per-stage time-series.
//! * Maintains sidecar atomic counters that the framework's own
//!   [`StageReporter`] reads to emit one counters line and up to
//!   four per-phase latency lines per stage at the configured
//!   cadence (default 60 s, see
//!   [`System::stats_period`](crate::System::stats_period)).
//!   The latency atomics are atomically reset every tick (see
//!   [`StageMetrics::take_period`]) so min / avg / max / samples
//!   reflect the current window, not lifetime extremes.
//!
//! When no `MeterProvider` is wired up the OTel calls return noop
//! instruments — recording is essentially free, the sidecar
//! atomics still drive the in-process reporter.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use opentelemetry::global;
use opentelemetry::metrics::{Counter, Histogram, ObservableGauge};
use opentelemetry::KeyValue;
use parking_lot::Mutex;
use savant_core::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
use savant_core::primitives::frame::VideoFrame;
use savant_core::primitives::WithAttributes;

use super::envelope::STAGE_INGRESS_NS;

/// Process-local monotonic time anchor used by the framework's
/// per-frame latency markers.  Established lazily on first use;
/// every subsequent [`monotonic_ns`] reads the same anchor so
/// deltas across `record_stage_ingress` / `take_stage_latencies`
/// are valid.
static FRAMEWORK_T0: OnceLock<Instant> = OnceLock::new();

/// Monotonic nanoseconds since the framework's anchor instant
/// — see [`FRAMEWORK_T0`].  Saturates to `i64::MAX` for absurdly
/// long-lived processes (≈292 years), which is fine because the
/// value is only used to compute differences within the same
/// process.
pub fn monotonic_ns() -> i64 {
    let t0 = *FRAMEWORK_T0.get_or_init(Instant::now);
    Instant::now()
        .duration_since(t0)
        .as_nanos()
        .min(i64::MAX as u128) as i64
}

/// Phase-stamp key used by [`stamp_phase`] / [`take_phase`] in
/// async batching stages (nvinfer, nvtracker).  Marks the moment a
/// frame is **submitted to the operator** — stamped on the actor
/// thread inside `submit_pairs`, just before the operator's
/// `add_frame`.  The delta between this stamp and the operator
/// callback firing on its worker thread is the per-frame
/// operator turnaround — recorded into the
/// `inference_latency` stream (reporter label `<stage>.infer`).
///
/// Note: this is **not** the moment preprocessing finishes.
/// Preprocessing in this framework is the user's
/// `BatchFormationCallback`, which runs inside the operator
/// after a batch has been formed; that's timed separately by
/// the framework as `preproc_latency` (see
/// [`StageMetrics::record_preproc_latency`]).
pub const PHASE_INFER_START: &str = "infer_start";

/// Phase-stamp key for the **e2e start** marker — placed by an
/// async batching stage (`nvinfer`/`nvtracker`) at handle entry
/// (before unseal) on each frame in the inbound delivery, and
/// read on the operator-callback thread *after* the user hook
/// (`on_inference` / `on_tracking`) has completed all
/// postprocessing for the frame.  The delta is the per-frame
/// end-to-end latency at this stage: handle entry → last
/// callback completion.
///
/// The attribute name composed by [`stamp_phase`] /
/// [`take_phase`] is `<stage>.e2e_start`; the stage prefix is
/// the unique [`StageName`](super::supervisor::StageName), so
/// multiple inference / tracker instances in the same pipeline
/// don't clash.
pub const PHASE_E2E_START: &str = "e2e_start";

/// Stamp a phase timestamp on a frame as a temporary, hidden
/// attribute under namespace [`STAGE_INGRESS_NS`] with attribute
/// name `<stage>.<phase>`.  Used by stages that decompose their
/// work into named phases and need cross-thread phase boundaries
/// to flow with the frame — e.g. nvinfer / nvtracker stamp
/// `infer_start` on the actor thread and read it on the
/// operator's callback thread to compute "operator turnaround"
/// (inference / tracking) duration.
///
/// Memory bound: stamps live on the frame's Arc-shared inner;
/// the [`Drop`] on the inner releases them when the frame's last
/// clone dies.  Use [`take_phase`] to consume the stamp at the
/// matching read site.
pub fn stamp_phase(frame: &VideoFrame, stage: &str, phase: &str, ns: i64) {
    let mut frame = frame.clone();
    let attr_name = format!("{stage}.{phase}");
    frame.set_temporary_attribute(
        STAGE_INGRESS_NS,
        &attr_name,
        &None,
        true,
        vec![AttributeValue::integer(ns, None)],
    );
}

/// Read and remove a phase stamp set by [`stamp_phase`].
/// Returns `None` when no stamp is present (frame arrived
/// without one, or a previous reader already consumed it).
pub fn take_phase(frame: &VideoFrame, stage: &str, phase: &str) -> Option<i64> {
    let mut frame = frame.clone();
    let attr_name = format!("{stage}.{phase}");
    let attr = frame.delete_attribute(STAGE_INGRESS_NS, &attr_name)?;
    let value = attr.values.first()?;
    if let AttributeValueVariant::Integer(ns) = value.value {
        Some(ns)
    } else {
        None
    }
}

/// Instrumentation library name used when fetching the
/// framework's [`opentelemetry::metrics::Meter`].  External
/// collectors / dashboards filter on this `instrumentation.name`
/// to isolate framework metrics.
pub const METER_NAME: &str = "savant_perception";

#[derive(Debug)]
struct StageInner {
    frames: AtomicU64,
    objects: AtomicU64,
    batches: AtomicU64,
    /// Last observed inbox depth — written by the loop driver on
    /// every recv, read by both the [`ObservableGauge`] callback
    /// (during OTel collection) and the [`StageReporter`] thread.
    queue_len: AtomicI64,
    /// Frame-latency running stats (ingress→egress) — populated
    /// from [`StageMetrics::record_frame_latency`].
    frame_lat: LatencyStats,
    /// Handle-latency running stats (per `handle()` call) —
    /// populated from [`StageMetrics::record_handle_latency`].
    handle_lat: LatencyStats,
    /// Phase-latency running stats — populated only by stages that
    /// decompose work into preproc / operator turnaround / postproc
    /// (nvinfer, nvtracker).  Other stages leave these idle and the
    /// reporter omits the matching lines.
    preproc_lat: LatencyStats,
    inference_lat: LatencyStats,
    postproc_lat: LatencyStats,
    /// End-to-end per-frame latency — populated only by
    /// async batching stages.  Spans handle entry (before unseal)
    /// to the last callback completion (after on_inference /
    /// on_tracking returns).  See [`PHASE_E2E_START`].
    e2e_lat: LatencyStats,
}

#[derive(Debug)]
struct LatencyStats {
    count: AtomicU64,
    sum_ns: AtomicU64,
    /// `u64::MAX` sentinel until the first sample lands.
    min_ns: AtomicU64,
    max_ns: AtomicU64,
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum_ns: AtomicU64::new(0),
            min_ns: AtomicU64::new(u64::MAX),
            max_ns: AtomicU64::new(0),
        }
    }
}

impl LatencyStats {
    fn record(&self, dur: Duration) {
        let ns = dur.as_nanos() as u64;
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_ns.fetch_add(ns, Ordering::Relaxed);
        // CAS loop on min.
        let mut current = self.min_ns.load(Ordering::Relaxed);
        while ns < current {
            match self
                .min_ns
                .compare_exchange_weak(current, ns, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
        // CAS loop on max.
        let mut current = self.max_ns.load(Ordering::Relaxed);
        while ns > current {
            match self
                .max_ns
                .compare_exchange_weak(current, ns, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
    }

    /// Atomically drain the running stats: read each counter, reset
    /// to its initial value (`0` for sums / counts / max,
    /// `u64::MAX` for min), and return the previous values.
    ///
    /// Concurrent recorders see the freshly reset state and
    /// contribute to the next period.  A recorder mid-flight (i.e.
    /// past its `count.fetch_add` but before its `min` / `max` CAS)
    /// can split its sample across two periods — in the worst case
    /// the next period's `count` is one bigger than its `min`/`max`
    /// observation set.  This is tolerable for the in-process
    /// reporter; the OTel histogram remains the authoritative
    /// surface for downstream collectors.
    fn take(&self) -> LatencySnapshot {
        LatencySnapshot {
            count: self.count.swap(0, Ordering::Relaxed),
            sum_ns: self.sum_ns.swap(0, Ordering::Relaxed),
            min_ns: self.min_ns.swap(u64::MAX, Ordering::Relaxed),
            max_ns: self.max_ns.swap(0, Ordering::Relaxed),
        }
    }
}

impl Default for StageInner {
    fn default() -> Self {
        Self {
            frames: AtomicU64::new(0),
            objects: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            queue_len: AtomicI64::new(0),
            frame_lat: LatencyStats::default(),
            handle_lat: LatencyStats::default(),
            preproc_lat: LatencyStats::default(),
            inference_lat: LatencyStats::default(),
            postproc_lat: LatencyStats::default(),
            e2e_lat: LatencyStats::default(),
        }
    }
}

/// Per-stage metric instruments + sidecar state.  Allocated by
/// [`System::register_actor`](crate::System::register_actor) for
/// actors and by
/// [`System::register_source`](crate::System::register_source) for
/// sources, then threaded into the stage's
/// [`Context`](crate::Context) /
/// [`SourceContext`](crate::SourceContext) so the loop driver
/// (actors) and source emission sites record per-message events
/// without any user-side instrumentation.
pub struct StageMetrics {
    name: String,
    attrs: Vec<KeyValue>,
    frames_counter: Counter<u64>,
    objects_counter: Counter<u64>,
    batches_counter: Counter<u64>,
    /// Per-frame ingress→egress latency.  Recorded by the stage's
    /// [`Router`](crate::Router) on every send: the loop driver
    /// stamps an ingress timestamp on each inbound frame as a
    /// temporary attribute, and `Router::send` reads the marker
    /// on outbound frames to compute the real `t_out − t_in`
    /// time the frame spent inside the stage.  For async batching
    /// stages (nvinfer, nvtracker, decoder) this captures the
    /// operator's queue + worker turnaround, not just the
    /// `handle()` enqueue time.
    frame_latency_histogram: Histogram<f64>,
    /// Per-call handler duration — `handle()` start to end.
    /// Always recorded.  Useful for stages whose frames don't
    /// egress (sinks like mp4_muxer / zmq_sink: no
    /// `Router::send` for the frame, no
    /// [`frame_latency_histogram`] sample, only handle time).
    handle_latency_histogram: Histogram<f64>,
    /// Optional per-stage phase breakdown — populated only by
    /// async batching stages that decompose their work into
    /// `preproc → operator turnaround → postproc`.  See
    /// [`record_preproc_latency`](StageMetrics::record_preproc_latency)
    /// / [`record_inference_latency`](StageMetrics::record_inference_latency)
    /// / [`record_postproc_latency`](StageMetrics::record_postproc_latency)
    /// for the recording sites — nvinfer / nvtracker drive
    /// these; other stages leave them empty.
    preproc_latency_histogram: Histogram<f64>,
    inference_latency_histogram: Histogram<f64>,
    postproc_latency_histogram: Histogram<f64>,
    /// Per-frame end-to-end latency for async batching stages —
    /// from handle entry (before unseal) to the last callback
    /// completion (after the user hook returns and the framework
    /// finishes its post-hook bookkeeping).  See
    /// [`record_e2e_latency`](StageMetrics::record_e2e_latency).
    e2e_latency_histogram: Histogram<f64>,
    /// Held only to keep the instrument and its callback alive.
    _queue_gauge: ObservableGauge<i64>,
    inner: Arc<StageInner>,
}

impl StageMetrics {
    /// Build per-stage metrics for the actor named `name`.  Pulls
    /// the framework [`opentelemetry::metrics::Meter`] from the
    /// global provider — when none is configured the resulting
    /// instruments are noops.
    pub fn new(name: impl Into<String>) -> Arc<Self> {
        let name = name.into();
        let meter = global::meter(METER_NAME);
        let inner = Arc::new(StageInner::default());
        let stage_attr = KeyValue::new("stage", name.clone());
        let attrs = vec![stage_attr.clone()];

        let frames_counter = meter
            .u64_counter("savant.stage.frames")
            .with_description("Frames processed by the stage")
            .init();
        let objects_counter = meter
            .u64_counter("savant.stage.objects")
            .with_description("Objects observed at the stage's inbound side")
            .init();
        let batches_counter = meter
            .u64_counter("savant.stage.batches")
            .with_description("Inbound message batches consumed by the stage")
            .init();
        let frame_latency_histogram = meter
            .f64_histogram("savant.stage.frame_latency_ms")
            .with_description(
                "Per-frame ingress→egress latency through this stage \
                 (for async batching stages, includes operator queue + \
                 worker turnaround, not just handler enqueue time)",
            )
            .with_unit("ms")
            .init();
        let handle_latency_histogram = meter
            .f64_histogram("savant.stage.handle_latency_ms")
            .with_description("Per-call handler duration")
            .with_unit("ms")
            .init();
        let preproc_latency_histogram = meter
            .f64_histogram("savant.stage.preproc_latency_ms")
            .with_description(
                "Per-batch preprocessing duration on the actor thread \
                 (handle() entry → operator submission)",
            )
            .with_unit("ms")
            .init();
        let inference_latency_histogram = meter
            .f64_histogram("savant.stage.inference_latency_ms")
            .with_description(
                "Per-frame operator turnaround — time spent inside the \
                 batching operator (preproc-done stamp → result callback)",
            )
            .with_unit("ms")
            .init();
        let postproc_latency_histogram = meter
            .f64_histogram("savant.stage.postproc_latency_ms")
            .with_description(
                "Per-batch postprocessing duration in the operator's \
                 result callback (on_inference / on_tracking hook)",
            )
            .with_unit("ms")
            .init();
        let e2e_latency_histogram = meter
            .f64_histogram("savant.stage.e2e_latency_ms")
            .with_description(
                "Per-frame end-to-end latency for async batching \
                 stages (handle entry → last callback completion)",
            )
            .with_unit("ms")
            .init();

        // Capture an Arc clone of the sidecar inner + the attribute
        // set into the gauge callback.  The callback fires whenever
        // the MeterProvider's reader collects metrics.
        let inner_cb = Arc::clone(&inner);
        let attrs_cb = vec![stage_attr];
        let queue_gauge = meter
            .i64_observable_gauge("savant.stage.queue_length")
            .with_description("Current inbox queue length")
            .with_callback(move |observer| {
                observer.observe(inner_cb.queue_len.load(Ordering::Relaxed), &attrs_cb);
            })
            .init();

        Arc::new(Self {
            name,
            attrs,
            frames_counter,
            objects_counter,
            batches_counter,
            frame_latency_histogram,
            handle_latency_histogram,
            preproc_latency_histogram,
            inference_latency_histogram,
            postproc_latency_histogram,
            e2e_latency_histogram,
            _queue_gauge: queue_gauge,
            inner,
        })
    }

    /// Stage label used as the `stage` attribute on every record.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Record a single inbound message: bumps the framework's
    /// per-stage frame / object / batch counters and stores the
    /// observed inbox depth for the gauge callback.
    ///
    /// Called by the loop driver on every inbound actor envelope —
    /// users of `Actor` don't touch this directly.  Source stages
    /// (mp4 demuxer, uri demuxer, zmq source) call it explicitly
    /// at every frame-emission site through their crate-internal
    /// `HookCtx::stage_metrics` accessor so the periodic 📊
    /// reporter shows the source's outbound FPS alongside every
    /// actor's inbound FPS.  Skips the frame / batch / object
    /// counters when `frame_count == 0` (e.g. for sentinel
    /// envelopes that carry no [`VideoFrame`](savant_core::primitives::frame::VideoFrame)).
    pub fn record_message(&self, frame_count: usize, object_count: usize, queue_len: usize) {
        if frame_count > 0 {
            self.frames_counter.add(frame_count as u64, &self.attrs);
            self.batches_counter.add(1, &self.attrs);
            self.inner
                .frames
                .fetch_add(frame_count as u64, Ordering::Relaxed);
            self.inner.batches.fetch_add(1, Ordering::Relaxed);
        }
        if object_count > 0 {
            self.objects_counter.add(object_count as u64, &self.attrs);
            self.inner
                .objects
                .fetch_add(object_count as u64, Ordering::Relaxed);
        }
        self.inner
            .queue_len
            .store(queue_len as i64, Ordering::Relaxed);
    }

    /// Record one ingress→egress sample for a single frame —
    /// the wall-clock time the frame spent inside this stage
    /// (loop driver receive → [`Router`](crate::Router) send).
    ///
    /// The ingress timestamp is carried on the frame itself as a
    /// temporary, hidden attribute under the
    /// `"telemetry.tracing"` namespace, so memory is bounded by
    /// the frame's [`Drop`] — no side-table that could bloat under
    /// user hooks that drop frames or stages that consume without
    /// forwarding.
    pub fn record_frame_latency(&self, dur: Duration) {
        let ms = dur.as_nanos() as f64 / 1_000_000.0;
        self.frame_latency_histogram.record(ms, &self.attrs);
        self.inner.frame_lat.record(dur);
    }

    /// Record one `handle()` call duration.  Always populated
    /// (one sample per inbound message), so sinks — which never
    /// forward frames and therefore never produce
    /// [`record_frame_latency`] samples — still surface a
    /// meaningful latency through this stream.
    pub fn record_handle_latency(&self, dur: Duration) {
        let ms = dur.as_nanos() as f64 / 1_000_000.0;
        self.handle_latency_histogram.record(ms, &self.attrs);
        self.inner.handle_lat.record(dur);
    }

    /// Record one preprocessing-phase sample — the wall-clock time
    /// the actor thread spent inside `handle()` before the batch
    /// was submitted to the underlying operator.  Recorded by
    /// stages that decompose work into preproc / inference /
    /// postproc (nvinfer, nvtracker).
    pub fn record_preproc_latency(&self, dur: Duration) {
        let ms = dur.as_nanos() as f64 / 1_000_000.0;
        self.preproc_latency_histogram.record(ms, &self.attrs);
        self.inner.preproc_lat.record(dur);
    }

    /// Record one inference-phase sample — the per-frame operator
    /// turnaround time (preproc-done stamp → result callback).
    /// Recorded once per frame by stages that decompose work into
    /// preproc / inference / postproc.
    pub fn record_inference_latency(&self, dur: Duration) {
        let ms = dur.as_nanos() as f64 / 1_000_000.0;
        self.inference_latency_histogram.record(ms, &self.attrs);
        self.inner.inference_lat.record(dur);
    }

    /// Record one postprocessing-phase sample — the wall-clock
    /// time the operator's result-callback hook (e.g. `on_inference`
    /// / `on_tracking`) ran.  Recorded by stages that decompose
    /// work into preproc / inference / postproc.
    pub fn record_postproc_latency(&self, dur: Duration) {
        let ms = dur.as_nanos() as f64 / 1_000_000.0;
        self.postproc_latency_histogram.record(ms, &self.attrs);
        self.inner.postproc_lat.record(dur);
    }

    /// Record one **end-to-end** sample for an async batching
    /// stage — a per-frame measurement spanning handle entry
    /// (before unseal) to the last callback completion (after
    /// the user hook returns and the framework finishes its
    /// post-hook bookkeeping).  Cross-thread carrier is the
    /// frame's [`PHASE_E2E_START`] attribute.
    pub fn record_e2e_latency(&self, dur: Duration) {
        let ms = dur.as_nanos() as f64 / 1_000_000.0;
        self.e2e_latency_histogram.record(ms, &self.attrs);
        self.inner.e2e_lat.record(dur);
    }

    /// Drain the per-period view of the sidecar.  Returns the
    /// cumulative counters (frames / objects / batches / queue
    /// length) **plus** a fresh snapshot of every latency stream
    /// — the latency atomics are atomically reset to their initial
    /// state (count/sum/max → 0, min → `u64::MAX`) so the next
    /// call sees only samples recorded after this one.
    ///
    /// This is what the [`StageReporter`] calls each tick: bootstrap
    /// outliers don't poison subsequent windows, and `min` / `max`
    /// reflect the current period rather than a sticky lifetime
    /// extreme.  The OTel histogram remains the cumulative surface
    /// for external collectors — only the in-process console line
    /// is period-scoped.
    pub fn take_period(&self) -> StageSnapshot {
        StageSnapshot {
            frames: self.inner.frames.load(Ordering::Relaxed),
            objects: self.inner.objects.load(Ordering::Relaxed),
            batches: self.inner.batches.load(Ordering::Relaxed),
            queue_len: self.inner.queue_len.load(Ordering::Relaxed),
            frame_latency: self.inner.frame_lat.take(),
            handle_latency: self.inner.handle_lat.take(),
            preproc_latency: self.inner.preproc_lat.take(),
            inference_latency: self.inner.inference_lat.take(),
            postproc_latency: self.inner.postproc_lat.take(),
            e2e_latency: self.inner.e2e_lat.take(),
        }
    }
}

impl std::fmt::Debug for StageMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StageMetrics")
            .field("name", &self.name)
            .field("inner", &self.inner)
            .finish()
    }
}

/// Snapshot of a [`StageMetrics`]'s sidecar atomics at a single
/// point in time.  Used by [`StageReporter`] to compute
/// period-over-period rate statistics (FPS, OPS, average latency
/// over the interval).
#[derive(Debug, Default, Clone, Copy)]
pub struct StageSnapshot {
    pub frames: u64,
    pub objects: u64,
    pub batches: u64,
    pub queue_len: i64,
    pub frame_latency: LatencySnapshot,
    pub handle_latency: LatencySnapshot,
    /// Preprocessing-phase latency — populated only by stages that
    /// decompose work into named phases (nvinfer, nvtracker).
    pub preproc_latency: LatencySnapshot,
    /// Operator-turnaround (inference / tracking) latency.
    pub inference_latency: LatencySnapshot,
    /// Postprocessing-phase latency.
    pub postproc_latency: LatencySnapshot,
    /// Per-frame end-to-end latency for async batching stages
    /// (handle entry → last callback completion).
    pub e2e_latency: LatencySnapshot,
}

/// Period-scoped running-stat snapshot for one of the per-stage
/// latency streams (`frame`, `handle`, `preproc`, `inference`,
/// `postproc`).  Returned by [`StageMetrics::take_period`], which
/// atomically resets the underlying atomics — the values describe
/// only samples observed since the previous take.
#[derive(Debug, Default, Clone, Copy)]
pub struct LatencySnapshot {
    pub count: u64,
    pub sum_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
}

/// Periodic console reporter for a set of [`StageMetrics`].  Spawns
/// one OS thread on [`StageReporter::start`] that wakes every
/// `period`, calls [`StageMetrics::take_period`] on each registered
/// stage, and logs at `info` level: one cumulative-counters line
/// (`📊 <stage> | frames … | objects … | queue …`) plus up to four
/// per-phase latency lines whose leading column carries the full
/// `<stage>.<phase>` label (`e2e`, `preproc`, `infer`,
/// `postproc`) — phase lines are emitted only when the matching
/// stream produced samples this window.
///
/// Stops cleanly on [`StageReporter::shutdown`] — emits one final
/// report so the closing window is always observable.
pub struct StageReporter {
    period: Duration,
    state: Arc<ReporterState>,
    handle: Option<thread::JoinHandle<()>>,
}

struct ReporterState {
    stages: Vec<(Arc<StageMetrics>, Mutex<StageBaseline>)>,
    shutdown: AtomicBool,
}

/// Per-stage baseline retained across reporter ticks — only the
/// cumulative counters need a baseline now, since
/// [`StageMetrics::take_period`] resets the latency atomics on
/// every tick (period-scoped min / avg / max / samples come back
/// directly in the snapshot).
#[derive(Debug, Clone, Copy)]
struct StageBaseline {
    frames: u64,
    objects: u64,
    at: Instant,
}

impl StageReporter {
    /// Build a reporter that prints stats for `stages` every
    /// `period`.  Call [`Self::start`] once to spawn the worker
    /// thread.
    pub fn new(period: Duration, stages: Vec<Arc<StageMetrics>>) -> Self {
        let now = Instant::now();
        let stages = stages
            .into_iter()
            .map(|s| {
                (
                    s,
                    Mutex::new(StageBaseline {
                        frames: 0,
                        objects: 0,
                        at: now,
                    }),
                )
            })
            .collect();
        Self {
            period,
            state: Arc::new(ReporterState {
                stages,
                shutdown: AtomicBool::new(false),
            }),
            handle: None,
        }
    }

    /// Spawn the reporter thread.  No-op when:
    ///
    /// * the reporter has no stages registered (avoids a useless
    ///   heartbeat thread for systems that only run sources), or
    /// * the configured `period` is [`Duration::ZERO`] — periodic
    ///   reporting is **disabled**.  The final
    ///   [`Self::report_now`] flush from
    ///   [`System::run`](crate::System::run) still emits the
    ///   closing snapshot, so a host that wants only the final
    ///   summary can pass `Duration::ZERO`.  Without this guard
    ///   the inner sleep loop in `run_reporter` would never make
    ///   progress and burn 100 % of a CPU core.
    pub fn start(&mut self) {
        if self.state.stages.is_empty() || self.period.is_zero() {
            return;
        }
        let state = Arc::clone(&self.state);
        let period = self.period;
        self.handle = Some(
            thread::Builder::new()
                .name("savant-stats".into())
                .spawn(move || run_reporter(state, period))
                .expect("failed to spawn stats reporter thread"),
        );
    }

    /// Force one synchronous report — used as the final flush from
    /// [`System::run`](crate::System::run) so the closing
    /// snapshot is always visible.
    pub fn report_now(&self) {
        report_once(&self.state);
    }

    /// Signal the reporter thread to exit and join it.  Idempotent.
    pub fn shutdown(&mut self) {
        self.state.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            // Best-effort join; if the worker panicked the framework
            // has bigger problems and we don't want to mask them
            // here.
            let _ = handle.join();
        }
    }
}

impl Drop for StageReporter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run_reporter(state: Arc<ReporterState>, period: Duration) {
    // Defensive: refuse `Duration::ZERO`.  The inner
    // `while elapsed < period` would otherwise never sleep and
    // the outer loop would call `report_once` on a tight spin,
    // pegging a CPU core.  `StageReporter::start` already
    // skips the spawn when `period.is_zero()`, but bare
    // invocations (tests, direct callers) deserve the same
    // safety.
    if period.is_zero() {
        return;
    }
    // Sleep in short chunks so shutdown isn't blocked by a long
    // outstanding wait on the very first interval.  Cap the chunk
    // at `period` so very short reporting intervals (sub-chunk)
    // still report at roughly their requested rate instead of
    // every 50 ms.
    let chunk = Duration::from_millis(50).min(period);
    loop {
        let mut elapsed = Duration::ZERO;
        while elapsed < period {
            if state.shutdown.load(Ordering::Relaxed) {
                return;
            }
            thread::sleep(chunk);
            elapsed += chunk;
        }
        if state.shutdown.load(Ordering::Relaxed) {
            return;
        }
        report_once(&state);
    }
}

fn report_once(state: &ReporterState) {
    let now = Instant::now();
    for (metrics, baseline) in &state.stages {
        // Single take_period() per tick: counters are cumulative
        // (returned as-is), latency atomics are atomically reset
        // — `snap.*_latency` is already period-scoped, no baseline
        // subtraction needed.
        let snap = metrics.take_period();
        let mut base = baseline.lock();
        let elapsed = now.saturating_duration_since(base.at);
        let elapsed_s = elapsed.as_secs_f64().max(1e-9);

        let frames_delta = snap.frames.saturating_sub(base.frames);
        let objects_delta = snap.objects.saturating_sub(base.objects);

        let fps = frames_delta as f64 / elapsed_s;
        let ops = objects_delta as f64 / elapsed_s;

        // Line 1: cumulative counters with a per-period delta /
        // rate alongside.  Always emitted, including for sources
        // whose latency lines are suppressed below.  ASCII pipes
        // only — Unicode box-drawing characters render at
        // inconsistent widths across terminals.
        log::info!(
            "📊 {name:<24} | frames {fr:>9} (+{fr_d:<6} -> {fps:>8.1} fps) \
             | objects {ob:>9} (+{ob_d:<6} -> {ops:>8.1} ops) \
             | queue {q:>3}",
            name = metrics.name(),
            fr = snap.frames,
            fr_d = frames_delta,
            fps = fps,
            ob = snap.objects,
            ob_d = objects_delta,
            ops = ops,
            q = snap.queue_len,
        );

        // Latency lines: one per populated stream.  Each carries
        // the full `<stage>.<phase>` label as its leading column
        // so the line is greppable on its own.
        //
        // The end-to-end (`e2e`) line prefers the explicit
        // `e2e_latency` stream (populated by async batching stages
        // with a per-frame stamp at handle entry and a read after
        // the last callback returns), falls back to the
        // `frame_latency` stream (generic ingress → egress for
        // stages that forward frames), then to `handle_latency`
        // for sinks whose frames don't egress.  Skipped entirely
        // when none has samples — sources don't measure handler
        // latency.
        let stage = metrics.name();
        let e2e_lat = if snap.e2e_latency.count > 0 {
            Some(snap.e2e_latency)
        } else if snap.frame_latency.count > 0 {
            Some(snap.frame_latency)
        } else if snap.handle_latency.count > 0 {
            Some(snap.handle_latency)
        } else {
            None
        };
        if let Some(lat) = e2e_lat {
            emit_latency_line(&format!("{stage}.e2e"), lat);
        }

        // Optional phase breakdown — only emitted when the matching
        // stream produced samples this period (nvinfer / nvtracker
        // populate them; other stages leave them empty).
        for (phase, lat) in [
            ("preproc", snap.preproc_latency),
            ("infer", snap.inference_latency),
            ("postproc", snap.postproc_latency),
        ] {
            if lat.count > 0 {
                emit_latency_line(&format!("{stage}.{phase}"), lat);
            }
        }

        *base = StageBaseline {
            frames: snap.frames,
            objects: snap.objects,
            at: now,
        };
    }
}

fn emit_latency_line(label: &str, lat: LatencySnapshot) {
    let avg_ms = if lat.count > 0 {
        (lat.sum_ns as f64 / lat.count as f64) / 1_000_000.0
    } else {
        0.0
    };
    let min_ms = if lat.min_ns == u64::MAX {
        0.0
    } else {
        lat.min_ns as f64 / 1_000_000.0
    };
    let max_ms = lat.max_ns as f64 / 1_000_000.0;
    log::info!(
        "   {label:<32} | latency: min {min:>6.2} ms | avg {avg:>6.2} ms | max {max:>6.2} ms | samples {n:>6}",
        label = label,
        min = min_ms,
        avg = avg_ms,
        max = max_ms,
        n = lat.count,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_then_take_period_round_trip() {
        let m = StageMetrics::new("test");
        m.record_message(2, 5, 3);
        m.record_handle_latency(Duration::from_micros(100));
        m.record_handle_latency(Duration::from_micros(200));
        m.record_frame_latency(Duration::from_micros(300));
        let s = m.take_period();
        assert_eq!(s.frames, 2);
        assert_eq!(s.objects, 5);
        assert_eq!(s.batches, 1);
        assert_eq!(s.queue_len, 3);
        assert_eq!(s.handle_latency.count, 2);
        assert_eq!(s.handle_latency.sum_ns, 300_000);
        assert_eq!(s.handle_latency.min_ns, 100_000);
        assert_eq!(s.handle_latency.max_ns, 200_000);
        assert_eq!(s.frame_latency.count, 1);
        assert_eq!(s.frame_latency.sum_ns, 300_000);
    }

    /// Latency atomics are reset by [`StageMetrics::take_period`]
    /// — bootstrap outliers must not leak into subsequent windows.
    #[test]
    fn take_period_resets_latency_atomics() {
        let m = StageMetrics::new("test");
        m.record_handle_latency(Duration::from_millis(250)); // bootstrap outlier
        let bootstrap = m.take_period();
        assert_eq!(bootstrap.handle_latency.count, 1);
        assert_eq!(bootstrap.handle_latency.max_ns, 250_000_000);

        // Second period: only a fast sample lands.  The 250 ms
        // outlier from the previous period must not influence
        // either `max` or the running average.
        m.record_handle_latency(Duration::from_micros(50));
        let steady = m.take_period();
        assert_eq!(steady.handle_latency.count, 1);
        assert_eq!(steady.handle_latency.min_ns, 50_000);
        assert_eq!(steady.handle_latency.max_ns, 50_000);
        assert_eq!(steady.handle_latency.sum_ns, 50_000);

        // Third period without samples: counts/sums are zero, min
        // is back at its `u64::MAX` sentinel.
        let idle = m.take_period();
        assert_eq!(idle.handle_latency.count, 0);
        assert_eq!(idle.handle_latency.sum_ns, 0);
        assert_eq!(idle.handle_latency.min_ns, u64::MAX);
        assert_eq!(idle.handle_latency.max_ns, 0);
    }

    #[test]
    fn reporter_shutdown_is_clean_with_no_stages() {
        let mut r = StageReporter::new(Duration::from_secs(1), Vec::new());
        r.start();
        r.shutdown();
    }
}
