//! `NvTracker`: appsrc → nvtracker → appsink via [`savant_gstreamer::pipeline`].
//!
//! ## GStreamer properties (manual batched flow)
//!
//! - **`appsrc`**: `format=time`, `stream-type=stream`, caps `video/x-raw(memory:NVMM)`.
//!   Each pushed buffer gets an explicit PTS from an internal monotonic counter (`next_pts`).
//!   A pad probe on appsrc's src pad answers DeepStream custom queries
//!   (`gst_nvquery_batch_size` / `gst_nvquery_numStreams_size`) with `config.max_batch_size`.
//! - **`appsink`**: `sync=false`, `emit-signals=false` — polled by the framework drain thread.
//! - **`nvtracker`**: `GstBaseTransform` in-place — operates on the same buffer, so no queue or
//!   meta bridge is needed.  The `sub-batches` property is **rejected** — each `NvTracker`
//!   instance is its own isolated tracker; create separate instances for different tracking
//!   workloads instead of sub-batching within one element.
//!
//! ## Per-source release ([`NvTracker::reset_stream`])
//!
//! `gst-nvtracker`'s low-level `NvMultiObjectTracker` library pins the most
//! recent frame buffer per active source as its prev-frame reference; that
//! pin is only released when the next batch flows through the tracker.  In a
//! long-running pipeline with churning sources this would accumulate one
//! pinned NVMM buffer per source ever seen, eventually exhausting the GPU.
//!
//! [`reset_stream`](NvTracker::reset_stream) replaces the legacy
//! `GST_NVEVENT_STREAM_RESET` emit with a deterministic release sequence:
//!
//! 1. allocate a one-shot single-buffer pool sized to `(tracker_width,
//!    tracker_height, input_format, gpu_id)`,
//! 2. acquire one buffer, wrap it in a one-slot
//!    [`NonUniformBatch`](deepstream_buffers::NonUniformBatch) tagged with
//!    [`NvTrackerServiceMeta`](crate::service_meta::NvTrackerServiceMeta) and
//!    no `SavantIdMeta`,
//! 3. submit the synthetic batch through the same `input_tx` that real
//!    submits use (so PTS ordering is preserved inside [`SubmitGate`]),
//! 4. immediately enqueue `GST_NVEVENT_PAD_DELETED` for the source,
//! 5. drop the one-shot pool.
//!
//! `nvtracker` consumes the service batch (releasing the previous prev-frame
//! pin), then handles `PAD_DELETED` by removing the source.  The synthetic
//! buffer remains pinned until the next regular batch flows, at which point
//! the tracker drops it as well.  All output samples carrying
//! `NvTrackerServiceMeta` are filtered out before reaching consumers, so the
//! caller observes nothing.
//!
//! The same path is invoked from:
//!
//! - explicit user calls (`reason = "manual"`),
//! - [`NvTracker::send_eos`] after emitting `savant.pipeline.source_eos`
//!   (`reason = "eos"`),
//! - the background stale-source evictor (`reason = "stale"`),
//! - [`NvTracker::shutdown`] / drop on each remaining source
//!   (`reason = "drop"`),
//! - the in-line continuity guard inside `finalize_batch_buffer` when a
//!   per-source resolution change or PTS regression is detected
//!   (`reason = "resolution_change…"` / `reason = "pts_regression…"`).

use crate::config::NvTrackerConfig;
use crate::detection_meta::attach_detection_meta;
use crate::error::{NvTrackerError, Result};
use crate::output::{extract_tracker_output, TrackerOutput};
use crate::roi::Roi;
use crate::service_meta::NvTrackerServiceMeta;
use crossbeam::channel::{Receiver, Sender};
use deepstream_buffers::{
    read_slot_dimensions, read_surface_header, BufferGenerator, MetaClearPolicy, NonUniformBatch,
    SavantIdMetaKind, SharedBuffer, SurfaceView,
};
use deepstream_sys;
use gstreamer as gst;
use gstreamer::prelude::*;
use log::{info, warn};
use lru::LruCache;
use parking_lot::Mutex;
use savant_gstreamer::pipeline::{
    build_source_eos_event, parse_source_eos_event, set_element_property, AppsrcPadProbe,
    GstPipeline, PipelineConfig, PipelineInput, PipelineOutput, PtsPolicy,
};
use savant_gstreamer::submit_gate::SubmitGate;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

extern "C" {
    fn gst_nvquery_is_batch_size(query: *mut gst::ffi::GstQuery) -> i32;
    fn gst_nvquery_batch_size_set(query: *mut gst::ffi::GstQuery, batch_size: u32);
    fn gst_nvquery_is_numStreams_size(query: *mut gst::ffi::GstQuery) -> i32;
    fn gst_nvquery_numStreams_size_set(query: *mut gst::ffi::GstQuery, num_streams_size: u32);
}

/// Discriminated output from [`NvTracker::recv`] / [`NvTracker::recv_timeout`] /
/// [`NvTracker::try_recv`].
#[derive(Debug)]
pub enum NvTrackerOutput {
    /// Completed tracking for one submitted batch buffer.
    Tracking(TrackerOutput),
    /// A downstream GStreamer event (not logical per-source EOS).
    Event(gst::Event),
    /// Logical end-of-stream for a specific source (`send_eos`).
    Eos { source_id: String },
    /// Pipeline or framework runtime error.
    Error(NvTrackerError),
}

/// Slot: source name + detections tagged with `class_id`.
type ClassifiedSlots = Vec<(String, Vec<(i32, Roi)>)>;

/// One frame to track: a source identifier, a single-surface GPU buffer, and
/// detections keyed by `class_id`.
#[derive(Debug, Clone)]
pub struct TrackedFrame {
    /// Human-readable stream name (e.g. `"cam-1"`); hashed to `pad_index` internally.
    pub source: String,
    /// Single-surface NVMM buffer acquired from a [`deepstream_buffers::BufferGenerator`].
    pub buffer: SharedBuffer,
    /// Detections grouped by DeepStream `class_id`.
    pub rois: HashMap<i32, Vec<Roi>>,
}

/// Shared state behind [`NvTracker`].  Held as `Arc<NvTrackerShared>` so the
/// stale-source evictor thread can call back into the same submit / reset
/// machinery the public API uses.
struct NvTrackerShared {
    input_tx: Sender<PipelineInput>,
    pipeline: Mutex<GstPipeline>,
    config: NvTrackerConfig,
    policy: MetaClearPolicy,
    /// Serialises the `next_pts + prepare_batch + finalize_batch_buffer +
    /// input_tx.send` sequence inside [`Self::submit`] and owns the
    /// monotonic PTS counter it protects.
    ///
    /// Same rationale as `NvInfer::submit_gate` — see the doc there for
    /// the full failure scenario this guards against.  Housing the
    /// counter inside [`SubmitGate`] makes it impossible to observe or
    /// advance the PTS without holding the gate, which is exactly the
    /// invariant a `Mutex<()>` + sibling `AtomicU64` pair would only
    /// enforce by convention.
    ///
    /// The same gate also serialises [`Self::reset_stream_with_reason`]:
    /// each reset pushes one service buffer (with a freshly assigned PTS)
    /// followed by `GST_NVEVENT_PAD_DELETED`, and must not race with
    /// concurrent real submits or other concurrent resets.
    submit_gate: SubmitGate,
    /// Wall-clock timestamp of the last submit/reset for each pad —
    /// used by the stale-source evictor to decide which sources have
    /// gone idle long enough to warrant an automatic
    /// [`Self::reset_stream_with_reason`].
    last_source_seen: Mutex<HashMap<u32, Instant>>,
    source_lru: Mutex<LruCache<u32, String>>,
    frame_counters: Mutex<HashMap<u32, i32>>,
    last_source_dims: Mutex<HashMap<u32, (u32, u32)>>,
    last_source_pts: Mutex<HashMap<u32, u64>>,
}

/// DeepStream nvtracker wrapper using the shared GStreamer pipeline framework.
pub struct NvTracker {
    shared: Arc<NvTrackerShared>,
    output_rx: Receiver<PipelineOutput>,
    /// Stale-source evictor join handle; dropped (and joined) on
    /// [`Self::shutdown`] / `Drop`.  `None` when
    /// [`NvTrackerConfig::stale_source_after`] is `None`.
    stale_evictor: Mutex<Option<StaleEvictor>>,
}

struct StaleEvictor {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl StaleEvictor {
    fn join(mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// LRU capacity for source_id reverse lookup (compile-time non-zero).
const SOURCE_LRU_CAPACITY: NonZeroUsize = NonZeroUsize::new(4096).unwrap();

impl NvTracker {
    /// Create and start the pipeline in `Playing`.
    pub fn new(config: NvTrackerConfig) -> Result<Self> {
        config.validate()?;
        let policy = config.meta_clear_policy;

        let name_display = if config.name.is_empty() {
            "nvtracker"
        } else {
            config.name.as_str()
        };

        gst::init().map_err(|e| NvTrackerError::GstInit(e.to_string()))?;

        let nvtracker = gst::ElementFactory::make("nvtracker")
            .name("nvtracker")
            .build()
            .map_err(|e| NvTrackerError::ElementCreationFailed {
                element: "nvtracker".into(),
                reason: e.to_string(),
            })?;

        nvtracker.set_property("tracker-width", config.tracker_width);
        nvtracker.set_property("tracker-height", config.tracker_height);
        nvtracker.set_property("ll-lib-file", &config.ll_lib_file);
        nvtracker.set_property("ll-config-file", &config.ll_config_file);
        nvtracker.set_property("gpu-id", config.gpu_id);
        nvtracker.set_property(
            "tracking-id-reset-mode",
            config.tracking_id_reset_mode.as_u32(),
        );
        if nvtracker.find_property("enable-past-frame").is_some() {
            set_element_property(&nvtracker, "enable-past-frame", "1").map_err(|reason| {
                NvTrackerError::InvalidProperty {
                    key: "enable-past-frame".into(),
                    reason,
                }
            })?;
        }

        for (key, value) in &config.element_properties {
            if key == "sub-batches" {
                return Err(NvTrackerError::ConfigError(
                    "\"sub-batches\" must not be set: each NvTracker instance is its own \
                     tracker; use separate NvTracker instances instead of sub-batching"
                        .into(),
                ));
            }
            set_element_property(&nvtracker, key, value).map_err(|reason| {
                NvTrackerError::InvalidProperty {
                    key: key.to_string(),
                    reason,
                }
            })?;
        }

        let elements: Vec<gst::Element> = vec![nvtracker.upcast()];

        let appsrc_caps = gst::Caps::builder("video/x-raw")
            .features(["memory:NVMM"])
            .field("format", config.input_format.gst_name())
            .build();

        let batch_size = config.max_batch_size;
        let appsrc_probe: Option<AppsrcPadProbe> = Some(Box::new(move |pad| {
            pad.add_probe(gst::PadProbeType::QUERY_UPSTREAM, move |_pad, info| {
                if let Some(query) = info.query_mut() {
                    let qptr = query.as_mut_ptr();
                    unsafe {
                        if gst_nvquery_is_batch_size(qptr) != 0 {
                            gst_nvquery_batch_size_set(qptr, batch_size);
                            return gst::PadProbeReturn::Handled;
                        }
                        if gst_nvquery_is_numStreams_size(qptr) != 0 {
                            gst_nvquery_numStreams_size_set(qptr, batch_size);
                            return gst::PadProbeReturn::Handled;
                        }
                    }
                }
                gst::PadProbeReturn::Ok
            });
        }));

        let pipeline_config = PipelineConfig {
            name: name_display.to_string(),
            appsrc_caps,
            elements,
            input_channel_capacity: config.input_channel_capacity,
            output_channel_capacity: config.output_channel_capacity,
            operation_timeout: Some(config.operation_timeout),
            drain_poll_interval: config.drain_poll_interval,
            idle_flush_interval: config.idle_flush_interval,
            appsrc_probe,
            pts_policy: Some(PtsPolicy::StrictPts),
            // The new `reset_stream` (service buffer +
            // `GST_NVEVENT_PAD_DELETED`) releases per-source prev-frame
            // pins reliably *while the pipeline runs*, but the
            // gst::Pipeline drop on shutdown still hangs inside
            // `NvMOT_DeInit` / GObject finalize for any source that has
            // ever submitted (verified by
            // `tests/test_reset_stream_release.rs::test_drop_resets_all_active_sources_without_hang`
            // when set to `false` — the finalizer thread reaches
            // `[NvMultiObjectTracker] De-initialized` and then the
            // gst::Pipeline drop blocks indefinitely).  The leak path
            // is the documented fallback in the unification plan.
            leak_on_finalize: true,
        };

        let (input_tx, output_rx, gst_pipeline) =
            GstPipeline::start(pipeline_config).map_err(NvTrackerError::from)?;

        info!("NvTracker initialized (name={})", name_display);

        let stale_after = config.stale_source_after;
        let shared = Arc::new(NvTrackerShared {
            input_tx,
            pipeline: Mutex::new(gst_pipeline),
            config,
            policy,
            submit_gate: SubmitGate::new(),
            last_source_seen: Mutex::new(HashMap::new()),
            source_lru: Mutex::new(LruCache::new(SOURCE_LRU_CAPACITY)),
            frame_counters: Mutex::new(HashMap::new()),
            last_source_dims: Mutex::new(HashMap::new()),
            last_source_pts: Mutex::new(HashMap::new()),
        });

        let evictor = stale_after.map(|after| spawn_stale_evictor(Arc::clone(&shared), after));

        Ok(Self {
            shared,
            output_rx,
            stale_evictor: Mutex::new(evictor),
        })
    }

    /// Submit a batch of frames for tracking.
    ///
    /// Blocks if the input channel is full (backpressure).
    ///
    /// # Concurrency
    ///
    /// Safe to call from multiple threads; the
    /// [`NvTrackerBatchingOperator`](crate::NvTrackerBatchingOperator) does
    /// so from both its `add_frame` path and its internal timer thread.
    /// The `next_pts + prepare_batch + finalize_batch_buffer +
    /// input_tx.send` window runs inside [`SubmitGate::submit_with`] —
    /// the pipeline feeder thread enforces [`PtsPolicy::StrictPts`] and
    /// must see buffers in PTS-increasing order.  See the matching
    /// comment on `NvInfer::submit` for the full failure scenario this
    /// guards against.
    pub fn submit(&self, frames: &[TrackedFrame], ids: Vec<SavantIdMetaKind>) -> Result<()> {
        self.shared.submit(frames, ids)
    }

    /// Block until the next output is available.
    pub fn recv(&self) -> Result<NvTrackerOutput> {
        loop {
            let output = self
                .output_rx
                .recv()
                .map_err(|_| NvTrackerError::ChannelDisconnected)?;
            if let Some(out) = self.shared.convert_output(output)? {
                return Ok(out);
            }
        }
    }

    /// Block until the next output or timeout. Returns `Ok(None)` on timeout.
    ///
    /// Service-batch outputs (carrying [`NvTrackerServiceMeta`]) are
    /// silently discarded inside the loop — they consume part of the
    /// caller-specified budget but never surface as `Some`.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<NvTrackerOutput>> {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            match self.output_rx.recv_timeout(remaining) {
                Ok(output) => {
                    if let Some(out) = self.shared.convert_output(output)? {
                        return Ok(Some(out));
                    }
                }
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => return Ok(None),
                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                    return Err(NvTrackerError::ChannelDisconnected);
                }
            }
        }
    }

    /// Non-blocking receive.  Skips service-batch outputs transparently.
    pub fn try_recv(&self) -> Result<Option<NvTrackerOutput>> {
        loop {
            match self.output_rx.try_recv() {
                Ok(output) => {
                    if let Some(out) = self.shared.convert_output(output)? {
                        return Ok(Some(out));
                    }
                }
                Err(crossbeam::channel::TryRecvError::Empty) => return Ok(None),
                Err(crossbeam::channel::TryRecvError::Disconnected) => {
                    return Err(NvTrackerError::ChannelDisconnected);
                }
            }
        }
    }

    /// Send a custom GStreamer event into the pipeline input.
    pub fn send_event(&self, event: gst::Event) -> Result<()> {
        self.shared.send_event(event)
    }

    /// Send a logical per-source EOS marker downstream (custom downstream
    /// event), then issue [`Self::reset_stream`] for the same source so the
    /// nvtracker prev-frame pin is released.
    ///
    /// `reset_stream` errors are logged (not propagated): EOS dispatch must
    /// not fail just because the tracker is already shutting down or the
    /// source had never submitted.
    pub fn send_eos(&self, source_id: &str) -> Result<()> {
        self.shared.send_eos(source_id)
    }

    /// Drive the per-source release sequence (service batch +
    /// `GST_NVEVENT_PAD_DELETED`).  See the module docs for the rationale
    /// and lifecycle.  `reason = "manual"`.
    pub fn reset_stream(&self, source_id: &str) -> Result<()> {
        self.shared.reset_stream(source_id)
    }

    pub fn is_failed(&self) -> bool {
        self.shared.is_failed()
    }

    /// Force-flush pending rescue-eligible custom-downstream events
    /// (including `savant.pipeline.source_eos`) through `nvtracker` when
    /// no buffers are in flight.
    ///
    /// See [`savant_gstreamer::pipeline::GstPipeline::flush_idle`] for
    /// semantics.  Returns the number of events flushed; `Ok(0)` means
    /// "nothing pending" or "tracker still busy".
    pub fn flush_idle(&self) -> Result<usize> {
        self.shared.flush_idle()
    }

    /// Graceful shutdown: stop the stale evictor, reset every active
    /// source (releases prev-frame pins), then drain outputs within
    /// `timeout` and stop the pipeline.
    pub fn graceful_shutdown(&self, timeout: Duration) -> Result<Vec<NvTrackerOutput>> {
        self.stop_stale_evictor();
        self.shared.reset_all_active_sources();
        let raw = {
            let mut guard = self.shared.pipeline.lock();
            guard
                .graceful_shutdown(timeout, &self.shared.input_tx, &self.output_rx)
                .map_err(NvTrackerError::from)?
        };
        let mut out = Vec::with_capacity(raw.len());
        for item in raw {
            if let Some(o) = self.shared.convert_output(item)? {
                out.push(o);
            }
        }
        Ok(out)
    }

    /// Abrupt shutdown (used by [`Drop`]).
    pub fn shutdown(&self) -> Result<()> {
        self.stop_stale_evictor();
        self.shared.reset_all_active_sources();
        self.shared
            .pipeline
            .lock()
            .shutdown()
            .map_err(NvTrackerError::from)?;
        Ok(())
    }

    fn stop_stale_evictor(&self) {
        if let Some(ev) = self.stale_evictor.lock().take() {
            ev.join();
        }
    }
}

impl Drop for NvTracker {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

impl NvTrackerShared {
    fn submit(&self, frames: &[TrackedFrame], ids: Vec<SavantIdMetaKind>) -> Result<()> {
        {
            let p = self.pipeline.lock();
            if p.is_draining() {
                return Err(NvTrackerError::ShuttingDown);
            }
            if p.is_failed() {
                return Err(NvTrackerError::PipelineFailed);
            }
        }
        self.submit_gate.submit_with(|next_pts| {
            let (batch, slots, input_pts) = self.prepare_batch(frames, ids)?;
            let buffer = self.finalize_batch_buffer(batch, &slots, &input_pts, next_pts)?;
            self.input_tx
                .send(PipelineInput::Buffer(buffer))
                .map_err(|_| NvTrackerError::ChannelDisconnected)?;
            // Mark each source as freshly seen so the stale-source evictor
            // does not race ahead and reset a source that is still active.
            let now = Instant::now();
            let mut seen = self.last_source_seen.lock();
            for (sid, _) in &slots {
                let pad = crc32fast::hash(sid.as_bytes());
                seen.insert(pad, now);
            }
            Ok(())
        })
    }

    fn send_event(&self, event: gst::Event) -> Result<()> {
        if self.pipeline.lock().is_draining() {
            return Err(NvTrackerError::ShuttingDown);
        }
        self.input_tx
            .send(PipelineInput::Event(event))
            .map_err(|_| NvTrackerError::ChannelDisconnected)?;
        Ok(())
    }

    fn send_eos(&self, source_id: &str) -> Result<()> {
        let event = build_source_eos_event(source_id);
        self.send_event(event)?;
        if let Err(e) = self.reset_stream_with_reason(source_id, "eos") {
            warn!(
                "nvtracker: send_eos({source_id}): post-EOS reset_stream failed: {e}; \
                 prev-frame pin will be released only on shutdown"
            );
        }
        Ok(())
    }

    fn reset_stream(&self, source_id: &str) -> Result<()> {
        if self.pipeline.lock().is_draining() {
            return Err(NvTrackerError::ShuttingDown);
        }
        self.reset_stream_with_reason(source_id, "manual")
    }

    fn reset_stream_with_reason(&self, source_id: &str, reason: &str) -> Result<()> {
        // Acquire the submit gate to prevent any concurrent
        // [`Self::submit`] from interleaving its PTS sequence with our
        // service-buffer push and pad-deleted event.
        self.submit_gate
            .submit_with(|next_pts| self.do_reset_under_gate(source_id, reason, next_pts))
    }

    fn is_failed(&self) -> bool {
        self.pipeline.lock().is_failed()
    }

    fn flush_idle(&self) -> Result<usize> {
        Ok(self.pipeline.lock().flush_idle()?)
    }

    /// Snapshot all currently active sources and run [`Self::reset_stream`]
    /// against each.  Idempotent and safe to call multiple times — sources
    /// removed by previous resets are dropped from the snapshot map.
    fn reset_all_active_sources(&self) {
        let pads: Vec<(u32, String)> = {
            let lru = self.source_lru.lock();
            self.frame_counters
                .lock()
                .keys()
                .copied()
                .map(|pad| {
                    let sid = lru
                        .peek(&pad)
                        .cloned()
                        .unwrap_or_else(|| format!("unknown-{pad:#x}"));
                    (pad, sid)
                })
                .collect()
        };
        if pads.is_empty() {
            return;
        }
        info!(
            "nvtracker: resetting {} active source(s) on shutdown: {:?}",
            pads.len(),
            pads.iter().map(|(_, s)| s.as_str()).collect::<Vec<_>>()
        );
        for (_, sid) in &pads {
            if let Err(e) = self.reset_stream_with_reason(sid, "drop") {
                warn!("nvtracker: reset_stream({sid}, drop) failed: {e}");
            }
        }
    }

    /// Inner reset implementation: assumes the caller already holds
    /// [`SubmitGate`].  Used both from [`Self::reset_stream_with_reason`]
    /// (acquires the gate) and from
    /// [`Self::enforce_source_continuity`] (already inside the gate while
    /// preparing a real submit).
    ///
    /// `next_pts` is the live PTS counter — the service buffer consumes
    /// one tick.
    fn do_reset_under_gate(&self, source_id: &str, reason: &str, next_pts: &mut u64) -> Result<()> {
        let pad = crc32fast::hash(source_id.as_bytes());
        info!(
            "nvtracker: reset_stream(source_id='{}', pad_index={}, reason={})",
            source_id, pad, reason
        );

        // Refresh the LRU before any work so future output samples
        // (e.g. the service buffer if it ever leaks past the filter)
        // can resolve the pad → source_id mapping.
        self.source_lru.lock().put(pad, source_id.to_string());

        // 1) Build a one-shot pool sized exactly for one tracker-resolution
        //    surface.  Min == max == 1: no oversubscription, no leftover
        //    buffers in the pool when the BufferGenerator drops.
        let gen = BufferGenerator::builder(
            self.config.input_format,
            self.config.tracker_width,
            self.config.tracker_height,
        )
        .gpu_id(self.config.gpu_id)
        .min_buffers(1)
        .max_buffers(1)
        .build()
        .map_err(|e| NvTrackerError::PipelineError(format!("service pool build failed: {e}")))?;

        // 2) Acquire one buffer and wrap it in a one-slot NonUniformBatch.
        let shared = gen.acquire(None).map_err(|e| {
            NvTrackerError::PipelineError(format!("service pool acquire failed: {e}"))
        })?;
        let view = SurfaceView::from_buffer(&shared, 0).map_err(|e| {
            NvTrackerError::PipelineError(format!("service SurfaceView::from_buffer failed: {e}"))
        })?;
        let mut nb = NonUniformBatch::new(self.config.gpu_id);
        nb.add(&view).map_err(|e| {
            NvTrackerError::PipelineError(format!("service NonUniformBatch::add failed: {e}"))
        })?;
        let batch_shared = nb.finalize(Vec::new()).map_err(|e| {
            NvTrackerError::PipelineError(format!("service NonUniformBatch::finalize failed: {e}"))
        })?;
        let mut batch =
            batch_shared
                .into_buffer()
                .map_err(|_| NvTrackerError::BufferOwnership {
                    operation: "service_batch".into(),
                })?;

        // 3) Attach NvDsBatchMeta (one frame, no detections) so nvtracker
        //    can iterate the batch, plus the NvTrackerServiceMeta marker
        //    so the output drain can drop the buffer instead of surfacing
        //    it as a TrackerOutput.
        let pts = *next_pts;
        *next_pts = next_pts.checked_add(1).ok_or_else(|| {
            NvTrackerError::PipelineError("service-buffer PTS counter overflow".into())
        })?;
        {
            let buf_ref = batch
                .get_mut()
                .ok_or_else(|| NvTrackerError::BufferNotWritable {
                    operation: "service_batch".into(),
                })?;
            attach_detection_meta(
                buf_ref,
                1,
                self.config.max_batch_size,
                &[(pad, Vec::new())],
                &[0],
            )?;
            NvTrackerServiceMeta::add(buf_ref, pad);
            buf_ref.set_pts(gst::ClockTime::from_nseconds(pts));
        }

        // 4) Push the service buffer.
        self.input_tx
            .send(PipelineInput::Buffer(batch))
            .map_err(|_| NvTrackerError::ChannelDisconnected)?;

        // 5) Push GST_NVEVENT_PAD_DELETED *after* the service buffer.
        let ev_ptr = unsafe { deepstream_sys::gst_nvevent_new_pad_deleted(pad) };
        if ev_ptr.is_null() {
            return Err(NvTrackerError::PipelineError(
                "gst_nvevent_new_pad_deleted returned null".into(),
            ));
        }
        let event = unsafe { gst::Event::from_glib_full(ev_ptr as *mut gst::ffi::GstEvent) };
        self.input_tx
            .send(PipelineInput::Event(event))
            .map_err(|_| NvTrackerError::ChannelDisconnected)?;

        // 6) Drop the one-shot generator: deactivates the pool.  The
        //    one outstanding service buffer keeps its memory alive via
        //    its own ref until the tracker releases it on the next
        //    regular batch.
        drop(gen);

        // 7) Cleanup per-source bookkeeping.
        self.frame_counters.lock().remove(&pad);
        self.last_source_dims.lock().remove(&pad);
        self.last_source_pts.lock().remove(&pad);
        self.last_source_seen.lock().remove(&pad);

        Ok(())
    }

    fn convert_output(&self, output: PipelineOutput) -> Result<Option<NvTrackerOutput>> {
        match output {
            PipelineOutput::Buffer(buffer) => {
                if NvTrackerServiceMeta::is_present(buffer.as_ref()) {
                    // Service buffer round-tripped through the tracker.
                    // Drop it here so the prev-frame pin it now holds is
                    // released the moment the buffer goes out of scope.
                    return Ok(None);
                }
                let resolve = |pad: u32| {
                    self.source_lru
                        .lock()
                        .get(&pad)
                        .cloned()
                        .unwrap_or_else(|| format!("unknown-{pad:#x}"))
                };
                let tracker_output =
                    extract_tracker_output(buffer, resolve, self.policy.clear_after())?;
                Ok(tracker_output.map(NvTrackerOutput::Tracking))
            }
            PipelineOutput::Eos => Ok(Some(NvTrackerOutput::Error(NvTrackerError::PipelineError(
                "unexpected hard GStreamer EOS in NvTracker output".into(),
            )))),
            PipelineOutput::Event(event) => {
                if let Some(source_id) = parse_source_eos_event(&event) {
                    Ok(Some(NvTrackerOutput::Eos { source_id }))
                } else {
                    Ok(Some(NvTrackerOutput::Event(event)))
                }
            }
            PipelineOutput::Error(e) => Ok(Some(NvTrackerOutput::Error(NvTrackerError::from(e)))),
        }
    }

    fn prepare_batch(
        &self,
        frames: &[TrackedFrame],
        ids: Vec<SavantIdMetaKind>,
    ) -> Result<(gst::Buffer, ClassifiedSlots, Vec<Option<u64>>)> {
        if frames.is_empty() {
            return Err(NvTrackerError::batch_meta(
                "prepare_batch",
                "frames must not be empty",
            ));
        }

        let mut nb = NonUniformBatch::new(self.config.gpu_id);
        for (i, frame) in frames.iter().enumerate() {
            let view = SurfaceView::from_buffer(&frame.buffer, 0).map_err(|e| {
                NvTrackerError::PipelineError(format!(
                    "SurfaceView::from_buffer failed for frame {i}: {e}"
                ))
            })?;
            nb.add(&view).map_err(|e| {
                NvTrackerError::PipelineError(format!(
                    "NonUniformBatch::add failed for frame {i}: {e}"
                ))
            })?;
        }

        let shared = nb.finalize(ids).map_err(|e| {
            NvTrackerError::PipelineError(format!("NonUniformBatch::finalize failed: {e}"))
        })?;

        let batch = shared
            .into_buffer()
            .map_err(|_| NvTrackerError::BufferOwnership {
                operation: "prepare_batch".into(),
            })?;

        let slots: ClassifiedSlots = frames
            .iter()
            .map(|f| {
                let flat: Vec<(i32, Roi)> = f
                    .rois
                    .iter()
                    .flat_map(|(&cid, rois)| rois.iter().map(move |r| (cid, r.clone())))
                    .collect();
                (f.source.clone(), flat)
            })
            .collect();
        let input_pts: Vec<Option<u64>> = frames.iter().map(|f| f.buffer.pts_ns()).collect();

        Ok((batch, slots, input_pts))
    }

    fn finalize_batch_buffer(
        &self,
        mut batch: gst::Buffer,
        slots: &ClassifiedSlots,
        input_pts: &[Option<u64>],
        next_pts: &mut u64,
    ) -> Result<gst::Buffer> {
        let (num_filled, max_batch_size) = read_surface_header(&batch)
            .map_err(|e| NvTrackerError::batch_meta("read_surface_header", e.to_string()))?;

        let slot_dims = if num_filled > 0 {
            read_slot_dimensions(&batch, num_filled)
                .map_err(|e| NvTrackerError::batch_meta("read_slot_dimensions", e.to_string()))?
        } else {
            Vec::new()
        };

        if slots.len() != num_filled as usize {
            return Err(NvTrackerError::batch_meta(
                "push_buffer",
                format!("slots.len() {} != num_filled {}", slots.len(), num_filled),
            ));
        }
        if input_pts.len() != slots.len() {
            return Err(NvTrackerError::batch_meta(
                "push_buffer",
                format!(
                    "input_pts.len() {} != slots.len() {}",
                    input_pts.len(),
                    slots.len()
                ),
            ));
        }

        validate_per_source_resolution(slots, &slot_dims, num_filled)?;
        // Continuity may push service buffers, advancing `next_pts`.  Do
        // it before reserving a PTS for the real batch so the real batch
        // remains strictly after any service batches.
        self.enforce_source_continuity(slots, &slot_dims, input_pts, next_pts)?;

        let meta_slots: Vec<(u32, Vec<(i32, Roi)>)> = slots
            .iter()
            .map(|(sid, rois)| {
                let pad = crc32fast::hash(sid.as_bytes());
                self.source_lru.lock().put(pad, sid.clone());
                (pad, rois.clone())
            })
            .collect();

        let frame_nums = {
            let mut counters = self.frame_counters.lock();
            let mut nums = Vec::with_capacity(slots.len());
            for (sid, _) in slots {
                let pad = crc32fast::hash(sid.as_bytes());
                let entry = counters.entry(pad).or_insert(0);
                let current = *entry;
                *entry = entry
                    .checked_add(1)
                    .ok_or_else(|| NvTrackerError::FrameNumOverflow {
                        pad_index: pad,
                        source_id: sid.clone(),
                    })?;
                nums.push(current);
            }
            nums
        };

        if self.policy.clear_before() {
            let buf_ref = batch
                .get_mut()
                .ok_or_else(|| NvTrackerError::BufferNotWritable {
                    operation: "meta_clear_policy_before".into(),
                })?;
            unsafe {
                deepstream::clear_all_frame_objects(
                    buf_ref.as_mut_ptr() as *mut deepstream_sys::GstBuffer
                );
            }
        }

        {
            let buf_ref = batch
                .get_mut()
                .ok_or_else(|| NvTrackerError::BufferNotWritable {
                    operation: "attach_detection_meta".into(),
                })?;
            attach_detection_meta(
                buf_ref,
                num_filled,
                max_batch_size,
                &meta_slots,
                &frame_nums,
            )?;
        }

        if !slot_dims.is_empty() {
            let buf_ref = batch
                .get_mut()
                .ok_or_else(|| NvTrackerError::BufferNotWritable {
                    operation: "patch_source_frame_dimensions".into(),
                })?;
            let buf_ptr = buf_ref.as_mut_ptr() as *mut deepstream_sys::GstBuffer;
            let batch_meta = unsafe { deepstream_sys::gst_buffer_get_nvds_batch_meta(buf_ptr) };
            if !batch_meta.is_null() {
                let mut frame_list = unsafe { (*batch_meta).frame_meta_list };
                let mut slot: usize = 0;
                while !frame_list.is_null() && slot < slot_dims.len() {
                    let frame_ptr =
                        unsafe { (*frame_list).data as *mut deepstream_sys::NvDsFrameMeta };
                    if !frame_ptr.is_null() {
                        let (w, h) = slot_dims.get(slot).copied().ok_or(
                            NvTrackerError::SlotIndexOutOfBounds {
                                index: slot as u32,
                                num_filled: slot_dims.len() as u32,
                                operation: "patch_source_frame_dimensions".into(),
                            },
                        )?;
                        unsafe {
                            (*frame_ptr).source_frame_width = w;
                            (*frame_ptr).source_frame_height = h;
                        }
                        slot += 1;
                    }
                    frame_list = unsafe { (*frame_list).next };
                }
            }
        }

        let pts = *next_pts;
        *next_pts = next_pts
            .checked_add(1)
            .ok_or_else(|| NvTrackerError::PipelineError("PTS counter overflow".into()))?;
        {
            let buf_ref = batch
                .get_mut()
                .ok_or_else(|| NvTrackerError::BufferNotWritable {
                    operation: "set_pts".into(),
                })?;
            buf_ref.set_pts(gst::ClockTime::from_nseconds(pts));
        }

        Ok(batch)
    }

    fn enforce_source_continuity(
        &self,
        slots: &ClassifiedSlots,
        slot_dims: &[(u32, u32)],
        input_pts: &[Option<u64>],
        next_pts: &mut u64,
    ) -> Result<()> {
        let mut projected_dims = HashMap::new();
        let mut projected_pts = HashMap::new();
        {
            let dims = self.last_source_dims.lock();
            let pts = self.last_source_pts.lock();
            for (sid, _) in slots.iter() {
                let pad = crc32fast::hash(sid.as_bytes());
                if let Some(dim) = dims.get(&pad).copied() {
                    projected_dims.insert(pad, dim);
                }
                if let Some(p) = pts.get(&pad).copied() {
                    projected_pts.insert(pad, p);
                }
            }
        }

        let mut reset_reasons: HashMap<u32, (String, Vec<String>)> = HashMap::new();

        for (idx, ((sid, _), (w, h))) in slots.iter().zip(slot_dims.iter()).enumerate() {
            let pad = crc32fast::hash(sid.as_bytes());
            let current_dim = (*w, *h);

            if let Some((prev_w, prev_h)) = projected_dims.get(&pad).copied() {
                if (prev_w, prev_h) != current_dim {
                    let entry = reset_reasons
                        .entry(pad)
                        .or_insert_with(|| (sid.clone(), Vec::new()));
                    entry.1.push(format!(
                        "resolution_change old={}x{} new={}x{}",
                        prev_w, prev_h, current_dim.0, current_dim.1
                    ));
                }
            }

            if let Some(current_pts) = input_pts[idx] {
                if let Some(prev_pts) = projected_pts.get(&pad).copied() {
                    if current_pts < prev_pts {
                        let entry = reset_reasons
                            .entry(pad)
                            .or_insert_with(|| (sid.clone(), Vec::new()));
                        entry.1.push(format!(
                            "pts_regression prev_pts={} new_pts={}",
                            prev_pts, current_pts
                        ));
                    }
                }
                projected_pts.insert(pad, current_pts);
            }

            projected_dims.insert(pad, current_dim);
        }

        for (_pad, (source_id, reasons)) in reset_reasons.iter() {
            let reason = reasons.join(", ");
            // Already inside `submit_gate.submit_with`; call the inner
            // helper directly to avoid re-entering the gate.
            self.do_reset_under_gate(source_id, &reason, next_pts)?;
        }

        {
            let mut dims = self.last_source_dims.lock();
            for (sid, _) in slots.iter() {
                let pad = crc32fast::hash(sid.as_bytes());
                if let Some(dim) = projected_dims.get(&pad).copied() {
                    dims.insert(pad, dim);
                }
            }
        }
        {
            let mut pts = self.last_source_pts.lock();
            for (sid, _) in slots.iter() {
                let pad = crc32fast::hash(sid.as_bytes());
                if let Some(p) = projected_pts.get(&pad).copied() {
                    pts.insert(pad, p);
                }
            }
        }

        Ok(())
    }
}

fn validate_per_source_resolution(
    slots: &ClassifiedSlots,
    slot_dims: &[(u32, u32)],
    num_filled: u32,
) -> Result<()> {
    let mut first_for_pad: HashMap<u32, (u32, u32, u32)> = HashMap::new();
    for (i, (sid, _)) in slots.iter().enumerate() {
        let pad = crc32fast::hash(sid.as_bytes());
        let (w, h) = slot_dims
            .get(i)
            .copied()
            .ok_or(NvTrackerError::SlotIndexOutOfBounds {
                index: i as u32,
                num_filled,
                operation: "validate_per_source_resolution".into(),
            })?;
        if let Some(&(w0, h0, slot_a)) = first_for_pad.get(&pad) {
            if (w0, h0) != (w, h) {
                return Err(NvTrackerError::ResolutionMismatch {
                    source_id: sid.clone(),
                    slot_a,
                    w_a: w0,
                    h_a: h0,
                    slot_b: i as u32,
                    w_b: w,
                    h_b: h,
                });
            }
        } else {
            first_for_pad.insert(pad, (w, h, i as u32));
        }
    }
    Ok(())
}

/// Spawn the stale-source evictor.  Polls every `min(after / 5, 1s)` and
/// invokes `reset_stream_with_reason(_, "stale")` for any pad whose last
/// `last_source_seen` instant is older than `after`.
fn spawn_stale_evictor(shared: Arc<NvTrackerShared>, after: Duration) -> StaleEvictor {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::clone(&stop);
    // Poll often enough that a freshly-stale source is reset within `after / 5`,
    // capped at 1 s so we don't waste cycles when `after` is very large.
    let tick = std::cmp::min(after / 5, Duration::from_secs(1)).max(Duration::from_millis(50));
    let handle = thread::Builder::new()
        .name("nvtracker-stale-evictor".into())
        .spawn(move || {
            while !stop_flag.load(Ordering::Acquire) {
                thread::sleep(tick);
                if stop_flag.load(Ordering::Acquire) {
                    break;
                }
                if shared.pipeline.lock().is_draining() {
                    continue;
                }
                let now = Instant::now();
                let stale: Vec<(u32, String)> = {
                    let seen = shared.last_source_seen.lock();
                    let lru = shared.source_lru.lock();
                    seen.iter()
                        .filter(|(_, last)| now.saturating_duration_since(**last) >= after)
                        .map(|(pad, _)| {
                            let sid = lru
                                .peek(pad)
                                .cloned()
                                .unwrap_or_else(|| format!("unknown-{pad:#x}"));
                            (*pad, sid)
                        })
                        .collect()
                };
                for (_, sid) in &stale {
                    if stop_flag.load(Ordering::Acquire) {
                        break;
                    }
                    if let Err(e) = shared.reset_stream_with_reason(sid, "stale") {
                        warn!("nvtracker: stale-evictor reset_stream({sid}) failed: {e}");
                    }
                }
            }
        })
        .expect("spawn nvtracker-stale-evictor");
    StaleEvictor {
        stop,
        handle: Some(handle),
    }
}

/// Default low-level tracker library path on a standard DeepStream install.
pub fn default_ll_lib_path() -> String {
    let p = Path::new("/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so");
    p.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config() -> NvTrackerConfig {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let lib = dir.join(format!("nvtracker_pipe_lib_{pid}.so"));
        let yml = dir.join(format!("nvtracker_pipe_cfg_{pid}.yml"));
        std::fs::write(&lib, b"x").unwrap();
        std::fs::write(&yml, b"y").unwrap();
        NvTrackerConfig::new(
            lib.to_string_lossy().into_owned(),
            yml.to_string_lossy().into_owned(),
        )
    }

    #[test]
    fn test_recv_timeout_returns_none() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut c = minimal_config();
        c.operation_timeout = Duration::from_secs(60);
        c.stale_source_after = None;
        let tracker = match NvTracker::new(c) {
            Ok(t) => t,
            Err(_) => return, // nvtracker element unavailable in CI
        };
        assert!(!tracker.is_failed());
        let r = tracker
            .recv_timeout(Duration::from_millis(10))
            .expect("recv_timeout");
        assert!(r.is_none());
        let _ = tracker.shutdown();
    }

    #[test]
    fn test_try_recv_returns_none() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut c = minimal_config();
        c.operation_timeout = Duration::from_secs(60);
        c.stale_source_after = None;
        let tracker = match NvTracker::new(c) {
            Ok(t) => t,
            Err(_) => return,
        };
        let r = tracker.try_recv().expect("try_recv");
        assert!(r.is_none());
        let _ = tracker.shutdown();
    }

    #[test]
    fn test_is_failed_false_on_healthy() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut c = minimal_config();
        c.operation_timeout = Duration::from_secs(60);
        c.stale_source_after = None;
        let tracker = match NvTracker::new(c) {
            Ok(t) => t,
            Err(_) => return,
        };
        assert!(!tracker.is_failed());
        let _ = tracker.shutdown();
    }

    /// Requires GPU + DeepStream nvtracker; ignored by default.
    #[test]
    #[ignore]
    fn test_submit_recv_single_frame() {
        let _ = env_logger::builder().is_test(true).try_init();
        let c = minimal_config();
        let tracker = NvTracker::new(c).expect("nvtracker new");
        // Real test would build NVMM frame + detections; placeholder documents hook.
        let _ = tracker;
    }

    #[test]
    #[ignore]
    fn test_submit_recv_multi_source() {}

    #[test]
    #[ignore]
    fn test_send_eos_round_trip() {}

    #[test]
    #[ignore]
    fn test_reset_stream_resets_tracking_ids() {}

    #[test]
    #[ignore]
    fn test_resolution_change_triggers_auto_reset() {}

    #[test]
    #[ignore]
    fn test_pts_regression_triggers_auto_reset() {}

    #[test]
    #[ignore]
    fn test_same_resolution_no_reset() {}
}
