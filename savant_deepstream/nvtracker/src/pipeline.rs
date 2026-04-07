//! `NvTracker`: appsrc → nvtracker → appsink.
//!
//! ## GStreamer properties (manual batched flow)
//!
//! - **`appsrc`**: `format=time`, `stream-type=stream`, caps `video/x-raw(memory:NVMM)`.
//!   Each pushed buffer gets an explicit PTS from an internal monotonic counter (`next_pts`), so
//!   `track_sync` can correlate request→response even without a wall-clock source.
//!   A pad probe on appsrc's src pad answers DeepStream custom queries
//!   (`gst_nvquery_batch_size` / `gst_nvquery_numStreams_size`) with `config.max_batch_size`.
//! - **`appsink`**: `sync=false`, `emit-signals=true` — low-latency pull of completed buffers.
//! - **`nvtracker`**: `GstBaseTransform` in-place — operates on the same buffer, so no queue or
//!   meta bridge is needed.  The `sub-batches` property is **rejected** — each `NvTracker`
//!   instance is its own isolated tracker; create separate instances for different tracking
//!   workloads instead of sub-batching within one element.

use crate::config::NvTrackerConfig;
use crate::detection_meta::attach_detection_meta;
use crate::error::{NvTrackerError, Result};
use crate::output::{extract_tracker_output, TrackerOutput};
use crate::roi::Roi;
use deepstream_buffers::{NonUniformBatch, SavantIdMetaKind, SharedBuffer, SurfaceView};
use deepstream_sys;
use glib::translate::from_glib_none;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_app::AppSinkCallbacks;
use log::{error, info};
use lru::LruCache;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

extern "C" {
    fn gst_nvquery_is_batch_size(query: *mut gst::ffi::GstQuery) -> i32;
    fn gst_nvquery_batch_size_set(query: *mut gst::ffi::GstQuery, batch_size: u32);
    fn gst_nvquery_is_numStreams_size(query: *mut gst::ffi::GstQuery) -> i32;
    fn gst_nvquery_numStreams_size_set(query: *mut gst::ffi::GstQuery, num_streams_size: u32);
}

/// Async completion callback.
pub type TrackerCallback = Box<dyn FnMut(TrackerOutput) + Send>;

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

struct SampleDelivery {
    callback: Mutex<Option<TrackerCallback>>,
    sync_tx: Mutex<std::collections::HashMap<u64, mpsc::Sender<TrackerOutput>>>,
}

/// DeepStream nvtracker wrapper.
///
/// GStreamer fields are wrapped in [`ManuallyDrop`] because the DeepStream
/// nvtracker plugin spawns CUDA worker threads that may never terminate during
/// GObject finalization. After [`shutdown`](Self::shutdown) transitions the
/// pipeline to `Null`, these wrappers are intentionally **not** dropped so the
/// process can exit cleanly. The OS reclaims all resources on exit.
pub struct NvTracker {
    pipeline: ManuallyDrop<gst::Pipeline>,
    appsrc: ManuallyDrop<gst_app::AppSrc>,
    #[allow(dead_code)]
    appsink: ManuallyDrop<gst_app::AppSink>,
    config: NvTrackerConfig,
    delivery: Arc<SampleDelivery>,
    next_pts: AtomicU64,
    /// Reverse lookup: `pad_index` (crc32 of `source_id`) → original string.
    source_lru: Arc<Mutex<LruCache<u32, String>>>,
    /// Per `pad_index`, next `frame_num` to assign (monotonic across batches).
    frame_counters: Arc<Mutex<HashMap<u32, i32>>>,
    /// Last observed frame resolution `(w, h)` per `pad_index`.
    last_source_dims: Arc<Mutex<HashMap<u32, (u32, u32)>>>,
    /// Last observed input frame PTS per `pad_index` (nanoseconds).
    last_source_pts: Arc<Mutex<HashMap<u32, u64>>>,
    /// Whether `shutdown()` has already been called.
    is_shut_down: bool,
    /// Terminal failed flag — set when an operation timeout is exceeded.
    failed: Arc<AtomicBool>,
    /// PTS → submission instant for in-flight buffers (both sync and async).
    in_flight: Arc<Mutex<HashMap<u64, Instant>>>,
    /// Watchdog thread handle.
    _watchdog_thread: Option<std::thread::JoinHandle<()>>,
    /// Shutdown flag for the watchdog thread.
    watchdog_shutdown: Arc<AtomicBool>,
}

/// LRU capacity for source_id reverse lookup (compile-time non-zero).
const SOURCE_LRU_CAPACITY: NonZeroUsize = NonZeroUsize::new(4096).unwrap();

impl NvTracker {
    /// Create and start the pipeline in `Playing`.
    pub fn new(config: NvTrackerConfig, callback: TrackerCallback) -> Result<Self> {
        config.validate()?;

        let name_display = if config.name.is_empty() {
            "nvtracker"
        } else {
            config.name.as_str()
        };

        gst::init().map_err(|e| NvTrackerError::GstInit(e.to_string()))?;

        let pipeline = gst::Pipeline::new();

        let appsrc = gst::ElementFactory::make("appsrc")
            .name("src")
            .build()
            .map_err(|e| NvTrackerError::ElementCreationFailed {
                element: "appsrc".into(),
                reason: e.to_string(),
            })?;

        let appsink = gst::ElementFactory::make("appsink")
            .name("sink")
            .build()
            .map_err(|e| NvTrackerError::ElementCreationFailed {
                element: "appsink".into(),
                reason: e.to_string(),
            })?;

        let appsrc_caps = gst::Caps::builder("video/x-raw")
            .features(["memory:NVMM"])
            .field("format", config.input_format.gst_name())
            .build();
        let appsrc_elem: &gst::Element = appsrc.upcast_ref();
        appsrc_elem.set_property("caps", &appsrc_caps);
        appsrc_elem.set_property_from_str("format", "time");
        appsrc_elem.set_property_from_str("stream-type", "stream");

        let appsink_elem: &gst::Element = appsink.upcast_ref();
        appsink_elem.set_property("sync", false);
        appsink_elem.set_property("emit-signals", true);

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
        // Keep past-frame metadata enabled by default so shadow/terminated lists can be emitted
        // by tracker backends that support them.
        if nvtracker.find_property("enable-past-frame").is_some() {
            Self::set_element_property(&nvtracker, "enable-past-frame", "1")?;
        }

        for (key, value) in &config.element_properties {
            if key == "sub-batches" {
                return Err(NvTrackerError::ConfigError(
                    "\"sub-batches\" must not be set: each NvTracker instance is its own \
                     tracker; use separate NvTracker instances instead of sub-batching"
                        .into(),
                ));
            }
            Self::set_element_property(&nvtracker, key, value)?;
        }

        let elements: Vec<gst::Element> = if config.queue_depth > 0 {
            let queue = gst::ElementFactory::make("queue")
                .name("queue")
                .build()
                .map_err(|e| NvTrackerError::ElementCreationFailed {
                    element: "queue".into(),
                    reason: e.to_string(),
                })?;
            queue.set_property("max-size-buffers", config.queue_depth);
            queue.set_property("max-size-bytes", 0u32);
            queue.set_property("max-size-time", 0u64);
            vec![
                appsrc.clone().upcast(),
                queue,
                nvtracker.clone().upcast(),
                appsink.clone().upcast(),
            ]
        } else {
            vec![
                appsrc.clone().upcast(),
                nvtracker.clone().upcast(),
                appsink.clone().upcast(),
            ]
        };
        for elem in &elements {
            pipeline.add(elem).map_err(|e| {
                NvTrackerError::PipelineError(format!("Failed to add element: {}", e))
            })?;
        }
        gst::Element::link_many(elements.iter()).map_err(|_| NvTrackerError::LinkFailed {
            chain: "appsrc->[queue]->nvtracker->appsink".into(),
        })?;

        let appsrc_typed: gst_app::AppSrc =
            appsrc.dynamic_cast::<gst_app::AppSrc>().map_err(|_| {
                NvTrackerError::ElementCreationFailed {
                    element: "appsrc".into(),
                    reason: "dynamic_cast to AppSrc failed".into(),
                }
            })?;

        {
            let batch_size = config.max_batch_size;
            let src_pad = appsrc_typed
                .static_pad("src")
                .ok_or_else(|| NvTrackerError::PipelineError("appsrc has no src pad".into()))?;
            src_pad.add_probe(gst::PadProbeType::QUERY_UPSTREAM, move |_pad, info| {
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
        }

        let appsink_typed: gst_app::AppSink =
            appsink.dynamic_cast::<gst_app::AppSink>().map_err(|_| {
                NvTrackerError::ElementCreationFailed {
                    element: "appsink".into(),
                    reason: "dynamic_cast to AppSink failed".into(),
                }
            })?;

        let delivery = Arc::new(SampleDelivery {
            callback: Mutex::new(Some(callback)),
            sync_tx: Mutex::new(std::collections::HashMap::new()),
        });
        let failed = Arc::new(AtomicBool::new(false));
        let in_flight: Arc<Mutex<HashMap<u64, Instant>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let delivery_clone = delivery.clone();
        let in_flight_cb = in_flight.clone();
        let lru_inner = Arc::new(Mutex::new(LruCache::new(SOURCE_LRU_CAPACITY)));

        let lru_cb = Arc::clone(&lru_inner);
        let callbacks = AppSinkCallbacks::builder()
            .new_sample(move |appsink| {
                let sample = appsink.pull_sample().map_err(|e| {
                    log::error!("appsink pull_sample error: {:?}", e);
                    gst::FlowError::Error
                })?;
                let buffer_ref = sample.buffer().ok_or_else(|| {
                    log::error!("sample has no buffer");
                    gst::FlowError::Error
                })?;
                let buffer: gst::Buffer = unsafe { from_glib_none(buffer_ref.as_ptr()) };
                let pts_key = buffer.pts().map(|t| t.nseconds());

                // Clear in-flight entry on successful delivery.
                if let Some(pts) = pts_key {
                    in_flight_cb.lock().remove(&pts);
                }

                let resolve = |pad: u32| {
                    lru_cb
                        .lock()
                        .get(&pad)
                        .cloned()
                        .unwrap_or_else(|| format!("unknown-{pad:#x}"))
                };
                let output = extract_tracker_output(buffer, resolve).map_err(|e| {
                    log::error!("extract_tracker_output: {}", e);
                    gst::FlowError::Error
                })?;

                let sync_sender = pts_key.and_then(|id| delivery_clone.sync_tx.lock().remove(&id));
                if let Some(tx) = sync_sender {
                    let _ = tx.send(output);
                } else if let Some(ref mut cb) = *delivery_clone.callback.lock() {
                    cb(output);
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build();
        appsink_typed.set_callbacks(callbacks);

        pipeline.set_state(gst::State::Playing).map_err(|e| {
            NvTrackerError::PipelineError(format!("Failed to start pipeline: {}", e))
        })?;

        let operation_timeout = config.operation_timeout;

        // Spawn watchdog thread for async in-flight deadline.
        let watchdog_shutdown = Arc::new(AtomicBool::new(false));
        let wd_in_flight = in_flight.clone();
        let wd_failed = failed.clone();
        let wd_shutdown = watchdog_shutdown.clone();
        let wd_timeout = operation_timeout;
        let wd_name = format!("{}-watchdog", name_display);
        let watchdog_thread = std::thread::Builder::new()
            .name(wd_name)
            .spawn(move || {
                let tick = wd_timeout / 2;
                loop {
                    std::thread::sleep(tick);
                    if wd_shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    if wd_failed.load(Ordering::Acquire) {
                        return;
                    }
                    let now = Instant::now();
                    let expired: Vec<u64> = wd_in_flight
                        .lock()
                        .iter()
                        .filter(|(_, submitted)| now.duration_since(**submitted) > wd_timeout)
                        .map(|(&pts, _)| pts)
                        .collect();
                    if !expired.is_empty() {
                        for pts in &expired {
                            wd_in_flight.lock().remove(pts);
                        }
                        error!(
                            "NvTracker: {} buffer(s) exceeded operation_timeout ({:?}), \
                             pipeline entering failed state",
                            expired.len(),
                            wd_timeout
                        );
                        wd_failed.store(true, Ordering::Release);
                        return;
                    }
                }
            })
            .ok();

        info!(
            "NvTracker initialized (name={}, queue_depth={})",
            name_display, config.queue_depth
        );

        Ok(Self {
            pipeline: ManuallyDrop::new(pipeline),
            appsrc: ManuallyDrop::new(appsrc_typed),
            appsink: ManuallyDrop::new(appsink_typed),
            config,
            delivery,
            next_pts: AtomicU64::new(0),
            source_lru: lru_inner,
            frame_counters: Arc::new(Mutex::new(HashMap::new())),
            last_source_dims: Arc::new(Mutex::new(HashMap::new())),
            last_source_pts: Arc::new(Mutex::new(HashMap::new())),
            is_shut_down: false,
            failed,
            in_flight,
            _watchdog_thread: watchdog_thread,
            watchdog_shutdown,
        })
    }

    /// Push frames through the tracker (async). A [`NonUniformBatch`] is built
    /// internally from the per-frame buffers.
    pub fn track(&self, frames: &[TrackedFrame], ids: Vec<SavantIdMetaKind>) -> Result<()> {
        if self.failed.load(Ordering::Acquire) {
            return Err(NvTrackerError::PipelineFailed);
        }
        let pts = self.next_pts.fetch_add(1, Ordering::Relaxed);
        self.in_flight.lock().insert(pts, Instant::now());
        let (batch, slots, input_pts) = match self.prepare_batch(frames, ids) {
            Ok(v) => v,
            Err(e) => {
                self.in_flight.lock().remove(&pts);
                return Err(e);
            }
        };
        if let Err(e) = self.push_buffer(batch, &slots, &input_pts, pts) {
            self.in_flight.lock().remove(&pts);
            return Err(e);
        }
        Ok(())
    }

    /// Push frames through the tracker; block until the result is available or
    /// `operation_timeout` is exceeded. On timeout, the pipeline enters a
    /// terminal failed state and must be recreated.
    pub fn track_sync(
        &self,
        frames: &[TrackedFrame],
        ids: Vec<SavantIdMetaKind>,
    ) -> Result<TrackerOutput> {
        if self.failed.load(Ordering::Acquire) {
            return Err(NvTrackerError::PipelineFailed);
        }
        let pts = self.next_pts.fetch_add(1, Ordering::Relaxed);
        self.in_flight.lock().insert(pts, Instant::now());
        let (tx, rx) = mpsc::channel();
        {
            self.delivery.sync_tx.lock().insert(pts, tx);
        }
        let (batch, slots, input_pts) = match self.prepare_batch(frames, ids) {
            Ok(v) => v,
            Err(e) => {
                self.in_flight.lock().remove(&pts);
                self.delivery.sync_tx.lock().remove(&pts);
                return Err(e);
            }
        };
        if let Err(e) = self.push_buffer(batch, &slots, &input_pts, pts) {
            self.in_flight.lock().remove(&pts);
            self.delivery.sync_tx.lock().remove(&pts);
            return Err(e);
        }
        match rx.recv_timeout(self.config.operation_timeout) {
            Ok(output) => {
                // in_flight already cleared by appsink callback
                Ok(output)
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.in_flight.lock().remove(&pts);
                self.delivery.sync_tx.lock().remove(&pts);
                error!(
                    "NvTracker: track_sync timed out after {:?}, pipeline entering failed state",
                    self.config.operation_timeout
                );
                self.failed.store(true, Ordering::Release);
                Err(NvTrackerError::PipelineFailed)
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.in_flight.lock().remove(&pts);
                self.delivery.sync_tx.lock().remove(&pts);
                error!("NvTracker: track_sync channel disconnected, pipeline entering failed state");
                self.failed.store(true, Ordering::Release);
                Err(NvTrackerError::PipelineFailed)
            }
        }
    }

    /// Build a [`NonUniformBatch`] from individual frame buffers and derive the
    /// internal slot representation `(source_id, classified_rois)`.
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

        let slots: Vec<(String, Vec<(i32, Roi)>)> = frames
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

    fn push_buffer(
        &self,
        mut batch: gst::Buffer,
        slots: &ClassifiedSlots,
        input_pts: &[Option<u64>],
        pts: u64,
    ) -> Result<()> {
        let (num_filled, max_batch_size) = read_surface_header(&batch)?;

        let slot_dims = if num_filled > 0 {
            read_slot_dimensions(&batch, num_filled)?
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
        self.enforce_source_continuity(slots, &slot_dims, input_pts)?;

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

        {
            let buf_ref = batch
                .get_mut()
                .ok_or_else(|| NvTrackerError::BufferNotWritable {
                    operation: "set_pts".into(),
                })?;
            buf_ref.set_pts(gst::ClockTime::from_nseconds(pts));
        }

        self.appsrc
            .push_buffer(batch)
            .map_err(|e| NvTrackerError::PipelineError(format!("appsrc push failed: {:?}", e)))?;

        Ok(())
    }

    /// Send `GST_NVEVENT_STREAM_RESET` for the stream identified by `source_id` (crc32 → pad id).
    pub fn reset_stream(&self, source_id: &str) -> Result<()> {
        self.reset_stream_with_reason(source_id, "manual")
    }

    fn reset_stream_with_reason(&self, source_id: &str, reason: &str) -> Result<()> {
        let pad = crc32fast::hash(source_id.as_bytes());
        info!(
            "Resetting tracker stream state: source_id='{}' pad_index={} reason={}",
            source_id, pad, reason
        );
        self.source_lru.lock().put(pad, source_id.to_string());
        self.frame_counters.lock().insert(pad, 0);
        self.last_source_dims.lock().remove(&pad);
        self.last_source_pts.lock().remove(&pad);

        let ev_ptr = unsafe { deepstream_sys::gst_nvevent_new_stream_reset(pad) };
        if ev_ptr.is_null() {
            return Err(NvTrackerError::PipelineError(
                "gst_nvevent_new_stream_reset returned null".into(),
            ));
        }
        let event = unsafe { gst::Event::from_glib_full(ev_ptr as *mut gst::ffi::GstEvent) };
        if !self.appsrc.send_event(event) {
            return Err(NvTrackerError::PipelineError(
                "appsrc send_event(stream_reset) rejected".into(),
            ));
        }
        Ok(())
    }

    fn enforce_source_continuity(
        &self,
        slots: &ClassifiedSlots,
        slot_dims: &[(u32, u32)],
        input_pts: &[Option<u64>],
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
            self.reset_stream_with_reason(source_id, &reason)?;
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

    /// Stop the pipeline gracefully: send EOS, wait for it to propagate, then
    /// transition to `Null`.  The GStreamer element wrappers are intentionally
    /// **not** freed afterwards (see [`Drop`] impl).
    pub fn shutdown(&mut self) -> Result<()> {
        if self.is_shut_down {
            return Ok(());
        }
        self.is_shut_down = true;
        self.watchdog_shutdown.store(true, Ordering::Release);
        if let Some(handle) = self._watchdog_thread.take() {
            let _ = handle.join();
        }
        let _ = self.appsrc.end_of_stream();
        let bus = self
            .pipeline
            .bus()
            .ok_or_else(|| NvTrackerError::PipelineError("Pipeline has no bus".into()))?;
        let _ = bus.timed_pop_filtered(
            gst::ClockTime::from_seconds(5),
            &[gst::MessageType::Eos, gst::MessageType::Error],
        );
        let _ = self.pipeline.set_state(gst::State::Null);
        let _ = self.pipeline.state(gst::ClockTime::from_seconds(5));
        Ok(())
    }

    fn set_element_property(element: &gst::Element, key: &str, value: &str) -> Result<()> {
        if element.find_property(key).is_none() {
            return Err(NvTrackerError::InvalidProperty {
                key: key.to_string(),
                reason: "property not found on nvtracker element".into(),
            });
        }
        let elem = element.clone();
        let k = key.to_string();
        let v = value.to_string();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            elem.set_property_from_str(&k, &v);
        }))
        .map_err(|_| NvTrackerError::InvalidProperty {
            key: key.to_string(),
            reason: format!("set_property_from_str failed for value '{value}'"),
        })?;
        Ok(())
    }
}

impl Drop for NvTracker {
    fn drop(&mut self) {
        let _ = self.shutdown();
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

fn read_surface_header(buffer: &gst::Buffer) -> Result<(u32, u32)> {
    let map = buffer.map_readable().map_err(|e| {
        NvTrackerError::batch_meta(
            "read_surface_header",
            format!("map_readable failed: {:?}", e),
        )
    })?;
    let data = map.as_slice();
    if data.len() < 12 {
        return Err(NvTrackerError::batch_meta(
            "read_surface_header",
            "buffer too small for NvBufSurface header (need 12 bytes)",
        ));
    }
    let batch_size = u32::from_ne_bytes([data[4], data[5], data[6], data[7]]);
    let num_filled = u32::from_ne_bytes([data[8], data[9], data[10], data[11]]);
    Ok((num_filled, batch_size))
}

fn read_slot_dimensions(buffer: &gst::Buffer, num_filled: u32) -> Result<Vec<(u32, u32)>> {
    use deepstream_buffers::ffi;

    let map = buffer.map_readable().map_err(|e| {
        NvTrackerError::batch_meta(
            "read_slot_dimensions",
            format!("map_readable failed: {:?}", e),
        )
    })?;
    let data = map.as_slice();

    let surface_size = std::mem::size_of::<ffi::NvBufSurface>();
    if data.len() < surface_size {
        return Err(NvTrackerError::batch_meta(
            "read_slot_dimensions",
            format!(
                "buffer too small for NvBufSurface struct (need {} bytes, have {})",
                surface_size,
                data.len()
            ),
        ));
    }

    let surf = unsafe { &*(data.as_ptr() as *const ffi::NvBufSurface) };
    if surf.surfaceList.is_null() {
        return Err(NvTrackerError::NullPointer {
            function: "NvBufSurface.surfaceList".into(),
        });
    }
    let mut dims = Vec::with_capacity(num_filled as usize);
    for i in 0..num_filled {
        let params = unsafe { &*surf.surfaceList.add(i as usize) };
        dims.push((params.width, params.height));
    }
    Ok(dims)
}

/// Default low-level tracker library path on a standard DeepStream install.
pub fn default_ll_lib_path() -> String {
    let p = Path::new("/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so");
    p.to_string_lossy().into_owned()
}
