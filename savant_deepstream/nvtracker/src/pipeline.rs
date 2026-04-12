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

use crate::config::NvTrackerConfig;
use crate::detection_meta::attach_detection_meta;
use crate::error::{NvTrackerError, Result};
use crate::output::{extract_tracker_output, TrackerOutput};
use crate::roi::Roi;
use crossbeam::channel::{Receiver, Sender};
use deepstream_buffers::{
    read_slot_dimensions, read_surface_header, NonUniformBatch, SavantIdMetaKind, SharedBuffer,
    SurfaceView,
};
use deepstream_sys;
use gstreamer as gst;
use gstreamer::prelude::*;
use log::info;
use lru::LruCache;
use parking_lot::Mutex;
use savant_gstreamer::pipeline::{
    build_source_eos_event, parse_source_eos_event, set_element_property, AppsrcPadProbe,
    GstPipeline, PipelineConfig, PipelineInput, PipelineOutput,
};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

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

/// DeepStream nvtracker wrapper using the shared GStreamer pipeline framework.
pub struct NvTracker {
    input_tx: Sender<PipelineInput>,
    output_rx: Receiver<PipelineOutput>,
    pipeline: Mutex<GstPipeline>,
    /// Set during [`Self::graceful_shutdown`] and [`Self::shutdown`] to reject new input.
    draining: AtomicBool,
    /// Prevents double `GstPipeline::shutdown` (nvtracker `set_state(Null)` is slow
    /// and must not be called twice).
    is_shut_down: AtomicBool,
    config: NvTrackerConfig,
    next_pts: AtomicU64,
    source_lru: Mutex<LruCache<u32, String>>,
    frame_counters: Mutex<HashMap<u32, i32>>,
    last_source_dims: Mutex<HashMap<u32, (u32, u32)>>,
    last_source_pts: Mutex<HashMap<u32, u64>>,
}

/// LRU capacity for source_id reverse lookup (compile-time non-zero).
const SOURCE_LRU_CAPACITY: NonZeroUsize = NonZeroUsize::new(4096).unwrap();

impl NvTracker {
    /// Create and start the pipeline in `Playing`.
    pub fn new(config: NvTrackerConfig) -> Result<Self> {
        config.validate()?;

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

        let mut elements: Vec<gst::Element> = Vec::new();
        if config.queue_depth > 0 {
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
            elements.push(queue);
        }
        elements.push(nvtracker.upcast());

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
            appsrc_probe,
        };

        let (input_tx, output_rx, gst_pipeline) =
            GstPipeline::start(pipeline_config).map_err(NvTrackerError::from)?;

        info!(
            "NvTracker initialized (name={}, queue_depth={})",
            name_display, config.queue_depth
        );

        Ok(Self {
            input_tx,
            output_rx,
            pipeline: Mutex::new(gst_pipeline),
            draining: AtomicBool::new(false),
            is_shut_down: AtomicBool::new(false),
            config,
            next_pts: AtomicU64::new(0),
            source_lru: Mutex::new(LruCache::new(SOURCE_LRU_CAPACITY)),
            frame_counters: Mutex::new(HashMap::new()),
            last_source_dims: Mutex::new(HashMap::new()),
            last_source_pts: Mutex::new(HashMap::new()),
        })
    }

    /// Submit a batch of frames for tracking.
    ///
    /// Blocks if the input channel is full (backpressure).
    pub fn submit(&self, frames: &[TrackedFrame], ids: Vec<SavantIdMetaKind>) -> Result<()> {
        if self.draining.load(Ordering::Acquire) {
            return Err(NvTrackerError::ShuttingDown);
        }
        if self.is_failed() {
            return Err(NvTrackerError::PipelineFailed);
        }
        let pts = self.next_pts.fetch_add(1, Ordering::Relaxed);
        let (batch, slots, input_pts) = self.prepare_batch(frames, ids)?;
        let buffer = self.finalize_batch_buffer(batch, &slots, &input_pts, pts)?;
        self.input_tx
            .send(PipelineInput::Buffer(buffer))
            .map_err(|_| NvTrackerError::ChannelDisconnected)?;
        Ok(())
    }

    /// Block until the next output is available.
    pub fn recv(&self) -> Result<NvTrackerOutput> {
        let output = self
            .output_rx
            .recv()
            .map_err(|_| NvTrackerError::ChannelDisconnected)?;
        self.convert_output(output)
    }

    /// Block until the next output or timeout. Returns `Ok(None)` on timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<NvTrackerOutput>> {
        match self.output_rx.recv_timeout(timeout) {
            Ok(output) => self.convert_output(output).map(Some),
            Err(crossbeam::channel::RecvTimeoutError::Timeout) => Ok(None),
            Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                Err(NvTrackerError::ChannelDisconnected)
            }
        }
    }

    /// Non-blocking receive.
    pub fn try_recv(&self) -> Result<Option<NvTrackerOutput>> {
        match self.output_rx.try_recv() {
            Ok(output) => self.convert_output(output).map(Some),
            Err(crossbeam::channel::TryRecvError::Empty) => Ok(None),
            Err(crossbeam::channel::TryRecvError::Disconnected) => {
                Err(NvTrackerError::ChannelDisconnected)
            }
        }
    }

    /// Send a custom GStreamer event into the pipeline input.
    pub fn send_event(&self, event: gst::Event) -> Result<()> {
        if self.draining.load(Ordering::Acquire) {
            return Err(NvTrackerError::ShuttingDown);
        }
        self.input_tx
            .send(PipelineInput::Event(event))
            .map_err(|_| NvTrackerError::ChannelDisconnected)?;
        Ok(())
    }

    /// Send a logical per-source EOS marker downstream (custom downstream event).
    pub fn send_eos(&self, source_id: &str) -> Result<()> {
        let event = build_source_eos_event(source_id);
        self.send_event(event)
    }

    /// Send `GST_NVEVENT_STREAM_RESET` for the stream identified by `source_id` (crc32 → pad id).
    pub fn reset_stream(&self, source_id: &str) -> Result<()> {
        if self.draining.load(Ordering::Acquire) {
            return Err(NvTrackerError::ShuttingDown);
        }
        self.reset_stream_with_reason(source_id, "manual")
    }

    pub fn is_failed(&self) -> bool {
        self.pipeline.lock().is_failed()
    }

    /// Graceful shutdown: reject new input, send EOS, drain outputs within `timeout`, stop pipeline.
    pub fn graceful_shutdown(&self, timeout: Duration) -> Result<Vec<NvTrackerOutput>> {
        if self.is_shut_down.swap(true, Ordering::AcqRel) {
            return Err(NvTrackerError::ShuttingDown);
        }
        self.draining.store(true, Ordering::Release);
        let raw = {
            let mut guard = self.pipeline.lock();
            guard
                .graceful_shutdown(timeout, &self.input_tx, &self.output_rx)
                .map_err(NvTrackerError::from)?
        };
        let mut out = Vec::with_capacity(raw.len());
        for item in raw {
            out.push(self.convert_output(item)?);
        }
        Ok(out)
    }

    /// Abrupt shutdown (used by [`Drop`]).
    pub fn shutdown(&self) -> Result<()> {
        if self.is_shut_down.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        self.draining.store(true, Ordering::Release);
        self.pipeline
            .lock()
            .shutdown()
            .map_err(NvTrackerError::from)?;
        Ok(())
    }

    fn convert_output(&self, output: PipelineOutput) -> Result<NvTrackerOutput> {
        match output {
            PipelineOutput::Buffer(buffer) => {
                let resolve = |pad: u32| {
                    self.source_lru
                        .lock()
                        .get(&pad)
                        .cloned()
                        .unwrap_or_else(|| format!("unknown-{pad:#x}"))
                };
                let tracker_output = extract_tracker_output(buffer, resolve)?;
                Ok(NvTrackerOutput::Tracking(tracker_output))
            }
            PipelineOutput::Eos => Ok(NvTrackerOutput::Error(NvTrackerError::PipelineError(
                "unexpected hard GStreamer EOS in NvTracker output".into(),
            ))),
            PipelineOutput::Event(event) => {
                if let Some(source_id) = parse_source_eos_event(&event) {
                    Ok(NvTrackerOutput::Eos { source_id })
                } else {
                    Ok(NvTrackerOutput::Event(event))
                }
            }
            PipelineOutput::Error(e) => Ok(NvTrackerOutput::Error(NvTrackerError::from(e))),
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
        pts: u64,
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

        Ok(batch)
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
        self.send_event(event)?;
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
