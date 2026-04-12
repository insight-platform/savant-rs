//! NvInfer pipeline implementation built on the `savant_gstreamer::pipeline` framework.

use crate::batch_meta_builder::{attach_batch_meta_with_rois, FULL_FRAME_SENTINEL};
use crate::config::NvInferConfig;
use crate::error::{NvInferError, Result};
use crate::meta_clear_policy::MetaClearPolicy;
use crate::nvinfer_types::DataType;
use crate::output::{BatchInferenceOutput, ElementOutput, TensorView};
use crate::roi::Roi;
use crossbeam::channel::{Receiver, Sender};
use deepstream::{BatchMeta, InferDims, InferTensorMeta};
use deepstream_buffers::{read_slot_dimensions, read_surface_header, SharedBuffer};
use gstreamer as gst;
use gstreamer::prelude::*;
use log::info;
use parking_lot::Mutex;
use savant_core::primitives::RBBox;
use savant_gstreamer::pipeline::{
    build_source_eos_event, parse_source_eos_event, set_element_property, GstPipeline,
    PipelineConfig, PipelineInput, PipelineOutput,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tempfile::NamedTempFile;

/// Output from the NvInfer pipeline.
#[derive(Debug)]
pub enum NvInferOutput {
    /// Inference results from a processed buffer.
    Inference(BatchInferenceOutput),
    /// A downstream GStreamer event captured at the pipeline output.
    Event(gst::Event),
    /// Logical end-of-stream for a specific source.
    Eos { source_id: String },
    /// Pipeline or framework runtime error (watchdog, bus error, etc.).
    Error(NvInferError),
}

/// The NvInfer inference engine.
///
/// Operates in secondary mode (`process-mode=2`): each submitted buffer must
/// carry [`NvDsObjectMeta`] entries (one per ROI) so that `Gst-nvinfer` crops
/// and processes each region independently. [`submit`](NvInfer::submit)
/// accepts an optional per-slot ROI map and attaches the metadata
/// automatically.
///
/// Uses the `savant_gstreamer::pipeline` framework internally. Channels are
/// hidden; interact via [`submit`](NvInfer::submit),
/// [`send_eos`](NvInfer::send_eos), and
/// [`recv`](NvInfer::recv)/[`recv_timeout`](NvInfer::recv_timeout)/[`try_recv`](NvInfer::try_recv).
pub struct NvInfer {
    input_tx: Sender<PipelineInput>,
    output_rx: Receiver<PipelineOutput>,
    pipeline: Mutex<GstPipeline>,
    #[allow(dead_code)]
    _config_file: NamedTempFile,
    /// Set during [`Self::graceful_shutdown`] and [`Self::shutdown`] to reject new input.
    draining: AtomicBool,
    /// Prevents double `GstPipeline::shutdown` (nvinfer `set_state(Null)` is slow
    /// and must not be called twice).
    is_shut_down: AtomicBool,
    /// Monotonic counter used as PTS for internal pipeline correlation.
    next_pts: AtomicU64,
    /// When and whether to clear object metas.
    policy: MetaClearPolicy,
    /// Whether D2H tensor copy is enabled.
    host_copy_enabled: bool,
}

impl NvInfer {
    /// Create and start a new NvInfer inference engine.
    ///
    /// The `nvinfer` GStreamer element is created internally from the
    /// configuration. An optional `queue` element is inserted when
    /// `config.queue_depth > 0`. The pipeline starts in `Playing` state.
    pub fn new(config: NvInferConfig) -> Result<Self> {
        let name_owned;
        let name_display = if config.name.is_empty() {
            let model_path = config
                .nvinfer_properties
                .get("model-engine-file")
                .or_else(|| config.nvinfer_properties.get("onnx-file"))
                .or_else(|| config.nvinfer_properties.get("tlt-encoded-model"));
            let base = model_path
                .map(|p| {
                    Path::new(p)
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_else(|| "unknown".into());
            name_owned = format!("nvinfer-{}-{}", config.gpu_id, base);
            name_owned.as_str()
        } else {
            config.name.as_str()
        };
        info!("NvInfer initializing (name={})", name_display);

        let config_file = config.validate_and_materialize()?;
        let config_path = config_file.path().to_string_lossy().to_string();

        let policy = config.meta_clear_policy;
        let host_copy_enabled = !config.disable_output_host_copy;

        gst::init().map_err(|e| NvInferError::GstInit(e.to_string()))?;

        let nvinfer = gst::ElementFactory::make("nvinfer")
            .name("nvinfer")
            .build()
            .map_err(|_| NvInferError::ElementCreationFailed("nvinfer".into()))?;

        nvinfer.set_property_from_str("config-file-path", &config_path);

        for (key, value) in &config.element_properties {
            set_element_property(&nvinfer, key, value).map_err(NvInferError::InvalidProperty)?;
        }

        let mut elements: Vec<gst::Element> = Vec::new();
        if config.queue_depth > 0 {
            let queue = gst::ElementFactory::make("queue")
                .name("queue")
                .build()
                .map_err(|_| NvInferError::ElementCreationFailed("queue".into()))?;
            queue.set_property("max-size-buffers", config.queue_depth);
            queue.set_property("max-size-bytes", 0u32);
            queue.set_property("max-size-time", 0u64);
            elements.push(queue);
        }
        elements.push(nvinfer.upcast());

        let appsrc_caps = gst::Caps::builder("video/x-raw")
            .features(["memory:NVMM"])
            .field("format", config.input_format.gst_name())
            .build();

        let pipeline_config = PipelineConfig {
            name: name_display.to_string(),
            appsrc_caps,
            elements,
            input_channel_capacity: config.input_channel_capacity,
            output_channel_capacity: config.output_channel_capacity,
            operation_timeout: Some(config.operation_timeout),
            drain_poll_interval: config.drain_poll_interval,
            appsrc_probe: None,
        };

        let (input_tx, output_rx, gst_pipeline) = GstPipeline::start(pipeline_config)?;

        info!(
            "NvInfer initialized (name={}, queue_depth={})",
            name_display, config.queue_depth
        );

        Ok(Self {
            input_tx,
            output_rx,
            pipeline: Mutex::new(gst_pipeline),
            _config_file: config_file,
            draining: AtomicBool::new(false),
            is_shut_down: AtomicBool::new(false),
            next_pts: AtomicU64::new(0),
            policy,
            host_copy_enabled,
        })
    }

    /// Submit a batched buffer for inference.
    ///
    /// The buffer is **consumed**: if the [`SharedBuffer`] has outstanding
    /// references, an error is returned.
    ///
    /// `rois` is an optional per-slot map of ROI lists. Key = slot index
    /// `0..(num_filled-1)`. If `None` or a slot has no entry, a full-frame
    /// sentinel object is attached for that slot so that `Gst-nvinfer` still
    /// receives a region to process.
    ///
    /// Blocks if the input channel is full (backpressure).
    pub fn submit(&self, batch: SharedBuffer, rois: Option<&HashMap<u32, Vec<Roi>>>) -> Result<()> {
        if self.draining.load(Ordering::Acquire) {
            return Err(NvInferError::ShuttingDown);
        }
        if self.is_failed() {
            return Err(NvInferError::PipelineFailed);
        }
        let pts = self.next_pts.fetch_add(1, Ordering::Relaxed);
        let batch = batch.into_buffer().map_err(|_| {
            NvInferError::PipelineError(
                "SharedBuffer has outstanding references; cannot take exclusive ownership".into(),
            )
        })?;
        let buffer = self.prepare_buffer(batch, rois, pts)?;
        self.input_tx
            .send(PipelineInput::Buffer(buffer))
            .map_err(|_| NvInferError::ChannelDisconnected)?;
        Ok(())
    }

    /// Send a logical per-source EOS marker downstream.
    ///
    /// This is encoded as a custom downstream GStreamer event and does **not**
    /// terminate the underlying GStreamer pipeline. The event passes through
    /// `nvinfer` unchanged and is surfaced by [`recv`](NvInfer::recv) as
    /// [`NvInferOutput::Eos`].
    pub fn send_eos(&self, source_id: &str) -> Result<()> {
        let event = build_source_eos_event(source_id);
        self.send_event(event)
    }

    /// Send a custom GStreamer event into the pipeline.
    pub fn send_event(&self, event: gst::Event) -> Result<()> {
        if self.draining.load(Ordering::Acquire) {
            return Err(NvInferError::ShuttingDown);
        }
        self.input_tx
            .send(PipelineInput::Event(event))
            .map_err(|_| NvInferError::ChannelDisconnected)?;
        Ok(())
    }

    /// Block until the next output is available.
    ///
    /// Returns [`NvInferOutput::Inference`] for processed buffers,
    /// [`NvInferOutput::Eos`] when a source EOS marker arrives,
    /// [`NvInferOutput::Event`] for downstream events, or
    /// [`NvInferOutput::Error`] for pipeline/framework failures.
    ///
    /// `Err` is reserved for channel disconnect only ([`NvInferError::ChannelDisconnected`]).
    pub fn recv(&self) -> Result<NvInferOutput> {
        let output = self
            .output_rx
            .recv()
            .map_err(|_| NvInferError::ChannelDisconnected)?;
        self.convert_output(output)
    }

    /// Block until the next output or timeout.
    ///
    /// Returns `Ok(None)` on timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<NvInferOutput>> {
        match self.output_rx.recv_timeout(timeout) {
            Ok(output) => self.convert_output(output).map(Some),
            Err(crossbeam::channel::RecvTimeoutError::Timeout) => Ok(None),
            Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                Err(NvInferError::ChannelDisconnected)
            }
        }
    }

    /// Non-blocking: return the next output if available, or `Ok(None)`.
    pub fn try_recv(&self) -> Result<Option<NvInferOutput>> {
        match self.output_rx.try_recv() {
            Ok(output) => self.convert_output(output).map(Some),
            Err(crossbeam::channel::TryRecvError::Empty) => Ok(None),
            Err(crossbeam::channel::TryRecvError::Disconnected) => {
                Err(NvInferError::ChannelDisconnected)
            }
        }
    }

    /// Check if the pipeline has entered a terminal failed state.
    pub fn is_failed(&self) -> bool {
        self.pipeline.lock().is_failed()
    }

    /// Graceful shutdown: reject new input, send EOS, drain outputs within `timeout`, stop pipeline.
    ///
    /// Returns all domain outputs produced before the pipeline EOS (terminal EOS is not included).
    pub fn graceful_shutdown(&self, timeout: Duration) -> Result<Vec<NvInferOutput>> {
        if self.is_shut_down.swap(true, Ordering::AcqRel) {
            return Err(NvInferError::ShuttingDown);
        }
        self.draining.store(true, Ordering::Release);
        let raw = {
            let mut guard = self.pipeline.lock();
            guard.graceful_shutdown(timeout, &self.input_tx, &self.output_rx)?
        };
        let mut out = Vec::with_capacity(raw.len());
        for item in raw {
            out.push(self.convert_output(item)?);
        }
        Ok(out)
    }

    /// Abrupt shutdown: stops threads and pipeline (used by [`Drop`]).
    pub fn shutdown(&self) -> Result<()> {
        if self.is_shut_down.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        self.draining.store(true, Ordering::Release);
        self.pipeline.lock().shutdown()?;
        Ok(())
    }

    /// Convert a raw [`PipelineOutput`] into a domain-specific [`NvInferOutput`].
    fn convert_output(&self, output: PipelineOutput) -> Result<NvInferOutput> {
        match output {
            PipelineOutput::Buffer(buffer) => {
                let batch_output =
                    extract_batch_output(buffer, self.policy, self.host_copy_enabled)?;
                Ok(NvInferOutput::Inference(batch_output))
            }
            PipelineOutput::Eos => Ok(NvInferOutput::Error(NvInferError::PipelineError(
                "unexpected hard GStreamer EOS in NvInfer output".into(),
            ))),
            PipelineOutput::Event(event) => {
                if let Some(source_id) = parse_source_eos_event(&event) {
                    Ok(NvInferOutput::Eos { source_id })
                } else {
                    Ok(NvInferOutput::Event(event))
                }
            }
            PipelineOutput::Error(e) => Ok(NvInferOutput::Error(NvInferError::FrameworkError(e))),
        }
    }

    /// Attach ROI metadata, patch frame dimensions, set PTS.
    fn prepare_buffer(
        &self,
        mut batch: gst::Buffer,
        rois: Option<&HashMap<u32, Vec<Roi>>>,
        pts: u64,
    ) -> Result<gst::Buffer> {
        let (num_filled, max_batch_size) = read_surface_header(&batch)?;

        let slot_dims = if num_filled > 0 {
            read_slot_dimensions(&batch, num_filled)?
        } else {
            Vec::new()
        };

        let merged_rois;
        let effective_rois = {
            let mut map = HashMap::with_capacity(slot_dims.len());
            for (slot, &(w, h)) in slot_dims.iter().enumerate() {
                let s = slot as u32;
                let has_user_rois = rois.and_then(|r| r.get(&s)).is_some_and(|v| !v.is_empty());
                if has_user_rois {
                    map.insert(s, rois.unwrap()[&s].clone());
                } else {
                    map.insert(
                        s,
                        vec![Roi {
                            id: 0,
                            bbox: RBBox::ltwh(0.0, 0.0, w as f32, h as f32)
                                .expect("non-zero surface dimensions"),
                        }],
                    );
                }
            }
            merged_rois = map;
            if merged_rois.is_empty() {
                None
            } else {
                Some(&merged_rois)
            }
        };

        {
            let buf_ref = batch
                .get_mut()
                .ok_or_else(|| NvInferError::PipelineError("Buffer is not writable".into()))?;
            attach_batch_meta_with_rois(
                buf_ref,
                num_filled,
                max_batch_size,
                self.policy,
                effective_rois,
            )?;
        }

        if !slot_dims.is_empty() {
            let buf_ref = batch
                .get_mut()
                .ok_or_else(|| NvInferError::PipelineError("Buffer not writable".into()))?;
            let buf_ptr = buf_ref.as_mut_ptr() as *mut deepstream_sys::GstBuffer;
            let batch_meta = unsafe { deepstream_sys::gst_buffer_get_nvds_batch_meta(buf_ptr) };
            if !batch_meta.is_null() {
                let mut frame_list = unsafe { (*batch_meta).frame_meta_list };
                let mut slot: usize = 0;
                while !frame_list.is_null() && slot < slot_dims.len() {
                    let frame_ptr =
                        unsafe { (*frame_list).data as *mut deepstream_sys::NvDsFrameMeta };
                    if !frame_ptr.is_null() {
                        unsafe {
                            (*frame_ptr).source_frame_width = slot_dims[slot].0;
                            (*frame_ptr).source_frame_height = slot_dims[slot].1;
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
                .ok_or_else(|| NvInferError::PipelineError("Buffer not writable".into()))?;
            buf_ref.set_pts(gst::ClockTime::from_nseconds(pts));
        }

        Ok(batch)
    }
}

impl Drop for NvInfer {
    fn drop(&mut self) {
        let _ = self.pipeline.lock().shutdown();
    }
}

/// Extract inference outputs from a completed buffer.
///
/// In secondary mode (`process-mode=2`), `Gst-nvinfer` attaches tensor outputs
/// to each `NvDsObjectMeta`'s `obj_user_meta_list`. This function iterates:
///
/// ```text
/// batch_meta -> frames -> frame.objects() -> object.user_meta() -> tensor
/// ```
fn extract_batch_output(
    buffer: gst::Buffer,
    policy: MetaClearPolicy,
    host_copy_enabled: bool,
) -> Result<BatchInferenceOutput> {
    let buffer_ref = buffer.as_ref();

    let batch_meta = unsafe {
        BatchMeta::from_gst_buffer(buffer_ref.as_ptr() as *mut _).map_err(|e| {
            NvInferError::PipelineError(format!("BatchMeta::from_gst_buffer: {:?}", e))
        })?
    };

    let frames = batch_meta.frames();
    let mut elements: Vec<ElementOutput> = Vec::new();

    for frame in frames.into_iter() {
        let slot_number = frame.batch_id();

        for obj in frame.objects() {
            let roi_id = if obj.unique_component_id() == FULL_FRAME_SENTINEL {
                None
            } else {
                Some(obj.object_id() as i64)
            };

            let mut tensors = Vec::new();
            for user_meta in obj.user_meta() {
                if user_meta.meta_type()
                    != deepstream_sys::NvDsMetaType_NVDSINFER_TENSOR_OUTPUT_META
                {
                    continue;
                }
                let raw_ptr = user_meta.user_meta_data();
                if let Some(tm) = unsafe {
                    InferTensorMeta::from_raw(raw_ptr as *mut deepstream_sys::NvDsInferTensorMeta)
                        .ok()
                } {
                    let layer_names = tm.layer_names();
                    let layer_dims = tm.layer_dimensions();
                    let layer_types = tm.layer_data_types();
                    let host_ptrs = tm.out_buf_ptrs_host();
                    let dev_ptrs = tm.out_buf_ptrs_dev();
                    for (j, name) in layer_names.iter().enumerate() {
                        let dims = layer_dims.get(j).cloned().unwrap_or(InferDims {
                            dimensions: vec![],
                            num_elements: 0,
                        });
                        let data_type: DataType = layer_types
                            .get(j)
                            .copied()
                            .map(DataType::from)
                            .unwrap_or(DataType::Float);
                        let byte_length = dims.num_elements as usize * data_type.element_size();
                        let host_ptr = host_ptrs.get(j).copied().unwrap_or(std::ptr::null_mut());
                        let device_ptr = dev_ptrs.get(j).copied().unwrap_or(std::ptr::null_mut());
                        tensors.push(TensorView {
                            name: name.clone(),
                            dims,
                            data_type,
                            host_ptr: host_ptr as *const _,
                            device_ptr: device_ptr as *const _,
                            byte_length,
                            host_copy_enabled,
                        });
                    }
                }
            }

            elements.push(ElementOutput {
                roi_id,
                slot_number,
                tensors,
            });
        }
    }

    let shared = SharedBuffer::from(buffer);
    Ok(BatchInferenceOutput::new(
        shared,
        elements,
        policy.clear_after(),
        host_copy_enabled,
    ))
}
