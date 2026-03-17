//! NvInfer pipeline implementation.

use crate::batch_meta_builder::{attach_batch_meta_with_rois, FULL_FRAME_SENTINEL};
use crate::config::NvInferConfig;
use crate::error::{NvInferError, Result};
use crate::meta_clear_policy::MetaClearPolicy;
use crate::nvinfer_types::DataType;
use crate::output::{BatchInferenceOutput, ElementOutput, TensorView};
use crate::roi::Roi;
use deepstream::{BatchMeta, InferDims, InferTensorMeta};
use deepstream_nvbufsurface::{bridge_savant_id_meta, SavantIdMeta, SavantIdMetaKind};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_app::AppSinkCallbacks;
use log::info;
use parking_lot::Mutex;
use savant_core::primitives::RBBox;
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;
use tempfile::NamedTempFile;

/// Callback type invoked when inference completes (async mode).
pub type InferCallback = Box<dyn FnMut(BatchInferenceOutput) + Send>;

/// Shared state for sync/async sample delivery.
struct SampleDelivery {
    /// User callback for async mode.
    callback: Mutex<Option<InferCallback>>,
    /// Per-batch_id senders for infer_sync callers waiting on specific batches.
    sync_tx: Mutex<HashMap<u64, mpsc::Sender<BatchInferenceOutput>>>,
}

/// The NvInfer inference engine.
///
/// Operates in secondary mode (`process-mode=2`): each submitted buffer must
/// carry [`NvDsObjectMeta`] entries (one per ROI) so that `Gst-nvinfer` crops
/// and processes each region independently.  [`submit`](NvInfer::submit) and
/// [`infer_sync`](NvInfer::infer_sync) accept an optional per-slot ROI map
/// and attach the metadata automatically.
pub struct NvInfer {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    #[allow(dead_code)] // Kept alive for callbacks; pipeline owns the element
    appsink: gst_app::AppSink,
    _config: NvInferConfig,
    #[allow(dead_code)] // Kept alive so temp config file persists
    _config_file: NamedTempFile,
    delivery: Arc<SampleDelivery>,
    /// Model input width (used for full-frame ROI fallback).
    input_width: u32,
    /// Model input height (used for full-frame ROI fallback).
    input_height: u32,
    /// When and whether to clear object metas.
    policy: MetaClearPolicy,
}

impl NvInfer {
    /// Create a new NvInfer inference engine.
    pub fn new(config: NvInferConfig, callback: InferCallback) -> Result<Self> {
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

        gst::init().map_err(|e| NvInferError::GstInit(e.to_string()))?;

        let config_file = config.validate_and_materialize()?;
        let config_path = config_file.path().to_string_lossy().to_string();

        let input_width = config.input_width.unwrap_or(0);
        let input_height = config.input_height.unwrap_or(0);
        let policy = config.meta_clear_policy;

        let pipeline = gst::Pipeline::new();

        let appsrc = gst::ElementFactory::make("appsrc")
            .name("src")
            .build()
            .map_err(|_| NvInferError::ElementCreationFailed("appsrc".into()))?;

        let appsink = gst::ElementFactory::make("appsink")
            .name("sink")
            .build()
            .map_err(|_| NvInferError::ElementCreationFailed("appsink".into()))?;

        let mut caps_builder = gst::Caps::builder("video/x-raw")
            .features(["memory:NVMM"])
            .field("format", config.input_format.as_str());
        if let Some(w) = config.input_width {
            caps_builder = caps_builder.field("width", w as i32);
        }
        if let Some(h) = config.input_height {
            caps_builder = caps_builder.field("height", h as i32);
        }
        let appsrc_caps = caps_builder.build();
        let appsrc_elem: &gst::Element = appsrc.upcast_ref();
        appsrc_elem.set_property("caps", &appsrc_caps);
        appsrc_elem.set_property_from_str("format", "time");
        appsrc_elem.set_property_from_str("stream-type", "stream");

        // Configure appsink: emit signals for callback, sync=false.
        let appsink_elem: &gst::Element = appsink.upcast_ref();
        appsink_elem.set_property("sync", false);
        appsink_elem.set_property("emit-signals", true);

        let nvinfer = gst::ElementFactory::make("nvinfer")
            .name("nvinfer")
            .build()
            .map_err(|_| NvInferError::ElementCreationFailed("nvinfer".into()))?;

        // Set config file path.
        nvinfer.set_property_from_str("config-file-path", &config_path);

        // Apply element properties.
        for (key, value) in &config.element_properties {
            Self::set_element_property(&nvinfer, key, value)?;
        }

        // Bridge SavantIdMeta across nvinfer so output buffers carry per-frame IDs.
        bridge_savant_id_meta(&nvinfer);

        let elements: Vec<gst::Element> = if config.queue_depth > 0 {
            let queue = gst::ElementFactory::make("queue")
                .name("queue")
                .build()
                .map_err(|_| NvInferError::ElementCreationFailed("queue".into()))?;
            queue.set_property("max-size-buffers", config.queue_depth);
            queue.set_property("max-size-bytes", 0u32);
            queue.set_property("max-size-time", 0u64);
            vec![
                appsrc.clone().upcast(),
                queue,
                nvinfer.clone().upcast(),
                appsink.clone().upcast(),
            ]
        } else {
            vec![
                appsrc.clone().upcast(),
                nvinfer.clone().upcast(),
                appsink.clone().upcast(),
            ]
        };

        for elem in &elements {
            pipeline.add(elem).map_err(|e| {
                NvInferError::PipelineError(format!("Failed to add element: {}", e))
            })?;
        }

        gst::Element::link_many(elements.iter())
            .map_err(|_| NvInferError::LinkFailed("appsrc->[queue]->nvinfer->appsink".into()))?;

        let appsrc_typed: gst_app::AppSrc = appsrc
            .dynamic_cast::<gst_app::AppSrc>()
            .map_err(|_| NvInferError::ElementCreationFailed("appsrc cast failed".into()))?;

        let appsink_typed: gst_app::AppSink = appsink
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| NvInferError::ElementCreationFailed("appsink cast failed".into()))?;

        let delivery = Arc::new(SampleDelivery {
            callback: Mutex::new(Some(callback)),
            sync_tx: Mutex::new(HashMap::new()),
        });
        let delivery_clone = delivery.clone();
        let callbacks = AppSinkCallbacks::builder()
            .new_sample(move |appsink| {
                let sample = appsink.pull_sample().map_err(|e| {
                    log::error!("appsink pull_sample error: {:?}", e);
                    gst::FlowError::Error
                })?;
                let batch_id_opt = sample.buffer().and_then(|b| b.pts()).map(|t| t.nseconds());
                let batch_id = batch_id_opt.unwrap_or(0);

                let output = extract_batch_output(sample, batch_id, policy).map_err(|e| {
                    log::error!("extract_batch_output error: {:?}", e);
                    gst::FlowError::Error
                })?;

                // If infer_sync is waiting for this batch_id, deliver there;
                // otherwise invoke the user callback.  Only attempt sync delivery
                // when PTS survived the round-trip (batch_id_opt.is_some()), so we
                // have a reliable batch_id.  When PTS is None we cannot know which
                // sync waiter (if any) owns this result, so we fall back to the
                // user callback to avoid misdelivery.
                let sync_sender =
                    batch_id_opt.and_then(|id| delivery_clone.sync_tx.lock().remove(&id));
                if let Some(tx) = sync_sender {
                    let _ = tx.send(output);
                } else {
                    if batch_id_opt.is_none() {
                        log::warn!(
                            "Buffer PTS is None; cannot determine batch_id, routing to callback"
                        );
                    }
                    if let Some(ref mut cb) = *delivery_clone.callback.lock() {
                        cb(output);
                    }
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build();
        appsink_typed.set_callbacks(callbacks);

        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| NvInferError::PipelineError(format!("Failed to start pipeline: {}", e)))?;

        info!(
            "NvInfer initialized (name={}, queue_depth={})",
            name_display, config.queue_depth
        );

        Ok(Self {
            pipeline,
            appsrc: appsrc_typed,
            appsink: appsink_typed,
            _config: config,
            _config_file: config_file,
            delivery,
            input_width,
            input_height,
            policy,
        })
    }

    /// Submit a batched buffer for inference.
    ///
    /// `batch_id` is user-chosen and must not be `u64::MAX` (that value maps to
    /// `GST_CLOCK_TIME_NONE` and cannot survive a PTS round-trip).
    ///
    /// `rois` is an optional per-slot map of ROI lists.  Key = slot index
    /// `0..(num_filled-1)`.  If `None` or a slot has no entry, a full-frame
    /// sentinel object is attached for that slot so that `Gst-nvinfer` still
    /// receives a region to process.
    ///
    /// Any existing object metas on the buffer's batch meta are cleared before
    /// the new ROI objects are written (see [`attach_batch_meta_with_rois`]).
    pub fn submit(
        &self,
        mut batch: gst::Buffer,
        batch_id: u64,
        rois: Option<&HashMap<u32, Vec<Roi>>>,
    ) -> Result<()> {
        if batch_id == u64::MAX {
            return Err(NvInferError::PipelineError(
                "batch_id must not be u64::MAX (reserved as GST_CLOCK_TIME_NONE)".into(),
            ));
        }

        let (num_filled, max_batch_size) = read_surface_header(&batch)?;

        let synthetic_rois;
        let effective_rois = if rois.is_none()
            && self.input_width == 0
            && self.input_height == 0
            && num_filled > 0
        {
            let dims = read_slot_dimensions(&batch, num_filled)?;
            let mut map = HashMap::with_capacity(dims.len());
            for (slot, &(w, h)) in dims.iter().enumerate() {
                map.insert(
                    slot as u32,
                    vec![Roi {
                        id: 0,
                        bbox: RBBox::ltwh(0.0, 0.0, w as f32, h as f32)
                            .expect("non-zero surface dimensions"),
                    }],
                );
            }
            synthetic_rois = map;
            Some(&synthetic_rois)
        } else {
            rois
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
                self.input_width,
                self.input_height,
            )?;
        }

        {
            let buf_ref = batch
                .get_mut()
                .ok_or_else(|| NvInferError::PipelineError("Buffer not writable".into()))?;
            buf_ref.set_pts(gst::ClockTime::from_nseconds(batch_id));
        }

        self.appsrc
            .push_buffer(batch)
            .map_err(|e| NvInferError::PipelineError(format!("appsrc push failed: {:?}", e)))?;

        Ok(())
    }

    /// Synchronous inference – blocks until results arrive (up to 30 s).
    ///
    /// Parameters are the same as [`submit`](NvInfer::submit).
    pub fn infer_sync(
        &self,
        batch: gst::Buffer,
        batch_id: u64,
        rois: Option<&HashMap<u32, Vec<Roi>>>,
    ) -> Result<BatchInferenceOutput> {
        let (tx, rx) = mpsc::channel();
        {
            let mut sync_map = self.delivery.sync_tx.lock();
            if sync_map.contains_key(&batch_id) {
                return Err(NvInferError::PipelineError(format!(
                    "batch_id {batch_id} is already in use by another infer_sync caller"
                )));
            }
            sync_map.insert(batch_id, tx);
        }
        if let Err(e) = self.submit(batch, batch_id, rois) {
            self.delivery.sync_tx.lock().remove(&batch_id);
            return Err(e);
        }
        match rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(output) => Ok(output),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.delivery.sync_tx.lock().remove(&batch_id);
                Err(NvInferError::PipelineError(format!(
                    "infer_sync timed out after 30s for batch_id {batch_id}"
                )))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.delivery.sync_tx.lock().remove(&batch_id);
                Err(NvInferError::PipelineError(format!(
                    "infer_sync channel disconnected for batch_id {batch_id}"
                )))
            }
        }
    }

    /// Graceful shutdown: send EOS, drain, stop pipeline.
    pub fn shutdown(&mut self) -> Result<()> {
        let _ = self.appsrc.end_of_stream();
        let bus = self
            .pipeline
            .bus()
            .ok_or_else(|| NvInferError::PipelineError("Pipeline has no bus".into()))?;
        let _ = bus.timed_pop_filtered(
            gst::ClockTime::from_seconds(10),
            &[gst::MessageType::Eos, gst::MessageType::Error],
        );
        self.pipeline
            .set_state(gst::State::Null)
            .map_err(|e| NvInferError::PipelineError(format!("set_state Null failed: {:?}", e)))?;
        Ok(())
    }

    /// Model input width (0 for flexible config).
    pub fn input_width(&self) -> u32 {
        self.input_width
    }

    /// Model input height (0 for flexible config).
    pub fn input_height(&self) -> u32 {
        self.input_height
    }

    fn set_element_property(element: &gst::Element, key: &str, value: &str) -> Result<()> {
        if element.find_property(key).is_none() {
            return Err(NvInferError::InvalidProperty(format!(
                "property '{key}' not found"
            )));
        }
        let elem = element.clone();
        let k = key.to_string();
        let v = value.to_string();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            elem.set_property_from_str(&k, &v);
        }))
        .map_err(|_| NvInferError::InvalidProperty(format!("failed to set '{key}' = '{value}'")))?;
        Ok(())
    }
}

impl Drop for NvInfer {
    fn drop(&mut self) {
        let _ = self.appsrc.end_of_stream();
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

/// Read numFilled and batchSize from the NvBufSurface descriptor in a single map.
///
/// NvBufSurface layout (first 12 bytes, native-endian):
///   offset 0: gpuId      (u32)
///   offset 4: batchSize  (u32)
///   offset 8: numFilled  (u32)
fn read_surface_header(buffer: &gst::Buffer) -> Result<(u32, u32)> {
    let map = buffer
        .map_readable()
        .map_err(|e| NvInferError::BatchMetaFailed(format!("map_readable failed: {:?}", e)))?;
    let data = map.as_slice();
    if data.len() < 12 {
        return Err(NvInferError::BatchMetaFailed(
            "Buffer too small for NvBufSurface".into(),
        ));
    }
    let batch_size = u32::from_ne_bytes([data[4], data[5], data[6], data[7]]);
    let num_filled = u32::from_ne_bytes([data[8], data[9], data[10], data[11]]);
    Ok((num_filled, batch_size))
}

/// Read per-slot (width, height) from the NvBufSurface surfaceList.
///
/// Uses the FFI `NvBufSurface` / `NvBufSurfaceParams` layout to extract
/// each filled slot's dimensions without going through the full
/// `SurfaceView` machinery.
fn read_slot_dimensions(buffer: &gst::Buffer, num_filled: u32) -> Result<Vec<(u32, u32)>> {
    use deepstream_nvbufsurface::ffi;

    let map = buffer
        .map_readable()
        .map_err(|e| NvInferError::BatchMetaFailed(format!("map_readable failed: {:?}", e)))?;
    let data = map.as_slice();

    let surface_size = std::mem::size_of::<ffi::NvBufSurface>();
    if data.len() < surface_size {
        return Err(NvInferError::BatchMetaFailed(
            "Buffer too small for NvBufSurface".into(),
        ));
    }

    let surf = unsafe { &*(data.as_ptr() as *const ffi::NvBufSurface) };
    if surf.surfaceList.is_null() {
        return Err(NvInferError::NullPointer(
            "NvBufSurface.surfaceList is null".into(),
        ));
    }
    let mut dims = Vec::with_capacity(num_filled as usize);
    for i in 0..num_filled {
        let params = unsafe { &*surf.surfaceList.add(i as usize) };
        dims.push((params.width, params.height));
    }
    Ok(dims)
}

fn savant_id_to_i64(k: &SavantIdMetaKind) -> i64 {
    match k {
        SavantIdMetaKind::Frame(id) | SavantIdMetaKind::Batch(id) => *id,
    }
}

/// Extract inference outputs from a completed sample.
///
/// In secondary mode (`process-mode=2`), `Gst-nvinfer` attaches tensor outputs
/// to each `NvDsObjectMeta`'s `obj_user_meta_list`.  This function iterates:
///
/// ```text
/// batch_meta → frames → frame.objects() → object.user_meta() → tensor
/// ```
///
/// `frame_id` is taken from [`SavantIdMeta`] (indexed by frame position).
/// `roi_id` is the `object_id` cast to `i64`; `None` when the object carries
/// the full-frame sentinel (`unique_component_id == FULL_FRAME_SENTINEL`).
fn extract_batch_output(
    sample: gst::Sample,
    batch_id: u64,
    policy: MetaClearPolicy,
) -> Result<BatchInferenceOutput> {
    let buffer = sample
        .buffer()
        .ok_or_else(|| NvInferError::PipelineError("Sample has no buffer".into()))?;

    let batch_meta = unsafe {
        BatchMeta::from_gst_buffer(buffer.as_ptr() as *mut _).map_err(|e| {
            NvInferError::PipelineError(format!("BatchMeta::from_gst_buffer: {:?}", e))
        })?
    };

    let ids: Vec<i64> = buffer
        .meta::<SavantIdMeta>()
        .map(|m| m.ids().iter().map(savant_id_to_i64).collect())
        .unwrap_or_default();

    let frames = batch_meta.frames();
    let mut elements: Vec<ElementOutput> = Vec::new();

    for (frame_idx, frame) in frames.into_iter().enumerate() {
        let frame_id = ids.get(frame_idx).copied();

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
                        });
                    }
                }
            }

            elements.push(ElementOutput {
                frame_id,
                roi_id,
                tensors,
            });
        }
    }

    Ok(BatchInferenceOutput::new(
        batch_id,
        sample,
        elements,
        policy.clear_after(),
    ))
}
