//! NvInfer pipeline implementation.

use crate::batch_meta_builder::{attach_batch_meta_with_rois, FULL_FRAME_SENTINEL};
use crate::config::NvInferConfig;
use crate::error::{NvInferError, Result};
use crate::meta_clear_policy::MetaClearPolicy;
use crate::nvinfer_types::DataType;
use crate::output::{BatchInferenceOutput, ElementOutput, TensorView};
use crate::roi::Roi;
use deepstream::{BatchMeta, InferDims, InferTensorMeta};
use deepstream_buffers::{bridge_savant_id_meta, SharedBuffer};
use glib::translate::from_glib_none;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_app::AppSinkCallbacks;
use log::info;
use parking_lot::Mutex;
use savant_core::primitives::RBBox;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;
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
    /// Monotonic counter used as PTS for internal pipeline correlation.
    next_pts: AtomicU64,
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

        let policy = config.meta_clear_policy;
        let host_copy_enabled = !config.disable_output_host_copy;

        let pipeline = gst::Pipeline::new();

        let appsrc = gst::ElementFactory::make("appsrc")
            .name("src")
            .build()
            .map_err(|_| NvInferError::ElementCreationFailed("appsrc".into()))?;

        let appsink = gst::ElementFactory::make("appsink")
            .name("sink")
            .build()
            .map_err(|_| NvInferError::ElementCreationFailed("appsink".into()))?;

        let appsrc_caps = gst::Caps::builder("video/x-raw")
            .features(["memory:NVMM"])
            .field("format", config.input_format.gst_name())
            .build();
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
        bridge_savant_id_meta(&nvinfer)?;

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
                let pts_key = sample.buffer().and_then(|b| b.pts()).map(|t| t.nseconds());

                let output =
                    extract_batch_output(sample, policy, host_copy_enabled).map_err(|e| {
                        log::error!("extract_batch_output error: {:?}", e);
                        gst::FlowError::Error
                    })?;

                // Route to infer_sync waiter if PTS matches, otherwise to the
                // user callback.
                let sync_sender = pts_key.and_then(|id| delivery_clone.sync_tx.lock().remove(&id));
                if let Some(tx) = sync_sender {
                    let _ = tx.send(output);
                } else {
                    if pts_key.is_none() {
                        log::warn!("Buffer PTS is None; routing to callback");
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
            next_pts: AtomicU64::new(0),
            policy,
        })
    }

    /// Submit a batched buffer for asynchronous inference.
    ///
    /// The buffer is **consumed**: if the [`SharedBuffer`] has outstanding
    /// references, an error is returned.
    ///
    /// `rois` is an optional per-slot map of ROI lists.  Key = slot index
    /// `0..(num_filled-1)`.  If `None` or a slot has no entry, a full-frame
    /// sentinel object is attached for that slot so that `Gst-nvinfer` still
    /// receives a region to process.
    ///
    /// Any existing object metas on the buffer's batch meta are cleared before
    /// the new ROI objects are written (see [`attach_batch_meta_with_rois`]).
    pub fn submit(&self, batch: SharedBuffer, rois: Option<&HashMap<u32, Vec<Roi>>>) -> Result<()> {
        let pts = self.next_pts.fetch_add(1, Ordering::Relaxed);
        let batch = batch.into_buffer().map_err(|_| {
            NvInferError::PipelineError(
                "SharedBuffer has outstanding references; cannot take exclusive ownership".into(),
            )
        })?;
        self.push_buffer(batch, rois, pts)
    }

    /// Synchronous inference with a configurable timeout.
    ///
    /// The buffer is **consumed**: if the [`SharedBuffer`] has outstanding
    /// references, an error is returned.
    ///
    /// Parameters are the same as [`submit`](NvInfer::submit).
    pub fn infer_sync_with_timeout(
        &self,
        batch: SharedBuffer,
        rois: Option<&HashMap<u32, Vec<Roi>>>,
        timeout: Duration,
    ) -> Result<BatchInferenceOutput> {
        let pts = self.next_pts.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel();
        {
            let mut sync_map = self.delivery.sync_tx.lock();
            sync_map.insert(pts, tx);
        }
        let batch = match batch.into_buffer() {
            Ok(buf) => buf,
            Err(_) => {
                self.delivery.sync_tx.lock().remove(&pts);
                return Err(NvInferError::PipelineError(
                    "SharedBuffer has outstanding references; cannot take exclusive ownership"
                        .into(),
                ));
            }
        };
        if let Err(e) = self.push_buffer(batch, rois, pts) {
            self.delivery.sync_tx.lock().remove(&pts);
            return Err(e);
        }
        match rx.recv_timeout(timeout) {
            Ok(output) => Ok(output),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.delivery.sync_tx.lock().remove(&pts);
                Err(NvInferError::PipelineError(format!(
                    "infer_sync timed out after {}ms",
                    timeout.as_millis()
                )))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.delivery.sync_tx.lock().remove(&pts);
                Err(NvInferError::PipelineError(
                    "infer_sync channel disconnected".into(),
                ))
            }
        }
    }

    /// Synchronous inference – blocks until results arrive (up to 30 s).
    ///
    /// The buffer is **consumed**: if the [`SharedBuffer`] has outstanding
    /// references, an error is returned.
    ///
    /// Parameters are the same as [`submit`](NvInfer::submit).
    pub fn infer_sync(
        &self,
        batch: SharedBuffer,
        rois: Option<&HashMap<u32, Vec<Roi>>>,
    ) -> Result<BatchInferenceOutput> {
        self.infer_sync_with_timeout(batch, rois, Duration::from_secs(30))
    }

    /// Internal: attach ROI metadata, set PTS, push to appsrc.
    fn push_buffer(
        &self,
        mut batch: gst::Buffer,
        rois: Option<&HashMap<u32, Vec<Roi>>>,
        pts: u64,
    ) -> Result<()> {
        let (num_filled, max_batch_size) = read_surface_header(&batch)?;

        let slot_dims = if num_filled > 0 {
            read_slot_dimensions(&batch, num_filled)?
        } else {
            Vec::new()
        };

        // Build effective ROIs: for any slot without explicit user ROIs,
        // synthesise a full-frame ROI from the actual surface slot dimensions.
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

        // Patch each frame meta's source_frame_width/height to match the
        // actual surface slot dimensions.  DeepStream nvinfer uses these
        // fields for its internal NvBufSurfTransform scaling; leaving them
        // at 0 produces undefined crop/resize behaviour.
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

        self.appsrc
            .push_buffer(batch)
            .map_err(|e| NvInferError::PipelineError(format!("appsrc push failed: {:?}", e)))?;

        Ok(())
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
    use deepstream_buffers::ffi;

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

/// Extract inference outputs from a completed sample.
///
/// In secondary mode (`process-mode=2`), `Gst-nvinfer` attaches tensor outputs
/// to each `NvDsObjectMeta`'s `obj_user_meta_list`.  This function iterates:
///
/// ```text
/// batch_meta → frames → frame.objects() → object.user_meta() → tensor
/// ```
///
/// User frame ids are not duplicated here: read them from the output buffer
/// ([`BatchInferenceOutput::buffer`](crate::output::BatchInferenceOutput::buffer))
/// via [`SharedBuffer::savant_ids`](deepstream_buffers::SharedBuffer::savant_ids).
/// [`ElementOutput::slot_number`] is
/// [`FrameMeta::batch_id`](deepstream::FrameMeta::batch_id) (surface slot).
/// `roi_id` is the `object_id` cast to `i64`; `None` when the object carries
/// the full-frame sentinel (`unique_component_id == FULL_FRAME_SENTINEL`).
fn extract_batch_output(
    sample: gst::Sample,
    policy: MetaClearPolicy,
    host_copy_enabled: bool,
) -> Result<BatchInferenceOutput> {
    let buffer = sample
        .buffer()
        .ok_or_else(|| NvInferError::PipelineError("Sample has no buffer".into()))?;

    let batch_meta = unsafe {
        BatchMeta::from_gst_buffer(buffer.as_ptr() as *mut _).map_err(|e| {
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

    // Use from_glib_none (gst_mini_object_ref) to get a ref-counted handle
    // WITHOUT deep-copying.  BufferRef::to_owned() calls gst_mini_object_copy()
    // which deep-copies NvDsBatchMeta and crashes in nvds_acquire_meta_from_pool.
    let owned: gst::Buffer = unsafe { from_glib_none(buffer.as_ptr()) };
    let shared = SharedBuffer::from(owned);
    Ok(BatchInferenceOutput::new(
        shared,
        elements,
        policy.clear_after(),
        host_copy_enabled,
    ))
}
