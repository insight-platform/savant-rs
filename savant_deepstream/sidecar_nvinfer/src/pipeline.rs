//! Sidecar inference pipeline implementation.

use crate::batch_meta_builder::attach_batch_meta;
use crate::config::SidecarConfig;
use crate::error::{Result, SidecarError};
use crate::nvinfer_types::{DataType, InferDims, InferTensorMeta};
use crate::output::{BatchInferenceOutput, ElementOutput, TensorView};
use deepstream::BatchMeta;
use deepstream_nvbufsurface::{bridge_savant_id_meta, SavantIdMeta, SavantIdMetaKind};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_app::AppSinkCallbacks;
use log::debug;
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

/// Callback type invoked when inference completes (async mode).
pub type InferCallback = Box<dyn FnMut(BatchInferenceOutput) + Send>;

/// Shared state for sync/async sample delivery.
struct SampleDelivery {
    /// User callback for async mode.
    callback: Mutex<Option<InferCallback>>,
    /// Per-batch_id senders for infer_sync callers waiting on specific batches.
    sync_tx: Mutex<HashMap<u64, mpsc::Sender<BatchInferenceOutput>>>,
}

/// The sidecar inference engine.
pub struct SidecarNvInfer {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    #[allow(dead_code)] // Kept alive for callbacks; pipeline owns the element
    appsink: gst_app::AppSink,
    _config: SidecarConfig,
    delivery: Arc<SampleDelivery>,
}

impl SidecarNvInfer {
    /// Create a new sidecar inference engine.
    pub fn new(config: SidecarConfig, callback: InferCallback) -> Result<Self> {
        let _ = gst::init();

        let config_path = config
            .config_file_path
            .canonicalize()
            .map_err(|e| SidecarError::PipelineError(format!("Config file path invalid: {}", e)))?
            .to_string_lossy()
            .to_string();

        let pipeline = gst::Pipeline::new();

        let appsrc = gst::ElementFactory::make("appsrc")
            .name("src")
            .build()
            .map_err(|_| SidecarError::ElementCreationFailed("appsrc".into()))?;

        let appsink = gst::ElementFactory::make("appsink")
            .name("sink")
            .build()
            .map_err(|_| SidecarError::ElementCreationFailed("appsink".into()))?;

        // Build appsrc caps from config input dimensions.
        let appsrc_caps = gst::Caps::builder("video/x-raw")
            .features(["memory:NVMM"])
            .field("format", config.input_format.as_str())
            .field("width", config.input_width as i32)
            .field("height", config.input_height as i32)
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
            .map_err(|_| SidecarError::ElementCreationFailed("nvinfer".into()))?;

        // Set config file path.
        nvinfer.set_property_from_str("config-file-path", &config_path);

        // Force output tensor meta so we can read results.
        Self::set_element_property(&nvinfer, "output-tensor-meta", "1")?;

        // Apply user properties.
        for (key, value) in &config.properties {
            Self::set_element_property(&nvinfer, key, value)?;
        }

        // Bridge SavantIdMeta across nvinfer so output buffers carry per-frame IDs.
        bridge_savant_id_meta(&nvinfer);

        let elements: Vec<gst::Element> = if config.queue_depth > 0 {
            let queue = gst::ElementFactory::make("queue")
                .name("queue")
                .build()
                .map_err(|_| SidecarError::ElementCreationFailed("queue".into()))?;
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
                SidecarError::PipelineError(format!("Failed to add element: {}", e))
            })?;
        }

        gst::Element::link_many(elements.iter())
            .map_err(|_| SidecarError::LinkFailed("appsrc->[queue]->nvinfer->appsink".into()))?;

        let appsrc_typed: gst_app::AppSrc = appsrc
            .dynamic_cast::<gst_app::AppSrc>()
            .map_err(|_| SidecarError::ElementCreationFailed("appsrc cast failed".into()))?;

        let appsink_typed: gst_app::AppSink = appsink
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| SidecarError::ElementCreationFailed("appsink cast failed".into()))?;

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
                let batch_id = sample
                    .buffer()
                    .and_then(|b| b.pts())
                    .map(|t| t.nseconds())
                    .unwrap_or(0);

                let output = extract_batch_output(sample, batch_id).map_err(|e| {
                    log::error!("extract_batch_output error: {:?}", e);
                    gst::FlowError::Error
                })?;

                // If infer_sync is waiting for this batch_id, deliver there;
                // otherwise invoke the user callback.  The sync_tx lock is
                // released before the callback runs so that concurrent
                // infer_sync callers (or a callback that itself calls
                // infer_sync) never deadlock on the non-reentrant Mutex.
                let sync_sender = delivery_clone.sync_tx.lock().unwrap().remove(&batch_id);
                if let Some(tx) = sync_sender {
                    let _ = tx.send(output);
                } else if let Some(ref mut cb) = *delivery_clone.callback.lock().unwrap() {
                    cb(output);
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build();
        appsink_typed.set_callbacks(callbacks);

        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| SidecarError::PipelineError(format!("Failed to start pipeline: {}", e)))?;

        debug!(
            "SidecarNvInfer pipeline built: queue_depth={}",
            config.queue_depth
        );

        Ok(Self {
            pipeline,
            appsrc: appsrc_typed,
            appsink: appsink_typed,
            _config: config,
            delivery,
        })
    }

    /// Submit a batched buffer for inference. batch_id is user-chosen.
    pub fn submit(&self, mut batch: gst::Buffer, batch_id: u64) -> Result<()> {
        let (num_filled, max_batch_size) = read_surface_header(&batch)?;
        attach_batch_meta(
            batch
                .get_mut()
                .ok_or_else(|| SidecarError::PipelineError("Buffer is not writable".into()))?,
            num_filled,
            max_batch_size,
        )?;

        {
            let buf_ref = batch
                .get_mut()
                .ok_or_else(|| SidecarError::PipelineError("Buffer not writable".into()))?;
            buf_ref.set_pts(gst::ClockTime::from_nseconds(batch_id));
        }

        self.appsrc
            .push_buffer(batch)
            .map_err(|e| SidecarError::PipelineError(format!("appsrc push failed: {:?}", e)))?;

        Ok(())
    }

    /// Synchronous inference (ignores callback, returns output directly).
    pub fn infer_sync(&self, batch: gst::Buffer, batch_id: u64) -> Result<BatchInferenceOutput> {
        let (tx, rx) = mpsc::channel();
        self.delivery.sync_tx.lock().unwrap().insert(batch_id, tx);
        if let Err(e) = self.submit(batch, batch_id) {
            self.delivery.sync_tx.lock().unwrap().remove(&batch_id);
            return Err(e);
        }
        match rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(output) => Ok(output),
            Err(e) => {
                self.delivery.sync_tx.lock().unwrap().remove(&batch_id);
                Err(SidecarError::PipelineError(format!(
                    "infer_sync timeout: {:?}",
                    e
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
            .ok_or_else(|| SidecarError::PipelineError("Pipeline has no bus".into()))?;
        let _ = bus.timed_pop_filtered(
            gst::ClockTime::from_seconds(10),
            &[gst::MessageType::Eos, gst::MessageType::Error],
        );
        self.pipeline
            .set_state(gst::State::Null)
            .map_err(|e| SidecarError::PipelineError(format!("set_state Null failed: {:?}", e)))?;
        Ok(())
    }

    fn set_element_property(element: &gst::Element, key: &str, value: &str) -> Result<()> {
        if element.find_property(key).is_none() {
            return Err(SidecarError::InvalidProperty(format!(
                "property '{}' not found",
                key
            )));
        }
        let elem = element.clone();
        let k = key.to_string();
        let v = value.to_string();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            elem.set_property_from_str(&k, &v);
        }))
        .map_err(|_| {
            SidecarError::InvalidProperty(format!("failed to set '{}' = '{}'", key, value))
        })?;
        Ok(())
    }
}

impl Drop for SidecarNvInfer {
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
        .map_err(|e| SidecarError::BatchMetaFailed(format!("map_readable failed: {:?}", e)))?;
    let data = map.as_slice();
    if data.len() < 12 {
        return Err(SidecarError::BatchMetaFailed(
            "Buffer too small for NvBufSurface".into(),
        ));
    }
    let batch_size = u32::from_ne_bytes([data[4], data[5], data[6], data[7]]);
    let num_filled = u32::from_ne_bytes([data[8], data[9], data[10], data[11]]);
    Ok((num_filled, batch_size))
}

fn savant_id_to_i64(k: &SavantIdMetaKind) -> i64 {
    match k {
        SavantIdMetaKind::Frame(id) | SavantIdMetaKind::Batch(id) => *id,
    }
}

fn extract_batch_output(sample: gst::Sample, batch_id: u64) -> Result<BatchInferenceOutput> {
    let buffer = sample
        .buffer()
        .ok_or_else(|| SidecarError::PipelineError("Sample has no buffer".into()))?;

    let batch_meta = unsafe {
        BatchMeta::from_gst_buffer(buffer.as_ptr() as *mut _).map_err(|e| {
            SidecarError::PipelineError(format!("BatchMeta::from_gst_buffer: {:?}", e))
        })?
    };

    let ids: Vec<i64> = buffer
        .meta::<SavantIdMeta>()
        .map(|m| m.ids().iter().map(savant_id_to_i64).collect())
        .unwrap_or_default();

    let frames = batch_meta.frames();
    let mut elements = Vec::with_capacity(frames.len());
    for (i, frame) in frames.into_iter().enumerate() {
        let id = ids.get(i).copied();
        let mut tensors = Vec::new();
        for user_meta in frame.user_meta() {
            if user_meta.meta_type() != deepstream_sys::NvDsMetaType_NVDSINFER_TENSOR_OUTPUT_META {
                continue;
            }
            let raw_ptr = user_meta.user_meta_data();
            let tensor_meta = unsafe {
                InferTensorMeta::from_raw(raw_ptr as *mut deepstream_sys::NvDsInferTensorMeta)
            };
            if let Some(tm) = tensor_meta {
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
                    let data_type = layer_types.get(j).copied().unwrap_or(DataType::Float);
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
        elements.push(ElementOutput { id, tensors });
    }

    Ok(BatchInferenceOutput::new(batch_id, sample, elements))
}
