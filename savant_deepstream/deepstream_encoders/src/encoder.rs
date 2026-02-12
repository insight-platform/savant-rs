//! GPU-accelerated video encoder backed by a DeepStream GStreamer pipeline.
//!
//! The encoder builds a GStreamer pipeline internally:
//!
//! ```text
//! appsrc (NVMM, encoder-native format) -> encoder -> parser -> appsink
//! ```
//!
//! When the user-facing format (e.g. RGBA) differs from the encoder's native
//! format (NV12/I420), the conversion is performed **outside** the GStreamer
//! pipeline using `NvBufSurfTransform` with a dedicated non-blocking CUDA
//! stream. This avoids the global GPU serialization barrier caused by
//! `nvvideoconvert` using the CUDA legacy default stream (stream 0).
//!
//! The user acquires NvBufSurface buffers from the embedded
//! [`NvBufSurfaceGenerator`], renders content into them, and then submits
//! them to the encoder. Encoded frames are collected from the appsink.

use crate::error::EncoderError;
use crate::{Codec, EncodedFrame, EncoderConfig};
use deepstream_nvbufsurface::{
    bridge_savant_id_meta, create_cuda_stream, destroy_cuda_stream,
    NvBufSurfaceGenerator, NvBufSurfaceMemType, TransformConfig,
};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use log::debug;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Properties that are known to enable B-frames and must be rejected.
const B_FRAME_PROPERTIES: &[&str] = &[
    "B-frames",
    "b-frames",
    "bframes",
    "Bframes",
    "num-B-Frames",
    "num-b-frames",
    "num-bframes",
];

/// GPU-accelerated video encoder.
///
/// Encapsulates a DeepStream encoding pipeline with an
/// [`NvBufSurfaceGenerator`] for buffer management. The encoder validates
/// that B-frames are never enabled and that PTS values are monotonically
/// increasing.
///
/// # Lifecycle
///
/// 1. Create with [`NvEncoder::new`].
/// 2. Access the [`NvBufSurfaceGenerator`] via [`generator()`](Self::generator)
///    to acquire NVMM buffers for rendering.
/// 3. Submit filled buffers with [`submit_frame`](Self::submit_frame).
/// 4. Pull encoded frames with [`pull_encoded`](Self::pull_encoded).
/// 5. Call [`finish`](Self::finish) to send EOS and drain remaining frames.
///    (also happens automatically on drop.)
pub struct NvEncoder {
    /// Internal GStreamer pipeline.
    pipeline: gst::Pipeline,
    /// AppSrc for pushing NVMM buffers.
    appsrc: gst_app::AppSrc,
    /// AppSink for pulling encoded frames.
    appsink: gst_app::AppSink,
    /// The NvBufSurface buffer generator for user-facing format (e.g. RGBA).
    generator: NvBufSurfaceGenerator,
    /// Codec used by this encoder.
    codec: Codec,
    /// Last PTS seen — used to detect reordering.
    last_pts_ns: Option<u64>,
    /// Whether EOS has been sent.
    finalized: bool,
    /// PTS -> (frame_id, duration_ns) map for reconstructing output metadata.
    pts_map: Arc<Mutex<HashMap<u64, (i64, Option<u64>)>>>,
    /// When format conversion is needed (e.g. RGBA → NV12), this holds a
    /// second generator for the encoder-native format and a dedicated
    /// non-blocking CUDA stream for the `NvBufSurfTransform` call.
    convert_ctx: Option<ConvertContext>,
}

/// Internal context for direct NvBufSurfTransform-based format conversion,
/// bypassing `nvvideoconvert` to avoid CUDA default-stream serialization.
struct ConvertContext {
    /// Generator for encoder-native format buffers (NV12 or I420).
    native_generator: NvBufSurfaceGenerator,
    /// Dedicated non-blocking CUDA stream (`cudaStreamNonBlocking`).
    cuda_stream: *mut std::ffi::c_void,
}

// Safety: the cuda_stream pointer is only used within the same GPU context
// and is not dereferenced on the Rust side.
unsafe impl Send for ConvertContext {}
unsafe impl Sync for ConvertContext {}

impl NvEncoder {
    /// Create a new GPU-accelerated encoder.
    ///
    /// Builds the internal GStreamer pipeline, configures the encoder element
    /// with the given properties, and starts the pipeline.
    ///
    /// # Arguments
    ///
    /// * `config` - Encoder configuration (codec, resolution, properties, …).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The codec is unsupported on this platform.
    /// - Any encoder property would enable B-frames.
    /// - GStreamer elements cannot be created or linked.
    pub fn new(config: &EncoderConfig) -> Result<Self, EncoderError> {
        // Ensure GStreamer is initialized.
        let _ = gst::init();

        // Validate: no B-frame properties allowed.
        for (key, _) in &config.encoder_properties {
            for bprop in B_FRAME_PROPERTIES {
                if key.eq_ignore_ascii_case(bprop) {
                    return Err(EncoderError::BFramesNotAllowed(key.clone()));
                }
            }
        }

        // Determine the encoder and parser element names.
        let (enc_name, parse_name, needs_convert) = match config.codec {
            Codec::H264 => (
                "nvv4l2h264enc",
                "h264parse",
                config.format != "NV12" && config.format != "I420",
            ),
            Codec::Hevc => (
                "nvv4l2h265enc",
                "h265parse",
                config.format != "NV12" && config.format != "I420",
            ),
            Codec::Jpeg => (
                "nvjpegenc",
                "jpegparse",
                config.format != "I420",
            ),
            Codec::Av1 => (
                "nvv4l2av1enc",
                "av1parse",
                config.format != "NV12" && config.format != "I420",
            ),
        };

        // Determine encoder-native format.
        let native_format = match config.codec {
            Codec::Jpeg => "I420",
            _ => "NV12",
        };

        // Create the user-facing buffer generator (e.g. RGBA).
        //
        // Pool size is hardcoded to 1: the NVENC hardware encoder may
        // continue DMA-reading from a buffer's GPU memory after the
        // GStreamer element has released its reference.  A pool of 1
        // forces full serialization so the buffer is never overwritten
        // while hardware is still reading from it.
        const POOL_SIZE: u32 = 1;

        let generator = NvBufSurfaceGenerator::builder(
            &config.format,
            config.width,
            config.height,
        )
        .fps(config.fps_num, config.fps_den)
        .gpu_id(config.gpu_id)
        .mem_type(NvBufSurfaceMemType::from(config.mem_type))
        .min_buffers(POOL_SIZE)
        .max_buffers(POOL_SIZE)
        .build()?;

        // When the user format differs from the encoder-native format, set up
        // a ConvertContext with a second generator + non-blocking CUDA stream.
        // This replaces the `nvvideoconvert` GStreamer element, avoiding the
        // CUDA default-stream serialization bottleneck.
        let convert_ctx = if needs_convert {
            let native_generator = NvBufSurfaceGenerator::builder(
                native_format,
                config.width,
                config.height,
            )
            .fps(config.fps_num, config.fps_den)
            .gpu_id(config.gpu_id)
            .mem_type(NvBufSurfaceMemType::from(config.mem_type))
            .min_buffers(POOL_SIZE)
            .max_buffers(POOL_SIZE)
            .build()?;

            let cuda_stream = create_cuda_stream().map_err(|e| {
                EncoderError::PipelineError(format!(
                    "Failed to create non-blocking CUDA stream: {}",
                    e
                ))
            })?;

            debug!(
                "ConvertContext created: {} -> {}, CUDA stream {:?}",
                config.format, native_format, cuda_stream,
            );

            Some(ConvertContext {
                native_generator,
                cuda_stream,
            })
        } else {
            None
        };

        // --- Build the GStreamer pipeline ---
        // No nvvideoconvert: appsrc -> encoder -> parser -> appsink
        let pipeline = gst::Pipeline::new();

        let appsrc = gst::ElementFactory::make("appsrc")
            .name("src")
            .build()
            .map_err(|_| EncoderError::ElementCreationFailed("appsrc".into()))?;

        let enc = gst::ElementFactory::make(enc_name)
            .name("enc")
            .build()
            .map_err(|_| EncoderError::ElementCreationFailed(enc_name.into()))?;

        let parse = gst::ElementFactory::make(parse_name)
            .name("parse")
            .build()
            .map_err(|_| EncoderError::ElementCreationFailed(parse_name.into()))?;

        let appsink = gst::ElementFactory::make("appsink")
            .name("sink")
            .build()
            .map_err(|_| EncoderError::ElementCreationFailed("appsink".into()))?;

        // Bridge SavantIdMeta across the encoder element.
        bridge_savant_id_meta(&enc);

        // Apply encoder properties.
        for (key, value) in &config.encoder_properties {
            Self::set_element_property(&enc, key, value)?;
        }

        // Forcibly disable B-frames on the encoder if the property exists.
        Self::force_disable_b_frames(&enc);

        // Configure appsrc with NVMM caps in the encoder-native format.
        // When conversion is needed, appsrc receives already-converted
        // native-format buffers (not the user's RGBA).
        let appsrc_caps = if convert_ctx.is_some() {
            convert_ctx.as_ref().unwrap().native_generator.nvmm_caps()
        } else {
            generator.nvmm_caps()
        };
        let appsrc_elem: &gst::Element = appsrc.upcast_ref();
        appsrc_elem.set_property("caps", &appsrc_caps);
        appsrc_elem.set_property_from_str("format", "time");
        appsrc_elem.set_property_from_str("stream-type", "stream");

        // Configure appsink.
        let appsink_elem: &gst::Element = appsink.upcast_ref();
        appsink_elem.set_property("sync", false);
        appsink_elem.set_property("emit-signals", false);

        // Assemble the pipeline chain: appsrc -> enc -> parse -> appsink
        let chain: Vec<&gst::Element> = vec![&appsrc, &enc, &parse, &appsink];

        for elem in &chain {
            pipeline.add(*elem).map_err(|e| {
                EncoderError::PipelineError(format!("Failed to add element: {}", e))
            })?;
        }
        for i in 0..chain.len() - 1 {
            chain[i].link(chain[i + 1]).map_err(|_| EncoderError::LinkFailed {
                from: chain[i].name().to_string(),
                to: chain[i + 1].name().to_string(),
            })?;
        }

        debug!(
            "NvEncoder pipeline built: {} elements, codec={:?}, convert={}",
            chain.len(),
            config.codec,
            convert_ctx.is_some(),
        );

        // Start the pipeline.
        pipeline.set_state(gst::State::Playing).map_err(|e| {
            EncoderError::PipelineError(format!("Failed to start pipeline: {}", e))
        })?;

        let appsrc_typed: gst_app::AppSrc = appsrc
            .dynamic_cast::<gst_app::AppSrc>()
            .map_err(|_| EncoderError::ElementCreationFailed("appsrc cast failed".into()))?;

        let appsink_typed: gst_app::AppSink = appsink
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| EncoderError::ElementCreationFailed("appsink cast failed".into()))?;

        Ok(Self {
            pipeline,
            appsrc: appsrc_typed,
            appsink: appsink_typed,
            generator,
            codec: config.codec,
            last_pts_ns: None,
            finalized: false,
            pts_map: Arc::new(Mutex::new(HashMap::new())),
            convert_ctx,
        })
    }

    /// Access the internal [`NvBufSurfaceGenerator`].
    ///
    /// Use this to acquire NVMM buffers for rendering before submitting
    /// them to the encoder.
    pub fn generator(&self) -> &NvBufSurfaceGenerator {
        &self.generator
    }

    /// The codec used by this encoder.
    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// Submit a filled NVMM buffer to the encoder.
    ///
    /// The buffer should have been acquired from [`generator()`](Self::generator).
    /// PTS values must be strictly monotonically increasing.
    ///
    /// # Arguments
    ///
    /// * `buffer` - A GStreamer buffer with NvBufSurface memory.
    /// * `frame_id` - User-defined frame identifier (carried through to output).
    /// * `pts_ns` - Presentation timestamp in nanoseconds. Must be strictly
    ///   greater than the previous PTS.
    /// * `duration_ns` - Optional duration in nanoseconds.
    ///
    /// # Errors
    ///
    /// Returns [`EncoderError::PtsReordered`] if PTS is not monotonically
    /// increasing. Returns [`EncoderError::AlreadyFinalized`] if EOS was
    /// already sent.
    pub fn submit_frame(
        &mut self,
        mut buffer: gst::Buffer,
        frame_id: i64,
        pts_ns: u64,
        duration_ns: Option<u64>,
    ) -> Result<(), EncoderError> {
        if self.finalized {
            return Err(EncoderError::AlreadyFinalized);
        }

        // Validate monotonic PTS.
        if let Some(prev) = self.last_pts_ns {
            if pts_ns <= prev {
                return Err(EncoderError::PtsReordered {
                    frame_id,
                    pts_ns,
                    prev_pts_ns: prev,
                });
            }
        }
        self.last_pts_ns = Some(pts_ns);

        // Store PTS -> (frame_id, duration) mapping.
        self.pts_map
            .lock()
            .unwrap()
            .insert(pts_ns, (frame_id, duration_ns));

        // When conversion is needed, transform the user buffer (e.g. RGBA)
        // into the encoder-native format (NV12/I420) using NvBufSurfTransform
        // with a dedicated non-blocking CUDA stream.
        let push_buffer = if let Some(ctx) = &self.convert_ctx {
            let transform_config = TransformConfig {
                padding: deepstream_nvbufsurface::Padding::None,
                interpolation: deepstream_nvbufsurface::Interpolation::Nearest,
                src_rect: None,
                compute_mode: deepstream_nvbufsurface::ComputeMode::Default,
                cuda_stream: ctx.cuda_stream,
            };

            let mut native_buf = ctx
                .native_generator
                .transform(&buffer, &transform_config, None)
                .map_err(|e| {
                    EncoderError::PipelineError(format!(
                        "NvBufSurfTransform (format conversion) failed: {}",
                        e
                    ))
                })?;

            // Set timestamps on the converted buffer.
            {
                let buf_ref = native_buf.get_mut().ok_or_else(|| {
                    EncoderError::BufferAcquisitionFailed(
                        "Converted buffer is not writable".into(),
                    )
                })?;
                buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
                if let Some(dur) = duration_ns {
                    buf_ref.set_duration(gst::ClockTime::from_nseconds(dur));
                }
            }
            native_buf
        } else {
            // No conversion needed — set timestamps and push directly.
            {
                let buf_ref = buffer.get_mut().ok_or_else(|| {
                    EncoderError::BufferAcquisitionFailed("Buffer is not writable".into())
                })?;
                buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
                if let Some(dur) = duration_ns {
                    buf_ref.set_duration(gst::ClockTime::from_nseconds(dur));
                }
            }
            buffer
        };

        // Push to appsrc.
        self.appsrc.push_buffer(push_buffer).map_err(|e| {
            EncoderError::PipelineError(format!("appsrc push failed: {:?}", e))
        })?;

        Ok(())
    }

    /// Pull one encoded frame from the encoder (non-blocking).
    ///
    /// Returns `Ok(Some(frame))` when an encoded frame is available,
    /// `Ok(None)` when no frame is ready yet, or `Err` on pipeline error.
    pub fn pull_encoded(&self) -> Result<Option<EncodedFrame>, EncoderError> {
        let sample = self
            .appsink
            .try_pull_sample(gst::ClockTime::from_mseconds(0));

        match sample {
            Some(sample) => {
                let buffer = sample.buffer().ok_or_else(|| {
                    EncoderError::PipelineError("Sample has no buffer".into())
                })?;

                let pts_ns = buffer
                    .pts()
                    .map(|t| t.nseconds())
                    .unwrap_or(0);

                let dts_ns = buffer.dts().map(|t| t.nseconds());
                let duration_ns = buffer.duration().map(|t| t.nseconds());

                // Look up frame_id from our PTS map.
                let (frame_id, orig_duration) = self
                    .pts_map
                    .lock()
                    .unwrap()
                    .remove(&pts_ns)
                    .unwrap_or((-1, duration_ns));

                // Use original duration if the encoder didn't set one.
                let final_duration = duration_ns.or(orig_duration);

                // Extract encoded data.
                let map = buffer.map_readable().map_err(|e| {
                    EncoderError::PipelineError(format!("Failed to map buffer: {:?}", e))
                })?;
                let data = map.as_slice().to_vec();

                Ok(Some(EncodedFrame {
                    frame_id,
                    pts_ns,
                    dts_ns,
                    duration_ns: final_duration,
                    data,
                    codec: self.codec,
                }))
            }
            None => Ok(None),
        }
    }

    /// Pull one encoded frame from the encoder (blocking with timeout).
    ///
    /// # Arguments
    ///
    /// * `timeout_ms` - Maximum time to wait in milliseconds.
    ///
    /// Returns `Ok(Some(frame))` when a frame arrives within the timeout,
    /// `Ok(None)` on timeout, or `Err` on pipeline error.
    pub fn pull_encoded_timeout(
        &self,
        timeout_ms: u64,
    ) -> Result<Option<EncodedFrame>, EncoderError> {
        let sample = self
            .appsink
            .try_pull_sample(gst::ClockTime::from_mseconds(timeout_ms));

        match sample {
            Some(sample) => {
                let buffer = sample.buffer().ok_or_else(|| {
                    EncoderError::PipelineError("Sample has no buffer".into())
                })?;

                let pts_ns = buffer.pts().map(|t| t.nseconds()).unwrap_or(0);
                let dts_ns = buffer.dts().map(|t| t.nseconds());
                let duration_ns = buffer.duration().map(|t| t.nseconds());

                let (frame_id, orig_duration) = self
                    .pts_map
                    .lock()
                    .unwrap()
                    .remove(&pts_ns)
                    .unwrap_or((-1, duration_ns));

                let final_duration = duration_ns.or(orig_duration);

                let map = buffer.map_readable().map_err(|e| {
                    EncoderError::PipelineError(format!("Failed to map buffer: {:?}", e))
                })?;
                let data = map.as_slice().to_vec();

                Ok(Some(EncodedFrame {
                    frame_id,
                    pts_ns,
                    dts_ns,
                    duration_ns: final_duration,
                    data,
                    codec: self.codec,
                }))
            }
            None => Ok(None),
        }
    }

    /// Send EOS and drain all remaining encoded frames.
    ///
    /// After this call, no more frames can be submitted. Any remaining
    /// frames in the pipeline are collected and returned.
    ///
    /// # Arguments
    ///
    /// * `drain_timeout_ms` - Per-frame timeout for draining (default: 2000ms).
    pub fn finish(
        &mut self,
        drain_timeout_ms: Option<u64>,
    ) -> Result<Vec<EncodedFrame>, EncoderError> {
        if self.finalized {
            return Ok(vec![]);
        }
        self.finalized = true;

        // Send EOS.
        self.appsrc.end_of_stream().map_err(|e| {
            EncoderError::PipelineError(format!("EOS failed: {:?}", e))
        })?;

        let timeout_ms = drain_timeout_ms.unwrap_or(2000);
        let mut frames = Vec::new();

        // Drain remaining frames from the appsink.
        loop {
            match self.pull_encoded_timeout(timeout_ms)? {
                Some(frame) => frames.push(frame),
                None => break,
            }
        }

        // Wait for EOS to propagate.
        let bus = self.pipeline.bus().ok_or_else(|| {
            EncoderError::PipelineError("Pipeline has no bus".into())
        })?;
        let _ = bus.timed_pop_filtered(
            gst::ClockTime::from_seconds(5),
            &[gst::MessageType::Eos, gst::MessageType::Error],
        );

        // Stop the pipeline.
        let _ = self.pipeline.set_state(gst::State::Null);

        debug!("NvEncoder finished, drained {} frames", frames.len());
        Ok(frames)
    }

    /// Check the pipeline bus for errors (non-blocking).
    ///
    /// Returns `Ok(())` if no error is pending, or `Err` with the error
    /// message.
    pub fn check_error(&self) -> Result<(), EncoderError> {
        if let Some(bus) = self.pipeline.bus() {
            if let Some(msg) = bus.pop_filtered(&[gst::MessageType::Error]) {
                let err = msg
                    .structure()
                    .and_then(|s| s.get::<String>("debug").ok())
                    .unwrap_or_default();
                return Err(EncoderError::PipelineError(err));
            }
        }
        Ok(())
    }

    // --- Internal helpers ---

    /// Forcibly disable B-frames on the encoder element if it has a
    /// recognized B-frame property.
    fn force_disable_b_frames(enc: &gst::Element) {
        // nvv4l2h264enc / nvv4l2h265enc use "num-B-Frames" (sometimes
        // "B-frames" depending on driver). Try known names and set to 0.
        for prop_name in B_FRAME_PROPERTIES {
            if enc.find_property(prop_name).is_some() {
                // set_property_from_str may panic if the value is invalid;
                // catch panics so we can log and continue.
                let enc_clone = enc.clone();
                let name = prop_name.to_string();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    enc_clone.set_property_from_str(&name, "0");
                }));
                match result {
                    Ok(()) => debug!("Disabled B-frames: {} = 0", prop_name),
                    Err(_) => debug!("Could not set {} to 0 on encoder", prop_name),
                }
            }
        }
    }

    /// Set a GStreamer element property from a string key/value pair.
    fn set_element_property(
        element: &gst::Element,
        key: &str,
        value: &str,
    ) -> Result<(), EncoderError> {
        // Reject B-frame properties.
        for bprop in B_FRAME_PROPERTIES {
            if key.eq_ignore_ascii_case(bprop) {
                return Err(EncoderError::BFramesNotAllowed(key.to_string()));
            }
        }

        // Check that the property exists on this element.
        if element.find_property(key).is_none() {
            return Err(EncoderError::InvalidProperty {
                name: key.to_string(),
                reason: "property not found on encoder element".to_string(),
            });
        }

        // set_property_from_str may panic on invalid values; catch and
        // convert to a proper error.
        let elem = element.clone();
        let k = key.to_string();
        let v = value.to_string();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            elem.set_property_from_str(&k, &v);
        }));

        result.map_err(|_| EncoderError::InvalidProperty {
            name: key.to_string(),
            reason: format!("failed to set value '{}' (set_property_from_str panicked)", value),
        })
    }
}

impl Drop for NvEncoder {
    fn drop(&mut self) {
        if !self.finalized {
            // Best-effort EOS + shutdown.
            let _ = self.appsrc.end_of_stream();
            let _ = self.pipeline.set_state(gst::State::Null);
            debug!("NvEncoder dropped (EOS sent)");
        }

        // Destroy the dedicated CUDA stream if we created one.
        if let Some(ctx) = self.convert_ctx.take() {
            if let Err(e) = destroy_cuda_stream(ctx.cuda_stream) {
                log::warn!("Failed to destroy CUDA stream on drop: {}", e);
            } else {
                debug!("ConvertContext CUDA stream destroyed");
            }
        }
    }
}
