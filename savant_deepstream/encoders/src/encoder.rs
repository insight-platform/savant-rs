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
//! [`BufferGenerator`], renders content into them, and then submits
//! them to the encoder. Encoded frames are collected from the appsink.

use crate::error::EncoderError;
use crate::{Codec, EncodedFrame, EncoderConfig, VideoFormat};
use deepstream_buffers::{
    bridge_savant_id_meta, pipeline::BufferGeneratorExt, BufferGenerator, CudaStream,
    TransformConfig,
};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use log::{debug, warn};
use std::collections::{HashMap, VecDeque};

/// Known GStreamer property names that control B-frame count on NVIDIA
/// encoders.  Used exclusively by [`force_disable_b_frames`] to ensure
/// B-frames are always off regardless of the encoder element's defaults.
const B_FRAME_PROPERTY_NAMES: &[&str] = &[
    "B-frames",
    "b-frames",
    "num-B-Frames",
    "num-b-frames",
    "num_B_Frames",
];

/// Map from output PTS → (frame_id, original duration).
type PtsMap = HashMap<u64, (u128, Option<u64>)>;

/// GPU-accelerated video encoder.
///
/// Encapsulates a DeepStream encoding pipeline with an
/// [`BufferGenerator`] for buffer management. The encoder validates
/// that B-frames are never enabled and that PTS values are monotonically
/// increasing.
///
/// # Lifecycle
///
/// 1. Create with [`NvEncoder::new`].
/// 2. Access the [`BufferGenerator`] via [`generator()`](Self::generator)
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
    generator: BufferGenerator,
    /// Codec used by this encoder.
    codec: Codec,
    /// Last **input** PTS — used to reject non-monotonic submissions.
    last_input_pts_ns: Option<u64>,
    /// Last **output** PTS — used to detect B-frame reordering in the
    /// encoded stream.  If the output PTS ever goes backwards, the
    /// encoder produced B-frames despite `force_disable_b_frames`.
    last_output_pts_ns: Option<u64>,
    /// Whether EOS has been sent.
    finalized: bool,
    /// PTS -> (frame_id, duration_ns) map for reconstructing output metadata.
    pts_map: PtsMap,
    /// Submission order of `frame_id` for intra-only codecs (JPEG, PNG, raw).
    ///
    /// Jetson `nvvideoconvert` / parsers may rewrite timestamps so the output
    /// buffer no longer matches [`Self::pts_map`].  Outputs are still in
    /// submission order, so we can fall back to popping this FIFO when the map
    /// misses (see [`Self::sample_to_frame`]).
    intra_submit_fifo: VecDeque<u128>,
    /// When format conversion is needed (e.g. RGBA → NV12), this holds a
    /// second generator for the encoder-native format and a dedicated
    /// non-blocking CUDA stream for the `NvBufSurfTransform` call.
    convert_ctx: Option<ConvertContext>,
}

/// Internal context for direct NvBufSurfTransform-based format conversion,
/// bypassing `nvvideoconvert` to avoid CUDA default-stream serialization.
struct ConvertContext {
    /// Generator for encoder-native format buffers (NV12 or I420).
    native_generator: BufferGenerator,
    /// Dedicated non-blocking CUDA stream (`cudaStreamNonBlocking`).
    cuda_stream: CudaStream,
}

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
        gst::init()
            .map_err(|e| EncoderError::PipelineError(format!("GStreamer init failed: {e}")))?;

        // Validate: encoder_params codec must match config.codec.
        if let Some(ref params) = config.encoder_params {
            let params_codec = params.codec();
            if params_codec != config.codec {
                return Err(EncoderError::InvalidProperty {
                    name: "encoder_params".to_string(),
                    reason: format!(
                        "properties variant is for codec {:?} but config codec is {:?}",
                        params_codec, config.codec
                    ),
                });
            }
        }

        // PNG uses a CPU-based GStreamer pipeline (nvvideoconvert -> pngenc).
        // Raw pseudoencoders download GPU frames to CPU memory.
        // Other codecs use NVIDIA hardware encoders.
        let is_png = config.codec == Codec::Png;
        let is_raw = matches!(config.codec, Codec::RawRgba | Codec::RawRgb);
        if is_png && config.format != VideoFormat::RGBA {
            return Err(EncoderError::InvalidProperty {
                name: "format".to_string(),
                reason: "PNG encoder requires VideoFormat::RGBA".to_string(),
            });
        }

        // H.264, HEVC and AV1 require NVENC hardware.  Fail early with a
        // clear error instead of letting the GStreamer element creation
        // produce a cryptic message or hang.
        let needs_nvenc = matches!(config.codec, Codec::H264 | Codec::Hevc | Codec::Av1);
        if needs_nvenc {
            let has_nvenc = nvidia_gpu_utils::has_nvenc(config.gpu_id).unwrap_or(false);
            if !has_nvenc {
                return Err(EncoderError::NvencNotAvailable {
                    codec: config.codec.name().to_string(),
                    gpu_id: config.gpu_id,
                });
            }
        }

        // Determine the encoder and parser element names (unused for PNG).
        let (enc_name, parse_name, needs_convert) = match config.codec {
            Codec::H264 => (
                "nvv4l2h264enc",
                "h264parse",
                config.format != VideoFormat::NV12 && config.format != VideoFormat::I420,
            ),
            Codec::Hevc => (
                "nvv4l2h265enc",
                "h265parse",
                config.format != VideoFormat::NV12 && config.format != VideoFormat::I420,
            ),
            Codec::Jpeg => ("nvjpegenc", "jpegparse", config.format != VideoFormat::I420),
            Codec::Av1 => (
                "nvv4l2av1enc",
                "av1parse",
                config.format != VideoFormat::NV12 && config.format != VideoFormat::I420,
            ),
            Codec::Png => ("pngenc", "identity", false),
            Codec::RawRgba => ("identity", "identity", false),
            Codec::RawRgb => ("identity", "identity", false),
        };

        // Determine encoder-native format.
        let native_format = match config.codec {
            Codec::Jpeg => VideoFormat::I420,
            Codec::Png => VideoFormat::RGBA,
            Codec::RawRgba => VideoFormat::RGBA,
            Codec::RawRgb => VideoFormat::RGBA,
            _ => VideoFormat::NV12,
        };

        // Buffer pool size for the user-facing (and native-format) generators.
        //
        // On dGPU, pool size 1: NVENC may continue DMA-reading from a buffer
        // after GStreamer releases its reference; a single buffer forces
        // serialization so memory is never overwritten while HW reads it.
        //
        // On Jetson, V4L2 video encoders (`nvv4l2h264enc` / `nvv4l2h265enc` /
        // `nvv4l2av1enc`) can keep several input buffers in flight before
        // releasing them; with a pool of 1, `acquire()` deadlocks.  A larger
        // pool (4) matches observed in-flight depth without requiring callers
        // to pull encoded output between submits.
        #[cfg(target_arch = "aarch64")]
        let pool_size: u32 = if matches!(config.codec, Codec::H264 | Codec::Hevc | Codec::Av1) {
            4
        } else {
            1
        };
        #[cfg(not(target_arch = "aarch64"))]
        let pool_size: u32 = 1;

        let generator = BufferGenerator::builder(config.format, config.width, config.height)
            .fps(config.fps_num, config.fps_den)
            .gpu_id(config.gpu_id)
            .mem_type(config.mem_type)
            .min_buffers(pool_size)
            .max_buffers(pool_size)
            .build()?;

        // When the user format differs from the encoder-native format, set up
        // a ConvertContext with a second generator + non-blocking CUDA stream.
        // This replaces the `nvvideoconvert` GStreamer element, avoiding the
        // CUDA default-stream serialization bottleneck.
        // PNG uses nvvideoconvert in-pipeline (NVMM -> CPU for pngenc).
        // Raw uses nvvideoconvert in-pipeline (NVMM -> CPU).
        let convert_ctx = if is_png || is_raw {
            None
        } else if needs_convert {
            let native_generator =
                BufferGenerator::builder(native_format, config.width, config.height)
                    .fps(config.fps_num, config.fps_den)
                    .gpu_id(config.gpu_id)
                    .mem_type(config.mem_type)
                    .min_buffers(pool_size)
                    .max_buffers(pool_size)
                    .build()?;

            let cuda_stream = CudaStream::new_non_blocking().map_err(|e| {
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
        let pipeline = gst::Pipeline::new();

        let appsrc = gst::ElementFactory::make("appsrc")
            .name("src")
            .build()
            .map_err(|_| EncoderError::ElementCreationFailed("appsrc".into()))?;

        let appsink = gst::ElementFactory::make("appsink")
            .name("sink")
            .build()
            .map_err(|_| EncoderError::ElementCreationFailed("appsink".into()))?;

        // Configure appsrc with NVMM caps in the encoder-native format.
        // When conversion is needed, appsrc receives already-converted
        // native-format buffers (not the user's RGBA).
        let appsrc_caps = if let Some(ctx) = &convert_ctx {
            ctx.native_generator.nvmm_caps_gst()
        } else {
            generator.nvmm_caps_gst()
        };
        let appsrc_elem: &gst::Element = appsrc.upcast_ref();
        appsrc_elem.set_property("caps", appsrc_caps);
        appsrc_elem.set_property_from_str("format", "time");
        appsrc_elem.set_property_from_str("stream-type", "stream");

        // Configure appsink.
        let appsink_elem: &gst::Element = appsink.upcast_ref();
        appsink_elem.set_property("sync", false);
        appsink_elem.set_property("emit-signals", false);

        let enc = if is_raw {
            None
        } else {
            Some(
                gst::ElementFactory::make(enc_name)
                    .name("enc")
                    .build()
                    .map_err(|_| EncoderError::ElementCreationFailed(enc_name.into()))?,
            )
        };

        if is_raw {
            let nvconv = gst::ElementFactory::make("nvvideoconvert")
                .name("nvconv")
                .build()
                .map_err(|_| EncoderError::ElementCreationFailed("nvvideoconvert".into()))?;

            #[cfg(target_arch = "aarch64")]
            nvconv.set_property_from_str("compute-hw", "1");

            let raw_format = if config.codec == Codec::RawRgba {
                "RGBA"
            } else {
                "RGB"
            };
            let caps = gst::Caps::builder("video/x-raw")
                .field("format", raw_format)
                .build();
            let capsfilter = gst::ElementFactory::make("capsfilter")
                .name("rawcaps")
                .property("caps", &caps)
                .build()
                .map_err(|_| EncoderError::ElementCreationFailed("capsfilter".into()))?;

            for elem in [&appsrc, &nvconv, &capsfilter, &appsink] {
                pipeline.add(elem).map_err(|e| {
                    EncoderError::PipelineError(format!("Failed to add element: {}", e))
                })?;
            }
            gst::Element::link_many([&appsrc, &nvconv, &capsfilter, &appsink]).map_err(|_| {
                EncoderError::LinkFailed("appsrc->nvvideoconvert->capsfilter->appsink".into())
            })?;
        } else if is_png {
            let enc = enc.as_ref().unwrap();
            let nvconv = gst::ElementFactory::make("nvvideoconvert")
                .name("nvconv")
                .build()
                .map_err(|_| EncoderError::ElementCreationFailed("nvvideoconvert".into()))?;

            // Apply PNG encoder properties (compression-level etc.).
            if let Some(ref params) = config.encoder_params {
                for (key, value) in params.to_gst_pairs() {
                    Self::set_element_property(enc, key, &value)?;
                }
            }

            for elem in [&appsrc, &nvconv, enc, &appsink] {
                pipeline.add(elem).map_err(|e| {
                    EncoderError::PipelineError(format!("Failed to add element: {}", e))
                })?;
            }
            gst::Element::link_many([&appsrc, &nvconv, enc, &appsink]).map_err(|_| {
                EncoderError::LinkFailed("appsrc->nvvideoconvert->pngenc->appsink".into())
            })?;
        } else {
            let enc = enc.as_ref().unwrap();
            // Hardware encoders: appsrc [-> nvvideoconvert] -> encoder -> parser -> appsink
            let parse = gst::ElementFactory::make(parse_name)
                .name("parse")
                .build()
                .map_err(|_| EncoderError::ElementCreationFailed(parse_name.into()))?;

            // On Jetson, nvjpegenc requires surfaces pinned/registered by
            // nvvideoconvert; NvDS pool surfaces from appsrc lack this
            // registration and cause hangs ("Surface not registered").
            #[cfg(target_arch = "aarch64")]
            let pre_enc: Option<gst::Element> = if config.codec == Codec::Jpeg {
                let conv = gst::ElementFactory::make("nvvideoconvert")
                    .name("pre_enc_conv")
                    .build()
                    .map_err(|_| EncoderError::ElementCreationFailed("nvvideoconvert".into()))?;
                conv.set_property("disable-passthrough", true);
                Some(conv)
            } else {
                None
            };
            #[cfg(not(target_arch = "aarch64"))]
            let pre_enc: Option<gst::Element> = None;

            // Bridge SavantIdMeta across the encoder element.
            bridge_savant_id_meta(enc);

            // Apply typed encoder properties.
            if let Some(ref params) = config.encoder_params {
                for (key, value) in params.to_gst_pairs() {
                    Self::set_element_property(enc, key, &value)?;
                }
            }

            // Forcibly disable B-frames on the encoder if the property exists.
            Self::force_disable_b_frames(enc);

            // Build the element chain, inserting nvvideoconvert when present.
            let mut all_elems: Vec<&gst::Element> = vec![&appsrc];
            if let Some(ref conv) = pre_enc {
                all_elems.push(conv);
            }
            all_elems.extend([enc, &parse, &appsink]);

            for elem in &all_elems {
                pipeline.add(*elem).map_err(|e| {
                    EncoderError::PipelineError(format!("Failed to add element: {}", e))
                })?;
            }
            gst::Element::link_many(&all_elems).map_err(|_| {
                EncoderError::LinkFailed("appsrc->[nvvideoconvert]->enc->parse->appsink".into())
            })?;
        }

        debug!(
            "NvEncoder pipeline built: codec={:?}, convert={}, png={}, raw={}",
            config.codec,
            convert_ctx.is_some(),
            is_png,
            is_raw,
        );

        // Start the pipeline.
        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| EncoderError::PipelineError(format!("Failed to start pipeline: {}", e)))?;

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
            last_input_pts_ns: None,
            last_output_pts_ns: None,
            finalized: false,
            pts_map: HashMap::new(),
            intra_submit_fifo: VecDeque::new(),
            convert_ctx,
        })
    }

    /// Access the internal [`BufferGenerator`].
    ///
    /// Use this to acquire NVMM buffers for rendering before submitting
    /// them to the encoder.
    pub fn generator(&self) -> &BufferGenerator {
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
        frame_id: u128,
        pts_ns: u64,
        duration_ns: Option<u64>,
    ) -> Result<(), EncoderError> {
        if self.finalized {
            return Err(EncoderError::AlreadyFinalized);
        }

        // Validate monotonic input PTS.
        if let Some(prev) = self.last_input_pts_ns {
            if pts_ns <= prev {
                return Err(EncoderError::PtsReordered {
                    frame_id,
                    pts_ns,
                    prev_pts_ns: prev,
                });
            }
        }
        self.last_input_pts_ns = Some(pts_ns);

        // Store PTS -> (frame_id, duration) mapping.
        self.pts_map.insert(pts_ns, (frame_id, duration_ns));

        if matches!(
            self.codec,
            Codec::Jpeg | Codec::Png | Codec::RawRgba | Codec::RawRgb
        ) {
            self.intra_submit_fifo.push_back(frame_id);
        }

        // When conversion is needed, transform the user buffer (e.g. RGBA)
        // into the encoder-native format (NV12/I420) using NvBufSurfTransform
        // with a dedicated non-blocking CUDA stream.
        let push_buffer = if let Some(ctx) = &self.convert_ctx {
            let transform_config = TransformConfig {
                padding: deepstream_buffers::Padding::None,
                dst_padding: None,
                interpolation: deepstream_buffers::Interpolation::Nearest,
                compute_mode: deepstream_buffers::ComputeMode::Default,
                cuda_stream: ctx.cuda_stream.clone(),
            };

            let src_shared = deepstream_buffers::SharedBuffer::from(buffer);
            let src_view =
                deepstream_buffers::SurfaceView::from_buffer(&src_shared, 0).map_err(|e| {
                    EncoderError::PipelineError(format!(
                        "Failed to create SurfaceView from source buffer: {}",
                        e
                    ))
                })?;
            let dst_shared = ctx
                .native_generator
                .transform(&src_view, &transform_config, None)
                .map_err(|e| {
                    EncoderError::PipelineError(format!(
                        "NvBufSurfTransform (format conversion) failed: {}",
                        e
                    ))
                })?;
            drop(src_view);
            drop(src_shared);
            let mut native_buf = dst_shared.into_buffer().map_err(|_| {
                EncoderError::BufferAcquisitionFailed(
                    "SharedBuffer has outstanding references".into(),
                )
            })?;

            {
                let buf_ref = native_buf.get_mut().ok_or_else(|| {
                    EncoderError::BufferAcquisitionFailed("Converted buffer is not writable".into())
                })?;
                buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
                buf_ref.set_dts(gst::ClockTime::from_nseconds(pts_ns));
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
                buf_ref.set_dts(gst::ClockTime::from_nseconds(pts_ns));
                if let Some(dur) = duration_ns {
                    buf_ref.set_duration(gst::ClockTime::from_nseconds(dur));
                }
            }
            buffer
        };

        // Push to appsrc.
        self.appsrc
            .push_buffer(push_buffer)
            .map_err(|e| EncoderError::PipelineError(format!("appsrc push failed: {:?}", e)))?;

        Ok(())
    }

    /// Pull one encoded frame from the encoder (non-blocking).
    ///
    /// Returns `Ok(Some(frame))` when an encoded frame is available,
    /// `Ok(None)` when no frame is ready yet, or `Err` on pipeline error
    /// or if B-frame reordering is detected in the output.
    pub fn pull_encoded(&mut self) -> Result<Option<EncodedFrame>, EncoderError> {
        let sample = self
            .appsink
            .try_pull_sample(gst::ClockTime::from_mseconds(0));
        self.sample_to_frame(sample)
    }

    /// Pull one encoded frame from the encoder (blocking with timeout).
    ///
    /// # Arguments
    ///
    /// * `timeout_ms` - Maximum time to wait in milliseconds.
    ///
    /// Returns `Ok(Some(frame))` when a frame arrives within the timeout,
    /// `Ok(None)` on timeout, or `Err` on pipeline error or if B-frame
    /// reordering is detected in the output.
    pub fn pull_encoded_timeout(
        &mut self,
        timeout_ms: u64,
    ) -> Result<Option<EncodedFrame>, EncoderError> {
        let sample = self
            .appsink
            .try_pull_sample(gst::ClockTime::from_mseconds(timeout_ms));
        self.sample_to_frame(sample)
    }

    /// Convert an appsink sample into an [`EncodedFrame`], validating
    /// that the output PTS is monotonically increasing and DTS ≤ PTS.
    ///
    /// These checks detect B-frame reordering: if B-frames were emitted
    /// despite `force_disable_b_frames`, the output decode order would
    /// differ from the presentation order and PTS would go backwards.
    fn sample_to_frame(
        &mut self,
        sample: Option<gst::Sample>,
    ) -> Result<Option<EncodedFrame>, EncoderError> {
        let sample = match sample {
            Some(s) => s,
            None => return Ok(None),
        };

        let buffer = sample
            .buffer()
            .ok_or_else(|| EncoderError::PipelineError("Sample has no buffer".into()))?;

        let buf_pts_ns = buffer.pts().map(|t| t.nseconds()).unwrap_or(0);
        let buf_dts_ns = buffer.dts().map(|t| t.nseconds());
        let duration_ns = buffer.duration().map(|t| t.nseconds());

        let is_intra_only = matches!(
            self.codec,
            Codec::Jpeg | Codec::Png | Codec::RawRgba | Codec::RawRgb
        );
        let buf_size = buffer.size() as u64;

        // Look up frame_id from our PTS map.  On Jetson the pre-encoder
        // nvvideoconvert retimestamps buffers (changing PTS) but preserves
        // the original PTS in DTS.  Try PTS first, fall back to DTS.
        let (pts_lookup, original_pts) = {
            let by_pts = self.pts_map.remove(&buf_pts_ns);
            if by_pts.is_some() {
                (by_pts, buf_pts_ns)
            } else if let Some(dts) = buf_dts_ns {
                let by_dts = self.pts_map.remove(&dts);
                (by_dts, dts)
            } else {
                (None, buf_pts_ns)
            }
        };

        // Intra-only codecs: keep a FIFO of submitted frame_ids.  If the map
        // misses (timestamp rewrite), correlate by strict submission order.
        let (frame_id, orig_duration, is_user_frame) = match pts_lookup {
            Some((id, dur)) => {
                if is_intra_only {
                    match self.intra_submit_fifo.pop_front() {
                        Some(fifo_id) if fifo_id != id => {
                            warn!(
                                "NvEncoder: intra FIFO head {fifo_id} != pts_map id {id} (codec={:?}) — trusting map",
                                self.codec
                            );
                        }
                        None => {
                            warn!(
                                "NvEncoder: intra FIFO empty on pts_map hit (codec={:?})",
                                self.codec
                            );
                        }
                        _ => {}
                    }
                }
                (Some(id), dur, true)
            }
            None if is_intra_only && buf_size > 0 => {
                let id = self.intra_submit_fifo.pop_front();
                if id.is_none() {
                    warn!(
                        "NvEncoder: encoded buffer has payload but intra FIFO empty (codec={:?}, size={buf_size})",
                        self.codec
                    );
                }
                (id, duration_ns, id.is_some())
            }
            None => (None, duration_ns, false),
        };
        let pts_ns = original_pts;

        // --- Output ordering validation ---
        //
        // Checks only apply to user-submitted frames.  Codec header
        // buffers emitted by some encoders (AV1 in particular) may carry
        // stale or meaningless PTS/DTS values.

        if is_user_frame {
            // 1. Output PTS must never go backwards.  Equal PTS is
            //    tolerated (duplicate PTS on codec headers is common).
            //    B-frame reordering causes PTS to *decrease*.
            if let Some(prev) = self.last_output_pts_ns {
                if pts_ns < prev {
                    return Err(EncoderError::OutputPtsReordered {
                        frame_id,
                        pts_ns,
                        prev_pts_ns: prev,
                    });
                }
            }
            self.last_output_pts_ns = Some(pts_ns);

            // 2. DTS must not exceed PTS (would indicate B-frame reordering).
            //    Intra-only codecs (JPEG, PNG, Raw) can never produce
            //    B-frames.  On Jetson the nvjpegenc pipeline may set DTS
            //    from a later frame's PTS, so the check is skipped.
            if !is_intra_only {
                if let Some(dts) = buf_dts_ns {
                    if dts > pts_ns {
                        return Err(EncoderError::OutputDtsExceedsPts {
                            frame_id,
                            dts_ns: dts,
                            pts_ns,
                        });
                    }
                }
            }
        }

        // For intra-only codecs, normalize DTS to equal PTS regardless
        // of what the GStreamer pipeline reported.
        let dts_ns = if is_intra_only {
            Some(pts_ns)
        } else {
            buf_dts_ns
        };

        // Use original duration if the encoder didn't set one.
        let final_duration = duration_ns.or(orig_duration);

        let keyframe = match self.codec {
            Codec::Jpeg | Codec::Png | Codec::RawRgba | Codec::RawRgb => true,
            _ => !buffer.flags().contains(gst::BufferFlags::DELTA_UNIT),
        };

        // Extract encoded data.
        let data = if matches!(self.codec, Codec::RawRgba | Codec::RawRgb) {
            Self::extract_raw_pixels(&sample, buffer)?
        } else {
            let map = buffer.map_readable().map_err(|e| {
                EncoderError::PipelineError(format!("Failed to map buffer: {:?}", e))
            })?;
            map.as_slice().to_vec()
        };

        Ok(Some(EncodedFrame {
            frame_id,
            pts_ns,
            dts_ns,
            duration_ns: final_duration,
            data,
            codec: self.codec,
            keyframe,
            time_base: (1, 1_000_000_000),
        }))
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
        self.appsrc
            .end_of_stream()
            .map_err(|e| EncoderError::PipelineError(format!("EOS failed: {:?}", e)))?;

        let timeout_ms = drain_timeout_ms.unwrap_or(2000);
        let mut frames = Vec::new();

        while let Some(frame) = self.pull_encoded_timeout(timeout_ms)? {
            frames.push(frame);
        }

        // Wait for EOS to propagate.
        let bus = self
            .pipeline
            .bus()
            .ok_or_else(|| EncoderError::PipelineError("Pipeline has no bus".into()))?;
        let _ = bus.timed_pop_filtered(
            gst::ClockTime::from_seconds(5),
            &[gst::MessageType::Eos, gst::MessageType::Error],
        );

        // Stop the pipeline.
        let _ = self.pipeline.set_state(gst::State::Null);

        if !self.intra_submit_fifo.is_empty() {
            warn!(
                "NvEncoder::finish: {} frame_id(s) left in intra_submit_fifo (codec={:?}) — outputs missing",
                self.intra_submit_fifo.len(),
                self.codec
            );
            self.intra_submit_fifo.clear();
        }

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
                if let gst::MessageView::Error(e) = msg.view() {
                    let err_msg = e.error().to_string();
                    let debug = e.debug().unwrap_or_default().to_string();
                    return Err(EncoderError::PipelineError(format!("{err_msg} ({debug})")));
                }
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
        for prop_name in B_FRAME_PROPERTY_NAMES {
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
            reason: format!(
                "failed to set value '{}' (set_property_from_str panicked)",
                value
            ),
        })
    }

    /// Extract tightly-packed pixel data from a raw video buffer,
    /// stripping any stride padding added by the video subsystem.
    fn extract_raw_pixels(
        sample: &gst::Sample,
        buffer: &gst::BufferRef,
    ) -> Result<Vec<u8>, EncoderError> {
        let caps = sample
            .caps()
            .ok_or_else(|| EncoderError::PipelineError("Raw sample has no caps".into()))?;
        let video_info = gst_video::VideoInfo::from_caps(caps).map_err(|e| {
            EncoderError::PipelineError(format!("Failed to parse video caps: {}", e))
        })?;

        let width = video_info.width() as usize;
        let height = video_info.height() as usize;
        let stride = video_info.stride()[0] as usize;
        let bpp = video_info.format_info().pixel_stride()[0] as usize;
        let row_bytes = width * bpp;

        let map = buffer
            .map_readable()
            .map_err(|e| EncoderError::PipelineError(format!("Failed to map buffer: {:?}", e)))?;
        let src = map.as_slice();

        if stride == row_bytes {
            Ok(src[..row_bytes * height].to_vec())
        } else {
            let mut data = Vec::with_capacity(row_bytes * height);
            for row in 0..height {
                data.extend_from_slice(&src[row * stride..row * stride + row_bytes]);
            }
            Ok(data)
        }
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

        // CudaStream's Drop handles CUDA stream destruction automatically.
        drop(self.convert_ctx.take());
    }
}
