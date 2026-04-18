//! Channel-based encoder pipeline built on the `savant_gstreamer::pipeline`
//! framework.
//!
//! Mirrors [`deepstream_decoders::pipeline`] in structure and lifecycle:
//!
//! * An internal [`GstPipeline`] owns the `appsrc → [nvvideoconvert] →
//!   encoder → parser → appsink` (or `appsrc → nvvideoconvert → capsfilter
//!   / pngenc → appsink`) chain, feeder / drain threads, and the
//!   rescue-probe pair for in-band `savant.*` custom events.
//! * The public [`NvEncoder::submit_frame`] /
//!   [`send_event`](NvEncoder::send_event) /
//!   [`send_source_eos`](NvEncoder::send_source_eos) /
//!   [`recv`](NvEncoder::recv) /
//!   [`graceful_shutdown`](NvEncoder::graceful_shutdown) /
//!   [`shutdown`](NvEncoder::shutdown) API takes `&self` so the encoder
//!   can be shared behind `Arc` without external locking.
//! * Output arrives on a crossbeam channel as [`NvEncoderOutput`], with
//!   per-source EOS markers surfaced via [`NvEncoderOutput::SourceEos`].
//!
//! Format conversion for RGBA/RGB/BGR → NV12/I420 for hardware encoders
//! is performed *outside* the pipeline using `NvBufSurfTransform` on a
//! dedicated non-blocking CUDA stream; this avoids the CUDA default
//! stream contention introduced by `nvvideoconvert` in-pipeline.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam::channel::{self, Receiver, Sender};
use deepstream_buffers::{
    bridge_savant_id_meta, pipeline::BufferGeneratorExt, BufferGenerator, CudaStream, Padding,
    SurfaceView, TransformConfig,
};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;
use log::{debug, info, warn};
use parking_lot::Mutex;
use savant_gstreamer::pipeline::{
    build_source_eos_event, parse_source_eos_event, GstPipeline, PipelineConfig, PipelineInput,
    PipelineOutput, PtsPolicy,
};

use crate::config::{EncoderConfig, NvEncoderConfig};
use crate::error::EncoderError;
use crate::{Codec, EncodedFrame, VideoFormat};

// ─── Constants ──────────────────────────────────────────────────────

/// GStreamer property names that enable B-frames on NVIDIA encoders.
/// Used by [`force_disable_b_frames`] to make sure B-frames are always
/// disabled regardless of element defaults.
const B_FRAME_PROPERTY_NAMES: &[&str] = &[
    "B-frames",
    "b-frames",
    "num-B-Frames",
    "num-b-frames",
    "num_B_Frames",
];

// ─── Output enum ────────────────────────────────────────────────────

/// Output from the channel-based encoder API.
#[derive(Debug)]
pub enum NvEncoderOutput {
    /// An encoded frame, correlated with the submitter's `frame_id` when
    /// possible.
    Frame(EncodedFrame),
    /// A downstream GStreamer event captured at the pipeline output that
    /// is **not** a [`SourceEos`](Self::SourceEos) marker.
    Event(gst::Event),
    /// Per-source logical EOS (custom downstream event passthrough).
    /// Delivered in-band, ordered with encoded frames. Does **not** stop
    /// the encoder — use [`NvEncoder::graceful_shutdown`] for that.
    SourceEos { source_id: String },
    /// End-of-stream: all buffered frames have been emitted.
    Eos,
    /// Pipeline or framework runtime error.
    Error(EncoderError),
}

// ─── Internal types ─────────────────────────────────────────────────

/// Per-frame metadata indexed by the submitted PTS.
#[derive(Debug, Clone)]
struct FrameMetadata {
    frame_id: u128,
    duration_ns: Option<u64>,
}

/// Direct NvBufSurfTransform conversion path (e.g. RGBA → NV12) with
/// a dedicated non-blocking CUDA stream. Avoids `nvvideoconvert`
/// default-stream serialization.
struct ConvertContext {
    native_generator: BufferGenerator,
    cuda_stream: CudaStream,
}

enum EncoderBackendState {
    /// GStreamer pipeline: appsrc → [nvvideoconvert] → enc → parser → appsink.
    ///
    /// `convert_ctx` is `Some` when a format conversion (RGBA → NV12/I420)
    /// is performed outside the pipeline — in that case `appsrc` receives
    /// buffers in the encoder-native format rather than the user-facing
    /// format.
    Pipeline {
        input_tx: Sender<PipelineInput>,
        output_rx: Receiver<PipelineOutput>,
        pipeline: Mutex<GstPipeline>,
        convert_ctx: Option<ConvertContext>,
    },
}

// ─── NvEncoder ──────────────────────────────────────────────────────

/// GPU-accelerated video encoder.
///
/// Channel-based, shareable behind `Arc`. Lifecycle mirrors
/// [`NvDecoder`](https://docs.rs/deepstream_decoders):
///
/// 1. Create with [`NvEncoder::new`].
/// 2. Acquire NVMM buffers via [`generator()`](Self::generator),
///    render into them, and submit with
///    [`submit_frame`](Self::submit_frame).
/// 3. Pull encoded outputs with
///    [`recv`](Self::recv) / [`recv_timeout`](Self::recv_timeout) /
///    [`try_recv`](Self::try_recv).
/// 4. Tear down with [`graceful_shutdown`](Self::graceful_shutdown) or
///    [`shutdown`](Self::shutdown) (also called implicitly on `Drop`).
pub struct NvEncoder {
    name: String,
    backend: EncoderBackendState,
    /// NVMM buffer pool for user-facing frames (the format the caller
    /// renders into; see `config.format()`).
    generator: Arc<Mutex<BufferGenerator>>,
    /// Codec used by this encoder.
    codec: Codec,
    /// User-facing video format (format callers render in).
    user_format: VideoFormat,
    /// Runtime flags.
    draining: AtomicBool,
    is_shut_down: AtomicBool,
    finalized: AtomicBool,
    /// Last submitted input PTS (for strict-monotonic validation).
    last_input_pts_ns: Mutex<Option<u64>>,
    /// Last delivered output PTS (for B-frame reordering detection).
    last_output_pts_ns: Mutex<Option<u64>>,
    /// Output-side PTS → (frame_id, duration) correlation map.
    pts_map: Mutex<HashMap<u64, FrameMetadata>>,
    /// Submission order of `frame_id` for intra-only codecs (JPEG, PNG,
    /// raw) whose pipelines may rewrite PTS.
    intra_submit_fifo: Mutex<VecDeque<u128>>,
    /// Stashed stream-level codec metadata (e.g. AV1 `OBU_SEQUENCE_HEADER`
    /// emitted as a standalone buffer before the first IDR on dGPU
    /// `nvv4l2av1enc`). Prepended to the next user frame's `data` to
    /// match Jetson's inline-header behaviour.
    pending_codec_header: Mutex<Option<Vec<u8>>>,
    /// Per-buffer frame_id counter for SavantIdMeta tagging (same
    /// mechanism as the decoder).  This is the identity used internally
    /// by the rescue-probe pair to keep events properly ordered; it is
    /// not surfaced on output.
    submit_counter: AtomicU64,
}

impl NvEncoder {
    /// Build and start a new encoder.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use deepstream_encoders::prelude::*;
    /// use std::time::Duration;
    ///
    /// gstreamer::init().unwrap();
    /// cuda_init(0).unwrap();
    ///
    /// let encoder_cfg = EncoderConfig::H264(
    ///     H264EncoderConfig::new(1920, 1080).format(VideoFormat::NV12),
    /// );
    /// let cfg = NvEncoderConfig::new(0, encoder_cfg)
    ///     .name("demo")
    ///     .operation_timeout(Duration::from_secs(5));
    /// let _encoder = NvEncoder::new(cfg).unwrap();
    /// ```
    pub fn new(config: NvEncoderConfig) -> Result<Self, EncoderError> {
        gst::init().map_err(|e| EncoderError::GstInit(e.to_string()))?;

        let codec = config.encoder.codec();
        validate_config(&config)?;

        let name = if config.name.is_empty() {
            format!("nvencoder-{}-{}", config.gpu_id, codec.name())
        } else {
            config.name.clone()
        };

        info!(
            "NvEncoder initializing (name={}, codec={:?}, gpu={}, size={}x{})",
            name,
            codec,
            config.gpu_id,
            config.encoder.width(),
            config.encoder.height()
        );

        // ── Buffer pools ─────────────────────────────────────────────
        //
        // Pool size: dGPU NVENC may continue DMA-reading from a buffer
        // after GStreamer releases its reference; a single-buffer pool
        // forces serialization so memory is never overwritten while HW
        // reads it.  Jetson V4L2 encoders hold several input buffers in
        // flight, so a pool of 1 would deadlock `acquire()` — use 4.
        //
        // Detected at runtime via `nvidia_gpu_utils::is_jetson_kernel`
        // so non-Jetson ARM hosts (e.g. Grace Hopper with H100/B300) do
        // not inherit Jetson-only tunings. Orin Nano is Jetson and also
        // uses the Jetson path for PNG/RAW/JPEG (pool=1); H264/HEVC/AV1
        // are unreachable there because `has_nvenc()` is false.
        let pool_size: u32 = if nvidia_gpu_utils::is_jetson_kernel()
            && matches!(codec, Codec::H264 | Codec::Hevc | Codec::Av1)
        {
            4
        } else {
            1
        };

        let user_format = config.encoder.format();
        let (fps_num, fps_den) = config.encoder.fps();
        let width = config.encoder.width();
        let height = config.encoder.height();

        let generator = BufferGenerator::builder(user_format, width, height)
            .fps(fps_num, fps_den)
            .gpu_id(config.gpu_id)
            .mem_type(config.mem_type)
            .min_buffers(pool_size)
            .max_buffers(pool_size)
            .build()?;

        // Decide whether we need an out-of-pipeline format conversion
        // (RGBA → NV12 / I420) via NvBufSurfTransform.  Only applies to
        // HW encoders; PNG and raw pseudoencoders use nvvideoconvert
        // in-pipeline because they need CPU-mappable memory downstream.
        let is_png = codec == Codec::Png;
        let is_raw = matches!(codec, Codec::RawRgba | Codec::RawRgb | Codec::RawNv12);
        let native_format = match codec {
            Codec::Jpeg => VideoFormat::I420,
            Codec::Png => VideoFormat::RGBA,
            Codec::RawRgba => VideoFormat::RGBA,
            Codec::RawRgb => VideoFormat::RGBA,
            Codec::RawNv12 => VideoFormat::NV12,
            _ => VideoFormat::NV12,
        };
        let needs_convert = !is_png
            && !is_raw
            && matches!(codec, Codec::H264 | Codec::Hevc | Codec::Av1 | Codec::Jpeg)
            && user_format != native_format;

        let convert_ctx = if needs_convert {
            let native_generator = BufferGenerator::builder(native_format, width, height)
                .fps(fps_num, fps_den)
                .gpu_id(config.gpu_id)
                .mem_type(config.mem_type)
                .min_buffers(pool_size)
                .max_buffers(pool_size)
                .build()?;
            let cuda_stream = CudaStream::new_non_blocking().map_err(|e| {
                EncoderError::PipelineError(format!(
                    "Failed to create non-blocking CUDA stream: {e}"
                ))
            })?;
            debug!(
                "NvEncoder convert_ctx created: {} -> {}, stream {:?}",
                user_format, native_format, cuda_stream
            );
            Some(ConvertContext {
                native_generator,
                cuda_stream,
            })
        } else {
            None
        };

        // ── Element chain ────────────────────────────────────────────

        let (elements, appsrc_caps) =
            build_pipeline_elements(&config.encoder, convert_ctx.is_some(), &generator)?;

        let pipeline_config = PipelineConfig {
            name: name.clone(),
            appsrc_caps,
            elements,
            input_channel_capacity: config.input_channel_capacity,
            output_channel_capacity: config.output_channel_capacity,
            operation_timeout: Some(config.operation_timeout),
            drain_poll_interval: config.drain_poll_interval,
            appsrc_probe: None,
            // Encoders enforce strict PTS monotonicity on input (no
            // B-frames).  StrictPts rejects duplicate or out-of-order PTS
            // before the buffer reaches the encoder element.
            pts_policy: Some(PtsPolicy::StrictPts),
            leak_on_finalize: false,
        };

        let (input_tx, output_rx, gst_pipeline) = GstPipeline::start(pipeline_config)?;

        let backend = EncoderBackendState::Pipeline {
            input_tx,
            output_rx,
            pipeline: Mutex::new(gst_pipeline),
            convert_ctx,
        };

        info!(
            "NvEncoder started (name={}, codec={:?}, gpu={})",
            name, codec, config.gpu_id
        );

        Ok(Self {
            name,
            backend,
            generator: Arc::new(Mutex::new(generator)),
            codec,
            user_format,
            draining: AtomicBool::new(false),
            is_shut_down: AtomicBool::new(false),
            finalized: AtomicBool::new(false),
            last_input_pts_ns: Mutex::new(None),
            last_output_pts_ns: Mutex::new(None),
            pts_map: Mutex::new(HashMap::new()),
            intra_submit_fifo: Mutex::new(VecDeque::new()),
            pending_codec_header: Mutex::new(None),
            submit_counter: AtomicU64::new(0),
        })
    }

    /// Shared NVMM buffer pool (user-facing format).
    pub fn generator(&self) -> Arc<Mutex<BufferGenerator>> {
        self.generator.clone()
    }

    /// The codec this encoder was configured for.
    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// `true` once the pipeline has entered a terminal failed state.
    pub fn is_failed(&self) -> bool {
        match &self.backend {
            EncoderBackendState::Pipeline { pipeline, .. } => pipeline.lock().is_failed(),
        }
    }

    /// Submit a filled NVMM buffer to the encoder.
    ///
    /// The buffer should have been acquired from
    /// [`generator()`](Self::generator).  When the user format differs
    /// from the encoder-native format (e.g. RGBA → NV12), the buffer is
    /// converted out of band via `NvBufSurfTransform` on a non-blocking
    /// CUDA stream before being pushed to `appsrc`.
    ///
    /// PTS values must be strictly monotonically increasing; a violation
    /// yields [`EncoderError::PtsReordered`] and does not enqueue the
    /// frame.
    ///
    /// Blocks when the input channel is full (backpressure).
    pub fn submit_frame(
        &self,
        buffer: gst::Buffer,
        frame_id: u128,
        pts_ns: u64,
        duration_ns: Option<u64>,
    ) -> Result<(), EncoderError> {
        if self.finalized.load(Ordering::Acquire) {
            return Err(EncoderError::AlreadyFinalized);
        }
        if self.draining.load(Ordering::Acquire) {
            return Err(EncoderError::ShuttingDown);
        }
        if self.is_failed() {
            return Err(EncoderError::PipelineFailed);
        }

        // Enforce strict monotonic input PTS locally (framework's
        // StrictPts policy would also catch this, but we want the rich
        // `PtsReordered { frame_id, .. }` error for callers).
        {
            let mut last = self.last_input_pts_ns.lock();
            if let Some(prev) = *last {
                if pts_ns <= prev {
                    return Err(EncoderError::PtsReordered {
                        frame_id,
                        pts_ns,
                        prev_pts_ns: prev,
                    });
                }
            }
            *last = Some(pts_ns);
        }

        self.pts_map.lock().insert(
            pts_ns,
            FrameMetadata {
                frame_id,
                duration_ns,
            },
        );
        if matches!(
            self.codec,
            Codec::Jpeg | Codec::Png | Codec::RawRgba | Codec::RawRgb | Codec::RawNv12
        ) {
            self.intra_submit_fifo.lock().push_back(frame_id);
        }

        let push_buffer = match &self.backend {
            EncoderBackendState::Pipeline { convert_ctx, .. } => {
                self.prepare_push_buffer(buffer, pts_ns, duration_ns, convert_ctx.as_ref())?
            }
        };

        match &self.backend {
            EncoderBackendState::Pipeline { input_tx, .. } => {
                self.submit_counter.fetch_add(1, Ordering::Relaxed);
                input_tx
                    .send(PipelineInput::Buffer(push_buffer))
                    .map_err(|_| EncoderError::ChannelDisconnected)?;
            }
        }
        Ok(())
    }

    /// Inject a custom downstream event into the pipeline, ordered with
    /// buffers already in flight.
    pub fn send_event(&self, event: gst::Event) -> Result<(), EncoderError> {
        if self.is_shut_down.load(Ordering::Acquire) {
            return Err(EncoderError::ShuttingDown);
        }
        match &self.backend {
            EncoderBackendState::Pipeline { input_tx, .. } => input_tx
                .send(PipelineInput::Event(event))
                .map_err(|_| EncoderError::ChannelDisconnected),
        }
    }

    /// Per-source logical EOS marker (custom downstream event).
    ///
    /// Surfaces as [`NvEncoderOutput::SourceEos`] on the output channel,
    /// ordered with encoded frames.  Delivered in-band via the rescue
    /// probe pair installed by [`GstPipeline`].  Does **not** stop the
    /// encoder.
    pub fn send_source_eos(&self, source_id: &str) -> Result<(), EncoderError> {
        self.send_event(build_source_eos_event(source_id))
    }

    /// Block until the next output is available.
    pub fn recv(&self) -> Result<NvEncoderOutput, EncoderError> {
        match &self.backend {
            EncoderBackendState::Pipeline { output_rx, .. } => {
                let out = output_rx
                    .recv()
                    .map_err(|_| EncoderError::ChannelDisconnected)?;
                self.map_output(out)
            }
        }
    }

    /// Block until the next output or timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<NvEncoderOutput>, EncoderError> {
        match &self.backend {
            EncoderBackendState::Pipeline { output_rx, .. } => {
                match output_rx.recv_timeout(timeout) {
                    Ok(out) => self.map_output(out).map(Some),
                    Err(channel::RecvTimeoutError::Timeout) => Ok(None),
                    Err(channel::RecvTimeoutError::Disconnected) => {
                        Err(EncoderError::ChannelDisconnected)
                    }
                }
            }
        }
    }

    /// Non-blocking: return the next output if available.
    pub fn try_recv(&self) -> Result<Option<NvEncoderOutput>, EncoderError> {
        match &self.backend {
            EncoderBackendState::Pipeline { output_rx, .. } => match output_rx.try_recv() {
                Ok(out) => self.map_output(out).map(Some),
                Err(channel::TryRecvError::Empty) => Ok(None),
                Err(channel::TryRecvError::Disconnected) => Err(EncoderError::ChannelDisconnected),
            },
        }
    }

    /// Graceful shutdown: reject new input, send EOS, drain and deliver
    /// in-flight outputs via `on_output`, then tear down the pipeline.
    ///
    /// `idle_timeout` bounds the time to wait *between* consecutive
    /// outputs.  `None` means wait indefinitely.  A GStreamer EOS or
    /// pipeline error always terminates the drain regardless of the idle
    /// timeout.
    pub fn graceful_shutdown<F>(
        &self,
        idle_timeout: Option<Duration>,
        mut on_output: F,
    ) -> Result<(), EncoderError>
    where
        F: FnMut(NvEncoderOutput),
    {
        if self.is_shut_down.swap(true, Ordering::AcqRel) {
            return Err(EncoderError::ShuttingDown);
        }
        self.draining.store(true, Ordering::Release);
        self.finalized.store(true, Ordering::Release);

        match &self.backend {
            EncoderBackendState::Pipeline {
                input_tx,
                output_rx,
                pipeline,
                ..
            } => {
                input_tx
                    .send(PipelineInput::Eos)
                    .map_err(|_| EncoderError::ChannelDisconnected)?;

                loop {
                    let raw = match idle_timeout {
                        Some(d) => match output_rx.recv_timeout(d) {
                            Ok(v) => v,
                            Err(channel::RecvTimeoutError::Timeout) => break,
                            Err(channel::RecvTimeoutError::Disconnected) => break,
                        },
                        None => match output_rx.recv() {
                            Ok(v) => v,
                            Err(_) => break,
                        },
                    };
                    let is_terminal = matches!(raw, PipelineOutput::Eos | PipelineOutput::Error(_));
                    if !matches!(raw, PipelineOutput::Eos) {
                        match self.map_output(raw)? {
                            NvEncoderOutput::Frame(f) => on_output(NvEncoderOutput::Frame(f)),
                            other => on_output(other),
                        }
                    }
                    if is_terminal {
                        break;
                    }
                }

                pipeline.lock().shutdown()?;
                self.log_residuals();
                Ok(())
            }
        }
    }

    /// Abrupt shutdown (used by [`Drop`]).
    pub fn shutdown(&self) -> Result<(), EncoderError> {
        if self.is_shut_down.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        self.draining.store(true, Ordering::Release);
        self.finalized.store(true, Ordering::Release);
        match &self.backend {
            EncoderBackendState::Pipeline { pipeline, .. } => pipeline.lock().shutdown()?,
        }
        self.log_residuals();
        Ok(())
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn prepare_push_buffer(
        &self,
        buffer: gst::Buffer,
        pts_ns: u64,
        duration_ns: Option<u64>,
        convert_ctx: Option<&ConvertContext>,
    ) -> Result<gst::Buffer, EncoderError> {
        if let Some(ctx) = convert_ctx {
            let transform_config = TransformConfig {
                padding: Padding::None,
                dst_padding: None,
                interpolation: deepstream_buffers::Interpolation::Nearest,
                compute_mode: deepstream_buffers::ComputeMode::Default,
                cuda_stream: ctx.cuda_stream.clone(),
            };
            let src_view = SurfaceView::from_gst_buffer(buffer, 0).map_err(|e| {
                EncoderError::BufferError(format!(
                    "Failed to create SurfaceView from source buffer: {e}"
                ))
            })?;
            let mut native_buf = ctx
                .native_generator
                .transform_to_buffer(&src_view, &transform_config, None)
                .map_err(|e| {
                    EncoderError::BufferError(format!(
                        "NvBufSurfTransform (format conversion) failed: {e}"
                    ))
                })?;
            set_timestamps(&mut native_buf, pts_ns, duration_ns)?;
            Ok(native_buf)
        } else {
            let mut buffer = buffer;
            set_timestamps(&mut buffer, pts_ns, duration_ns)?;
            Ok(buffer)
        }
    }

    /// Translate a raw [`PipelineOutput`] into an [`NvEncoderOutput`].
    ///
    /// Implements the codec-header stashing and the PTS / DTS / intra-FIFO
    /// correlation originally in `sample_to_frame`.
    fn map_output(&self, output: PipelineOutput) -> Result<NvEncoderOutput, EncoderError> {
        match output {
            PipelineOutput::Buffer(buffer) => {
                let frame = self.buffer_to_frame(buffer)?;
                match frame {
                    Some(f) => Ok(NvEncoderOutput::Frame(f)),
                    None => {
                        // Buffer was consumed as stashed codec header.
                        // Poll for the next real output so the caller
                        // always observes a frame or terminal marker.
                        self.recv_until_non_header()
                    }
                }
            }
            PipelineOutput::Eos => Ok(NvEncoderOutput::Eos),
            PipelineOutput::Event(event) => {
                if let Some(source_id) = parse_source_eos_event(&event) {
                    Ok(NvEncoderOutput::SourceEos { source_id })
                } else {
                    Ok(NvEncoderOutput::Event(event))
                }
            }
            PipelineOutput::Error(e) => Ok(NvEncoderOutput::Error(EncoderError::FrameworkError(e))),
        }
    }

    /// Continuation after a codec-header-only buffer was stashed. Pulls
    /// the next output; loops until a real frame or terminal marker is
    /// produced.
    fn recv_until_non_header(&self) -> Result<NvEncoderOutput, EncoderError> {
        match &self.backend {
            EncoderBackendState::Pipeline { output_rx, .. } => loop {
                let raw = output_rx
                    .recv()
                    .map_err(|_| EncoderError::ChannelDisconnected)?;
                match raw {
                    PipelineOutput::Buffer(buffer) => {
                        if let Some(f) = self.buffer_to_frame(buffer)? {
                            return Ok(NvEncoderOutput::Frame(f));
                        }
                    }
                    PipelineOutput::Eos => return Ok(NvEncoderOutput::Eos),
                    PipelineOutput::Event(event) => {
                        return if let Some(sid) = parse_source_eos_event(&event) {
                            Ok(NvEncoderOutput::SourceEos { source_id: sid })
                        } else {
                            Ok(NvEncoderOutput::Event(event))
                        }
                    }
                    PipelineOutput::Error(e) => {
                        return Ok(NvEncoderOutput::Error(EncoderError::FrameworkError(e)))
                    }
                }
            },
        }
    }

    /// Convert an appsink buffer into an [`EncodedFrame`], or return `Ok(None)`
    /// when the buffer is a codec-header-only pre-IDR sample that was
    /// stashed for inlining into the next user frame.
    fn buffer_to_frame(&self, buffer: gst::Buffer) -> Result<Option<EncodedFrame>, EncoderError> {
        let buf_pts_ns = buffer.pts().map(|t| t.nseconds()).unwrap_or(0);
        let buf_dts_ns = buffer.dts().map(|t| t.nseconds());
        let duration_ns = buffer.duration().map(|t| t.nseconds());

        let is_intra_only = matches!(
            self.codec,
            Codec::Jpeg | Codec::Png | Codec::RawRgba | Codec::RawRgb | Codec::RawNv12
        );
        let buf_size = buffer.size() as u64;

        // Look up frame_id from our PTS map.  On Jetson the pre-encoder
        // nvvideoconvert retimestamps buffers (changing PTS) but
        // preserves the original PTS in DTS.  Try PTS first, fall back
        // to DTS.
        let (pts_lookup, original_pts) = {
            let mut map = self.pts_map.lock();
            let by_pts = map.remove(&buf_pts_ns);
            if let Some(v) = by_pts {
                (Some(v), buf_pts_ns)
            } else if let Some(dts) = buf_dts_ns {
                let by_dts = map.remove(&dts);
                match by_dts {
                    Some(v) => (Some(v), dts),
                    None => (None, buf_pts_ns),
                }
            } else {
                (None, buf_pts_ns)
            }
        };

        let (frame_id, orig_duration, is_user_frame) = match pts_lookup {
            Some(meta) => {
                if is_intra_only {
                    let mut fifo = self.intra_submit_fifo.lock();
                    match fifo.pop_front() {
                        Some(fid) if fid != meta.frame_id => {
                            warn!(
                                "NvEncoder: intra FIFO head {fid} != pts_map id {} (codec={:?}) — trusting map",
                                meta.frame_id, self.codec
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
                (Some(meta.frame_id), meta.duration_ns, true)
            }
            None if is_intra_only && buf_size > 0 => {
                let id = self.intra_submit_fifo.lock().pop_front();
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

        // ── Codec-header stashing (Jetson parity) ────────────────────
        //
        // A non-user-frame buffer with payload on a non-intra-only codec
        // is a stream-level codec header (e.g. dGPU `nvv4l2av1enc`
        // emits the AV1 `OBU_SEQUENCE_HEADER` as a standalone buffer
        // before the first IDR; Jetson's V4L2 encoder inlines it into
        // the first IDR automatically).  Match the Jetson behavior by
        // stashing the header bytes and prepending them to the next
        // user frame's `data`.
        if !is_user_frame && !is_intra_only && buf_size > 0 {
            let map = buffer.map_readable().map_err(|e| {
                EncoderError::PipelineError(format!("Failed to map codec header buffer: {e:?}"))
            })?;
            let bytes = map.as_slice();
            let had_header_flag = buffer.flags().contains(gst::BufferFlags::HEADER);
            let mut guard = self.pending_codec_header.lock();
            match &mut *guard {
                Some(existing) => existing.extend_from_slice(bytes),
                slot @ None => *slot = Some(bytes.to_vec()),
            }
            debug!(
                "NvEncoder: stashed codec header ({} bytes, header_flag={}, total pending={}, codec={:?})",
                bytes.len(),
                had_header_flag,
                guard.as_ref().map(|v: &Vec<u8>| v.len()).unwrap_or(0),
                self.codec
            );
            return Ok(None);
        }

        // ── Ordering validation (user frames only) ──────────────────
        if is_user_frame {
            let mut last = self.last_output_pts_ns.lock();
            if let Some(prev) = *last {
                if pts_ns < prev {
                    return Err(EncoderError::OutputPtsReordered {
                        frame_id,
                        pts_ns,
                        prev_pts_ns: prev,
                    });
                }
            }
            *last = Some(pts_ns);

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

        // For intra-only codecs, normalize DTS = PTS.
        let dts_ns = if is_intra_only {
            Some(pts_ns)
        } else {
            buf_dts_ns
        };

        let final_duration = duration_ns.or(orig_duration);

        let keyframe = match self.codec {
            Codec::Jpeg | Codec::Png | Codec::RawRgba | Codec::RawRgb | Codec::RawNv12 => true,
            _ => !buffer.flags().contains(gst::BufferFlags::DELTA_UNIT),
        };

        let mut data = if self.codec == Codec::RawNv12 {
            let map = buffer.map_readable().map_err(|e| {
                EncoderError::PipelineError(format!("Failed to map NV12 buffer: {e:?}"))
            })?;
            map.as_slice().to_vec()
        } else if matches!(self.codec, Codec::RawRgba | Codec::RawRgb) {
            extract_raw_pixels(&buffer, self.codec)?
        } else {
            let map = buffer
                .map_readable()
                .map_err(|e| EncoderError::PipelineError(format!("Failed to map buffer: {e:?}")))?;
            map.as_slice().to_vec()
        };

        // Prepend stashed codec header on first user frame.
        if is_user_frame {
            let mut guard = self.pending_codec_header.lock();
            if let Some(header) = Option::take(&mut *guard) {
                let header_len = header.len();
                let mut combined = header;
                combined.reserve(data.len());
                combined.extend_from_slice(&data);
                data = combined;
                debug!(
                    "NvEncoder: inlined {} bytes of codec header into frame (frame_id={:?}, codec={:?})",
                    header_len, frame_id, self.codec
                );
            }
        }

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

    fn log_residuals(&self) {
        let fifo = self.intra_submit_fifo.lock();
        if !fifo.is_empty() {
            warn!(
                "NvEncoder '{}' shutdown: {} frame_id(s) left in intra_submit_fifo (codec={:?}) — outputs missing",
                self.name, fifo.len(), self.codec
            );
        }
        let header = self.pending_codec_header.lock();
        if let Some(h) = (*header).as_ref() {
            debug!(
                "NvEncoder '{}' shutdown: dropped orphan codec header ({} bytes, codec={:?})",
                self.name,
                h.len(),
                self.codec
            );
        }
        let pts = self.pts_map.lock();
        if !pts.is_empty() {
            warn!(
                "NvEncoder '{}' shutdown: {} frame(s) still pending in pts_map",
                self.name,
                pts.len()
            );
        }
        // Silence unused-warning on user_format when no compile-time
        // raw-NV12 stride logic needs it.  Kept for future scaling /
        // padding heuristics.
        let _ = self.user_format;
    }
}

impl Drop for NvEncoder {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

// ─── Helpers ────────────────────────────────────────────────────────

fn set_timestamps(
    buffer: &mut gst::Buffer,
    pts_ns: u64,
    duration_ns: Option<u64>,
) -> Result<(), EncoderError> {
    let buf = buffer
        .get_mut()
        .ok_or_else(|| EncoderError::BufferError("Buffer is not writable".into()))?;
    buf.set_pts(gst::ClockTime::from_nseconds(pts_ns));
    buf.set_dts(gst::ClockTime::from_nseconds(pts_ns));
    if let Some(dur) = duration_ns {
        buf.set_duration(gst::ClockTime::from_nseconds(dur));
    }
    Ok(())
}

/// Extract tightly-packed pixel data from a raw video buffer, stripping
/// any stride padding using `gst_video::VideoMeta` when present.  Falls
/// back to assuming unpadded rows of `width * bpp` when the meta is
/// absent.
fn extract_raw_pixels(buffer: &gst::Buffer, codec: Codec) -> Result<Vec<u8>, EncoderError> {
    // We don't have access to caps on the drain-side buffer, so rely on
    // GstVideoMeta (which elements attach when stride != default).  The
    // raw pipeline is appsrc → nvvideoconvert → capsfilter → appsink;
    // dimensions and format are fixed by the capsfilter.
    let bpp = match codec {
        Codec::RawRgba => 4usize,
        Codec::RawRgb => 3usize,
        _ => unreachable!("extract_raw_pixels only for raw RGB/RGBA"),
    };

    let map = buffer
        .map_readable()
        .map_err(|e| EncoderError::PipelineError(format!("Failed to map buffer: {e:?}")))?;
    let src = map.as_slice();

    if let Some(meta) = buffer.meta::<gst_video::VideoMeta>() {
        let width = meta.width() as usize;
        let height = meta.height() as usize;
        let stride = meta.stride().first().copied().unwrap_or(0) as usize;
        let row_bytes = width * bpp;

        if stride == 0 || stride == row_bytes {
            return Ok(src[..row_bytes * height.min(src.len() / row_bytes.max(1))].to_vec());
        }
        let mut out = Vec::with_capacity(row_bytes * height);
        for row in 0..height {
            let start = row * stride;
            let end = start + row_bytes;
            if end > src.len() {
                break;
            }
            out.extend_from_slice(&src[start..end]);
        }
        Ok(out)
    } else {
        Ok(src.to_vec())
    }
}

// ─── Config validation ─────────────────────────────────────────────

fn validate_config(config: &NvEncoderConfig) -> Result<(), EncoderError> {
    let codec = config.encoder.codec();
    let user_format = config.encoder.format();

    if matches!(codec, Codec::Vp8 | Codec::Vp9) {
        return Err(EncoderError::UnsupportedCodec(codec.name().to_string()));
    }

    if codec == Codec::Png && user_format != VideoFormat::RGBA {
        return Err(EncoderError::InvalidProperty {
            name: "format".into(),
            reason: "PNG encoder requires VideoFormat::RGBA".into(),
        });
    }

    if matches!(codec, Codec::H264 | Codec::Hevc | Codec::Av1) {
        let has_nvenc = nvidia_gpu_utils::has_nvenc(config.gpu_id).unwrap_or(false);
        if !has_nvenc {
            return Err(EncoderError::NvencNotAvailable {
                codec: codec.name().to_string(),
                gpu_id: config.gpu_id,
            });
        }
    }

    Ok(())
}

// ─── Pipeline element chain ────────────────────────────────────────

/// Build the element chain between `appsrc` and `appsink` and compute
/// the appsrc caps.
///
/// `has_convert_ctx = true` means the caller performs format conversion
/// out of band (via [`ConvertContext`]); `appsrc` in that case receives
/// buffers already in the encoder-native format.
fn build_pipeline_elements(
    config: &EncoderConfig,
    has_convert_ctx: bool,
    user_generator: &BufferGenerator,
) -> Result<(Vec<gst::Element>, gst::Caps), EncoderError> {
    let codec = config.codec();

    // ── Appsrc caps ─────────────────────────────────────────────────
    //
    // When we convert out of band, appsrc emits encoder-native NVMM
    // (NV12 / I420); otherwise it emits the user-facing format from the
    // generator directly.  We build a second generator just for the
    // NV12 caps path to reuse `nvmm_caps_gst` without a full pool.
    let appsrc_caps = if has_convert_ctx {
        let native_format = match codec {
            Codec::Jpeg => VideoFormat::I420,
            _ => VideoFormat::NV12,
        };
        let temp = BufferGenerator::builder(native_format, config.width(), config.height())
            .fps(config.fps().0, config.fps().1)
            .min_buffers(1)
            .max_buffers(1)
            .build()?;
        temp.nvmm_caps_gst()
    } else {
        user_generator.nvmm_caps_gst()
    };

    // ── Element chain ───────────────────────────────────────────────
    let elements = match codec {
        Codec::H264 | Codec::Hevc | Codec::Av1 | Codec::Jpeg => {
            let (enc_factory, parse_factory) = match codec {
                Codec::H264 => ("nvv4l2h264enc", "h264parse"),
                Codec::Hevc => ("nvv4l2h265enc", "h265parse"),
                Codec::Av1 => ("nvv4l2av1enc", "av1parse"),
                Codec::Jpeg => ("nvjpegenc", "jpegparse"),
                _ => unreachable!(),
            };

            let enc = make_elem(enc_factory, "enc")?;
            // Bridge Savant ID meta around the encoder (same as the
            // GstPipeline does around the whole chain — but encoders
            // often drop arbitrary metas, so an inner bridge is safe
            // and matches the old encoder.rs behavior).
            bridge_savant_id_meta(&enc).map_err(|e| {
                EncoderError::PipelineError(format!("bridge meta around encoder failed: {e}"))
            })?;

            apply_encoder_props(&enc, config.to_gst_pairs())?;
            force_disable_b_frames(&enc);

            if matches!(codec, Codec::H264 | Codec::Hevc) {
                enc.set_property_from_str("insert-sps-pps", "1");
            }

            let parse = make_elem(parse_factory, "parse")?;
            if matches!(codec, Codec::H264 | Codec::Hevc) {
                parse.set_property("config-interval", -1i32);
            }

            // On Jetson, nvjpegenc requires surfaces pinned/registered
            // by nvvideoconvert; NvDS pool surfaces from appsrc lack
            // this registration and cause hangs ("Surface not
            // registered").
            //
            // Runtime-detected via `is_jetson_kernel` so non-Jetson ARM
            // hosts (Grace Hopper) use the dGPU path.
            let pre_enc: Option<gst::Element> =
                if nvidia_gpu_utils::is_jetson_kernel() && codec == Codec::Jpeg {
                    let conv = make_elem("nvvideoconvert", "pre_enc_conv")?;
                    conv.set_property("disable-passthrough", true);
                    Some(conv)
                } else {
                    None
                };

            let mut elems = Vec::new();
            if let Some(c) = pre_enc {
                elems.push(c);
            }
            elems.push(enc);
            elems.push(parse);
            elems
        }
        Codec::Png => {
            // appsrc (NVMM) → nvvideoconvert (GPU→CPU RGBA) → pngenc
            let conv = make_elem("nvvideoconvert", "nvconv")?;
            let enc = make_elem("pngenc", "enc")?;
            apply_encoder_props(&enc, config.to_gst_pairs())?;
            vec![conv, enc]
        }
        Codec::RawRgba | Codec::RawRgb | Codec::RawNv12 => {
            let raw_format = match codec {
                Codec::RawRgba => "RGBA",
                Codec::RawRgb => "RGB",
                Codec::RawNv12 => "NV12",
                _ => unreachable!(),
            };
            let conv = make_elem("nvvideoconvert", "nvconv")?;
            // `compute-hw=1` routes nvvideoconvert to the VIC on Jetson
            // (higher throughput than GPU for simple format/layout
            // conversions). No effect on dGPU nvvideoconvert.
            // Runtime-detected — see note on `pool_size` above.
            if nvidia_gpu_utils::is_jetson_kernel() {
                conv.set_property_from_str("compute-hw", "1");
            }

            let caps = gst::Caps::builder("video/x-raw")
                .field("format", raw_format)
                .build();
            let capsfilter = gst::ElementFactory::make("capsfilter")
                .name("rawcaps")
                .property("caps", &caps)
                .build()
                .map_err(|_| EncoderError::ElementCreationFailed("capsfilter".into()))?;
            vec![conv, capsfilter]
        }
        Codec::Vp8 | Codec::Vp9 => unreachable!("VP8/VP9 are decode-only"),
    };

    Ok((elements, appsrc_caps))
}

fn make_elem(factory: &str, name: &str) -> Result<gst::Element, EncoderError> {
    gst::ElementFactory::make(factory)
        .name(name)
        .build()
        .map_err(|_| EncoderError::ElementCreationFailed(factory.to_string()))
}

fn apply_encoder_props(
    element: &gst::Element,
    pairs: Vec<(&'static str, String)>,
) -> Result<(), EncoderError> {
    for (key, value) in pairs {
        if element.find_property(key).is_none() {
            return Err(EncoderError::InvalidProperty {
                name: key.to_string(),
                reason: "property not found on encoder element".to_string(),
            });
        }
        let elem = element.clone();
        let k = key.to_string();
        let v = value.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            elem.set_property_from_str(&k, &v);
        }));
        if result.is_err() {
            return Err(EncoderError::InvalidProperty {
                name: key.to_string(),
                reason: format!(
                    "failed to set value '{}' (set_property_from_str panicked)",
                    value
                ),
            });
        }
    }
    Ok(())
}

fn force_disable_b_frames(enc: &gst::Element) {
    for prop_name in B_FRAME_PROPERTY_NAMES {
        if enc.find_property(prop_name).is_some() {
            let enc_clone = enc.clone();
            let name = prop_name.to_string();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                enc_clone.set_property_from_str(&name, "0");
            }));
            match result {
                Ok(()) => debug!("Disabled B-frames: {prop_name} = 0"),
                Err(_) => debug!("Could not set {prop_name} to 0 on encoder"),
            }
        }
    }
}

// ─── Low-level EOS trait (not in prelude) ──────────────────────────

/// Low-level EOS for the "submit → send_eos → recv loop" pattern.
///
/// Intentionally not part of [`crate::prelude`] — regular consumers
/// should use [`NvEncoder::graceful_shutdown`] which sends EOS and
/// drains in a single call.
pub trait NvEncoderExt {
    /// Signal end-of-stream.
    ///
    /// Sends actual GStreamer EOS through the pipeline to flush any
    /// remaining encoded frames.  The caller should continue calling
    /// [`NvEncoder::recv`] until [`NvEncoderOutput::Eos`] is received.
    ///
    /// Idempotent: calling after finalization is a no-op.
    fn send_eos(&self) -> Result<(), EncoderError>;
}

impl NvEncoderExt for NvEncoder {
    fn send_eos(&self) -> Result<(), EncoderError> {
        if self.finalized.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        match &self.backend {
            EncoderBackendState::Pipeline { input_tx, .. } => input_tx
                .send(PipelineInput::Eos)
                .map_err(|_| EncoderError::ChannelDisconnected)?,
        }
        Ok(())
    }
}
