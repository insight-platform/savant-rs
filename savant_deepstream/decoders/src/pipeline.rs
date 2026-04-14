//! Channel-based decoder pipeline built on the `savant_gstreamer::pipeline` framework.
//!
//! Three backend strategies:
//! - **Pipeline**: GStreamer pipeline (appsrc → parser → GPU decoder → appsink)
//!   for H264/HEVC/VP8/VP9/AV1/GPU-JPEG via [`GstPipeline`].
//! - **RawUpload**: synchronous CPU→GPU pixel upload (RGBA/RGB).
//! - **ImageDecode**: CPU image decode (PNG, CPU-JPEG) via the `image` crate
//!   followed by a GPU upload.
//!
//! All three strategies expose the same pull-based API: [`NvDecoder::submit_packet`],
//! [`NvDecoder::recv`] / [`NvDecoder::recv_timeout`] / [`NvDecoder::try_recv`].

use crate::config::{DecoderConfig, JpegBackend, NvDecoderConfig};
use crate::{Codec, DecodedFrame, DecoderError, VideoFormat};
use crossbeam::channel::{self, Receiver, Sender};
use deepstream_buffers::{
    BufferGenerator, NvBufSurfaceMemType, SavantIdMeta, SavantIdMetaKind, SharedBuffer,
    SurfaceView, TransformConfig,
};
use gstreamer as gst;
use gstreamer::prelude::*;
use log::{info, warn};
use parking_lot::Mutex;
use savant_gstreamer::pipeline::{
    build_source_eos_event, parse_source_eos_event, GstPipeline, PipelineConfig, PipelineInput,
    PipelineOutput, PtsPolicy,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ── Internal types ──────────────────────────────────────────────────

struct FrameMetadata {
    frame_id: u128,
    original_pts_ns: u64,
    dts_ns: Option<u64>,
    duration_ns: Option<u64>,
}

enum DecoderBackendState {
    /// GPU decoder pipeline: appsrc → parser → nvv4l2decoder/nvjpegdec → appsink.
    Pipeline {
        input_tx: Sender<PipelineInput>,
        output_rx: Receiver<PipelineOutput>,
        pipeline: Mutex<GstPipeline>,
    },
    /// Synchronous CPU→GPU raw pixel upload (RGBA / RGB).
    RawUpload {
        output_tx: Sender<NvDecoderOutput>,
        output_rx: Receiver<NvDecoderOutput>,
        width: u32,
        height: u32,
        expected_bpp: u32,
    },
    /// CPU-only image decode (PNG, CPU-JPEG) via the `image` crate.
    ImageDecode {
        output_tx: Sender<NvDecoderOutput>,
        output_rx: Receiver<NvDecoderOutput>,
    },
}

// ── Public types ────────────────────────────────────────────────────

/// Output from the channel-based decoder API.
#[derive(Debug)]
pub enum NvDecoderOutput {
    /// A decoded RGBA frame.
    Frame(DecodedFrame),
    /// A downstream GStreamer event captured at the pipeline output.
    Event(gst::Event),
    /// Per-source logical EOS (custom downstream event passthrough).
    /// Does **not** stop the decoder — use [`NvDecoder::graceful_shutdown`]
    /// for that.
    SourceEos { source_id: String },
    /// End-of-stream: all buffered frames have been delivered.
    Eos,
    /// Pipeline or framework runtime error.
    Error(DecoderError),
}

/// GPU-accelerated video / image / raw-frame decoder.
///
/// Uses the [`savant_gstreamer::pipeline`] framework for GPU decoder backends
/// and synchronous processing for raw-upload and CPU image-decode paths.
/// All output is delivered via the pull-based [`recv`](Self::recv) /
/// [`recv_timeout`](Self::recv_timeout) / [`try_recv`](Self::try_recv) API.
///
/// `submit_packet` takes `&self` so the decoder can be shared behind `Arc`.
pub struct NvDecoder {
    name: String,
    backend: DecoderBackendState,
    pool: Arc<Mutex<BufferGenerator>>,
    transform_config: TransformConfig,
    gpu_id: u32,
    codec: Codec,
    draining: AtomicBool,
    is_shut_down: AtomicBool,
    finalized: AtomicBool,
    /// Maps monotonic PTS → original frame metadata for PTS correlation on output.
    pts_metadata: Mutex<HashMap<u64, FrameMetadata>>,
    /// FIFO fallback for intra-only codecs whose decoders may not preserve PTS.
    intra_submit_fifo: Mutex<VecDeque<u128>>,
    /// Auxiliary GPU pool for CPU→GPU upload when decoded dimensions ≠ pool dimensions.
    aux_pool: Mutex<Option<BufferGenerator>>,
}

impl NvDecoder {
    /// Create and start a new decoder.
    ///
    /// `pool` is the RGBA output buffer pool. Every decoded frame is transformed
    /// into this pool before delivery. `transform_config` controls GPU
    /// colour-conversion / scaling.
    pub fn new(
        config: NvDecoderConfig,
        pool: BufferGenerator,
        transform_config: TransformConfig,
    ) -> Result<Self, DecoderError> {
        gst::init().map_err(|e| DecoderError::GstInit(e.to_string()))?;

        let codec = config.decoder.codec();
        validate_config(&config.decoder, config.gpu_id)?;

        let name = if config.name.is_empty() {
            format!("nvdecoder-{}-{}", config.gpu_id, codec.name())
        } else {
            config.name.clone()
        };

        info!(
            "NvDecoder initializing (name={}, codec={:?}, gpu={})",
            name, codec, config.gpu_id
        );

        let pool = Arc::new(Mutex::new(pool));

        let backend = if is_raw_format(codec) {
            let (width, height, expected_bpp) = match &config.decoder {
                DecoderConfig::RawRgba(c) => (c.width, c.height, 4u32),
                DecoderConfig::RawRgb(c) => (c.width, c.height, 3u32),
                _ => unreachable!(),
            };
            let (tx, rx) = channel::bounded(config.output_channel_capacity);
            DecoderBackendState::RawUpload {
                output_tx: tx,
                output_rx: rx,
                width,
                height,
                expected_bpp,
            }
        } else if is_image_decode(&config.decoder) {
            let (tx, rx) = channel::bounded(config.output_channel_capacity);
            DecoderBackendState::ImageDecode {
                output_tx: tx,
                output_rx: rx,
            }
        } else {
            let (elements, appsrc_caps) = build_pipeline_elements(&config.decoder)?;
            let pipeline_config = PipelineConfig {
                name: name.clone(),
                appsrc_caps,
                elements,
                input_channel_capacity: config.input_channel_capacity,
                output_channel_capacity: config.output_channel_capacity,
                operation_timeout: Some(config.operation_timeout),
                drain_poll_interval: config.drain_poll_interval,
                appsrc_probe: None,
                pts_policy: Some(PtsPolicy::StrictDecodeOrder),
                leak_on_finalize: false,
            };
            let (input_tx, output_rx, gst_pipeline) = GstPipeline::start(pipeline_config)?;
            DecoderBackendState::Pipeline {
                input_tx,
                output_rx,
                pipeline: Mutex::new(gst_pipeline),
            }
        };

        info!("NvDecoder initialized (name={})", name);

        Ok(Self {
            name,
            backend,
            pool,
            transform_config,
            gpu_id: config.gpu_id,
            codec,
            draining: AtomicBool::new(false),
            is_shut_down: AtomicBool::new(false),
            finalized: AtomicBool::new(false),
            pts_metadata: Mutex::new(HashMap::new()),
            intra_submit_fifo: Mutex::new(VecDeque::new()),
            aux_pool: Mutex::new(None),
        })
    }

    /// The codec this decoder was configured for.
    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// Check if the pipeline has entered a terminal failed state.
    pub fn is_failed(&self) -> bool {
        match &self.backend {
            DecoderBackendState::Pipeline { pipeline, .. } => pipeline.lock().is_failed(),
            _ => false,
        }
    }

    /// Submit a compressed packet (or raw frame) for decoding.
    ///
    /// Timestamp ordering is enforced by the [`GstPipeline`] feeder thread
    /// via [`PtsPolicy::StrictDecodeOrder`]: DTS (when set) or PTS must be
    /// strictly ascending. A violation produces
    /// [`NvDecoderOutput::Error`] on the output channel.
    ///
    /// Blocks if the input channel is full (backpressure).
    pub fn submit_packet(
        &self,
        data: &[u8],
        frame_id: u128,
        pts_ns: u64,
        dts_ns: Option<u64>,
        duration_ns: Option<u64>,
    ) -> Result<(), DecoderError> {
        if self.finalized.load(Ordering::Acquire) {
            return Err(DecoderError::AlreadyFinalized);
        }
        if self.draining.load(Ordering::Acquire) {
            return Err(DecoderError::ShuttingDown);
        }
        if self.is_failed() {
            return Err(DecoderError::PipelineFailed);
        }

        match &self.backend {
            DecoderBackendState::Pipeline { input_tx, .. } => {
                {
                    let mut meta = self.pts_metadata.lock();
                    meta.insert(
                        pts_ns,
                        FrameMetadata {
                            frame_id,
                            original_pts_ns: pts_ns,
                            dts_ns,
                            duration_ns,
                        },
                    );
                }
                if is_intra_only(self.codec) {
                    self.intra_submit_fifo.lock().push_back(frame_id);
                }

                let mut buffer = gst::Buffer::from_mut_slice(data.to_vec());
                {
                    let buf = buffer.get_mut().ok_or_else(|| {
                        DecoderError::BufferError("Buffer is not writable".to_string())
                    })?;
                    buf.set_pts(gst::ClockTime::from_nseconds(pts_ns));
                    if let Some(dts) = dts_ns {
                        buf.set_dts(gst::ClockTime::from_nseconds(dts));
                    }
                    if let Some(dur) = duration_ns {
                        buf.set_duration(gst::ClockTime::from_nseconds(dur));
                    }
                    SavantIdMeta::replace(buf, vec![SavantIdMetaKind::Frame(frame_id)]);
                }

                input_tx
                    .send(PipelineInput::Buffer(buffer))
                    .map_err(|_| DecoderError::ChannelDisconnected)?;
            }
            DecoderBackendState::RawUpload {
                output_tx,
                width,
                height,
                expected_bpp,
                ..
            } => {
                let frame = raw_upload_frame(
                    *width,
                    *height,
                    *expected_bpp,
                    &self.pool,
                    data,
                    frame_id,
                    pts_ns,
                    dts_ns,
                    duration_ns,
                    self.codec,
                )?;
                output_tx
                    .send(NvDecoderOutput::Frame(frame))
                    .map_err(|_| DecoderError::ChannelDisconnected)?;
            }
            DecoderBackendState::ImageDecode { output_tx, .. } => {
                let frame = image_decode_and_upload(
                    &self.pool,
                    &self.transform_config,
                    data,
                    frame_id,
                    pts_ns,
                    dts_ns,
                    duration_ns,
                    self.codec,
                    self.gpu_id,
                    &self.aux_pool,
                )?;
                output_tx
                    .send(NvDecoderOutput::Frame(frame))
                    .map_err(|_| DecoderError::ChannelDisconnected)?;
            }
        }

        Ok(())
    }

    /// Inject a custom downstream GStreamer event into the pipeline.
    ///
    /// The event is ordered with buffers already in the pipeline, so it
    /// appears in the output stream at the correct position.
    /// Only supported for GPU pipeline backends.
    pub fn send_event(&self, event: gst::Event) -> Result<(), DecoderError> {
        if self.is_shut_down.load(Ordering::Acquire) {
            return Err(DecoderError::ShuttingDown);
        }
        match &self.backend {
            DecoderBackendState::Pipeline { input_tx, .. } => {
                input_tx
                    .send(PipelineInput::Event(event))
                    .map_err(|_| DecoderError::ChannelDisconnected)?;
                Ok(())
            }
            _ => Err(DecoderError::PipelineError(
                "events not supported on non-pipeline backends".into(),
            )),
        }
    }

    /// Per-source logical EOS marker (custom downstream event).
    ///
    /// Surfaces as [`NvDecoderOutput::SourceEos`] on the output channel,
    /// ordered with decoded frames. Does **not** stop the decoder.
    pub fn send_source_eos(&self, source_id: &str) -> Result<(), DecoderError> {
        self.send_event(build_source_eos_event(source_id))
    }

    /// Block until the next output is available.
    ///
    /// `Err` is reserved for channel disconnect only.
    pub fn recv(&self) -> Result<NvDecoderOutput, DecoderError> {
        match &self.backend {
            DecoderBackendState::Pipeline { output_rx, .. } => {
                let output = output_rx
                    .recv()
                    .map_err(|_| DecoderError::ChannelDisconnected)?;
                self.convert_output(output)
            }
            DecoderBackendState::RawUpload { output_rx, .. }
            | DecoderBackendState::ImageDecode { output_rx, .. } => output_rx
                .recv()
                .map_err(|_| DecoderError::ChannelDisconnected),
        }
    }

    /// Block until the next output or timeout. Returns `Ok(None)` on timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<NvDecoderOutput>, DecoderError> {
        match &self.backend {
            DecoderBackendState::Pipeline { output_rx, .. } => {
                match output_rx.recv_timeout(timeout) {
                    Ok(output) => self.convert_output(output).map(Some),
                    Err(channel::RecvTimeoutError::Timeout) => Ok(None),
                    Err(channel::RecvTimeoutError::Disconnected) => {
                        Err(DecoderError::ChannelDisconnected)
                    }
                }
            }
            DecoderBackendState::RawUpload { output_rx, .. }
            | DecoderBackendState::ImageDecode { output_rx, .. } => {
                match output_rx.recv_timeout(timeout) {
                    Ok(v) => Ok(Some(v)),
                    Err(channel::RecvTimeoutError::Timeout) => Ok(None),
                    Err(channel::RecvTimeoutError::Disconnected) => {
                        Err(DecoderError::ChannelDisconnected)
                    }
                }
            }
        }
    }

    /// Non-blocking: return the next output if available, or `Ok(None)`.
    pub fn try_recv(&self) -> Result<Option<NvDecoderOutput>, DecoderError> {
        match &self.backend {
            DecoderBackendState::Pipeline { output_rx, .. } => match output_rx.try_recv() {
                Ok(output) => self.convert_output(output).map(Some),
                Err(channel::TryRecvError::Empty) => Ok(None),
                Err(channel::TryRecvError::Disconnected) => Err(DecoderError::ChannelDisconnected),
            },
            DecoderBackendState::RawUpload { output_rx, .. }
            | DecoderBackendState::ImageDecode { output_rx, .. } => match output_rx.try_recv() {
                Ok(v) => Ok(Some(v)),
                Err(channel::TryRecvError::Empty) => Ok(None),
                Err(channel::TryRecvError::Disconnected) => Err(DecoderError::ChannelDisconnected),
            },
        }
    }

    /// Graceful shutdown: reject new input, send EOS, drain and convert
    /// in-flight outputs one at a time via `on_output`, then tear down the
    /// pipeline.
    ///
    /// Each drained NVMM buffer is converted to RGBA using the shared output
    /// pool (same pool used by [`recv`](Self::recv) / [`try_recv`](Self::try_recv)),
    /// so at most one extra RGBA buffer is live at a time.
    ///
    /// `idle_timeout` is the maximum time to wait **between** consecutive
    /// outputs without receiving anything new.  `None` means wait
    /// indefinitely.  A GStreamer EOS or pipeline error always terminates the
    /// drain immediately regardless of the idle timeout.
    pub fn graceful_shutdown<F>(
        &self,
        idle_timeout: Option<Duration>,
        mut on_output: F,
    ) -> Result<(), DecoderError>
    where
        F: FnMut(NvDecoderOutput),
    {
        if self.is_shut_down.swap(true, Ordering::AcqRel) {
            return Err(DecoderError::ShuttingDown);
        }
        self.draining.store(true, Ordering::Release);
        match &self.backend {
            DecoderBackendState::Pipeline {
                input_tx,
                output_rx,
                pipeline,
            } => {
                input_tx
                    .send(PipelineInput::Eos)
                    .map_err(|_| DecoderError::ChannelDisconnected)?;

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
                        on_output(self.convert_output(raw)?);
                    }
                    if is_terminal {
                        break;
                    }
                }

                pipeline.lock().shutdown()?;
                Ok(())
            }
            DecoderBackendState::RawUpload { .. } | DecoderBackendState::ImageDecode { .. } => {
                Ok(())
            }
        }
    }

    /// Abrupt shutdown: stops threads and pipeline (used by [`Drop`]).
    pub fn shutdown(&self) -> Result<(), DecoderError> {
        if self.is_shut_down.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        self.draining.store(true, Ordering::Release);
        if let DecoderBackendState::Pipeline { pipeline, .. } = &self.backend {
            pipeline.lock().shutdown()?;
        }
        Ok(())
    }

    /// Convert a raw [`PipelineOutput`] into a domain-specific [`NvDecoderOutput`],
    /// using the decoder's shared pool for NVMM→RGBA conversion.
    fn convert_output(&self, output: PipelineOutput) -> Result<NvDecoderOutput, DecoderError> {
        let pool = self.pool.lock();
        self.convert_output_with_pool(output, &pool)
    }

    /// Convert a raw [`PipelineOutput`] using an explicit RGBA pool reference.
    fn convert_output_with_pool(
        &self,
        output: PipelineOutput,
        pool: &BufferGenerator,
    ) -> Result<NvDecoderOutput, DecoderError> {
        match output {
            PipelineOutput::Buffer(buffer) => {
                let pts = buffer.pts().map(|t| t.nseconds()).unwrap_or(0);

                let mut meta_guard = self.pts_metadata.lock();
                let metadata = meta_guard.remove(&pts);

                let (frame_id, original_pts, dts_ns, duration_ns) = if let Some(m) = metadata {
                    if is_intra_only(self.codec) {
                        self.intra_submit_fifo.lock().pop_front();
                    }
                    (Some(m.frame_id), m.original_pts_ns, m.dts_ns, m.duration_ns)
                } else if is_intra_only(self.codec) && buffer.size() > 0 {
                    // FIFO fallback for decoders that don't preserve PTS.
                    let fid = self.intra_submit_fifo.lock().pop_front();
                    if let Some(fid) = fid {
                        meta_guard.retain(|_, m| m.frame_id != fid);
                    }
                    (fid, pts, None, None)
                } else {
                    (None, pts, None, None)
                };
                drop(meta_guard);

                let shared = nvmm_to_rgba(buffer, pool, &self.transform_config)?;

                Ok(NvDecoderOutput::Frame(DecodedFrame {
                    frame_id,
                    pts_ns: original_pts,
                    dts_ns,
                    duration_ns,
                    buffer: Some(shared),
                    codec: self.codec,
                    format: VideoFormat::RGBA,
                }))
            }
            PipelineOutput::Eos => Ok(NvDecoderOutput::Eos),
            PipelineOutput::Event(event) => {
                if let Some(source_id) = parse_source_eos_event(&event) {
                    Ok(NvDecoderOutput::SourceEos { source_id })
                } else {
                    Ok(NvDecoderOutput::Event(event))
                }
            }
            PipelineOutput::Error(e) => Ok(NvDecoderOutput::Error(DecoderError::FrameworkError(e))),
        }
    }
}

impl Drop for NvDecoder {
    fn drop(&mut self) {
        if let DecoderBackendState::Pipeline { pipeline, .. } = &self.backend {
            let _ = pipeline.lock().shutdown();
        }
        let pts = self.pts_metadata.lock();
        if !pts.is_empty() {
            warn!(
                "NvDecoder '{}' dropped with {} frame(s) still pending",
                self.name,
                pts.len()
            );
        }
    }
}

// ── Low-level EOS (opt-in trait, not in prelude) ────────────────────

/// Low-level EOS for the "submit → send_eos → recv loop" pattern.
///
/// This trait is intentionally **not** part of the [`prelude`](crate::prelude)
/// so that regular consumers use [`NvDecoder::graceful_shutdown`] instead,
/// avoiding double-EOS ambiguity.  Import the trait explicitly when you need
/// the manual drain workflow (benchmarks, integration tests).
pub trait NvDecoderExt {
    /// Signal end-of-stream.
    ///
    /// For Pipeline backends, sends actual GStreamer EOS through the pipeline
    /// to flush any buffered frames (critical for B-frame codecs). The caller
    /// should continue calling [`NvDecoder::recv`] until
    /// [`NvDecoderOutput::Eos`] is received.
    ///
    /// Prefer [`NvDecoder::graceful_shutdown`] which sends EOS internally and
    /// drains via callback in a single call.
    ///
    /// Idempotent: calling after finalization is a no-op.
    fn send_eos(&self) -> Result<(), DecoderError>;
}

impl NvDecoderExt for NvDecoder {
    fn send_eos(&self) -> Result<(), DecoderError> {
        if self.finalized.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        match &self.backend {
            DecoderBackendState::Pipeline { input_tx, .. } => {
                input_tx
                    .send(PipelineInput::Eos)
                    .map_err(|_| DecoderError::ChannelDisconnected)?;
            }
            DecoderBackendState::RawUpload { output_tx, .. }
            | DecoderBackendState::ImageDecode { output_tx, .. } => {
                let _ = output_tx.send(NvDecoderOutput::Eos);
            }
        }
        Ok(())
    }
}

// ── Config validation ───────────────────────────────────────────────

fn validate_config(config: &DecoderConfig, gpu_id: u32) -> Result<(), DecoderError> {
    use crate::config::{H264StreamFormat, HevcStreamFormat};

    match config {
        DecoderConfig::RawRgba(c) if c.width == 0 || c.height == 0 => {
            return Err(DecoderError::InvalidProperty {
                name: "width/height".to_string(),
                reason: "raw config dimensions must be non-zero".to_string(),
            });
        }
        DecoderConfig::RawRgb(c) if c.width == 0 || c.height == 0 => {
            return Err(DecoderError::InvalidProperty {
                name: "width/height".to_string(),
                reason: "raw config dimensions must be non-zero".to_string(),
            });
        }
        DecoderConfig::H264(cfg)
            if matches!(
                cfg.stream_format,
                H264StreamFormat::Avc | H264StreamFormat::Avc3
            ) && cfg.codec_data.is_none() =>
        {
            return Err(DecoderError::InvalidProperty {
                name: "codec_data".to_string(),
                reason: format!("{} stream format requires codec_data", cfg.stream_format),
            });
        }
        DecoderConfig::Hevc(cfg)
            if matches!(
                cfg.stream_format,
                HevcStreamFormat::Hvc1 | HevcStreamFormat::Hev1
            ) && cfg.codec_data.is_none() =>
        {
            return Err(DecoderError::InvalidProperty {
                name: "codec_data".to_string(),
                reason: format!("{} stream format requires codec_data", cfg.stream_format),
            });
        }
        _ => {}
    }

    let codec = config.codec();
    if matches!(
        codec,
        Codec::H264 | Codec::Hevc | Codec::Vp8 | Codec::Vp9 | Codec::Av1
    ) && gst::ElementFactory::find("nvv4l2decoder").is_none()
    {
        return Err(DecoderError::NvdecNotAvailable {
            codec: codec.name().to_string(),
            gpu_id,
        });
    }

    if let DecoderConfig::Jpeg(cfg) = config {
        if cfg.backend == JpegBackend::Gpu && gst::ElementFactory::find("nvjpegdec").is_none() {
            return Err(DecoderError::NvdecNotAvailable {
                codec: "jpeg (GPU)".to_string(),
                gpu_id,
            });
        }
    }

    Ok(())
}

// ── Pipeline element chain ──────────────────────────────────────────

fn build_pipeline_elements(
    config: &DecoderConfig,
) -> Result<(Vec<gst::Element>, gst::Caps), DecoderError> {
    let caps = build_appsrc_caps(config);
    let elements = match config {
        DecoderConfig::H264(cfg) => {
            let parser = make_elem("h264parse", "parse")?;
            parser.set_property("config-interval", -1i32);
            let dec = make_elem("nvv4l2decoder", "dec")?;
            apply_v4l2_props(&dec, cfg.to_gst_pairs())?;
            vec![parser, dec]
        }
        DecoderConfig::Hevc(cfg) => {
            let parser = make_elem("h265parse", "parse")?;
            parser.set_property("config-interval", -1i32);
            let dec = make_elem("nvv4l2decoder", "dec")?;
            apply_v4l2_props(&dec, cfg.to_gst_pairs())?;
            vec![parser, dec]
        }
        DecoderConfig::Vp8(cfg) => {
            let parse = make_elem("identity", "parse")?;
            let dec = make_elem("nvv4l2decoder", "dec")?;
            apply_v4l2_props(&dec, cfg.to_gst_pairs())?;
            vec![parse, dec]
        }
        DecoderConfig::Vp9(cfg) => {
            let parse = make_elem("identity", "parse")?;
            let dec = make_elem("nvv4l2decoder", "dec")?;
            apply_v4l2_props(&dec, cfg.to_gst_pairs())?;
            vec![parse, dec]
        }
        DecoderConfig::Av1(cfg) => {
            let parser = make_elem("av1parse", "parse")?;
            let dec = make_elem("nvv4l2decoder", "dec")?;
            apply_v4l2_props(&dec, cfg.to_gst_pairs())?;
            vec![parser, dec]
        }
        DecoderConfig::Jpeg(cfg) => {
            let parser = make_elem("jpegparse", "parse")?;
            match cfg.backend {
                JpegBackend::Gpu => {
                    let dec = make_elem("nvjpegdec", "dec")?;
                    vec![parser, dec]
                }
                JpegBackend::Cpu => unreachable!("JPEG CPU uses ImageDecode backend"),
            }
        }
        DecoderConfig::Png(_) => unreachable!("PNG uses ImageDecode backend"),
        DecoderConfig::RawRgba(_) | DecoderConfig::RawRgb(_) => {
            unreachable!("raw formats use RawUpload backend")
        }
    };
    Ok((elements, caps))
}

fn build_appsrc_caps(config: &DecoderConfig) -> gst::Caps {
    match config {
        DecoderConfig::H264(cfg) => {
            let mut b = gst::Caps::builder("video/x-h264")
                .field("stream-format", cfg.stream_format.gst_name())
                .field("alignment", "au");
            if let Some(cd) = &cfg.codec_data {
                b = b.field("codec_data", gst::Buffer::from_slice(cd.clone()));
            }
            b.build()
        }
        DecoderConfig::Hevc(cfg) => {
            let mut b = gst::Caps::builder("video/x-h265")
                .field("stream-format", cfg.stream_format.gst_name())
                .field("alignment", "au");
            if let Some(cd) = &cfg.codec_data {
                b = b.field("codec_data", gst::Buffer::from_slice(cd.clone()));
            }
            b.build()
        }
        DecoderConfig::Vp8(_) => gst::Caps::builder("video/x-vp8").build(),
        DecoderConfig::Vp9(_) => gst::Caps::builder("video/x-vp9").build(),
        DecoderConfig::Av1(_) => gst::Caps::builder("video/x-av1").build(),
        DecoderConfig::Jpeg(cfg) => match cfg.backend {
            JpegBackend::Gpu => gst::Caps::builder("image/jpeg").build(),
            JpegBackend::Cpu => unreachable!("JPEG CPU uses ImageDecode backend"),
        },
        DecoderConfig::Png(_) => unreachable!("PNG uses ImageDecode backend"),
        DecoderConfig::RawRgba(_) | DecoderConfig::RawRgb(_) => {
            unreachable!("raw formats do not use GStreamer appsrc caps")
        }
    }
}

fn make_elem(factory: &str, name: &str) -> Result<gst::Element, DecoderError> {
    gst::ElementFactory::make(factory)
        .name(name)
        .build()
        .map_err(|_| DecoderError::ElementCreationFailed(factory.to_string()))
}

fn apply_v4l2_props(
    element: &gst::Element,
    pairs: Vec<(&'static str, String)>,
) -> Result<(), DecoderError> {
    for (name, val) in pairs {
        if element.has_property(name) {
            if let Ok(parsed) = val.parse::<u32>() {
                element.set_property(name, parsed);
            } else if let Ok(parsed) = val.parse::<i32>() {
                element.set_property(name, parsed);
            } else {
                element.set_property_from_str(name, &val);
            }
        }
    }
    Ok(())
}

// ── Codec classification ────────────────────────────────────────────

fn is_raw_format(codec: Codec) -> bool {
    matches!(codec, Codec::RawRgba | Codec::RawRgb)
}

fn is_image_decode(config: &DecoderConfig) -> bool {
    match config {
        DecoderConfig::Png(_) => true,
        DecoderConfig::Jpeg(cfg) => cfg.backend == JpegBackend::Cpu,
        _ => false,
    }
}

fn is_intra_only(codec: Codec) -> bool {
    matches!(
        codec,
        Codec::Jpeg | Codec::Png | Codec::RawRgba | Codec::RawRgb
    )
}

// ── Frame processing ────────────────────────────────────────────────

/// Transform an NVMM (V4L2 pool) buffer into an RGBA buffer from the output pool.
fn nvmm_to_rgba(
    buffer: gst::Buffer,
    pool: &BufferGenerator,
    transform_config: &TransformConfig,
) -> Result<SharedBuffer, DecoderError> {
    let src = SharedBuffer::from(buffer);
    let dst = pool
        .acquire(None)
        .map_err(|e| DecoderError::BufferError(format!("RGBA pool acquire failed: {e}")))?;
    src.transform_into(0, &dst, 0, transform_config, None)
        .map_err(|e| DecoderError::BufferError(format!("NVMM→RGBA transform failed: {e}")))?;
    Ok(dst)
}

/// Upload raw CPU pixels directly into a GPU buffer.
#[allow(clippy::too_many_arguments)]
fn raw_upload_frame(
    width: u32,
    height: u32,
    expected_bpp: u32,
    pool: &Mutex<BufferGenerator>,
    data: &[u8],
    frame_id: u128,
    pts_ns: u64,
    dts_ns: Option<u64>,
    duration_ns: Option<u64>,
    codec: Codec,
) -> Result<DecodedFrame, DecoderError> {
    let expected_len = (width * height * expected_bpp) as usize;
    if data.len() != expected_len {
        return Err(DecoderError::BufferError(format!(
            "unexpected data size {} for {}x{} raw input (expected {} bytes for {} bpp)",
            data.len(),
            width,
            height,
            expected_len,
            expected_bpp,
        )));
    }

    let shared = pool
        .lock()
        .acquire(None)
        .map_err(|e| DecoderError::BufferError(format!("pool acquire failed: {e}")))?;

    let upload_data: std::borrow::Cow<'_, [u8]> = if expected_bpp == 3 {
        std::borrow::Cow::Owned(rgb_to_rgba(data))
    } else {
        std::borrow::Cow::Borrowed(data)
    };
    let view = SurfaceView::from_buffer(&shared, 0)
        .map_err(|e| DecoderError::BufferError(format!("SurfaceView::from_buffer failed: {e}")))?;
    view.upload(&upload_data, width, height, 4)
        .map_err(|e| DecoderError::BufferError(format!("upload failed: {e}")))?;

    Ok(DecodedFrame {
        frame_id: Some(frame_id),
        pts_ns,
        dts_ns,
        duration_ns,
        buffer: Some(shared),
        codec,
        format: VideoFormat::RGBA,
    })
}

fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for pixel in rgb.chunks_exact(3) {
        rgba.extend_from_slice(pixel);
        rgba.push(255);
    }
    rgba
}

/// Decode a JPEG/PNG image on the CPU and upload RGBA pixels into the GPU pool.
#[allow(clippy::too_many_arguments)]
fn image_decode_and_upload(
    pool: &Mutex<BufferGenerator>,
    transform_config: &TransformConfig,
    data: &[u8],
    frame_id: u128,
    pts_ns: u64,
    dts_ns: Option<u64>,
    duration_ns: Option<u64>,
    codec: Codec,
    gpu_id: u32,
    aux_pool: &Mutex<Option<BufferGenerator>>,
) -> Result<DecodedFrame, DecoderError> {
    let img = image::load_from_memory(data)
        .map_err(|e| DecoderError::BufferError(format!("image decode failed: {e}")))?;
    let rgba = img.to_rgba8();
    let src_w = rgba.width();
    let src_h = rgba.height();
    let pixels = rgba.as_raw();

    let pool_guard = pool.lock();
    let pool_w = pool_guard.width();
    let pool_h = pool_guard.height();
    let dst = pool_guard
        .acquire(None)
        .map_err(|e| DecoderError::BufferError(format!("pool acquire failed: {e}")))?;
    drop(pool_guard);

    if src_w == pool_w && src_h == pool_h {
        let view = SurfaceView::from_buffer(&dst, 0).map_err(|e| {
            DecoderError::BufferError(format!("SurfaceView::from_buffer failed: {e}"))
        })?;
        view.upload(pixels, src_w, src_h, 4)
            .map_err(|e| DecoderError::BufferError(format!("upload failed: {e}")))?;
    } else {
        let tmp = {
            let mut aux = aux_pool.lock();
            let needs_new = match aux.as_ref() {
                Some(p) => p.width() != src_w || p.height() != src_h,
                None => true,
            };
            if needs_new {
                let p = BufferGenerator::builder(VideoFormat::RGBA, src_w, src_h)
                    .gpu_id(gpu_id)
                    .mem_type(NvBufSurfaceMemType::Default)
                    .min_buffers(2)
                    .max_buffers(2)
                    .build()
                    .map_err(|e| {
                        DecoderError::BufferError(format!("aux pool creation failed: {e}"))
                    })?;
                *aux = Some(p);
            }
            aux.as_mut()
                .unwrap()
                .acquire(None)
                .map_err(|e| DecoderError::BufferError(format!("aux pool acquire failed: {e}")))?
        };

        let view = SurfaceView::from_buffer(&tmp, 0).map_err(|e| {
            DecoderError::BufferError(format!("SurfaceView::from_buffer failed: {e}"))
        })?;
        view.upload(pixels, src_w, src_h, 4)
            .map_err(|e| DecoderError::BufferError(format!("upload failed: {e}")))?;
        drop(view);

        tmp.transform_into(0, &dst, 0, transform_config, None)
            .map_err(|e| DecoderError::BufferError(format!("image→RGBA transform failed: {e}")))?;
    }

    Ok(DecodedFrame {
        frame_id: Some(frame_id),
        pts_ns,
        dts_ns,
        duration_ns,
        buffer: Some(dst),
        codec,
        format: VideoFormat::RGBA,
    })
}
