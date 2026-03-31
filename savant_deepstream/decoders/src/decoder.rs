use crate::config::{DecoderConfig, JpegBackend};
use crate::{Codec, DecodedFrame, DecoderError, DecoderEvent, VideoFormat};
use deepstream_buffers::{
    bridge_savant_id_meta, BufferGenerator, SavantIdMeta, SavantIdMetaKind, SharedBuffer,
    TransformConfig,
};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use log::warn;
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

type PtsMap = HashMap<u64, (u128, Option<u64>, Option<u64>)>;
type EventCallback = Arc<Mutex<Box<dyn FnMut(DecoderEvent) + Send>>>;

struct FrameIdState {
    pts_map: PtsMap,
    /// (frame_id, pts_ns key used in pts_map) for intra-only codecs.
    intra_submit_fifo: VecDeque<(u128, u64)>,
    codec: Codec,
}

struct PipelineState {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    appsink: gst_app::AppSink,
    drain_thread: Option<JoinHandle<()>>,
    eos_at_pad: Arc<AtomicBool>,
}

struct RawUploadState {
    format: VideoFormat,
    width: u32,
    height: u32,
}

struct ImageDecodeState {
    aux_pool: Option<BufferGenerator>,
}

enum DecoderBackend {
    Pipeline(PipelineState),
    RawUpload(RawUploadState),
    ImageDecode(ImageDecodeState),
}

/// GPU-accelerated video / image / raw-frame decoder.
///
/// Decoded frames are transformed to RGBA before delivery via the
/// caller-provided output pool, so downstream never needs an
/// additional colour-conversion step.
pub struct NvDecoder {
    gpu_id: u32,
    config: DecoderConfig,
    codec: Codec,
    pool: Arc<Mutex<BufferGenerator>>,
    transform_config: TransformConfig,
    backend: DecoderBackend,
    shared: Arc<Mutex<FrameIdState>>,
    callback: EventCallback,
    last_input_pts_ns: Option<u64>,
    finalized: bool,
}

impl NvDecoder {
    /// Create a new decoder.
    ///
    /// `pool` is the RGBA output buffer pool.  Every decoded frame is
    /// acquired from this pool (and optionally scaled/converted via
    /// `NvBufSurfTransform`) before delivery.  The caller controls pool
    /// dimensions and depth.
    ///
    /// `transform_config` controls how the GPU transform handles format
    /// conversion and scaling (interpolation, padding, compute mode).
    ///
    /// `on_event` is invoked on a dedicated drain thread for every
    /// decoded frame, EOS, and pipeline error.
    pub fn new<F>(
        gpu_id: u32,
        config: &DecoderConfig,
        pool: BufferGenerator,
        transform_config: TransformConfig,
        on_event: F,
    ) -> Result<Self, DecoderError>
    where
        F: FnMut(DecoderEvent) + Send + 'static,
    {
        gst::init()
            .map_err(|e| DecoderError::PipelineError(format!("GStreamer init failed: {e}")))?;

        Self::validate_config(config, gpu_id)?;

        let codec = config.codec();
        let shared = Arc::new(Mutex::new(FrameIdState {
            pts_map: HashMap::new(),
            intra_submit_fifo: VecDeque::new(),
            codec,
        }));
        let callback: EventCallback = Arc::new(Mutex::new(Box::new(on_event)));
        let pool = Arc::new(Mutex::new(pool));

        let backend = if is_raw_format(codec) {
            DecoderBackend::RawUpload(Self::build_raw_upload(config)?)
        } else if is_image_decode(config) {
            DecoderBackend::ImageDecode(ImageDecodeState { aux_pool: None })
        } else {
            DecoderBackend::Pipeline(Self::build_pipeline(config)?)
        };

        Ok(Self {
            gpu_id,
            config: config.clone(),
            codec,
            pool,
            transform_config,
            backend,
            shared,
            callback,
            last_input_pts_ns: None,
            finalized: false,
        })
    }

    /// The codec this decoder was configured for.
    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// Submit a compressed packet (or raw frame) for decoding.
    ///
    /// `pts_ns` must be **strictly monotonically increasing** across
    /// successive calls.  It serves as the submission ordering key, not
    /// necessarily the presentation timestamp.  Callers with B-frame
    /// content should pass a DTS-based ordering value (e.g.
    /// `dts_ns.unwrap_or(pts_ns)`) to satisfy monotonicity.
    ///
    /// Returns [`DecoderError::PtsReordered`] if `pts_ns` is not greater
    /// than the previous submission's value.
    pub fn submit_packet(
        &mut self,
        data: &[u8],
        frame_id: u128,
        pts_ns: u64,
        dts_ns: Option<u64>,
        duration_ns: Option<u64>,
    ) -> Result<(), DecoderError> {
        if self.finalized {
            return Err(DecoderError::AlreadyFinalized);
        }

        self.maybe_restart_from_bus_error();

        if let Some(prev) = self.last_input_pts_ns {
            if pts_ns <= prev {
                return Err(DecoderError::PtsReordered {
                    frame_id,
                    pts_ns,
                    prev_pts_ns: prev,
                });
            }
        }
        self.last_input_pts_ns = Some(pts_ns);

        match &mut self.backend {
            DecoderBackend::Pipeline(state) => {
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
                let push_res = state.appsrc.push_buffer(buffer);
                push_res.map_err(|e| {
                    DecoderError::PipelineError(format!("appsrc push failed: {:?}", e))
                })?;
                {
                    let mut st = self.shared.lock();
                    st.pts_map.insert(pts_ns, (frame_id, dts_ns, duration_ns));
                    if is_intra_only(self.codec) {
                        st.intra_submit_fifo.push_back((frame_id, pts_ns));
                    }
                }
                Self::ensure_drain_thread(
                    state,
                    &self.shared,
                    &self.callback,
                    &self.pool,
                    &self.transform_config,
                    self.gpu_id,
                );
            }
            DecoderBackend::RawUpload(raw) => {
                let frame = raw_upload_frame(
                    raw,
                    &self.pool,
                    data,
                    frame_id,
                    pts_ns,
                    dts_ns,
                    duration_ns,
                    self.codec,
                )?;
                (self.callback.lock())(DecoderEvent::Frame(frame));
            }
            DecoderBackend::ImageDecode(state) => {
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
                    &mut state.aux_pool,
                )?;
                (self.callback.lock())(DecoderEvent::Frame(frame));
            }
        }
        Ok(())
    }

    /// Signal end-of-stream.  The [`DecoderEvent::Eos`] callback will
    /// fire once all buffered frames have been delivered.
    pub fn send_eos(&mut self) -> Result<(), DecoderError> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;
        match &mut self.backend {
            DecoderBackend::Pipeline(state) => {
                state
                    .appsrc
                    .end_of_stream()
                    .map_err(|e| DecoderError::PipelineError(format!("EOS failed: {:?}", e)))?;
                Self::ensure_drain_thread(
                    state,
                    &self.shared,
                    &self.callback,
                    &self.pool,
                    &self.transform_config,
                    self.gpu_id,
                );
            }
            DecoderBackend::RawUpload(_) | DecoderBackend::ImageDecode(_) => {
                (self.callback.lock())(DecoderEvent::Eos);
            }
        }
        Ok(())
    }

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

    fn ensure_drain_thread(
        state: &mut PipelineState,
        shared: &Arc<Mutex<FrameIdState>>,
        callback: &EventCallback,
        pool: &Arc<Mutex<BufferGenerator>>,
        transform_config: &TransformConfig,
        gpu_id: u32,
    ) {
        if state.drain_thread.is_some() {
            return;
        }
        let sink = state.appsink.clone();
        let pipeline = state.pipeline.clone();
        let shared = Arc::clone(shared);
        let cb = Arc::clone(callback);
        let eos_flag = state.eos_at_pad.clone();
        let pool = Arc::clone(pool);
        let tf = transform_config.clone();
        match std::thread::Builder::new()
            .name("decoder-drain".into())
            .spawn(move || drain_loop(sink, pipeline, shared, cb, pool, tf, gpu_id, eos_flag))
        {
            Ok(handle) => state.drain_thread = Some(handle),
            Err(e) => {
                (callback.lock())(DecoderEvent::Error(DecoderError::PipelineError(format!(
                    "failed to spawn drain thread: {e}"
                ))));
            }
        }
    }

    fn maybe_restart_from_bus_error(&mut self) {
        let reason = if let DecoderBackend::Pipeline(state) = &self.backend {
            let bus = match state.pipeline.bus() {
                Some(b) => b,
                None => return,
            };
            match bus.pop_filtered(&[gst::MessageType::Error]) {
                Some(msg) => {
                    if let gst::MessageView::Error(e) = msg.view() {
                        Some(format!("{} ({})", e.error(), e.debug().unwrap_or_default()))
                    } else {
                        None
                    }
                }
                None => None,
            }
        } else {
            None
        };
        if let Some(reason) = reason {
            self.restart_pipeline(reason);
        }
    }

    fn restart_pipeline(&mut self, reason: String) {
        let lost = {
            let mut st = self.shared.lock();
            let lost = st.pts_map.len().max(st.intra_submit_fifo.len());
            st.pts_map.clear();
            st.intra_submit_fifo.clear();
            lost
        };
        self.last_input_pts_ns = None;
        self.finalized = false;

        if let DecoderBackend::Pipeline(state) = &mut self.backend {
            let _ = state.pipeline.set_state(gst::State::Null);
            // Signal EOS so the drain loop exits instead of polling forever.
            state.eos_at_pad.store(true, Ordering::Release);
            if let Some(handle) = state.drain_thread.take() {
                let _ = handle.join();
            }
        }

        match Self::build_pipeline(&self.config) {
            Ok(new_state) => {
                self.backend = DecoderBackend::Pipeline(new_state);
                (self.callback.lock())(DecoderEvent::PipelineRestarted {
                    reason,
                    lost_frame_count: lost,
                });
            }
            Err(e) => {
                (self.callback.lock())(DecoderEvent::Error(DecoderError::PipelineError(format!(
                    "pipeline restart failed after '{reason}': {e}"
                ))));
            }
        }
    }

    fn build_raw_upload(config: &DecoderConfig) -> Result<RawUploadState, DecoderError> {
        let (format, width, height) = match config {
            DecoderConfig::RawRgba(c) => (VideoFormat::RGBA, c.width, c.height),
            DecoderConfig::RawRgb(c) => (VideoFormat::RGB, c.width, c.height),
            _ => unreachable!("build_raw_upload called for non-raw config"),
        };
        Ok(RawUploadState {
            format,
            width,
            height,
        })
    }

    fn build_pipeline(config: &DecoderConfig) -> Result<PipelineState, DecoderError> {
        let pipeline = gst::Pipeline::new();
        let appsrc = gst::ElementFactory::make("appsrc")
            .name("src")
            .build()
            .map_err(|_| DecoderError::ElementCreationFailed("appsrc".to_string()))?;
        let appsink = gst::ElementFactory::make("appsink")
            .name("sink")
            .build()
            .map_err(|_| DecoderError::ElementCreationFailed("appsink".to_string()))?;
        let appsrc = appsrc
            .dynamic_cast::<gst_app::AppSrc>()
            .map_err(|_| DecoderError::PipelineError("appsrc cast failed".to_string()))?;
        let appsink = appsink
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| DecoderError::PipelineError("appsink cast failed".to_string()))?;

        let appsrc_caps = build_appsrc_caps(config);
        let appsrc_elem = appsrc.upcast_ref::<gst::Element>();
        appsrc_elem.set_property("caps", appsrc_caps);
        appsrc_elem.set_property_from_str("format", "time");
        appsrc_elem.set_property_from_str("stream-type", "stream");
        appsrc_elem.set_property("is-live", true);
        appsrc_elem.set_property("min-latency", 0i64);

        let appsink_elem = appsink.upcast_ref::<gst::Element>();
        appsink_elem.set_property("sync", false);
        appsink_elem.set_property("emit-signals", false);

        let eos_at_pad = Arc::new(AtomicBool::new(false));
        {
            let eos_flag = eos_at_pad.clone();
            if let Some(sink_pad) = appsink_elem.static_pad("sink") {
                sink_pad.add_probe(gst::PadProbeType::EVENT_DOWNSTREAM, move |_pad, info| {
                    if let Some(ev) = info.event() {
                        if ev.type_() == gst::EventType::Eos {
                            eos_flag.store(true, Ordering::Release);
                        }
                    }
                    gst::PadProbeReturn::Ok
                });
            }
        }

        match config {
            DecoderConfig::H264(cfg) => {
                let parser = make_elem("h264parse", "parse")?;
                let dec = make_elem("nvv4l2decoder", "dec")?;
                apply_v4l2_props(&dec, cfg.to_gst_pairs())?;
                bridge_savant_id_meta(&dec)?;
                add_link(
                    &pipeline,
                    &[appsrc.upcast_ref(), &parser, &dec, appsink.upcast_ref()],
                )?;
            }
            DecoderConfig::Hevc(cfg) => {
                let parser = make_elem("h265parse", "parse")?;
                let dec = make_elem("nvv4l2decoder", "dec")?;
                apply_v4l2_props(&dec, cfg.to_gst_pairs())?;
                bridge_savant_id_meta(&dec)?;
                add_link(
                    &pipeline,
                    &[appsrc.upcast_ref(), &parser, &dec, appsink.upcast_ref()],
                )?;
            }
            DecoderConfig::Vp8(cfg) => {
                let parse = make_elem("identity", "parse")?;
                let dec = make_elem("nvv4l2decoder", "dec")?;
                apply_v4l2_props(&dec, cfg.to_gst_pairs())?;
                bridge_savant_id_meta(&dec)?;
                add_link(
                    &pipeline,
                    &[appsrc.upcast_ref(), &parse, &dec, appsink.upcast_ref()],
                )?;
            }
            DecoderConfig::Vp9(cfg) => {
                let parse = make_elem("identity", "parse")?;
                let dec = make_elem("nvv4l2decoder", "dec")?;
                apply_v4l2_props(&dec, cfg.to_gst_pairs())?;
                bridge_savant_id_meta(&dec)?;
                add_link(
                    &pipeline,
                    &[appsrc.upcast_ref(), &parse, &dec, appsink.upcast_ref()],
                )?;
            }
            DecoderConfig::Av1(cfg) => {
                let parser = make_elem("av1parse", "parse")?;
                let dec = make_elem("nvv4l2decoder", "dec")?;
                apply_v4l2_props(&dec, cfg.to_gst_pairs())?;
                bridge_savant_id_meta(&dec)?;
                add_link(
                    &pipeline,
                    &[appsrc.upcast_ref(), &parser, &dec, appsink.upcast_ref()],
                )?;
            }
            DecoderConfig::Jpeg(cfg) => {
                let parser = make_elem("jpegparse", "parse")?;
                match cfg.backend {
                    JpegBackend::Gpu => {
                        let dec = make_elem("nvjpegdec", "dec")?;
                        bridge_savant_id_meta(&dec)?;
                        add_link(
                            &pipeline,
                            &[appsrc.upcast_ref(), &parser, &dec, appsink.upcast_ref()],
                        )?;
                    }
                    JpegBackend::Cpu => {
                        unreachable!("JPEG CPU uses ImageDecode backend, not Pipeline")
                    }
                }
            }
            DecoderConfig::Png(_) => {
                unreachable!("PNG uses ImageDecode backend, not Pipeline")
            }
            DecoderConfig::RawRgba(_) | DecoderConfig::RawRgb(_) => {
                unreachable!("raw formats use RawUpload backend, not Pipeline")
            }
        }

        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| DecoderError::PipelineError(format!("Failed to start pipeline: {e:?}")))?;
        Ok(PipelineState {
            pipeline,
            appsrc,
            appsink,
            drain_thread: None,
            eos_at_pad,
        })
    }
}

impl Drop for NvDecoder {
    fn drop(&mut self) {
        match &mut self.backend {
            DecoderBackend::Pipeline(state) => {
                let _ = state.appsrc.end_of_stream();
                if let Some(handle) = state.drain_thread.take() {
                    let _ = handle.join();
                }
                let _ = state.pipeline.set_state(gst::State::Null);
            }
            DecoderBackend::RawUpload(_) | DecoderBackend::ImageDecode(_) => {}
        }
        let st = self.shared.lock();
        if !st.intra_submit_fifo.is_empty() {
            warn!(
                "NvDecoder dropped with {} frame IDs still pending",
                st.intra_submit_fifo.len()
            );
        }
    }
}

/// Upload raw CPU pixels directly into an NvBufSurface-backed GPU buffer.
#[allow(clippy::too_many_arguments)]
fn raw_upload_frame(
    raw: &RawUploadState,
    pool: &Mutex<BufferGenerator>,
    data: &[u8],
    frame_id: u128,
    pts_ns: u64,
    dts_ns: Option<u64>,
    duration_ns: Option<u64>,
    codec: Codec,
) -> Result<DecodedFrame, DecoderError> {
    let shared = pool
        .lock()
        .acquire(None)
        .map_err(|e| DecoderError::BufferError(format!("pool acquire failed: {e}")))?;

    let upload_data = match raw.format {
        VideoFormat::RGBA => {
            let expected = (raw.width * raw.height * 4) as usize;
            if data.len() != expected {
                return Err(DecoderError::BufferError(format!(
                    "unexpected data size {} for {}x{} RGBA (expected {})",
                    data.len(),
                    raw.width,
                    raw.height,
                    expected,
                )));
            }
            std::borrow::Cow::Borrowed(data)
        }
        VideoFormat::RGB => {
            let expected = (raw.width * raw.height * 3) as usize;
            if data.len() != expected {
                return Err(DecoderError::BufferError(format!(
                    "unexpected data size {} for {}x{} RGB (expected {})",
                    data.len(),
                    raw.width,
                    raw.height,
                    expected,
                )));
            }
            std::borrow::Cow::Owned(rgb_to_rgba(data))
        }
        _ => {
            return Err(DecoderError::BufferError(format!(
                "unsupported raw format: {:?}",
                raw.format
            )));
        }
    };
    let view = deepstream_buffers::SurfaceView::from_buffer(&shared, 0)
        .map_err(|e| DecoderError::BufferError(format!("SurfaceView::from_buffer failed: {e}")))?;
    view.upload(&upload_data, raw.width, raw.height, 4)
        .map_err(|e| DecoderError::BufferError(format!("upload failed: {e}")))?;

    Ok(DecodedFrame {
        frame_id: Some(frame_id),
        pts_ns,
        dts_ns,
        duration_ns,
        buffer: shared,
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

/// Decode a JPEG or PNG image using the Rust `image` crate and upload
/// the resulting RGBA pixels into the caller-provided GPU pool.
///
/// When decoded dimensions match the pool, pixels are uploaded directly.
/// Otherwise a temporary auxiliary surface handles the upload and a GPU
/// transform scales/pads into the output buffer.
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
    aux_pool: &mut Option<BufferGenerator>,
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
        let view = deepstream_buffers::SurfaceView::from_buffer(&dst, 0).map_err(|e| {
            DecoderError::BufferError(format!("SurfaceView::from_buffer failed: {e}"))
        })?;
        view.upload(pixels, src_w, src_h, 4)
            .map_err(|e| DecoderError::BufferError(format!("upload failed: {e}")))?;
    } else {
        let tmp_pool = match aux_pool {
            Some(p) if p.width() == src_w && p.height() == src_h => p,
            _ => {
                let p = BufferGenerator::builder(VideoFormat::RGBA, src_w, src_h)
                    .gpu_id(gpu_id)
                    .mem_type(deepstream_buffers::NvBufSurfaceMemType::Default)
                    .min_buffers(2)
                    .max_buffers(2)
                    .build()
                    .map_err(|e| {
                        DecoderError::BufferError(format!("aux pool creation failed: {e}"))
                    })?;
                aux_pool.insert(p)
            }
        };

        let tmp = tmp_pool
            .acquire(None)
            .map_err(|e| DecoderError::BufferError(format!("aux pool acquire failed: {e}")))?;
        let view = deepstream_buffers::SurfaceView::from_buffer(&tmp, 0).map_err(|e| {
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
        buffer: dst,
        codec,
        format: VideoFormat::RGBA,
    })
}

#[allow(clippy::too_many_arguments)]
fn drain_loop(
    sink: gst_app::AppSink,
    _pipeline: gst::Pipeline,
    shared: Arc<Mutex<FrameIdState>>,
    callback: EventCallback,
    pool: Arc<Mutex<BufferGenerator>>,
    transform_config: TransformConfig,
    gpu_id: u32,
    eos_at_pad: Arc<AtomicBool>,
) {
    let mut aux_pool: Option<BufferGenerator> = None;

    loop {
        let eos_seen = eos_at_pad.load(Ordering::Acquire);

        match sink.try_pull_sample(gst::ClockTime::from_mseconds(0)) {
            Some(sample) => {
                let result = sample_to_frame(
                    &shared,
                    sample,
                    &pool,
                    &transform_config,
                    &mut aux_pool,
                    gpu_id,
                );
                match result {
                    Ok(frame) => (callback.lock())(DecoderEvent::Frame(frame)),
                    Err(e) => (callback.lock())(DecoderEvent::Error(e)),
                }
            }
            None if sink.is_eos() || eos_seen => {
                (callback.lock())(DecoderEvent::Eos);
                break;
            }
            None => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

fn sample_to_frame(
    shared: &Mutex<FrameIdState>,
    sample: gst::Sample,
    pool: &Mutex<BufferGenerator>,
    transform_config: &TransformConfig,
    aux_pool: &mut Option<BufferGenerator>,
    gpu_id: u32,
) -> Result<DecodedFrame, DecoderError> {
    let buffer = sample
        .buffer()
        .ok_or_else(|| DecoderError::PipelineError("Sample has no buffer".to_string()))?;

    let pts_ns = buffer.pts().map(|t| t.nseconds()).unwrap_or(0);
    let dts_ns = buffer.dts().map(|t| t.nseconds());
    let duration_ns = buffer.duration().map(|t| t.nseconds());
    let buf_size = buffer.size() as u64;

    let mut st = shared.lock();

    // Lookup the frame_id associated with this decoded buffer.
    // Primary key: the output buffer's PTS (matches input PTS stored in
    // pts_map).  Fallback: when the decoder doesn't preserve PTS
    // (uncommon with nvv4l2decoder), try the output buffer's DTS as a
    // secondary key.
    let (lookup, out_pts) = {
        let by_pts = st.pts_map.remove(&pts_ns);
        if by_pts.is_some() {
            (by_pts, pts_ns)
        } else if let Some(dts) = dts_ns {
            let by_dts = st.pts_map.remove(&dts);
            (by_dts, dts)
        } else {
            (None, pts_ns)
        }
    };

    let frame_id = match lookup {
        Some((id, _, _)) => {
            if is_intra_only(st.codec) {
                let _ = st.intra_submit_fifo.pop_front();
            }
            Some(id)
        }
        None if is_intra_only(st.codec) && buf_size > 0 => {
            st.intra_submit_fifo.pop_front().map(|(id, pts_key)| {
                st.pts_map.remove(&pts_key);
                id
            })
        }
        None => None,
    };

    let codec = st.codec;
    drop(st);

    let caps = sample.caps();
    let is_nvmm = caps
        .and_then(|c| c.features(0))
        .map(|f| f.contains("memory:NVMM"))
        .unwrap_or(false);

    let src_format = caps
        .and_then(|caps| {
            caps.structure(0)
                .and_then(|s| s.get::<String>("format").ok())
        })
        .and_then(|s| VideoFormat::from_name(s.as_str()))
        .unwrap_or_else(|| native_output_format(codec));

    let structure = caps.and_then(|c| c.structure(0).map(|s| s.to_owned()));
    let width = structure
        .as_ref()
        .and_then(|s| s.get::<i32>("width").ok())
        .unwrap_or(0) as u32;
    let height = structure
        .as_ref()
        .and_then(|s| s.get::<i32>("height").ok())
        .unwrap_or(0) as u32;

    let shared_buf = if is_nvmm {
        nvmm_to_rgba(buffer, pool, transform_config)?
    } else {
        cpu_to_rgba(
            buffer,
            src_format,
            width,
            height,
            pool,
            transform_config,
            aux_pool,
            gpu_id,
        )?
    };

    Ok(DecodedFrame {
        frame_id,
        pts_ns: out_pts,
        dts_ns,
        duration_ns,
        buffer: shared_buf,
        codec,
        format: VideoFormat::RGBA,
    })
}

/// Transform an NVMM (V4L2 pool) buffer into an RGBA buffer from the
/// caller-provided output pool.
///
/// `NvBufSurfTransform` handles format conversion (NV12/I420 → RGBA),
/// scaling, and the copy in a single GPU operation.  The V4L2 output
/// buffer is released as soon as the transform completes, preventing
/// buffer-pool starvation.
fn nvmm_to_rgba(
    buffer: &gst::BufferRef,
    pool: &Mutex<BufferGenerator>,
    transform_config: &TransformConfig,
) -> Result<SharedBuffer, DecoderError> {
    let src = SharedBuffer::from(buffer.to_owned());

    let dst = pool
        .lock()
        .acquire(None)
        .map_err(|e| DecoderError::BufferError(format!("RGBA pool acquire failed: {e}")))?;

    src.transform_into(0, &dst, 0, transform_config, None)
        .map_err(|e| DecoderError::BufferError(format!("NVMM→RGBA transform failed: {e}")))?;

    Ok(dst)
}

/// Upload a CPU-memory GStreamer buffer into an RGBA buffer from the
/// caller-provided output pool.
///
/// When source dimensions match the pool dimensions, pixels are uploaded
/// directly (fast path).  Otherwise a temporary auxiliary surface is used
/// for the upload, followed by a GPU transform to the output buffer.
#[allow(clippy::too_many_arguments)]
fn cpu_to_rgba(
    buffer: &gst::BufferRef,
    src_format: VideoFormat,
    width: u32,
    height: u32,
    pool: &Mutex<BufferGenerator>,
    transform_config: &TransformConfig,
    aux_pool: &mut Option<BufferGenerator>,
    gpu_id: u32,
) -> Result<SharedBuffer, DecoderError> {
    let map = buffer
        .map_readable()
        .map_err(|e| DecoderError::PipelineError(format!("Failed to map CPU buffer: {:?}", e)))?;
    let cpu_data = map.as_slice();

    let bpp = match src_format {
        VideoFormat::RGBA | VideoFormat::BGRx => 4u32,
        _ => {
            return Err(DecoderError::BufferError(format!(
                "unsupported CPU upload format: {:?}",
                src_format
            )))
        }
    };

    let pool_guard = pool.lock();
    let pool_w = pool_guard.width();
    let pool_h = pool_guard.height();
    let dst = pool_guard
        .acquire(None)
        .map_err(|e| DecoderError::BufferError(format!("RGBA pool acquire failed: {e}")))?;
    drop(pool_guard);

    if width == pool_w && height == pool_h && src_format == VideoFormat::RGBA {
        let view = deepstream_buffers::SurfaceView::from_buffer(&dst, 0).map_err(|e| {
            DecoderError::BufferError(format!("SurfaceView::from_buffer failed: {e}"))
        })?;
        view.upload(cpu_data, width, height, bpp)
            .map_err(|e| DecoderError::BufferError(format!("upload failed: {e}")))?;
    } else {
        let upload_format = if src_format == VideoFormat::BGRx {
            VideoFormat::BGRx
        } else {
            VideoFormat::RGBA
        };
        let tmp_pool = match aux_pool {
            Some(p) if p.width() == width && p.height() == height => p,
            _ => {
                let p = BufferGenerator::builder(upload_format, width, height)
                    .gpu_id(gpu_id)
                    .mem_type(deepstream_buffers::NvBufSurfaceMemType::Default)
                    .min_buffers(2)
                    .max_buffers(2)
                    .build()
                    .map_err(|e| {
                        DecoderError::BufferError(format!(
                            "failed to create CPU upload aux pool: {e}"
                        ))
                    })?;
                aux_pool.insert(p)
            }
        };

        let tmp = tmp_pool
            .acquire(None)
            .map_err(|e| DecoderError::BufferError(format!("aux pool acquire failed: {e}")))?;
        let view = deepstream_buffers::SurfaceView::from_buffer(&tmp, 0).map_err(|e| {
            DecoderError::BufferError(format!("SurfaceView::from_buffer failed: {e}"))
        })?;
        view.upload(cpu_data, width, height, bpp)
            .map_err(|e| DecoderError::BufferError(format!("upload failed: {e}")))?;
        drop(view);

        tmp.transform_into(0, &dst, 0, transform_config, None)
            .map_err(|e| DecoderError::BufferError(format!("CPU→RGBA transform failed: {e}")))?;
    }

    Ok(dst)
}

fn make_elem(factory: &str, name: &str) -> Result<gst::Element, DecoderError> {
    gst::ElementFactory::make(factory)
        .name(name)
        .build()
        .map_err(|_| DecoderError::ElementCreationFailed(factory.to_string()))
}

fn add_link(pipeline: &gst::Pipeline, elems: &[&gst::Element]) -> Result<(), DecoderError> {
    for elem in elems {
        pipeline
            .add(*elem)
            .map_err(|e| DecoderError::PipelineError(format!("Failed to add element: {e}")))?;
    }
    gst::Element::link_many(elems)
        .map_err(|_| DecoderError::LinkFailed("Failed to link decoder pipeline".to_string()))
}

fn apply_v4l2_props(
    element: &gst::Element,
    pairs: Vec<(&'static str, String)>,
) -> Result<(), DecoderError> {
    for (name, val) in pairs {
        if element.has_property(name, None) {
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

fn native_output_format(codec: Codec) -> VideoFormat {
    match codec {
        Codec::H264 | Codec::Hevc | Codec::Vp8 | Codec::Vp9 | Codec::Av1 => VideoFormat::NV12,
        Codec::Jpeg => VideoFormat::I420,
        Codec::Png | Codec::RawRgba => VideoFormat::RGBA,
        Codec::RawRgb => VideoFormat::BGRx,
        Codec::RawNv12 => unreachable!("RawNv12 not supported as decoder input"),
    }
}
