# Rust Test Patterns & Templates

## Test Categories

### NOGPU Tests (bypass, drop, worker, engine, geometry)
- Require `gstreamer::init().unwrap()` at start
- Use `SurfaceView::wrap(gst::Buffer::new())` as stub view (test-only, gated by `testing` feature)
- Helper: `make_surface_view()` in `tests/common/mod.rs`
- Cover: Drop, Bypass codec specs, EOS, shutdown, spec hot-swap, idle eviction, geometry transforms

### GPU Tests (encode, conditional, render, full pipeline)
- Require `gstreamer::init()` + `cuda_init(0)`
- Use `BufferGenerator` for real GPU buffers, then `SurfaceView::from_buffer(buf, 0).unwrap()` (by value, consuming buf)
- Helper: `make_gpu_surface_view(&gen, idx, dur_ns)` in `tests/common/mod.rs`
- Cover: full encode pipeline, Skia rendering, conditional gates, on_render/on_gpumat callbacks

---

## Cargo.toml Test Dependencies
```toml
[dev-dependencies]
env_logger = "0.11"
serial_test = { workspace = true }
gstreamer-video = { workspace = true }
```

## Test File Location
Integration tests: `savant_deepstream/picasso/tests/test_*.rs`
Unit tests: inline `#[cfg(test)] mod tests` in source files

---

## Common Helpers

### Make VideoFrameProxy (NOGPU)
```rust
fn make_frame(source_id: &str) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id, "30/1", 320, 240,
        VideoFrameContent::None, VideoFrameTranscodingMethod::Copy,
        &None, None, (1, 1_000_000_000), 0, None, None,
    ).unwrap()
}
```

### Make VideoFrameProxy with custom size
```rust
fn make_frame_sized(source_id: &str, w: i64, h: i64) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id, "30/1", w, h,
        VideoFrameContent::None, VideoFrameTranscodingMethod::Copy,
        &None, None, (1, 1_000_000_000), 0, None, None,
    ).unwrap()
}
```

### Make SurfaceView (NOGPU)
```rust
fn make_surface_view() -> deepstream_buffers::SurfaceView {
    deepstream_buffers::SurfaceView::wrap(make_gst_buffer())
}
```
Wraps a plain `gst::Buffer::new()` — surface params are zeroed, suitable for Drop/Bypass paths.

### Make GStreamer Buffer (NOGPU, internal helper)
```rust
fn make_gst_buffer() -> gstreamer::Buffer {
    gstreamer::init().unwrap();
    gstreamer::Buffer::new()
}
```

### Make GPU SurfaceView
```rust
fn make_gpu_surface_view(
    gen: &BufferGenerator,
    idx: u64,
    dur_ns: u64,
) -> deepstream_buffers::SurfaceView {
    let shared = gen.acquire(Some(idx as i64)).unwrap();
    deepstream_buffers::SurfaceView::from_buffer(&shared, 0).unwrap()
}
```
Acquires a GPU buffer and creates a validated `SurfaceView` for encode tests. On Jetson, `from_buffer` triggers EGL-CUDA registration for the surface. For batched buffers accessed multiple times, use `SurfaceView::from_buffer(&shared, i)`.

### Make GPU Buffer (internal helper)
```rust
fn make_gpu_buffer(gen: &BufferGenerator, idx: u64, dur_ns: u64) -> gstreamer::Buffer {
    let shared = gen.acquire(Some(idx as i64)).unwrap();
    shared.into_buffer().expect("sole owner")
}
```

### Make Frame with Attribute
```rust
fn make_frame_with_attr(source_id: &str, idx: u64, ns: &str, name: &str) -> VideoFrameProxy {
    let frame = make_frame(source_id);
    let mut fm = frame.clone();
    fm.set_pts((idx * FRAME_DUR_NS) as i64).unwrap();
    fm.set_duration(Some(FRAME_DUR_NS as i64)).unwrap();  // optional, for encode
    fm.set_persistent_attribute(ns, name, &None, false, vec![]);
    frame
}
```

### Add Object to Frame
```rust
fn add_object(frame: &VideoFrameProxy, cx: f32, cy: f32, w: f32, h: f32) -> i64 {
    use savant_core::primitives::object::VideoObjectBuilder;
    let obj = VideoObjectBuilder::default()
        .id(0)
        .namespace("det".to_string())
        .label("car".to_string())
        .detection_box(RBBox::new(cx, cy, w, h, None))
        .build().unwrap();
    frame.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId).unwrap().get_id()
}
```

---

## Callback Impl Patterns

### Counting callback with AtomicUsize
```rust
struct CountingBypassCb {
    count: Arc<AtomicUsize>,
}
impl OnBypassFrame for CountingBypassCb {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::VideoFrame(_) => {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            OutputMessage::EndOfStream(_) => {}
        }
    }
}
```

For EOS counting on bypass sources (EOS flows through `on_bypass_frame`):
```rust
struct CountingBypassCbWithEos {
    eos_count: Arc<AtomicUsize>,
}
impl OnBypassFrame for CountingBypassCbWithEos {
    fn call(&self, output: OutputMessage) {
        if let OutputMessage::EndOfStream(_) = output {
            self.eos_count.fetch_add(1, Ordering::SeqCst);
        }
    }
}
```

### Encoded + EOS counter
```rust
struct CountingEncodedCb {
    count: Arc<AtomicUsize>,
    eos_count: Arc<AtomicUsize>,
}
impl OnEncodedFrame for CountingEncodedCb {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::EndOfStream(_) => { self.eos_count.fetch_add(1, Ordering::SeqCst); }
            OutputMessage::VideoFrame(_)  => { self.count.fetch_add(1, Ordering::SeqCst); }
        }
    }
}
```

### Eviction callback
```rust
struct TerminateEviction;
impl OnEviction for TerminateEviction {
    fn call(&self, _source_id: &str) -> EvictionDecision {
        EvictionDecision::TerminateImmediately
    }
}
```

### Render counter (GPU tests)
```rust
struct RenderCounter(Arc<AtomicUsize>);
impl OnRender for RenderCounter {
    fn call(&self, _: &str, _: &mut deepstream_buffers::SkiaRenderer, _: &VideoFrameProxy) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}
```

### GpuMatRecorder pattern (on_gpumat tests)
`OnGpuMat::call` receives `(&self, source_id, frame, view)` — the `cuda_stream` is obtained from `view.cuda_stream().as_raw() as usize`. The `data_ptr: usize` stored in `GpuMatCall` is for equality checks only — must NOT be dereferenced after the callback scope. The actual CUDA read (`cudaMemcpy2D`) happens inside the callback while the pointer is still valid. Storing the pointer for later use is undefined behaviour because the buffer may be recycled after the encode pipeline returns.

### StreamResetRecorder pattern (on_stream_reset tests)
```rust
struct StreamResetRecorder {
    resets: Arc<Mutex<Vec<(String, StreamResetReason)>>>,
}
impl OnStreamReset for StreamResetRecorder {
    fn call(&self, source_id: &str, reason: StreamResetReason) {
        self.resets.lock().unwrap().push((source_id.to_string(), reason));
    }
}
```

---

## Engine-Level Test Templates

### Bypass Multi-Source (NOGPU)
```rust
#[test]
fn engine_bypass_multi_source() {
    gstreamer::init().unwrap();

    let bypass_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(CountingBypassCb { count: bypass_count.clone() })),
        ..Default::default()
    };
    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 60,
            ..Default::default()
        },
        callbacks,
    );

    let spec = SourceSpec { codec: CodecSpec::Bypass, ..Default::default() };
    engine.set_source_spec("src-a", spec.clone()).unwrap();
    engine.set_source_spec("src-b", spec).unwrap();

    for _ in 0..3 {
        engine.send_frame("src-a", make_frame("src-a"), make_surface_view(), None).unwrap();
        engine.send_frame("src-b", make_frame("src-b"), make_surface_view(), None).unwrap();
    }

    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(bypass_count.load(Ordering::SeqCst), 6);
    engine.shutdown();
}
```

### Post-Shutdown Rejection (NOGPU)
```rust
#[test]
fn engine_shutdown_rejects_new_frames() {
    gstreamer::init().unwrap();
    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 60,
            ..Default::default()
        },
        Callbacks::default(),
    );
    engine.shutdown();
    let result = engine.send_frame("x", make_frame("x"), make_surface_view(), None);
    assert!(result.is_err());
}
```

### EOS Sentinel (NOGPU)
Bypass EOS flows through `on_bypass_frame`; Drop EOS through `on_encoded_frame`.
Example for bypass:
```rust
#[test]
fn engine_eos_sends_sentinel() {
    gstreamer::init().unwrap();
    let eos_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(CountingBypassCbWithEos { eos_count: eos_count.clone() })),
        ..Default::default()
    };
    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 60,
            ..Default::default()
        }, callbacks,
    );
    engine.set_source_spec("s", SourceSpec { codec: CodecSpec::Bypass, ..Default::default() }).unwrap();
    engine.send_frame("s", make_frame("s"), make_surface_view(), None).unwrap();
    engine.send_eos("s").unwrap();
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(eos_count.load(Ordering::SeqCst), 1);
    engine.shutdown();
}
```
For Drop sources, use `on_encoded_frame` with `CountingEncodedCb` instead.

### Spec Hot-Swap (NOGPU)
```rust
#[test]
fn engine_spec_hot_swap() {
    gstreamer::init().unwrap();
    let bypass_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(CountingBypassCb { count: bypass_count.clone() })),
        ..Default::default()
    };
    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 60,
            ..Default::default()
        },
        callbacks,
    );

    // Start with Drop — no bypass callbacks
    engine.set_source_spec("s", SourceSpec { codec: CodecSpec::Drop, ..Default::default() }).unwrap();
    for _ in 0..3 {
        engine.send_frame("s", make_frame("s"), make_surface_view(), None).unwrap();
    }
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(bypass_count.load(Ordering::SeqCst), 0);

    // Swap to Bypass
    engine.set_source_spec("s", SourceSpec { codec: CodecSpec::Bypass, ..Default::default() }).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    for _ in 0..3 {
        engine.send_frame("s", make_frame("s"), make_surface_view(), None).unwrap();
    }
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(bypass_count.load(Ordering::SeqCst), 3);
    engine.shutdown();
}
```

---

## Worker-Level Test Template (NOGPU)
```rust
#[test]
fn worker_bypass_fires_callback() {
    gstreamer::init().unwrap();
    let bypass_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Arc::new(Callbacks {
        on_bypass_frame: Some(Arc::new(CountingBypassCb { count: bypass_count.clone() })),
        ..Default::default()
    });
    let spec = SourceSpec { codec: CodecSpec::Bypass, ..Default::default() };
    let worker = SourceWorker::spawn("test".into(), spec, callbacks, Duration::from_secs(60), 8, PtsResetPolicy::EosOnDecreasingPts);

    for _ in 0..5 {
        worker.send(WorkerMessage::Frame(make_frame("test"), make_surface_view(), None)).unwrap();
    }
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(bypass_count.load(Ordering::SeqCst), 5);
    drop(worker);
}
```

---

## NVENC / Hardware Detection Helpers

```rust
fn has_nvenc() -> bool {
    nvidia_gpu_utils::has_nvenc(0).unwrap_or(false)
}

fn has_nvjpegenc() -> bool {
    let _ = gstreamer::init();
    gstreamer::ElementFactory::find("nvjpegenc").is_some()
}

fn is_jetson() -> bool {
    cfg!(target_arch = "aarch64")
}
```

### Platform-Aware Encoder Config Builders

```rust
fn h264_encoder_config(w: u32, h: u32) -> EncoderConfig {
    if is_jetson() {
        EncoderConfig::new(Codec::H264, w, h)
            .format(VideoFormat::RGBA)
            .properties(EncoderProperties::H264Jetson(H264JetsonProps {
                preset_level: Some(JetsonPresetLevel::UltraFast),
                ..Default::default()
            }))
    } else {
        EncoderConfig::new(Codec::H264, w, h)
            .format(VideoFormat::RGBA)
            .properties(EncoderProperties::H264Dgpu(H264DgpuProps {
                preset: Some(DgpuPreset::P1),
                tuning_info: Some(TuningPreset::LowLatency),
                ..Default::default()
            }))
    }
}

fn make_default_encoder_config(w: u32, h: u32) -> Option<EncoderConfig> {
    if has_nvenc() {
        Some(h264_encoder_config(w, h))
    } else if has_nvjpegenc() {
        Some(EncoderConfig::new(Codec::Jpeg, w, h)
            .format(VideoFormat::RGBA)
            .properties(EncoderProperties::Jpeg(JpegProps { quality: Some(85) })))
    } else {
        Some(EncoderConfig::new(Codec::Png, w, h).format(VideoFormat::RGBA))
    }
}
```

⚠ Always use platform-aware builders in tests/benches. Hardcoding `H264DgpuProps` fails on Jetson even when NVENC is available.

---

## GPU Encode Test Template
```rust
#[test]
fn encode_pipeline_basic() {
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    const W: u32 = 640;
    const H: u32 = 480;
    const DUR: u64 = 33_333_333;

    let enc_config = make_default_encoder_config(W, H)
        .expect("No encoder available — skipping");

    let enc_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(EncodedCounter(enc_count.clone()))),
        ..Default::default()
    };
    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            ..Default::default()
        },
        callbacks,
    );

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(enc_config),
        },
        ..Default::default()
    };
    engine.set_source_spec("src", spec).unwrap();

    let gen = BufferGenerator::new(
        VideoFormat::RGBA, W, H, 30, 1, 0, NvBufSurfaceMemType::Default,
    ).unwrap();

    for i in 0..10u64 {
        let frame = make_frame_sized("src", W as i64, H as i64);
        let mut fm = frame.clone();
        fm.set_pts((i * DUR) as i64).unwrap();
        fm.set_duration(Some(DUR as i64)).unwrap();
        let view = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("src", frame, view, None).unwrap();
    }

    std::thread::sleep(Duration::from_secs(2));
    engine.send_eos("src").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();

    assert!(enc_count.load(Ordering::SeqCst) > 0);
}
```

---

## Geometry Test Template (NOGPU)
```rust
#[test]
fn symmetric_letterbox_center() {
    let frame = make_frame_sized("t", 800, 600);
    let obj_id = add_object(&frame, 400.0, 300.0, 80.0, 60.0);
    rewrite_frame_transformations(
        &frame, 800, 800,
        &TransformConfig { padding: Padding::Symmetric, ..Default::default() },
        None,
    ).unwrap();
    // Object should shift down by 100px (pad_top = (800-600)/2 = 100)
    let obj = frame.get_all_objects().into_iter().find(|o| o.get_id() == obj_id).unwrap();
    let det = obj.get_detection_box();
    assert!((det.get_yc() - 400.0).abs() < 0.5);
}
```

---

## Async Drain Test Pattern (GPU)

With the async drain thread, encoded output arrives independently of frame
submission. Use poll-with-deadline instead of fixed sleep:

```rust
#[test]
#[serial_test::serial]
fn e2e_async_drain_delivers_independently() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(), eos_count: eos_count.clone(),
        })),
        ..Default::default()
    };
    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            ..Default::default()
        },
        callbacks,
    );
    engine.set_source_spec("s", jpeg_spec()).unwrap();
    let gen = make_generator();

    let n = 10u64;
    for i in 0..n {
        let mut frame = make_frame_sized("s", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let view = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("s", frame, view, None).unwrap();
    }

    // Do NOT send more frames — only wait for the drain thread to deliver.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while enc_count.load(Ordering::SeqCst) < n as usize {
        assert!(std::time::Instant::now() < deadline, "timed out");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(enc_count.load(Ordering::SeqCst), n as usize);
    engine.shutdown();
}
```

---

## Benchmark Patterns

### SurfaceView in Benchmarks
Picasso benchmarks use `SurfaceView::from_buffer(buf, 0).unwrap()` (by value) instead of `SurfaceView::wrap(buf)`. With POOLED meta, EGL-CUDA registration survives pool recycles, so this is now efficient for repeated encode cycles.

### Benchmark Callback Pattern

⚠ With the async drain thread, the `OnEncodedFrame` callback fires from the
drain thread. Using `mpsc::sync_channel(0)` (rendezvous) will block the drain
thread until someone calls `recv()`. Use a buffered channel + `try_send`:

```rust
struct EncodedSignal(mpsc::SyncSender<()>);

impl OnEncodedFrame for EncodedSignal {
    fn call(&self, output: OutputMessage) {
        if let OutputMessage::VideoFrame(_) = output {
            let _ = self.0.try_send(());  // never block the drain thread
        }
    }
}

// Create with capacity > 0:
let (tx, rx) = mpsc::sync_channel::<()>(16);

// Drain stale signals between criterion samples:
while rx.try_recv().is_ok() {}
```

---

## Timing Guidance
- NOGPU tests: `sleep(100-300ms)` sufficient for worker thread processing
- GPU encode tests: prefer poll-with-deadline pattern (see Async Drain Test Pattern) over fixed sleeps
- Fixed `sleep(1-3s)` still works for simple GPU tests
- Always `send_eos` → wait for EOS sentinel → `shutdown` for encode tests
- Idle timeout tests: set `Duration::from_millis(200)`, sleep 500ms to trigger
- ⚠ Encoder rejects non-monotonic PTS (`PtsReordered`). Ensure frame counters keep incrementing across criterion closure invocations (define counter outside the closure).
