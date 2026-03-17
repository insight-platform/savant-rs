# Test Patterns & Templates

## Test Categories

### Hardware requirement tags
| Tag | Meaning | Guard |
|---|---|---|
| GPU | Needs CUDA + GStreamer | `gstreamer::init()` + `cuda_init(0)` |
| NVENC | Needs NVENC hardware | `has_nvenc()` |
| JPEG | Needs nvjpegenc element | `has_nvjpegenc()` |
| CPU | Works everywhere | PNG, Raw codecs |

---

## Common Helpers

### init + detection
```rust
fn init() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).expect("CUDA init failed");
}

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

### Fallback codec selection
```rust
fn make_default_config(w: u32, h: u32) -> Option<EncoderConfig> {
    if has_nvenc() {
        Some(EncoderConfig::new(Codec::Hevc, w, h))
    } else if has_nvjpegenc() {
        Some(EncoderConfig::new(Codec::Jpeg, w, h).format(VideoFormat::I420))
    } else {
        None
    }
}
```

### Platform-aware H264 config
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
```

### Platform-aware HEVC config
```rust
fn hevc_encoder_config(w: u32, h: u32) -> EncoderConfig {
    if is_jetson() {
        EncoderConfig::new(Codec::Hevc, w, h)
            .format(VideoFormat::RGBA)
            .properties(EncoderProperties::HevcJetson(HevcJetsonProps {
                preset_level: Some(JetsonPresetLevel::UltraFast),
                ..Default::default()
            }))
    } else {
        EncoderConfig::new(Codec::Hevc, w, h)
            .format(VideoFormat::RGBA)
            .properties(EncoderProperties::HevcDgpu(HevcDgpuProps {
                preset: Some(DgpuPreset::P1),
                tuning_info: Some(TuningPreset::LowLatency),
                ..Default::default()
            }))
    }
}
```

### Default encoder config with fallback (platform-aware)
```rust
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

---

## Test Templates

### Encoder creation test (with NVENC guard)
```rust
#[test]
#[serial]
fn test_encoder_creation_codec() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    let config = EncoderConfig::new(Codec::H264, 640, 480);
    let encoder = NvEncoder::new(&config);
    assert!(encoder.is_ok(), "Failed: {:?}", encoder.err());
}
```

### Submit + pull test (generic pattern)
```rust
#[test]
#[serial]
fn test_codec_submit_and_pull() {
    init();
    // guard: has_nvenc() / has_nvjpegenc() / always for PNG/Raw
    let config = EncoderConfig::new(codec, 320, 240).format(format);
    let mut encoder = NvEncoder::new(&config).unwrap();
    let dur = 33_333_333u64;
    for i in 0..5u128 {
        let buf = encoder.generator().acquire(Some(i as i64)).unwrap();
        let shared = SharedBuffer::from(buf);
        encoder.submit_frame(shared, i, i as u64 * dur, Some(dur)).unwrap();
    }
    let frames = encoder.finish(Some(5000)).unwrap();
    assert!(!frames.is_empty());
    for f in &frames {
        assert!(!f.data.is_empty());
        assert_eq!(f.codec, codec);
    }
}
```

### Platform-specific property creation test
```rust
#[test]
#[serial]
fn test_encoder_creation_h264_jetson_props() {
    init();
    if !is_jetson() || !has_nvenc() {
        eprintln!("Skipping — requires Jetson + NVENC");
        return;
    }
    let config = EncoderConfig::new(Codec::H264, 640, 480)
        .format(VideoFormat::RGBA)
        .properties(EncoderProperties::H264Jetson(H264JetsonProps {
            preset_level: Some(JetsonPresetLevel::UltraFast),
            maxperf_enable: Some(true),
            ..Default::default()
        }));
    assert!(NvEncoder::new(&config).is_ok());
}
```

### Raw pixel round-trip test
```rust
#[test]
#[serial]
fn test_raw_rgba_pixel_data_round_trip() {
    init();
    let (w, h, bpp) = (64u32, 48u32, 4usize);
    let config = EncoderConfig::new(Codec::RawRgba, w, h).format(VideoFormat::RGBA);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let mut input_pixels = vec![0u8; (w as usize) * (h as usize) * bpp];
    // ... fill with pattern ...

    let buf = encoder.generator().acquire(Some(0)).unwrap();
    let view = SurfaceView::from_buffer(buf, 0).unwrap();
    deepstream_buffers::upload_to_surface(&view, &input_pixels, w, h, 4)
        .expect("upload failed");
    let shared = SharedBuffer::from(view.into_buffer().unwrap());
    encoder.submit_frame(shared, 42, 0, Some(33_333_333)).unwrap();
    let frames = encoder.finish(Some(5000)).unwrap();
    assert_eq!(frames[0].data, input_pixels);
}
```

---

## Cargo.toml (dev-dependencies)
```toml
[dev-dependencies]
env_logger = "0.11"
serial_test = { workspace = true }
criterion = { workspace = true }
nvidia_gpu_utils = { workspace = true }
```

## Build & Test
```bash
cargo fmt -p deepstream_encoders
cargo clippy -p deepstream_encoders --tests --benches -- -D warnings
cargo test -p deepstream_encoders -- --test-threads=1
```
⚠ Use `--test-threads=1` or `#[serial]` — GStreamer + CUDA state is process-global.
