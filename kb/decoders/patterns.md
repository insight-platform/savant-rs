# Decoder Test Patterns

## Init Helpers

```rust
fn init() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).expect("CUDA init failed");
}
```

## Capability Guards

- `has_nvdec()`: `ElementFactory::find("nvv4l2decoder").is_some()`
- `has_nvjpegdec()`: `ElementFactory::find("nvjpegdec").is_some()`
- `has_nvenc()`: `nvidia_gpu_utils::has_nvenc(0).unwrap_or(false)`

## Construction Pattern

```rust
let mut decoder = NvDecoder::new(
    0,
    &config,
    make_rgba_pool(width, height),
    identity_transform_config(),
    move |ev| {
        let _ = tx.send(ev);
    },
)?;
```

## E2E Event-Driven Pattern

1. Encode reference packets (`deepstream_encoders`) if needed.
2. Submit packets with `(frame_id, pts_ns, dts_ns, duration_ns)`.
3. Call `send_eos()`.
4. Drain events from channel until `Eos`.
5. Validate at least:
   - output `DecodedFrame.format == RGBA`
   - `frame_id` propagation
   - `pts_ns` propagation (set match for pipeline codecs, exact order for synchronous backends)

## DOS / Garbage Pattern

1. Submit valid packets (optional)
2. Submit garbage packet bytes
3. Observe one of:
   - `DecoderEvent::Error(...)`
   - `DecoderEvent::PipelineRestarted { .. }`
   - no frames + timeout/stall
4. For PNG/JPEG CPU (`image` crate path), malformed bytes are expected to fail
   immediately from `submit_packet(...) -> Err(BufferError(...))`.
5. If restart happened, submit valid packets again and verify recovery.

## Build/Test Commands

```bash
cargo fmt -p deepstream_decoders
cargo clippy -p deepstream_decoders
cargo test -p deepstream_decoders -- --test-threads=1
```

Use `#[serial]` on integration tests because CUDA + GStreamer are process-global.
