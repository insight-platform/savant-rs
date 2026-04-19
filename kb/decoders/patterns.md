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

After `init()` / `cuda_init(0)`, integration tests assume NVDEC (`nvv4l2decoder`) and hardware JPEG (`nvjpegdec`) are present — no separate element probes.

- `has_nvenc()`: `nvidia_gpu_utils::has_nvenc(0).unwrap_or(false)` (encode-side tests/benches still skip when NVENC is missing)

## `VideoFrameProxy` codec / time base

`VideoFrameProxy::get_codec()` is `Option<VideoCodec>` (not a free-form string). `get_fps()` / `set_fps` and `get_time_base()` / `set_time_base` use `(i64, i64)` rationals. On the wire, protobuf uses the `VideoCodec` enum and `Rational32` messages for `fps` and `time_base`. Downstream decoders resolve `VideoCodec` to `DecoderConfig`; `VideoCodec::SwJpeg` selects CPU JPEG.

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
3. Call `graceful_shutdown(...)` or `shutdown()` depending on the scenario.
4. Drain outputs from `recv`/`recv_timeout` until `Eos`.
5. Validate at least:
   - output `DecodedFrame.format == RGBA`
   - `frame_id` propagation
   - `pts_ns` propagation (set match for pipeline codecs, exact order for synchronous backends)

## DOS / Garbage Pattern

1. Submit valid packets (optional)
2. Submit garbage packet bytes
3. Observe one of:
   - `NvDecoderOutput::Error(...)`
   - terminal `PipelineFailed` behavior via recv outputs
   - no frames + timeout/stall
4. For PNG/JPEG CPU (`image` crate path), malformed bytes are expected to fail
   immediately from `submit_packet(...) -> Err(BufferError(...))`.
5. If restart happened, submit valid packets again and verify recovery.

## Stream Detection E2E Pattern

Tests in `tests/test_stream_detect_e2e.rs` verify `detect_stream_config`
and `is_random_access_point` against real asset files from
`assets/manifest.json`:

1. **Annex-B (raw files)**: read `.h264`/`.h265`, split into AUs via
   `split_annexb_nalus` + `group_nalus_to_access_units`, feed first AU →
   assert `ByteStream` with no `codec_data`.
2. **AVCC/HVCC (MP4 demuxed)**: use `Mp4Demuxer::demux_all()` (no parser,
   raw container packets) to collect length-prefixed packets from `.mp4`
   files → feed to `detect_stream_config` → assert `Avc`/`Hvc1` with
   valid `codec_data` (version byte = 1, reasonable length).
3. **Raw file prefix**: feed first 4 KiB of each raw bitstream directly
   (no AU splitting) → assert Annex-B detection.
4. **RAP (Annex-B)**: first grouped access unit of each raw `.h264`/`.h265`
   asset → `is_random_access_point` is `true`.
5. **RAP (MP4)**: `Mp4Demuxer::demux_all()` packets — first RAP index
   matches a keyframe; B-frame assets include at least one non-RAP packet.

```rust
use savant_gstreamer::mp4_demuxer::Mp4Demuxer;

// Callback-based: collect all raw (unparsed) packets in one call.
let (packets, codec) = Mp4Demuxer::demux_all(mp4_path)
    .unwrap_or_else(|e| panic!("demuxer failed: {e}"));

for pkt in &packets {
    if let Some(cfg) = detect_stream_config(codec.unwrap(), &pkt.data) {
        // verify stream format and codec_data
        break;
    }
}
```

These tests do **not** need `#[serial]` or NVDEC hardware — they only
exercise NAL parsing, not the decode pipeline.

## Build/Test Commands

```bash
cargo fmt -p savant-deepstream-decoders
cargo clippy -p savant-deepstream-decoders
cargo test -p savant-deepstream-decoders -- --test-threads=1
```

Use `#[serial]` on integration tests because CUDA + GStreamer are process-global.
