# Critical Caveats & Platform-Specific Notes

## 1. NVENC Availability

- **Orin Nano (8GB and 4GB):** Does NOT have NVENC hardware. `nvidia_gpu_utils::has_nvenc(0)` returns `false`. H264, HEVC, AV1 codecs will fail with `NvencNotAvailable`.
- **Other Jetson models** (AGX Orin, Orin NX, Xavier, etc.): Have NVENC.
- **Some datacenter dGPUs** (H100, A100, A30, B200, etc.): No NVENC. `has_nvenc()` uses NVML `encoder_capacity(H264)` to detect.

Tests using H264/HEVC/AV1 **must** guard with:
```rust
if !has_nvenc() {
    eprintln!("NVENC not available — skipping");
    return;
}
```

---

## 2. Platform-Specific Encoder Properties

The same GStreamer element (`nvv4l2h264enc`, `nvv4l2h265enc`) has **different property sets** on dGPU vs Jetson. Using the wrong property struct will cause GStreamer element errors.

| Platform | Detect | H264 Props | HEVC Props | Key Properties |
|---|---|---|---|---|
| dGPU | `!cfg!(target_arch = "aarch64")` | `H264DgpuProps` | `HevcDgpuProps` | `preset-id` (DgpuPreset), `tuning-info-id` (TuningPreset) |
| Jetson | `cfg!(target_arch = "aarch64")` | `H264JetsonProps` | `HevcJetsonProps` | `preset-level` (JetsonPresetLevel), `maxperf-enable` |

Platform detection for test/bench code:
```rust
fn is_jetson() -> bool {
    cfg!(target_arch = "aarch64")
}
```

⚠ JPEG, PNG, RawRgba, RawRgb, and RawNv12 codecs have platform-neutral properties — no branching needed.

---

## 3. Jetson nvjpegenc "Surface not registered" Hang

On Jetson, `nvjpegenc` requires surfaces to be "pinned" by its own mechanism. Surfaces from the NvDS buffer pool are NOT registered, causing `NVJPGGetSurfPinHandle: Surface not registered` and silent pipeline hang (appsrc blocks on backpressure).

**Fix in encoder.rs:** On aarch64, `nvvideoconvert` with `disable-passthrough=true` is inserted before `nvjpegenc`. This forces surface re-allocation through nvvideoconvert's pool, creating surfaces compatible with NVJPG.

This does NOT affect `nvv4l2h264enc` / `nvv4l2h265enc` — those handle NvDS pool surfaces directly.

---

## 4. Jetson VIC Limitations for Raw Codecs

The Video Image Compositor (VIC) on Jetson does NOT support NV12→RGBA/RGB format conversion. Using the default VIC compute path causes "RGB/BGR Format transformation is not supported by VIC" errors.

**Fix in encoder.rs:** For raw pseudoencoders on aarch64, `nvvideoconvert` gets `compute-hw=1` to force GPU-based processing instead of VIC:
```rust
#[cfg(target_arch = "aarch64")]
nvconv.set_property_from_str("compute-hw", "1");
```

⚠ This affects Raw codecs only. NVENC codecs don't use `nvvideoconvert` for format conversion (they use NvBufSurfTransform directly).
⚠ The `compute-hw` property only exists on Jetson `nvvideoconvert`.

---

## 5. SurfaceView::upload Takes 5 Arguments

```rust
pub fn upload(&self, data: &[u8], width: u32, height: u32, channels: u32)
    -> Result<(), NvBufSurfaceError>
```

Method on `SurfaceView` (not a free function). The `channels` parameter is commonly forgotten. Use `4` for RGBA, `3` for RGB.

---

## 6. Buffer Pool Size = 1

Internal buffer pools are configured with exactly 1 buffer (`min_buffers=1, max_buffers=1`). The NVENC hardware encoder may continue DMA-reading from a buffer's GPU memory after GStreamer releases its reference. A pool of 1 forces serialization.

⚠ Do NOT increase pool size to "improve performance" — this will cause intermittent stale-data artifacts.

---

## 7. B-Frame Enforcement

B-frames are categorically disabled:
1. `EncoderProperties` types have no B-frame fields
2. `force_disable_b_frames()` sets all known B-frame property variants to 0
3. Runtime validation detects B-frame reordering in output (PTS goes backwards → `OutputPtsReordered`)

---

## 8. AV1 Codec Header Inlining (dGPU parity with Jetson)

AV1 encoders emit an `OBU_SEQUENCE_HEADER` at stream start. The two platforms surface it differently at the raw GStreamer layer:

- **Jetson** (`nvv4l2av1enc`, MMAPI/V4L2): inlines the sequence header into the first IDR's compressed buffer — one `appsink` sample per user frame.
- **dGPU** (`nvv4l2av1enc`, NVENC shim): emits the sequence header as a **separate ~32-byte buffer** before the first IDR — two `appsink` samples for the first frame.

`NvEncoder::sample_to_frame` normalises this to Jetson-style behavior on both platforms:

1. A sample that fails the `pts_map` correlation on a non-intra-only codec (H.264 / HEVC / AV1) is treated as a stream-level header and its bytes are stashed in `NvEncoder::pending_codec_header` (concatenated if multiple header OBUs arrive).
2. `sample_to_frame` returns `Ok(None)` for such samples; `pull_encoded` / `pull_encoded_timeout` loop internally so the stash is invisible to callers.
3. When the next user frame arrives, the stashed bytes are prepended to its `data`. The resulting `EncodedFrame` has `frame_id = Some(id)` and a bitstream that starts with `OBU_SEQUENCE_HEADER`.
4. An orphan header at `finish()` (no user frame ever followed) is dropped with a `debug!` — benign, as no consumer was waiting on it.

Consequences for consumers:

- Every `EncodedFrame` returned from `NvEncoder` corresponds to a submitted frame. `frame_id` is always `Some` for non-intra-only codecs.
- Picasso no longer logs `ERROR: drain: cannot correlate encoded payload …` for AV1. The branch was downgraded to `warn!` as a diagnostic safety net.
- Output PTS validation is relaxed to allow equal PTS (only strictly backwards PTS is rejected), preserving tolerance for any residual edge cases where codec headers share a PTS with the following frame.

Test reference: `test_av1_sequence_header_is_inlined` in `savant_deepstream/encoders/tests/test_encoder.rs` verifies the invariant under Picasso's actual AV1 config (`Av1DgpuProps` with P1 preset / LowLatency tuning).

---

## 9. Test Execution Notes

- Use `#[serial]` (from `serial_test` crate) on all integration tests — GStreamer + CUDA state is process-global.
- Use `--test-threads=1` when running `cargo test`.
- `cargo test -p deepstream_encoders` for the encoder crate only.
- `gstreamer::init()` is idempotent but must be called before any GStreamer operation.
- `cuda_init(0)` must be called before creating any `NvEncoder`.

---

## 10. Jetson Memory Access Model

On Jetson, `NvBufSurfaceMemType::Default` maps to `SurfaceArray` (VIC-managed), which is NOT CUDA-addressable. The `SurfaceView::upload` method uses `cudaMemcpy2D` (device-to-device via EGL-CUDA mapping) on both platforms. For CPU upload paths, create a `SurfaceView` first — it resolves the CUDA pointer via `EglCudaMeta` on Jetson.
