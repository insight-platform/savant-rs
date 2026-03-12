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

⚠ JPEG, PNG, and Raw codecs have platform-neutral properties — no branching needed.
⚠ AV1 is dGPU-only. `EncoderProperties::from_pairs(Codec::Av1, Platform::Jetson, _)` returns `Err(UnsupportedCodec)`.

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

## 5. upload_to_surface Takes 5 Arguments

```rust
pub unsafe fn upload_to_surface(
    buf: &gst::Buffer,
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,    // <-- commonly forgotten 5th arg
) -> Result<(), NvBufSurfaceError>
```

⚠ The `channels` parameter was added for multi-format support. Use `4` for RGBA, `3` for RGB. Omitting it causes a compile error that's easy to misinterpret.

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

## 8. AV1 Codec Header Buffers

AV1 encoders emit a sequence header buffer before the first data frame, often with the same PTS. The output PTS validation is relaxed to allow equal PTS (only strictly backwards PTS is rejected). Codec header buffers are not in the pts_map and skip ordering validation.

---

## 9. Test Execution Notes

- Use `#[serial]` (from `serial_test` crate) on all integration tests — GStreamer + CUDA state is process-global.
- Use `--test-threads=1` when running `cargo test`.
- `cargo test -p deepstream_encoders` for the encoder crate only.
- `gstreamer::init()` is idempotent but must be called before any GStreamer operation.
- `cuda_init(0)` must be called before creating any `NvEncoder`.

---

## 10. Jetson Memory Access Model

On Jetson, `NvBufSurfaceMemType::Default` maps to `SurfaceArray` (VIC-managed), which is NOT CUDA-addressable. The `upload_to_surface` function handles this transparently using `NvBufSurfaceMap` → CPU write → `NvBufSurfaceSyncForDevice` → `NvBufSurfaceUnMap` on Jetson, vs direct `cuMemcpyHtoD_v2` on dGPU.
