# Encoder Caveats

The `NvEncoder` contract is the direct symmetric counterpart of
`NvDecoder` (see `kb/decoders/caveats.md`). Many of the decoder
caveats -- PTS monotonicity, EOS semantics, custom-event rescue via
the `GstPipeline` probe pair, and test-runtime guidance -- apply
verbatim to encoders.

The items below call out the encoder-specific behaviour.

## 1) NVENC Availability

- **Orin Nano (8GB and 4GB):** does NOT have NVENC hardware.
  `nvidia_gpu_utils::has_nvenc(0)` returns `false`. H.264, HEVC, AV1
  codecs will fail with `NvencNotAvailable`.
- **Other Jetson models** (AGX Orin, Orin NX, Xavier): have NVENC.
- **Some datacenter dGPUs** (H100, A100, A30, B200): no NVENC.
  `has_nvenc()` queries NVML `encoder_capacity(H264)`.

Tests that exercise H.264/HEVC/AV1 must guard with:

```rust
if !has_nvenc() {
    eprintln!("NVENC not available -- skipping");
    return;
}
```

## 2) Platform-Specific Encoder Properties

The same GStreamer element (`nvv4l2h264enc`, `nvv4l2h265enc`) exposes
**different property sets** on dGPU vs Jetson. Using the wrong
property struct causes GStreamer element errors.

| Platform | `cfg!` | H264 props | HEVC props | Key settings |
|---|---|---|---|---|
| dGPU | `!cfg!(target_arch = "aarch64")` | `H264DgpuProps` | `HevcDgpuProps` | `preset-id` (DgpuPreset), `tuning-info-id` (TuningPreset) |
| Jetson | `cfg!(target_arch = "aarch64")` | `H264JetsonProps` | `HevcJetsonProps` | `preset-level` (JetsonPresetLevel), `maxperf-enable` |

JPEG, PNG, RawRgba, RawRgb, RawNv12 are platform-neutral -- no
branching required.

### 2a) `insert-sps-pps` is Jetson-only

`insert-sps-pps` is a **Jetson-only** property of the L4T
`nvv4l2h264enc` / `nvv4l2h265enc`.  The dGPU elements with the same
factory names (NVENC-backed) do not expose it, so
`set_property_from_str("insert-sps-pps", "1")` will panic with
`property 'insert-sps-pps' of type 'nvv4l2h264enc' not found`.

Rules inside `pipeline.rs`:

- gate the `insert-sps-pps` property on
  `nvidia_gpu_utils::is_jetson_kernel()` — never set it unconditionally;
- on dGPU, rely on `h264parse` / `h265parse` with
  `config-interval=-1` (already set unconditionally) to rewrite SPS/PPS
  in front of every keyframe in the byte-stream;
- keep runtime detection (`is_jetson_kernel`), not
  `cfg!(target_arch = "aarch64")`, so non-Jetson ARM hosts (Grace Hopper)
  still use the dGPU path.

The same gating pattern applies to any other L4T-only property you may
add in the future (e.g. `EnableTwopassCBR`, `num-Ref-Frames` —
already listed in `H264JetsonProps`).

## 3) Jetson `nvjpegenc` "Surface not registered" Hang

On Jetson, `nvjpegenc` requires surfaces pinned via its own
registration mechanism. Surfaces from the NvDS buffer pool are NOT
registered and cause `NVJPGGetSurfPinHandle: Surface not registered`
followed by a silent pipeline hang (appsrc blocks on backpressure).

The `NvEncoder` pipeline builder on aarch64 inserts an
`nvvideoconvert disable-passthrough=true` element before `nvjpegenc`
so surfaces are re-allocated through nvvideoconvert's pool (which is
registered with NVJPG). `nvv4l2h264enc` / `nvv4l2h265enc` do not need
this and accept the NvDS pool surfaces directly.

## 4) Jetson VIC Limitations for Raw Codecs

The Video Image Compositor (VIC) on Jetson does NOT support
NV12 → RGBA/RGB conversion. Using the default VIC compute path
triggers `RGB/BGR Format transformation is not supported by VIC`.

For raw pseudoencoders on aarch64, the pipeline builder sets
`compute-hw=1` on `nvvideoconvert` to force the GPU-based path:

```rust
#[cfg(target_arch = "aarch64")]
nvconv.set_property_from_str("compute-hw", "1");
```

NVENC codecs are unaffected: they use `NvBufSurfTransform` for format
conversion and skip `nvvideoconvert` entirely.

## 5) SurfaceView::upload

```rust
pub fn upload(
    &self,
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
) -> Result<(), NvBufSurfaceError>
```

Method on `SurfaceView` (not a free function). `channels` is commonly
forgotten -- `4` for RGBA, `3` for RGB, `1.5` semantics do not apply
(NV12 uses the dedicated upload path).

## 6) Buffer Pool Size = 1

The internal NVMM pool is configured with
`min_buffers=1, max_buffers=1`. The NVENC hardware may keep DMA-reading
from a buffer's GPU memory after GStreamer releases its reference. A
pool of 1 forces serialization and prevents stale-data artifacts.

Do NOT increase the pool size to "improve performance" -- the symptom
is intermittent visible corruption in encoded output that only appears
under sustained load.

## 7) B-Frame Enforcement

B-frames are categorically disabled:

1. `EncoderProperties` types do not expose B-frame fields.
2. `NvEncoder` pipeline construction explicitly sets all known B-frame
   GStreamer properties to `0`.
3. Runtime validation detects B-frame reordering in output (PTS moves
   backwards) and returns `OutputPtsReordered`.

## 8) Codec Header Stashing

Some encoders emit codec-level header buffers (e.g. AV1
`OBU_SEQUENCE_HEADER`, HEVC VPS/SPS/PPS) as *standalone* GStreamer
buffers that do not correspond to any submitted frame. Behaviour
varies by platform:

- **Jetson** (`nvv4l2av1enc` MMAPI/V4L2): inlines the sequence header
  into the first IDR's compressed buffer -- one `appsink` sample per
  submitted frame.
- **dGPU** (`nvv4l2av1enc` NVENC shim): emits the sequence header as a
  separate ~32-byte buffer before the first IDR -- two `appsink`
  samples for the first user frame.

`NvEncoder` normalises this to Jetson-style behaviour on both
platforms:

1. A sample whose PTS cannot be correlated with any submitted frame
   (for non-intra-only codecs: H.264 / HEVC / AV1) is treated as a
   stream-level header. Its bytes are stashed in a pending buffer
   (multiple header OBUs are concatenated in order).
2. The stash is invisible to callers; `recv*` loops internally skip
   header-only samples.
3. When the next user frame arrives, the stashed bytes are prepended
   to its `data`. The resulting `EncodedFrame` has
   `frame_id = Some(id)` and a bitstream that starts with
   `OBU_SEQUENCE_HEADER` (AV1) or the H.26x parameter sets.
4. An orphan stash at `graceful_shutdown` / `Drop` time (no user frame
   ever followed) is dropped with a `debug!` log -- benign, since no
   consumer is waiting on it.

Consequences for consumers:

- Every `EncodedFrame` returned from `NvEncoder` corresponds to a
  submitted frame. `frame_id` is always `Some` for non-intra-only
  codecs.
- Picasso's drain layer does not need to correlate header-only
  payloads. The "cannot correlate encoded payload" path is a
  defensive `warn!` that should never fire under the stashing
  contract.
- Output PTS validation allows *equal* PTS across consecutive samples
  (only strictly decreasing PTS is rejected) to tolerate any residual
  edge cases where a codec header shares a PTS with the following
  frame.

Test reference:
`test_av1_sequence_header_is_inlined` in
`savant_deepstream/encoders/tests/test_encoder.rs` verifies the
invariant under Picasso's AV1 configuration (`Av1DgpuProps` with
P1 preset / LowLatency tuning).

## 9) In-band Custom Event Delivery (mirror of decoder §12)

`NvEncoder` delegates pipeline lifecycle to `GstPipeline`, which
installs the same entry/exit probe pair as the decoder side. That
probe pair ensures `savant.*`-namespaced custom-downstream events --
including `source_eos` -- survive any encoder-internal drain that
would otherwise silently discard events attached to as-yet-unpushed
frames. See `kb/decoders/caveats.md` §12 for the mechanism; the
encoder side relies on it unchanged.

Practical consequence: `NvEncoder::send_source_eos(src)` during an
active encode delivers `SourceEos` on the callback before `Eos`, on
both dGPU and Jetson, regardless of whether teardown is initiated by
`graceful_shutdown` or by upstream EOS.

## 10) PTS Monotonicity Is Strict

`pts_ns` passed to `submit_frame` must be strictly increasing. Equal
or decreasing values produce `EncoderError::PtsReordered`. This
matches the decoder contract verbatim.

## 11) Test Runtime Notes

- Use `#[serial]` on integration tests -- GStreamer + CUDA state is
  process-global.
- Run with `--test-threads=1`:
  `cargo test -p savant-deepstream-encoders -- --test-threads=1`.
- `gstreamer::init()` is idempotent but must precede any GStreamer
  operation.
- `cuda_init(0)` must precede any `NvEncoder::new`.

## 12) Jetson Memory Access Model

On Jetson, `NvBufSurfaceMemType::Default` maps to `SurfaceArray`
(VIC-managed), which is NOT CUDA-addressable from userspace.
`SurfaceView::upload` uses `cudaMemcpy2D` (device-to-device via
EGL-CUDA mapping) on both platforms; on Jetson the mapping is
resolved through `EglCudaMeta` when the `SurfaceView` is constructed.
For CPU -> NVMM uploads, always go through a `SurfaceView` -- do not
attempt direct CPU writes to the `NvBufSurface` plane pointer.

## 13) Python Bindings Live in `savant_rs.deepstream`

The PyO3 encoder classes (`EncoderConfig`, `EncoderProperties`,
per-codec props, `Platform`, `RateControl`, etc.) are registered on
`savant_rs.deepstream`, mirroring the decoder bindings. They are NOT
exposed from `savant_rs.picasso`. Code that used to import from
`savant_rs.picasso` (e.g. `from savant_rs.picasso import
EncoderConfig, H264DgpuProps`) must be updated to
`from savant_rs.deepstream import EncoderConfig, H264DgpuProps`.
Picasso continues to own only Picasso-specific spec types
(`CodecSpec`, `GeneralSpec`, `SourceSpec`, `ObjectDrawSpec`,
`Callbacks`, `PicassoEngine`).
