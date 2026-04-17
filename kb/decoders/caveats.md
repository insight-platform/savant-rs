# Decoder Caveats

## 1) NVDEC Availability

- V4L2 decode paths (`nvv4l2decoder`) need NVIDIA decode support and plugin
  availability.
- Guard NVDEC tests with runtime checks.

## 2) JPEG Backend Selection Matters

- `JpegDecoderConfig::gpu()` uses `nvjpegdec` (requires plugin/hardware path).
- `JpegDecoderConfig::cpu()` uses Rust `image` crate decode (no `nvjpegdec` dependency).

## 3) PNG and JPEG CPU Do Not Use GStreamer Decode

- PNG and JPEG CPU are decoded via Rust `image` crate from packet bytes.
- Invalid PNG/JPEG CPU bytes fail immediately in `submit_packet` with `BufferError`
  (no deferred pipeline event expected).

## 4) Output Is Always RGBA to Callers

- Decoder `recv*` outputs deliver `DecodedFrame.format == RGBA` for all supported codecs.
- Conversion/scaling into RGBA is done via upload + transform into the provided pool.

## 5) Pool Pressure

- Decoder output pools are finite.
- Holding decoded buffers too long can stall throughput.

## 6) PTS Monotonicity Is Strict

- `pts_ns` passed to `submit_packet` must be strictly increasing.
- Equal or decreasing values produce `DecoderError::PtsReordered`.

## 7) Test Runtime Notes

- Use `#[serial]` and `--test-threads=1` for integration tests.
- Initialize CUDA and GStreamer before creating decoders in test processes.

## 8) Stream Format Detection

- `detect_stream_config` only supports `Codec::H264` and `Codec::Hevc`.
  All other codecs return `None`.
- For length-prefixed (AVCC/HVCC) detection to succeed, the access unit
  must contain in-band parameter sets: SPS + PPS for H264, VPS + SPS + PPS
  for HEVC. Non-IDR frames without parameter sets return `None`.
- MP4 containers may strip in-band parameter sets from non-keyframe packets
  (parameter sets live in `codec_data` sideband). Use `Mp4Demuxer::demux_all()`
  (no parser) to get raw container packets; only keyframe AUs with in-band
  params will be detectable as AVCC/HVCC.
- NAL parsing depends on `cros-codecs` crate. The `Header` trait must be
  in scope for `.len()` on NALU headers (`use cros_codecs::codec::h264::nalu::Header;`).

### `is_random_access_point`

- Single access unit only: does **not** accumulate VPS/SPS/PPS from earlier
  packets. If parameter sets are in a prior AU, this returns `false` even
  for a keyframe AU.
- HEVC IRAP uses `NaluType::is_irap()` (IDR, CRA, BLA, reserved IRAP types).
- VP8/VP9/AV1 always `false`; intra codecs (JPEG, PNG, raw) are `true` when
  non-empty.

> **Note:** `RawNv12` is handled by `is_random_access_point()` for completeness 
> but is NOT a supported `DecoderConfig` variant. It cannot be used to 
> instantiate an `NvDecoder`.

## 9) config-interval Property

- `h264parse` and `h265parse` are set with `config-interval=-1` in both
  `NvDecoder` pipeline and `Mp4Demuxer` parsed mode.
- This ensures SPS/PPS/VPS are re-injected into every IDR, which is
  critical for mid-stream decoder restarts and stream format detection.

## 10) VP9 Parser Is Platform-Aware

- The `NvDecoder` VP9 pipeline picks its parser based on
  `nvidia_gpu_utils::is_jetson_kernel()`:
  - **dGPU (Turing / Ampere / Ada / Blackwell)**:
    `appsrc → vp9parse → nvv4l2decoder`.
  - **Jetson (Tegra) NVDEC**:
    `appsrc → identity → nvv4l2decoder` (preserves the pre-existing
    known-good behaviour).
- The dGPU DeepStream `nvv4l2decoder` rejects bare `video/x-vp9` sink
  caps; it requires `width`, `height`, and usually `profile` /
  `chroma-format` / `bit-depth-luma` / `bit-depth-chroma`. Feeding bare
  caps via `identity` results in
  `pre_eventfunc_check: caps video/x-vp9 not accepted` and
  `streaming stopped, reason not-negotiated (-4)` — 0 frames decoded.
- `vp9parse` (from `gst-plugins-bad`, `videoparsersbad`) parses the
  uncompressed VP9 frame header and enriches the caps with all required
  fields. It also emits `alignment=super-frame` which dGPU NVDEC handles
  correctly.
- Jetson NVDEC tolerates bare `video/x-vp9` caps; `vp9parse` was not
  validated on Jetson at the time of the gating, so we deliberately do
  not switch Jetson to `vp9parse` to avoid a drive-by regression.
- There is no `vp8parse` in upstream `gst-plugins-bad`, so VP8 remains on
  `identity` on both platforms. Dimensions are not mandatory for
  `nvv4l2decoder` VP8 sink caps in the versions we target.
- `Mp4Demuxer::new_parsed` already uses `vp9parse` internally for MP4
  VP9 sources, independent of this decoder-side gating.

## 11) NVDEC Bit-Depth Support Differs by Platform (H.264 gotcha)

- H.264 4:2:0 10-bit (High 10 profile) is **NOT** supported on dGPU
  NVDEC 4th gen (Turing) or 5th gen (Ampere, Ada). It fails at runtime
  with `NvV4l2VideoDec: Feature not supported on this GPU (Error Code: 801)`,
  `Failed to process frame`, `streaming stopped, reason error (-5)`.
- H.264 4:2:0 10-bit **IS** supported on:
  - dGPU NVDEC 6th gen (Blackwell, e.g. RTX 5xxx / RTX PRO Blackwell).
  - Jetson Orin Tegra NVDEC.
- HEVC Main10 (4:2:0 10-bit) and VP9 Profile 2 (4:2:0 10-bit) are
  supported on all NVDEC generations we target (Turing and later).
- HEVC 4:2:2 (8/10/12-bit) is also a split: Turing/Ampere/Ada = NO;
  Blackwell = YES. We don't test 4:2:2 yet; if added, gate by platform.
- Source: NVIDIA Video Encode / Decode Support Matrix
  (https://developer.nvidia.com/video-encode-decode-support-matrix).
  Per-generation NVDEC codec × bit-depth × chroma subsampling rows.
- The e2e test manifest (`savant_deepstream/decoders/assets/manifest.json`)
  encodes this via `supported_platforms`; H.264 10-bit entries list only
  `blackwell` and `jetson_orin`, so integration tests skip them on
  Turing/Ampere/Ada rather than failing with the cryptic error-801.
- When adding new assets, check the matrix first and restrict
  `supported_platforms` accordingly; add an `unsupported_note` key with
  the rationale so future readers don't re-expand the list.

## 12) dGPU `nvv4l2decoder` Holds Tail Frames Until Flush

- **Symptom**: a test submits N frames, waits for N callback invocations,
  and times out with `got M < N` (typically M = N − 2..4).  The same test
  passes on Jetson Orin.
- **Root cause**: dGPU NVDEC (via DeepStream `nvv4l2decoder`) keeps a
  tail of decoded frames in its output queue.  They are released to
  downstream (and thus to our callback) only when one of the following
  happens:
  - more input arrives that pushes them out;
  - the decoder receives a **real** GStreamer `GstEvent::Eos` (which
    triggers a flush);
  - the pipeline is taken to `NULL` (which also drains).

  Jetson NVDEC is shallower and typically releases frames as soon as
  they're decoded, which is why Jetson-targeted tests don't need an
  explicit flush.

- **Important — `source_eos` is NOT a flush trigger**.  It is a custom
  downstream event used by the multistream pipeline to mark the end of
  one logical source while keeping the GStreamer pipeline alive for
  other sources.  It does **not** propagate as a real EOS to
  `nvv4l2decoder` and does **not** cause a drain.
- The only public `FlexibleDecoder` API that drains the decoder is
  `graceful_shutdown()`.  It sends a real EOS, waits for the pipeline to
  deliver every pending frame through the user callback, and then tears
  the decoder down.  The underlying `NvDecoder` exposes the same via
  `send_eos()` + `recv_timeout` loop.

### Test idiom for `test_flexible_decoder_real.rs` (and similar)

Do **not** rely on `collector.wait_for_frames(...)` after `submit(...)`
for correctness assertions.  The correct shape is:

```rust
let submitted = submit_access_units(&dec, &aus, entry, n, 0);

// Real EOS + synchronous drain via callback.
dec.graceful_shutdown().unwrap();

assert_eq!(collector.frame_count(), submitted.len());
```

For codec-change tests, the internal drain happens automatically when
`FlexibleDecoder` retires the old `NvDecoder` during `activate()`, so
per-phase `wait_for_frames` is also unnecessary — a single final
`graceful_shutdown()` before the aggregate assertion is enough:

```rust
submit_access_units(&dec, &phase1_aus, ...);
submit_access_units(&dec, &phase2_aus, ...);  // triggers phase-1 drain
collector.wait_for(|o| matches!(o, CollectedOutput::ParameterChange { .. }), TIMEOUT);
dec.graceful_shutdown().unwrap();             // drains phase-2 tail
assert_eq!(collector.frame_count(), all_uuids.len());
```

For tests that also exercise `source_eos`, keep the
`collector.wait_for(SourceEos, ...)` assertion (it validates the custom
event path) but still rely on `graceful_shutdown()` for the
frame-count drain.

### Additional dGPU limitation: custom events do not survive `nvv4l2decoder`

The DeepStream dGPU `nvv4l2decoder` wraps a V4L2 driver and does **not**
preserve arbitrary custom downstream events (including the one built by
`build_source_eos_event`) across its src pad.  The event is queued
behind buffers in the decoder's input queue, and when the decoder
flushes (real EOS or shutdown), the event is dropped rather than
forwarded to the appsink.  Consequences:

- On dGPU, `FlexibleDecoder::source_eos(src_id)` called while a decoder
  is *active* has no observable effect at the callback level:
  `CollectedOutput::SourceEos` never arrives.
- On Jetson Orin, the Tegra `nvv4l2decoder` preserves the custom event,
  so the callback does observe `SourceEos`.
- When `source_eos` is called while `FlexibleDecoder` is in
  `Idle`/`Detecting` state, it is emitted directly through the callback
  (bypassing GStreamer) on both platforms and is fully reliable.

The `test_source_eos_between_codec_changes` integration test gates its
`SourceEos` assertion on `nvidia_gpu_utils::is_jetson_kernel()` for
this reason.  If you need a reliable per-source EOS marker on dGPU, use
the `Idle` path (e.g. between sessions, after `graceful_shutdown`) or
implement the marker at the application layer above the decoder.
