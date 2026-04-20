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
  params will be detectable as AVCC/HVCC. `demux_all()` returns
  `(Vec<DemuxedPacket>, Option<VideoInfo>)`; use `info.codec` when you need
  the codec.
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

- The `NvDecoder` VP9 pipeline uses `vp9parse` on **both** dGPU and
  Jetson:
    `appsrc → vp9parse → nvv4l2decoder`.
- The dGPU DeepStream `nvv4l2decoder` rejects bare `video/x-vp9` sink
  caps; it requires `width`, `height`, and usually `profile` /
  `chroma-format` / `bit-depth-luma` / `bit-depth-chroma`. Feeding bare
  caps via `identity` results in
  `pre_eventfunc_check: caps video/x-vp9 not accepted` and
  `streaming stopped, reason not-negotiated (-4)` — 0 frames decoded.
- `vp9parse` (from `gst-plugins-bad`, `videoparsersbad`) parses the
  uncompressed VP9 frame header and enriches the caps with all required
  fields. It also emits `alignment=super-frame` which both dGPU and
  Tegra NVDEC handle correctly.
- **Why Jetson also needs `vp9parse`**: Jetson `nvv4l2decoder` *will*
  accept bare `video/x-vp9` caps fed through `identity`, but the
  pre-parsed, super-frame-aligned packets produced by
  `Mp4Demuxer::new_parsed` / `Mp4Demuxer::demux_all_parsed` (which
  return `(Vec<DemuxedPacket>, Option<VideoInfo>)` and themselves use
  `vp9parse` internally) do not decode end-to-end
  through the bare-caps `identity` path — empirically 0 frames are
  emitted by the Tegra NVDEC. Running `vp9parse` again in the
  `NvDecoder` pipeline enriches the caps with the fields the Tegra
  decoder needs and produces frames correctly on Jetson Orin. This was
  introduced on 2026-04-15 together with the `jetson_orin` addition for
  VP9 assets in `manifest.json`.
- There is no `vp8parse` in upstream `gst-plugins-bad`, so VP8 remains on
  `identity` on both platforms. Dimensions are not mandatory for
  `nvv4l2decoder` VP8 sink caps in the versions we target.
- `Mp4Demuxer::new_parsed` already uses `vp9parse` internally for MP4
  VP9 sources; running `vp9parse` again in the decoder pipeline is
  idempotent for well-formed streams (it re-reads the uncompressed
  header and re-emits the same enriched caps).

## 11) NVDEC Bit-Depth Support Differs by Platform (H.264 gotcha)

- H.264 4:2:0 10-bit (High 10 profile) is **NOT** supported on dGPU
  NVDEC 4th gen (Turing) or 5th gen (Ampere, Ada). It fails at runtime
  with `NvV4l2VideoDec: Feature not supported on this GPU (Error Code: 801)`,
  `Failed to process frame`, `streaming stopped, reason error (-5)`.
- H.264 4:2:0 10-bit is **ALSO NOT** supported on Jetson Orin Tegra
  NVDEC. Orin's H.264 decode path is 8-bit only (Baseline/Main/High).
  A High10 stream fed to `nvv4l2decoder` on Orin fails with:
  ```
  NvMMLiteBlockCreate : Block : BlockType = 261 NVMMLITE_NVVIDEODEC
  <NvVideoBufferProcessing:NNNN> video_parser_parse Unsupported Codec
  ```
  and zero frames decoded. This is an empirical finding confirmed on
  JetPack 6, consistent with the Jetson Linux Multimedia API guide
  which lists only Baseline/Main/High (8-bit) for H.264 decode on Orin.
- H.264 4:2:0 10-bit **IS** supported on:
  - dGPU NVDEC 6th gen (Blackwell, e.g. RTX 5xxx / RTX PRO Blackwell).
  - Jetson Thor (Blackwell-class Tegra).
- For 10-bit coverage on Jetson Orin, use VP9 Profile 2 or HEVC Main10
  instead — both are supported by Orin's NVDEC.
- HEVC Main10 (4:2:0 10-bit) and VP9 Profile 2 (4:2:0 10-bit) are
  supported on all NVDEC generations we target (Turing and later,
  including Jetson Orin).
- HEVC 4:2:2 (8/10/12-bit) is also a split: Turing/Ampere/Ada = NO;
  Blackwell = YES. We don't test 4:2:2 yet; if added, gate by platform.
- Source: NVIDIA Video Encode / Decode Support Matrix
  (https://developer.nvidia.com/video-encode-decode-support-matrix)
  plus Jetson Linux Multimedia API Reference for Orin-specific codec
  coverage (the dGPU matrix alone is not sufficient for Jetson).
- The e2e test manifest (`savant_deepstream/decoders/assets/manifest.json`)
  encodes this via `supported_platforms`; H.264 10-bit entries list only
  `blackwell`, so integration tests skip them on Turing/Ampere/Ada and
  on Jetson Orin rather than failing with the cryptic platform-specific
  errors above.
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

### Custom downstream events and `GstVideoDecoder` drain

Empirical finding (verified on Jetson Orin, April 2026, via downstream
event probes on every pad of a live `appsrc → h264parse → nvv4l2decoder
→ appsink` pipeline):

> Serialized custom-downstream events (`GST_EVENT_CUSTOM_DOWNSTREAM`,
> e.g. the one built by `build_source_eos_event`) are queued by the
> `GstVideoDecoder` base class against the currently-parsed frame
> (`GstVideoCodecFrame::events` / `current_frame_events`) and are only
> released onto the decoder's src pad when that frame is pushed
> downstream. A real `GstEvent::Eos` triggers a drain of pending
> *frames* but does **not** flush pending *events* — any event still
> attached to a yet-to-arrive frame is silently discarded during the
> drain.

This is the **base-class contract**, so it affects every decoder that
builds on `GstVideoDecoder` — `nvv4l2decoder` on Jetson (Tegra) and
dGPU (DeepStream/V4L2), `avdec_*`, `vaapi*`, `d3d11*`, etc.

### Generic rescue mechanism in `GstPipeline`

Rather than patch each test / integration with trailing "carrier"
buffers (which is brittle and test-specific), the generic
`GstPipeline` runner installs a **probe pair** around the whole
processing chain that delivers rescue-eligible custom-downstream
events reliably, in-band, without any help from callers:

```text
    ┌────────┐   ENTER probe    ┌────────┐  ... any chain ...  ┌────────┐   EXIT probe    ┌──────────┐
    │ appsrc ├─ EVENT_DOWNSTREAM ─┤h264parse├─────────────────────┤ decoder├─ EVENT_DOWNSTREAM ─┤ appsink  │
    └────────┘  record pending    └────────┘   (may drop)        └────────┘  clear pending /   └──────────┘
                                                                              rescue on EOS
```

* The **entry probe** on `appsrc.src` records every rescue-eligible
  event on a shared `pending_rescue_events` list.
  "Rescue-eligible" means a *serialized* custom-downstream event whose
  structure name is in the `savant.*` namespace (see
  `runner::is_rescueable_event`). The namespace restriction keeps
  element-specific serialized events (`GstForceKeyUnit`, still-frame,
  etc.) out of the rescue path — those are meant for a specific
  element and re-injecting them on EOS would be semantically wrong.
* The **exit probe** on `appsink.sink`:
  - For normal events that arrive: matches them against the pending
    list by `gst::Event::seqnum()` (stable across `clone()` and across
    element boundaries for events that are "passing by") and removes
    them from the list.
  - For `GstEvent::Eos`: takes ownership of whatever is still in the
    pending list (those are the events the intermediate decoder
    swallowed on drain) and re-injects each one downstream via
    `pad.peer().push_event(event)`. The peer is the src pad of the
    last processing element before appsink, so `push_event` delivers
    the event downstream, where it re-enters the exit probe and is
    forwarded to the output channel the normal way. Only after the
    rescue does the `Eos` itself propagate.

This makes **"every rescue-eligible event injected at `appsrc` is
observed at `appsink` before `Eos`"** an invariant of the pipeline
runner, independent of which decoder / transform chain is
instantiated. In particular:

- `FlexibleDecoder::source_eos(src_id)` during an active decode
  delivers `SourceEos` to the callback on **both** Jetson and dGPU,
  whether or not any further buffers are submitted for the source,
  and whether the decoder is torn down by `graceful_shutdown` or by a
  codec-change-driven real EOS.
- When `source_eos` is called while `FlexibleDecoder` is in
  `Idle`/`Detecting` state, it is still emitted directly through the
  callback (bypassing GStreamer) on both platforms.

The `test_source_eos_between_codec_changes` integration test
therefore asserts `SourceEos` propagation unconditionally on every
platform, with no carrier-buffer hack and no
`is_jetson_kernel()` gate.
