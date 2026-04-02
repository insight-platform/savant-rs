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

- Decoder callbacks deliver `DecodedFrame.format == RGBA` for all supported codecs.
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
  (parameter sets live in `codec_data` sideband). Use `Mp4Demuxer::new()`
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

## 9) config-interval Property

- `h264parse` and `h265parse` are set with `config-interval=-1` in both
  `NvDecoder` pipeline and `Mp4Demuxer` parsed mode.
- This ensures SPS/PPS/VPS are re-injected into every IDR, which is
  critical for mid-stream decoder restarts and stream format detection.
