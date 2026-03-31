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
