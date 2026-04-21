# Codec Header Inlining (NvEncoder)

## What

`NvEncoder` automatically inlines encoder-emitted stream-level metadata
(e.g. AV1 `OBU_SEQUENCE_HEADER`) into the next user frame, so every
`EncodedFrame` returned from `pull_encoded` / `pull_encoded_timeout` /
`finish` corresponds to a submitted frame and has `frame_id = Some(id)`.

## Why

The AV1 encoder element `nvv4l2av1enc` behaves differently between
platforms at the raw GStreamer layer:

| Platform | nvv4l2av1enc backend | Sequence header emission |
|---|---|---|
| Jetson | MMAPI / V4L2 | Inlined into the first IDR buffer |
| dGPU | NVENC shim | Standalone ~32-byte buffer before the first IDR |

Without normalisation, dGPU consumers would see an extra `EncodedFrame`
with `frame_id = None` for the header, forcing every consumer (e.g.
Picasso's drain loop) to special-case it. Picasso logged an ERROR on
every AV1 source startup.

## How

Field and flow inside
[`savant_deepstream/encoders/src/encoder.rs`](../../savant_deepstream/encoders/src/encoder.rs):

- `NvEncoder::pending_codec_header: Option<Vec<u8>>` — the stash slot.
- `sample_to_frame`:
  1. Look up `frame_id` via the `pts_map` (existing logic).
  2. If `!is_user_frame && !is_intra_only && buf_size > 0`: this is a
     stream-level header. Copy / extend into `pending_codec_header` and
     return `Ok(None)`.
  3. For a user frame, after extracting `data`, drain
     `pending_codec_header.take()` and prepend it to `data`.
- `pull_encoded` / `pull_encoded_timeout` loop until they get a user
  frame or a truly empty sample, so stashing is transparent to callers.
- `finish` drains an orphan header (no user frame followed) with a
  `debug!` — benign; no consumer was waiting.

Intra-only codecs (JPEG / PNG / Raw) never emit stream-level headers
and are excluded from the stash path. A non-user-frame payload on those
codecs is surfaced with `frame_id = None` as before; Picasso warns.

## Test

See `test_av1_sequence_header_is_inlined` in
[`savant_deepstream/encoders/tests/test_encoder.rs`](../../savant_deepstream/encoders/tests/test_encoder.rs).
It uses the same `Av1DgpuProps` (P1 preset + LowLatency tuning) that
Picasso uses — the configuration under which a standalone 32-byte
header buffer was observed on Ada / Ampere dGPUs.

## Related KB

- [`caveats.md`](./caveats.md) §8 — top-level explanation and platform asymmetry.
- [`api.md`](./api.md) `EncodedFrame` table — describes when `frame_id` is `None`.
