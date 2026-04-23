# savant_perception — Knowledge Base

Notes for future developers working on the end-to-end DeepStream samples
in this crate (`cars-demo` under `examples/` and any future examples).

The samples wire up the full pipeline:

```
MP4 demux -> NVDEC decode -> NvInfer (YOLOv11n) -> NvTracker (NvDCF)
          -> Picasso (render + encode) -> MP4 mux
```

See the individual notes:

- `engine_cache.md` — platform-specific TensorRT engine caching (first-run
  build, later-run reuse).
- `submit_serialisation.md` — **the** concurrency bug behind both the
  `StrictPts` crash and the `PTS reset detected` / `muxer closed`
  flood on dGPU; fixed upstream in `savant_deepstream` (NvInfer +
  NvTracker + both batching operators).  Read this first before
  tweaking any ordering-related policy.
- `pts_reset_policy.md` — why we do **not** need to override
  Picasso's default `PtsResetPolicy` once the submit-ordering bug is
  fixed, and what would need to change if a future sample wants to
  segment output MP4s on deliberate PTS resets.
- `pipeline_termination.md` — how EOS propagates from the decoder all the
  way to the MP4 muxer and why the muxer thread must only exit on the
  **final** EOS (not on mid-stream Picasso encoder-recreation events).
