# DeepStream Pytest Report

**Date:** 2026-03-15
**Build:** `SAVANT_FEATURES=deepstream` debug wheel
**Total:** 805 | **Passed:** 805 | **Failed:** 0 | **Skipped:** 0 (0 fixable)

## Summary

All 805 tests passed with zero failures and zero skips. The decomposition of
`deepstream.rs` into submodules (`enums`, `config`, `buffer`, `surface_view`,
`generators`, `skia`, `functions`) introduced no regressions.

## Failure Groups

None.

## Fixable Skips

None.

## Unavoidable Skips

None.

## Passing Tests

<details>
<summary>All passing test files (23)</summary>

- `test_capi.py` — 3 passed
- `test_deepstream_batched_buffer.py` — 27 passed
- `test_deepstream_config.py` — 5 passed
- `test_deepstream_generator.py` — 3 passed
- `test_deepstream_gst_buffer.py` — 8 passed
- `test_deepstream_meta.py` — 3 passed
- `test_deepstream_non_uniform.py` — 4 passed
- `test_deepstream_surface_view.py` — 14 passed
- `test_deepstream_transform.py` — 4 passed
- `test_draw_spec.py` — 36 passed
- `test_etcd.py` — 3 passed
- `test_match.py` — 30 passed
- `test_nvinfer.py` — 4 passed
- `test_picasso_full.py` — 50 passed
- `test_picasso_pipeline.py` — 9 passed
- `test_picasso_spec.py` — 60 passed
- `test_primitives.py` — 2 passed
- `test_symbol_mapper.py` — 39 passed
- `test_video_frame.py` — 136 passed
- `test_video_object.py` — 321 passed
- `test_webserver.py` — 2 passed
- `test_zmq.py` — 42 passed

</details>
