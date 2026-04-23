# TensorRT engine caching

## Goal

The YOLOv11n TensorRT engine should be **built once** and **reused on every
subsequent run**, regardless of which GPU / TensorRT / DeepStream version
is present on the machine.

## Layout

Cached engines live under:

```
savant_perception/assets/cache/<model-name>/<platform-tag>/<engine-file>
```

- `<model-name>` — e.g. `yolo11n`.
- `<platform-tag>` — produced by
  `nvidia_gpu_utils::gpu_platform_tag(gpu_id)`.  Examples: `ada`,
  `ampere`, `orin_nano_8gb`, etc.  If the utility fails we fall back to
  `unknown`.
- `<engine-file>` — `yolo11n.onnx_b1_gpu<gpu_id>_fp16.engine` (matches
  DeepStream's own naming so nvinfer can consume it).

Helpers:

- `savant_perception::assets::engine_cache_platform_tag(gpu_id)` — returns
  the platform tag string.
- `savant_perception::assets::engine_cache_dir(model_name, gpu_id)` — creates
  (if missing) and returns the directory above.
- `savant_perception::cars_tracking::model::yolo11n_engine_cache_path(gpu_id)`
  — returns the full expected path.

## Why platform-specific?

TensorRT plan files are **not portable across GPU architectures, driver
versions, or TensorRT versions**.  When an incompatible engine is loaded
DeepStream logs:

```
ERROR: [TRT]: IRuntime::deserializeCudaEngine: ...
Platform specific tag mismatch detected.
TensorRT plan files are only supported on the target runtime platform
they were created on.
```

DeepStream reacts by **rebuilding** the engine (slow — minutes) next to
the ONNX file.  We then need to move that fresh engine into the cache
directory so the next run can reuse it.

## Replacement semantics

`deepstream_nvinfer::engine_cache::promote_built_engine(from, to)` was
changed from idempotent to **replacing**:

- `from` missing, `to` present — no-op (Ok(false)).
- `from` present — always overwrite `to` (Ok(true)).

This guarantees a freshly-built (possibly incompatible-with-old-cache)
engine replaces the stale one.  Combined with the platform-specific
directory above, this makes the cache self-healing across driver /
architecture changes without any manual intervention.

## Pitfalls

- **Do not hand-edit the cache filename** — DeepStream derives the
  expected engine name from the ONNX filename and infer config
  (batch-size, precision, gpu-id).  Changing the format breaks reuse.
- **Do not commit cached engines** — they are multi-MB binary blobs
  specific to one machine.  `.gitignore` should cover
  `savant_perception/assets/cache/`.
- If reuse mysteriously stops working, remove the whole
  `savant_perception/assets/cache/<model-name>/` tree and let the next run
  rebuild from scratch.  Then look for `Platform specific tag mismatch`
  in the logs to see what actually changed.
