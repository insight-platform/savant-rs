# Picasso Python API Reference

Module: `savant_rs.picasso`

## Import Guard (required at top of every test file)
```python
import pytest
_mod = pytest.importorskip("savant_rs.picasso")
if not hasattr(_mod, "PicassoEngine"):
    pytest.skip("deepstream feature disabled", allow_module_level=True)
```

---

## PicassoEngine
Central pipeline manager. Spawns per-source worker threads + watchdog.

```
SIG: __init__(general: GeneralSpec, callbacks: Callbacks) → None
```

### Methods
| Method | Signature | Notes |
|---|---|---|
| `set_source_spec` | `(source_id: str, spec: SourceSpec) → None` | Creates/replaces worker. ⚠ RuntimeError if shut down |
| `remove_source_spec` | `(source_id: str) → None` | Stops worker. ⚠ RuntimeError if shut down |
| `send_frame` | `(source_id: str, frame: VideoFrame, buf: Union[SurfaceView, SharedBuffer, int, Any], src_rect: Optional[Rect]=None) → None` | GPU. `buf` dispatch order: **PySurfaceView** → `__cuda_array_interface__` object (e.g. CuPy) → SharedBuffer/int (legacy). Optional per-frame crop. ⚠ RuntimeError if shut down, ValueError if buf is null |
| `send_eos` | `(source_id: str) → None` | Signals end-of-stream. ⚠ RuntimeError if shut down |
| `shutdown` | `() → None` | Idempotent. Releases GIL during join. |

### repr
- Running: `"PicassoEngine(running)"`
- After shutdown: `"PicassoEngine(shut_down)"`

---

## GeneralSpec
```
SIG: __init__(name: str = "picasso", idle_timeout_secs: int = 30, inflight_queue_size: int = 8, pts_reset_policy: Optional[PtsResetPolicy] = None) → None
```
- `name`: get/set, str. Optional instance name for logging and future extensibility. DEF: "picasso"
- `idle_timeout_secs`: get/set, u64. DEF: 30
- `inflight_queue_size`: get/set, int. Capacity of the per-worker message queue. Controls how many frames can be buffered between `send_frame` and the worker consuming them. Larger values absorb bursts but increase memory/latency. DEF: 8
- `pts_reset_policy`: get/set, PtsResetPolicy. Policy for handling non-monotonic (decreasing) PTS values. Always returns a value; DEF: `PtsResetPolicy.eos_on_decreasing_pts()`
- repr: `"GeneralSpec(name=..., idle_timeout_secs=N, inflight_queue_size=N)"`

---

## EvictionDecision
Factory statics only (no direct __init__):

| Factory | Meaning |
|---|---|
| `EvictionDecision.keep_for(secs: int)` | Keep source alive N more seconds |
| `EvictionDecision.terminate()` | Drain encoder (EOS) then stop worker |
| `EvictionDecision.terminate_immediately()` | Stop worker immediately, no drain |

repr includes variant name and value.

---

## Callbacks
```
SIG: __init__(
    on_encoded_frame: Optional[Callable[[OutputMessage], Any]] = None,
    on_bypass_frame: Optional[Callable[[OutputMessage], Any]] = None,
    on_render: Optional[Callable[[str, int, int, int, VideoFrame], Any]] = None,
    on_object_draw_spec: Optional[Callable[..., Optional[ObjectDraw]]] = None,
    on_gpumat: Optional[Callable[[str, VideoFrame, int, int, int, int, int], Any]] = None,
    on_eviction: Optional[Callable[[str], EvictionDecision]] = None,
    on_stream_reset: Optional[Callable[[str, StreamResetReason], Any]] = None,
) → None
```

All 7 slots: get/set as properties. All Optional, DEF: None.

### Callback Signatures
| Callback | Args | Return |
|---|---|---|
| `on_encoded_frame` | `(output: OutputMessage)` | None |
| `on_bypass_frame` | `(output: OutputMessage)` | None |
| `on_render` | `(source_id: str, fbo_id: int, width: int, height: int, frame: VideoFrame)` | None |
| `on_object_draw_spec` | `(source_id: str, object: BorrowedVideoObject, current_spec: Optional[ObjectDraw])` | `Optional[ObjectDraw]` |
| `on_gpumat` | `(source_id: str, frame: VideoFrame, data_ptr: int, pitch: int, width: int, height: int, cuda_stream: int)` | None |
| `on_eviction` | `(source_id: str)` | `EvictionDecision` |
| `on_stream_reset` | `(source_id: str, reason: StreamResetReason)` | None |

⚠ Callbacks execute on Rust worker threads with GIL acquired. Keep them fast.
⚠ on_eviction MUST return EvictionDecision. On error → defaults to Terminate.
⚠ on_stream_reset is informational — no return value required.
⚠ **Thread affinity**: PyO3 objects marked `!Send` (e.g. `Mp4Muxer` from
  `savant_gstreamer`) **cannot** be used inside callbacks — they panic if
  accessed from a thread other than the one that created them. Relay data
  to the owning thread via `queue.Queue` instead.

### Encode Pipeline Execution Order

When `CodecSpec.encode()` is used, the per-frame processing order is:

1. **GPU transform** — input NvBufSurface scaled/padded to encoder resolution
2. **Load into Skia FBO** — transformed pixels become the canvas background
3. **Draw spec rendering** — `ObjectDrawSpec` entries rendered as Skia overlays
4. **`on_render` callback** — fires AFTER draw spec; draws ON TOP of bboxes
5. **Copy Skia FBO → NvBufSurface** — composited result written back
6. **`on_gpumat` callback** — fires on the final NvBufSurface CUDA memory
7. **Hardware encode** — frame submitted to encoder

⚠ To place content **behind** draw-spec bboxes, pre-fill the input
NvBufSurface before calling `send_frame` (e.g. via `nvgstbuf_as_gpu_mat`).
⚠ `on_render` is for **overlays** (sidebar, HUD, watermarks), not backgrounds.
⚠ `on_gpumat` sees the fully-composited frame; modifications go directly to
the encoder.

repr: `"Callbacks(active=[on_encoded_frame, ...])"` listing non-None slots.

---

## CodecSpec
Factory statics only:

| Factory | Meaning | GPU |
|---|---|---|
| `CodecSpec.drop_frames()` | Discard frame | NOGPU |
| `CodecSpec.bypass()` | Pass-through, transform bboxes back | GPU |
| `CodecSpec.encode(transform: TransformConfig, encoder: EncoderConfig)` | Transform + render + encode | GPU |

### Properties (bool, read-only)
`is_drop`, `is_bypass`, `is_encode`

---

## SourceSpec
```
SIG: __init__(
    codec: Optional[CodecSpec] = None,        # DEF: CodecSpec.drop_frames()
    conditional: Optional[ConditionalSpec] = None,  # DEF: empty
    draw: Optional[ObjectDrawSpec] = None,     # DEF: empty
    font_family: str = "sans-serif",
    idle_timeout_secs: Optional[int] = None,
    use_on_render: bool = False,
    use_on_gpumat: bool = False,
    callback_order: CallbackInvocationOrder = CallbackInvocationOrder.SkiaGpuMat,
) → None
```

### Properties (get/set)
| Prop | Type | DEF |
|---|---|---|
| `codec` | CodecSpec | drop_frames() |
| `conditional` | ConditionalSpec | empty |
| `draw` | ObjectDrawSpec | empty |
| `font_family` | str | "sans-serif" |
| `idle_timeout_secs` | Optional[int] | None |
| `use_on_render` | bool | False |
| `use_on_gpumat` | bool | False |
| `callback_order` | CallbackInvocationOrder | SkiaGpuMat |

---

## ConditionalSpec
```
SIG: __init__(
    encode_attribute: Optional[tuple[str, str]] = None,
    render_attribute: Optional[tuple[str, str]] = None,
) → None
```
Both get/set. Tuple is `(namespace, attribute_name)`. None = unconditional.

---

## ObjectDrawSpec
```
SIG: __init__() → None  # empty spec
```

| Method | Signature |
|---|---|
| `insert` | `(namespace: str, label: str, draw: ObjectDraw) → None` |
| `lookup` | `(namespace: str, label: str) → Optional[ObjectDraw]` |
| `is_empty` | `() → bool` |
| `len` / `__len__` | `() → int` |

Keyed by `(namespace, label)` pair. Uses `ObjectDraw` from `savant_rs.draw_spec`.

---

## EncoderConfig
```
SIG: __init__(codec: Codec, width: int, height: int) → None
```

### Builder methods (return self for chaining)
| Method | Signature |
|---|---|
| `format` | `(fmt: VideoFormat) → EncoderConfig` |
| `fps` | `(num: int, den: int) → EncoderConfig` |
| `gpu_id` | `(id: int) → EncoderConfig` |
| `properties` | `(props: EncoderProperties) → EncoderConfig` |

### Properties (read-only getters after building)
| Prop | Type |
|---|---|
| `fps_num` | int |
| `fps_den` | int |
| `mem_type` | MemType |
| `encoder_params` | Optional[EncoderProperties] |

⚠ **Builder methods shadow property setters.** The builder methods `format()`,
`gpu_id()` are exposed with the same Python name as the underlying property.
At runtime the property setter is **read-only** — always use the builder
method call form:
```python
cfg = EncoderConfig(Codec.H264, 1280, 720)
cfg.format(VideoFormat.RGBA)   # ✅ builder call
cfg.gpu_id(0)                  # ✅ builder call
cfg.fps(30, 1)
cfg.properties(props)

# cfg.gpu_id = 0              # ❌ AttributeError: read-only
# cfg.format = VideoFormat.NV12  # ❌ AttributeError: read-only
```

---

## EncoderProperties
Factory statics only:

| Factory | Args |
|---|---|
| `EncoderProperties.h264_dgpu(props: H264DgpuProps)` | dGPU H.264 |
| `EncoderProperties.hevc_dgpu(props: HevcDgpuProps)` | dGPU HEVC |
| `EncoderProperties.h264_jetson(props: H264JetsonProps)` | Jetson H.264 |
| `EncoderProperties.hevc_jetson(props: HevcJetsonProps)` | Jetson HEVC |
| `EncoderProperties.jpeg(props: JpegProps)` | JPEG |
| `EncoderProperties.png(props: PngProps)` | PNG (CPU-based, lossless) |
| `EncoderProperties.av1_dgpu(props: Av1DgpuProps)` | dGPU AV1 |

---

## OutputMessage
Received in `on_encoded_frame` callback. Tagged union.

| Prop/Method | Type | Notes |
|---|---|---|
| `is_video_frame` | bool (getter) | |
| `is_eos` | bool (getter) | |
| `as_video_frame()` | VideoFrame | ⚠ RuntimeError if EOS |
| `as_eos()` | EndOfStream | ⚠ RuntimeError if VideoFrame |

### Accessing encoded bitstream data
The `VideoFrame` returned by `as_video_frame()` carries the encoded
bitstream in its `content` field:
```python
vf = output.as_video_frame()
if vf.content.is_internal():
    data: bytes = vf.content.get_data()   # raw H.264/HEVC/AV1/JPEG/PNG bytes
    pts_ns = vf.pts                       # nanoseconds (time_base = 1/1e9)
    dts_ns = vf.dts                       # Optional[int]
    duration_ns = vf.duration             # Optional[int]
```

---

## on_bypass_frame and OutputMessage
`on_bypass_frame` now receives `OutputMessage` (same shape as `on_encoded_frame`).
Bypass frames are delivered as `OutputMessage.as_video_frame()`; EOS for bypass
sources is delivered as `OutputMessage.as_eos()` through this callback (not
`on_encoded_frame`).

---

## PtsResetPolicy
Policy for handling non-monotonic (decreasing) PTS values.
Factory statics only (no direct __init__):

| Factory | Meaning |
|---|---|
| `PtsResetPolicy.eos_on_decreasing_pts()` | Emit synthetic EOS before recreating the encoder (default) |
| `PtsResetPolicy.recreate_on_decreasing_pts()` | Silently recreate the encoder without emitting EOS |

repr includes variant name.

---

## StreamResetReason
Reason the worker's encoder was reset. Passed to the `on_stream_reset` callback.

### Properties (read-only)
| Prop | Type | Notes |
|---|---|---|
| `last_pts_ns` | int | PTS of the last successfully accepted frame (nanoseconds) |
| `new_pts_ns` | int | PTS of the incoming frame that triggered the reset (nanoseconds) |

---

## CallbackInvocationOrder
Controls when the `on_gpumat` callback fires relative to Skia rendering.

| Variant | Meaning |
|---|---|
| `CallbackInvocationOrder.SkiaGpuMat` | Skia render then `on_gpumat` (default) |
| `CallbackInvocationOrder.GpuMatSkia` | `on_gpumat` then Skia render |
| `CallbackInvocationOrder.GpuMatSkiaGpuMat` | `on_gpumat` before **and** after Skia render |

| Method | Signature |
|---|---|
| `from_name` | `(name: str) → CallbackInvocationOrder` — create from string name (`SkiaGpuMat`, `GpuMatSkia`, `GpuMatSkiaGpuMat`) |

---

## SurfaceView
Unified GPU surface descriptor. Wraps either an NvBufSurface slot or a raw
CUDA pointer (e.g. from CuPy). Preferred `buf` argument for `send_frame`.

Module: `savant_rs.deepstream`

### Factory Methods
| Method | Signature | Notes |
|---|---|---|
| `from_buffer` | `(buf: Union[SharedBuffer, int], slot_index: int = 0, cuda_stream: int = 0) → SurfaceView` | Create a view of an NvBufSurface-backed buffer (SharedBuffer or raw `int` pointer). `slot_index` selects the slot in a batched NvBufSurface. |
| `from_cuda_array` | `(obj: Any) → SurfaceView` | Create from any object exposing `__cuda_array_interface__` (e.g. CuPy ndarray). Must be contiguous RGBA `uint8` on GPU. |

### Properties (read-only)
| Prop | Type | Notes |
|---|---|---|
| `data_ptr` | int | CUDA device pointer |
| `pitch` | int | Row stride in bytes |
| `width` | int | Width in pixels |
| `height` | int | Height in pixels |
| `gpu_id` | int | CUDA device ordinal |
| `channels` | int | Number of channels (e.g. 4 for RGBA) |
| `color_format` | int | Raw `NvBufSurfaceColorFormat` value |

### `__cuda_array_interface__`
`SurfaceView` exposes `__cuda_array_interface__` (v3), so it can be consumed
by CuPy, Numba, or any library that supports the protocol:
```python
import cupy as cp
view = SurfaceView.from_buffer(buf, slot_index=0)
arr = cp.asarray(view)   # zero-copy GPU array (H, W, C) uint8
```
