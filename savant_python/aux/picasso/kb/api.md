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
| `send_frame` | `(source_id: str, frame: VideoFrame, buf_ptr: int) → None` | GPU. ⚠ RuntimeError if shut down, ValueError if buf_ptr==0 |
| `send_eos` | `(source_id: str) → None` | Signals end-of-stream. ⚠ RuntimeError if shut down |
| `shutdown` | `() → None` | Idempotent. Releases GIL during join. |

### repr
- Running: `"PicassoEngine(running)"`
- After shutdown: `"PicassoEngine(shut_down)"`

---

## GeneralSpec
```
SIG: __init__(idle_timeout_secs: int = 30) → None
```
- `idle_timeout_secs`: get/set, u64. DEF: 30
- repr: `"GeneralSpec(idle_timeout_secs=N)"`

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
    on_encoded_frame: Optional[Callable[[EncodedOutput], Any]] = None,
    on_bypass_frame: Optional[Callable[[BypassOutput], Any]] = None,
    on_render: Optional[Callable[[str, int, int, int, VideoFrame], Any]] = None,
    on_object_draw_spec: Optional[Callable[..., Optional[ObjectDraw]]] = None,
    on_gpumat: Optional[Callable[[str, VideoFrame, int, int, int, int], Any]] = None,
    on_eviction: Optional[Callable[[str], EvictionDecision]] = None,
) → None
```

All 6 slots: get/set as properties. All Optional, DEF: None.

### Callback Signatures
| Callback | Args | Return |
|---|---|---|
| `on_encoded_frame` | `(output: EncodedOutput)` | None |
| `on_bypass_frame` | `(output: BypassOutput)` | None |
| `on_render` | `(source_id: str, fbo_id: int, width: int, height: int, frame: VideoFrame)` | None |
| `on_object_draw_spec` | `(source_id: str, object: BorrowedVideoObject, current_spec: Optional[ObjectDraw])` | `Optional[ObjectDraw]` |
| `on_gpumat` | `(source_id: str, frame: VideoFrame, data_ptr: int, pitch: int, width: int, height: int)` | None |
| `on_eviction` | `(source_id: str)` | `EvictionDecision` |

⚠ Callbacks execute on Rust worker threads with GIL acquired. Keep them fast.
⚠ on_eviction MUST return EvictionDecision. On error → defaults to Terminate.

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
| `properties` | `(props: EncoderProperties) → EncoderConfig` |

### Properties (get/set)
| Prop | Type |
|---|---|
| `format` | VideoFormat |
| `fps_num` | int |
| `fps_den` | int |
| `gpu_id` | int |
| `mem_type` | MemType |
| `encoder_params` | Optional[EncoderProperties] |

⚠ Builder `.format()` shadows the property getter. Use property access after building.

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

## EncodedOutput
Received in `on_encoded_frame` callback. Tagged union.

| Prop/Method | Type | Notes |
|---|---|---|
| `is_video_frame` | bool (getter) | |
| `is_eos` | bool (getter) | |
| `as_video_frame()` | VideoFrame | ⚠ RuntimeError if EOS |
| `as_eos()` | EndOfStream | ⚠ RuntimeError if VideoFrame |

---

## BypassOutput
Received in `on_bypass_frame` callback.

| Prop | Type |
|---|---|
| `source_id` | str (getter) |
| `frame` | VideoFrame (getter) |
