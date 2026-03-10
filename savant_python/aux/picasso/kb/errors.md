# Error Conditions

## PicassoEngine

| Operation | Condition | Exception |
|---|---|---|
| `set_source_spec` | After `shutdown()` | `RuntimeError("engine is shut down")` |
| `remove_source_spec` | After `shutdown()` | `RuntimeError("engine is shut down")` |
| `send_frame` | After `shutdown()` | `RuntimeError("engine is shut down")` |
| `send_frame` | `buf` is null (empty `DsNvBufSurfaceGstBuffer` or `int(0)`) | `ValueError("buf_ptr is null")` |
| `send_eos` | After `shutdown()` | `RuntimeError("engine is shut down")` |
| `shutdown` | Multiple calls | No error (idempotent) |

## OutputMessage

| Method | Condition | Exception |
|---|---|---|
| `as_video_frame()` | Output is EOS | `RuntimeError` |
| `as_eos()` | Output is VideoFrame | `RuntimeError` |

## Enum.from_name

| Type | Condition | Exception |
|---|---|---|
| `Platform.from_name(name)` | Unknown name | `ValueError` |
| `RateControl.from_name(name)` | Unknown name | `ValueError` |
| All `*.from_name()` | Unknown name | `ValueError` |

## Negative Test Patterns

### Post-shutdown rejection
```python
engine = PicassoEngine(GeneralSpec(), Callbacks())
engine.shutdown()
with pytest.raises(RuntimeError, match="shut down"):
    engine.set_source_spec("x", SourceSpec())
with pytest.raises(RuntimeError, match="shut down"):
    engine.remove_source_spec("x")
with pytest.raises(RuntimeError):
    engine.send_eos("x")
```

### Invalid enum name
```python
with pytest.raises(ValueError):
    Platform.from_name("unknown")
```
