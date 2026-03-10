# NvInfer KB — Error Conditions

## RuntimeError from NvInfer engine

| Trigger | When |
|---|---|
| `NvInfer.__init__` with invalid config | GStreamer pipeline fails to link / PLAYING |
| `submit()` after `shutdown()` | Engine already stopped |
| `infer_sync()` after `shutdown()` | Engine already stopped |
| `shutdown()` twice | Second call raises |
| `infer_sync()` timeout | 30 s deadline expires without result |
| `submit()` with `batch_id = 2**64 - 1` | Maps to `GST_CLOCK_TIME_NONE` |

## RuntimeError from output access

| Trigger | When |
|---|---|
| `TensorView.as_bytes()` after output dropped | Arc guard is `None` |
| `TensorView.as_numpy()` after output dropped | Arc guard is `None` |
| `ElementOutput.tensors` after output dropped | Arc guard is `None` |
| `BatchInferenceOutput.elements` after output dropped | Arc guard is `None` |

These arise when a user stores a child reference (`ElementOutput`,
`TensorView`) beyond the lifetime of the parent `BatchInferenceOutput`.
Under normal usage (accessing tensors inside the callback or after
`infer_sync`), this does not occur.

## Negative-path test patterns

```python
def test_shutdown_twice():
    engine = make_engine()
    engine.shutdown()
    with pytest.raises(RuntimeError, match="already shut down"):
        engine.shutdown()

def test_submit_after_shutdown():
    engine = make_engine()
    engine.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        engine.submit(buf, batch_id=1)

def test_infer_after_shutdown():
    engine = make_engine()
    engine.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        engine.infer_sync(buf, batch_id=1)
```

⚠ These tests still require GPU + DeepStream runtime because the engine
constructor (`NvInfer.__init__`) creates a real GStreamer pipeline.
