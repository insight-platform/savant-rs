# NvInfer KB — Error Conditions

## RuntimeError from NvInfer engine

| Trigger | When |
|---|---|
| `NvInfer.__init__` with invalid config | GStreamer pipeline fails to link / PLAYING |
| `submit()` after `shutdown()` | Engine already stopped |
| `infer_sync()` after `shutdown()` | Engine already stopped |
| `shutdown()` twice | Second call raises |
| `infer_sync()` timeout | `operation_timeout_ms` deadline expires without result; pipeline enters terminal failed state (`PipelineFailed`) and must be recreated |
| `submit()` / `infer_sync()` after failed state | Pipeline previously entered failed state due to timeout; all calls raise `RuntimeError` with "pipeline failed" |
| `submit()` with consumed SharedBuffer | SharedBuffer already consumed |

## RuntimeError from output access

| Trigger | When |
|---|---|
| `ElementOutput.tensors` after output dropped | Arc guard is `None` |
| `BatchInferenceOutput.elements` after output dropped | Arc guard is `None` |
| `BatchInferenceOutput.buffer()` after output dropped | Arc guard is `None` |

Message: `"BatchInferenceOutput has been released"`

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
        engine.submit(buf)

def test_infer_after_shutdown():
    engine = make_engine()
    engine.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        engine.infer_sync(buf)
```

⚠ These tests still require GPU + DeepStream runtime because the engine
constructor (`NvInfer.__init__`) creates a real GStreamer pipeline.
