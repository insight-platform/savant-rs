# NvInfer KB — Error Conditions

## RuntimeError from NvInfer engine

| Trigger | When |
|---|---|
| `NvInfer.__init__` with invalid config | GStreamer pipeline fails to link / PLAYING |
| `submit()` / `recv()` / `recv_timeout()` / `try_recv()` / `send_eos()` / `is_failed()` after `shutdown()` or `graceful_shutdown()` | Engine already stopped |
| `submit()` / `send_eos()` / `send_custom_downstream_event()` after `graceful_shutdown()` begins (before inner taken) | `RuntimeError` (`ShuttingDown`) |
| `shutdown()` twice | Second call raises |
| `recv()` / `recv_timeout()` / `try_recv()` after channel disconnect | `RuntimeError` (channel disconnected) |
| `submit()` after failed state | Pipeline previously entered terminal failed state; raises `RuntimeError` with "pipeline failed" |
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
Under normal usage (accessing tensors while holding `BatchInferenceOutput`
from `NvInferOutput.as_inference()`), this does not occur.

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

def test_recv_after_shutdown():
    engine = make_engine()
    engine.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        engine.recv()
```

⚠ These tests still require GPU + DeepStream runtime because the engine
constructor (`NvInfer.__init__`) creates a real GStreamer pipeline.
