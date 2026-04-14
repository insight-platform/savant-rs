# Test Patterns & Templates

## Runtime test pattern (`NvInfer`)

1. Build `NvInferConfig` and create `NvInfer::new(config)`.
2. Build an input batch (`SharedBuffer`) from `deepstream_buffers`.
3. Call `submit(shared, rois)`.
4. Pull results through `recv`/`recv_timeout` and match `NvInferOutput`.

Example skeleton:

```rust
nvinfer.submit(shared, Some(&rois))?;

match nvinfer.recv_timeout(Duration::from_secs(5))? {
    Some(NvInferOutput::Inference(output)) => {
        for elem in output.elements() {
            for t in &elem.tensors {
                if let Ok(vals) = t.as_f32s() {
                    // validate outputs
                }
            }
        }
    }
    Some(NvInferOutput::Error(e)) => panic!("nvinfer error: {e}"),
    _ => panic!("unexpected output"),
}
```

## Full-frame inference pattern

Pass `rois = None` to let runtime synthesize per-slot full-frame ROIs from actual slot dimensions.

## Batching operator pattern

1. Configure `NvInferBatchingOperatorConfig`.
2. Provide `BatchFormationCallback` and `OperatorResultCallback`.
3. Feed frames via `add_frame`.
4. Inside callback, process tensors from `OperatorInferenceOutput`.
5. Use `take_deliveries()` + `SealedDeliveries::unseal()` for downstream buffer handoff.

## Build/test commands

```bash
cargo test -p nvinfer
cargo bench -p nvinfer --bench nvinfer_bench
```
