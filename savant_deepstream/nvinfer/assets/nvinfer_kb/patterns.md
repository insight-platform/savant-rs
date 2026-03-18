# Test Patterns & Templates

## Test Categories

All integration tests require GPU (CUDA + GStreamer + DeepStream).
Use `common::init()` for one-time setup.

---

## Common Init (tests/common/mod.rs)

```rust
pub fn init() {
    INIT.call_once(|| {
        let _ = env_logger::try_init();
        gst::init().unwrap();
        cuda_init(0).expect("Failed to initialize CUDA - is a GPU available?");
    });
}
```

---

## Model Property Helpers

Tests use `identity_properties()`, `age_gender_properties()`, etc. from
`tests/common/mod.rs`. These return `HashMap<String, String>` with absolute
paths to ONNX and engine files under `nvinfer/assets/`.

Jetson scaling is injected automatically: `scaling-compute-hw=1` on aarch64.

---

## NvInferConfig Builder

```rust
let props = identity_properties();
let config = NvInferConfig::new(props, "RGBA", 12, 12)
    .gpu_id(0)
    .queue_depth(0)
    .meta_clear_policy(MetaClearPolicy::Before);
```

For heterogeneous batches (variable slot dimensions):
```rust
let config = NvInferConfig::new_flexible(props, "RGBA");
// Must supply explicit rois for every slot
```

---

## Build Uniform Batch for infer_sync

```rust
let src_gen = BufferGenerator::builder(VideoFormat::RGBA, 640, 480)
    .gpu_id(0)
    .build()
    .unwrap();
let batched_gen = UniformBatchGenerator::builder(VideoFormat::RGBA, 640, 640, 4)
    .gpu_id(0)
    .build()
    .unwrap();

let config = platform_transform_config();
let ids: Vec<SavantIdMetaKind> = (0..4).map(SavantIdMetaKind::Frame).collect();
let mut batch = batched_gen.acquire_batch(config, ids).unwrap();

for i in 0..4u32 {
    let src = src_gen.acquire(Some(i as i64)).unwrap();
    let view = SurfaceView::from_buffer(&src, 0).unwrap();
    batch.transform_slot(i, &view, None).unwrap();
}
batch.finalize().unwrap();
let shared = batch.shared_buffer();
```

---

## infer_sync with ROIs

```rust
let mut rois = HashMap::new();
rois.insert(0, vec![
    Roi { id: 42, bbox: RBBox::ltwh(10.0, 10.0, 50.0, 50.0).unwrap() },
]);
rois.insert(1, vec![
    Roi { id: 1, bbox: RBBox::ltwh(0.0, 0.0, 100.0, 100.0).unwrap() },
]);

let output = nvinfer.infer_sync(shared, Some(&rois)).unwrap();
assert_eq!(output.num_elements(), 2);
```

---

## Async submit with Callback

```rust
let (tx, rx) = mpsc::channel();
let callback: InferCallback = Box::new(move |output| {
    let _ = tx.send(output);
});

let nvinfer = NvInfer::new(config, callback).unwrap();
nvinfer.submit(shared, None).unwrap();

let output = rx.recv_timeout(Duration::from_secs(30)).unwrap();
```

---

## TensorView Consumption

```rust
for elem in output.elements() {
    let frame_id = elem.frame_id;
    let roi_id = elem.roi_id;
    for t in &elem.tensors {
        match t.data_type {
            DataType::Float => {
                let slice = unsafe { t.as_slice::<f32>() };
                // process slice
            }
            DataType::Half => { /* ... */ }
            _ => {}
        }
    }
}
```

---

## Cargo.toml Dev Dependencies

```toml
[dev-dependencies]
env_logger = "0.11"
serial_test = { workspace = true }
nvidia_gpu_utils = { path = "../nvidia_gpu_utils" }
```

---

## Build & Test

```bash
cargo test -p nvinfer
cargo bench -p nvinfer --bench nvinfer_bench
```

Use `#[serial]` when tests share GStreamer/CUDA state.
