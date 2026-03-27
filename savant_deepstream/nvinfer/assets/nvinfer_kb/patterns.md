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

## Shared age-gender test helpers (`tests/common/age_gender_test_utils.rs`)

`place_non_overlapping`, `load_face_images` (with explicit face width/height),
`decode_age`, `decode_gender`, and `to_candle_tensor` live in this module so
`test_age_gender`, `test_roi_crop_fidelity`, and `test_oversized_batch` do not
duplicate them.

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

// Builder pattern:
let batched_gen = UniformBatchGenerator::builder(VideoFormat::RGBA, 640, 640, 4)
    .gpu_id(0)
    .build()
    .unwrap();

// Direct constructor (format, width, height, batch_size, pool_size, gpu_id, mem_type):
let batched_gen = UniformBatchGenerator::new(
    VideoFormat::RGBA, 640, 640, 4, 2, 0, NvBufSurfaceMemType::Default,
).unwrap();

let config = platform_transform_config();
let ids: Vec<SavantIdMetaKind> = (0..4).map(|i| SavantIdMetaKind::Frame(i as i64)).collect();
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
    let _slot = elem.slot_number;
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
candle-core = "0.9"
cudarc = { version = "0.17", features = ["cuda-12020"] }
half = "2.5"
criterion = { workspace = true }
env_logger = "0.11"
image = "0.25"
nvidia_gpu_utils = { path = "../nvidia_gpu_utils" }
rand = "0.9"
serde = { workspace = true }
serde_json = { workspace = true }
serial_test = { workspace = true }
```

---

## Build & Test

```bash
cargo test -p nvinfer
cargo bench -p nvinfer --bench nvinfer_bench
```

Use `#[serial]` when tests share GStreamer/CUDA state.

---

# Batching Operator Patterns

## NvInferBatchingOperatorConfig

```rust
use nvinfer::{NvInferBatchingOperatorConfig, NvInferConfig, ModelInputScaling};
use std::time::Duration;

let nvinfer_config = NvInferConfig::new(props, "RGBA", 112, 112)
    .gpu_id(0)
    .queue_depth(1)
    .scaling(ModelInputScaling::KeepAspectRatio);

let config = NvInferBatchingOperatorConfig {
    max_batch_size: 8,
    same_source_allowed: false,
    max_batch_wait: Duration::from_millis(100),
    nvinfer: nvinfer_config,
};
```

---

## Batching Operator with Callbacks

```rust
use nvinfer::*;
use std::sync::Arc;

let batch_formation: BatchFormationCallback = Arc::new(|frames| {
    let ids = frames.iter().enumerate()
        .map(|(i, _)| SavantIdMetaKind::Frame(i as i64))
        .collect();
    let rois = frames.iter().map(|_| RoiKind::FullFrame).collect();
    BatchFormationResult { ids, rois }
});

let result_callback: OperatorResultCallback = Box::new(|output| {
    for frame_output in output.frames() {
        let source = frame_output.frame.get_source_id();
        for elem in &frame_output.elements {
            // elem derefs to ElementOutput
            let roi_id = elem.roi_id;
            let slot = elem.slot_number;
            // Access tensors
            for t in &elem.tensors { /* ... */ }
        }
    }
});

let mut operator = NvInferBatchingOperator::new(
    config, batch_formation, result_callback,
)?;

operator.add_frame(frame, buffer)?;
operator.flush()?;
operator.shutdown()?;
```

---

## Coordinate Scaling in Result Callback

```rust
let result_callback: OperatorResultCallback = Box::new(|output| {
    for frame_output in output.frames() {
        for elem in &frame_output.elements {
            // Scale points from model space to frame coordinates
            let points = vec![(50.0f32, 50.0f32), (100.0, 100.0)];
            let frame_points = elem.scale_points(&points);

            // Scale LTWH boxes
            let ltwh_boxes = vec![[10.0f32, 20.0, 30.0, 40.0]];
            let frame_ltwh = elem.scale_ltwh(&ltwh_boxes);

            // Scale LTRB boxes
            let ltrb_boxes = vec![[10.0f32, 20.0, 40.0, 60.0]];
            let frame_ltrb = elem.scale_ltrb(&ltrb_boxes);

            // Scale rotated bounding boxes
            let rbboxes = vec![RBBox::new(50.0, 50.0, 20.0, 30.0, Some(45.0))];
            let frame_rbboxes = elem.scale_rbboxes(&rbboxes);

            // Get the scaler directly (Copy, useful for py.detach)
            let scaler = elem.coordinate_scaler();
            let (x, y) = scaler.scale_point(50.0, 50.0);
        }
    }
});
```

---

## CoordinateScaler Standalone

```rust
use nvinfer::{CoordinateScaler, ModelInputScaling};

// ROI at (100, 200), size 200x400, model input 100x200, Fill mode
let scaler = CoordinateScaler::new(
    100.0, 200.0,  // roi_left, roi_top
    200.0, 400.0,  // roi_w, roi_h
    100.0, 200.0,  // model_w, model_h
    ModelInputScaling::Fill,
);
// scale_x = 200/100 = 2, scale_y = 400/200 = 2
let (x, y) = scaler.scale_point(50.0, 100.0);
assert_eq!(x, 200.0);  // 100 + 50*2
assert_eq!(y, 400.0);  // 200 + 100*2
```

---

## Batch Formation with ROIs

```rust
let batch_formation: BatchFormationCallback = Arc::new(|frames| {
    let ids = frames.iter().enumerate()
        .map(|(i, _)| SavantIdMetaKind::Frame(i as i64))
        .collect();
    let rois = frames.iter().map(|frame| {
        RoiKind::Rois(vec![
            Roi { id: 1, bbox: RBBox::new(160.0, 120.0, 100.0, 100.0, None) },
            Roi { id: 2, bbox: RBBox::new(300.0, 200.0, 80.0, 60.0, None) },
        ])
    }).collect();
    BatchFormationResult { ids, rois }
});
```

---

## OperatorElement Unit Tests (no GPU)

```rust
use nvinfer::batching_operator::scaler::CoordinateScaler;
use nvinfer::ModelInputScaling;

#[test]
fn fill_identity() {
    let s = CoordinateScaler::new(0.0, 0.0, 100.0, 100.0, 100.0, 100.0, ModelInputScaling::Fill);
    let (x, y) = s.scale_point(50.0, 50.0);
    assert!((x - 50.0).abs() < 1e-4);
    assert!((y - 50.0).abs() < 1e-4);
}

#[test]
fn keep_ar_sym_with_offset() {
    let s = CoordinateScaler::new(
        100.0, 200.0, 200.0, 100.0, 100.0, 100.0,
        ModelInputScaling::KeepAspectRatioSymmetric,
    );
    // s=0.5, inv_s=2, pad_x=0, pad_y=25
    // offset_y = 200 - 25*2 = 150
    let (x, y) = s.scale_point(0.0, 25.0);
    assert!((x - 100.0).abs() < 1e-4);
    assert!((y - 200.0).abs() < 1e-4);
}
```
