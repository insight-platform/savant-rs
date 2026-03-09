# NvInfer KB — Test Patterns & Helpers

## Test file header

```python
import json
import os
import random
import threading
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest

try:
    from savant_rs.nvinfer import (
        NvInfer,
        NvInferConfig,
        Roi,
    )
    from savant_rs.deepstream import (
        DsNvNonUniformSurfaceBuffer,
        DsNvSurfaceBufferGenerator,
        DsNvUniformSurfaceBufferGenerator,
        Rect,
        TransformConfig,
        init_cuda,
    )
    HAS_DS = True
except ImportError:
    HAS_DS = False

pytestmark = pytest.mark.skipif(not HAS_DS, reason="DeepStream runtime not available")
```

## GPU runtime guard

Use at the top of test modules to skip the entire file when there is no
GPU/DeepStream runtime:

```python
pytestmark = pytest.mark.skipif(not HAS_DS, reason="DeepStream runtime not available")
```

For individual tests that need model assets:

```python
ASSETS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "savant_deepstream",
    "nvinfer",
    "assets",
)

@pytest.mark.skipif(
    not _has_assets() or not _has_model(), reason="Model assets missing"
)
def test_something():
    ...
```

## E2E lifecycle pattern (synchronous)

```python
def test_e2e():
    init_cuda(0)

    config = NvInferConfig(
        nvinfer_properties=age_gender_properties(),
        input_format="RGBA",
        input_width=1920,
        input_height=1080,
    )
    engine = NvInfer(config, callback=lambda output: None)

    try:
        # ... prepare buffer, ROIs ...
        output = engine.infer_sync(batch=gst_buffer, batch_id=1, rois=rois)

        assert output.batch_id == 1
        assert output.num_elements == expected_count

        for elem in output.elements:
            for tensor in elem.tensors:
                arr = tensor.as_numpy()
                # ... decode + validate ...
    finally:
        engine.shutdown()
```

## E2E lifecycle pattern (asynchronous callback)

```python
def test_e2e_async():
    init_cuda(0)

    result_holder: List = []
    done = threading.Event()

    def on_output(output):
        result_holder.append(output)
        done.set()

    config = NvInferConfig(
        nvinfer_properties=age_gender_properties(),
        input_format="RGBA",
        input_width=1920,
        input_height=1080,
    )
    engine = NvInfer(config, callback=on_output)

    try:
        engine.submit(batch=gst_buffer, batch_id=3, rois=rois)
        assert done.wait(timeout=30), "callback not invoked within 30 s"
        output = result_holder[0]
        # ... validate output ...
    finally:
        engine.shutdown()
```

## Helper: age_gender_properties

```python
def age_gender_properties() -> Dict[str, str]:
    d = ASSETS_DIR
    return {
        "gpu-id": "0",
        "gie-unique-id": "2",
        "net-scale-factor": "0.007843137254902",
        "offsets": "127.5;127.5;127.5",
        "onnx-file": os.path.join(d, "age_gender_mobilenet_v2_dynBatch.onnx"),
        "model-engine-file": os.path.join(
            d, "age_gender_mobilenet_v2_dynBatch.onnx_b32_gpu0_fp16.engine",
        ),
        "batch-size": "32",
        "network-mode": "2",
        "network-type": "100",
        "infer-dims": "3;112;112",
        "model-color-format": "0",
    }
```

## Helper: load face images

```python
from PIL import Image

def load_face_images() -> List[Tuple[str, np.ndarray]]:
    images_dir = os.path.join(ASSETS_DIR, "age_gender")
    entries = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    result = []
    for fname in entries:
        img = Image.open(os.path.join(images_dir, fname)).convert("RGBA")
        result.append((fname, np.array(img)))
    return result
```

## Helper: place_non_overlapping

**Important**: ROI coordinates must be even (aligned to 2 pixels).
`gstnvinfer.cpp` applies `GST_ROUND_UP_2` to `crop_rect_params->left` and
`->top`, rounding odd coordinates up by 1 pixel.  This shifts the crop
window relative to the actual object location, so the model receives
different pixels and produces different results.

```python
ALIGN = 2

def place_non_overlapping(
    faces: List[np.ndarray],
    canvas_w: int,
    canvas_h: int,
    rng: random.Random,
) -> List[Tuple[int, int, int, int]]:
    """Return (left, top, width, height) for each face, snapped to even coordinates."""
    placements: List[Tuple[int, int, int, int]] = []
    for face in faces:
        fh, fw = face.shape[:2]
        max_x = (canvas_w - fw) // ALIGN
        max_y = (canvas_h - fh) // ALIGN
        for attempt in range(10_000):
            x = rng.randint(0, max_x) * ALIGN
            y = rng.randint(0, max_y) * ALIGN
            overlaps = any(
                x < px + pw and x + fw > px and y < py + ph and y + fh > py
                for px, py, pw, ph in placements
            )
            if not overlaps:
                placements.append((x, y, fw, fh))
                break
            if attempt == 9_999:
                raise RuntimeError(f"Failed to place face {len(placements)} without overlap")
    return placements
```

## Helper: decode_age / decode_gender

```python
def decode_age(tensor_data: np.ndarray) -> float:
    """Expected-value decoding: model already outputs softmax probabilities."""
    probs = tensor_data.astype(np.float32)
    return float(np.dot(probs, np.arange(101)))

def decode_gender(tensor_data: np.ndarray) -> str:
    data = tensor_data.astype(np.float32)
    idx = int(np.argmax(data))
    return "male" if idx == 0 else "female"
```

## GPU upload: host numpy -> pitched device surface

Use the existing `nvbuf_as_gpu_mat` context manager from `savant_rs.deepstream`.
It wraps a raw CUDA device pointer as an OpenCV `GpuMat` and synchronises
the CUDA stream on exit.

```python
from savant_rs.deepstream import nvbuf_as_gpu_mat

src_buf, data_ptr, pitch = src_gen.acquire_surface_with_ptr(0)
with nvbuf_as_gpu_mat(data_ptr, pitch, W, H) as (gpu_mat, stream):
    gpu_mat.upload(np.ascontiguousarray(canvas), stream)
```

## Uniform batching pattern

```python
src_gen = DsNvSurfaceBufferGenerator(
    format="RGBA", width=W, height=H, gpu_id=0, pool_size=1,
)
src_buf, data_ptr, pitch = src_gen.acquire_surface_with_ptr(0)
_upload_canvas_to_gpu(canvas, data_ptr, pitch)

batched_gen = DsNvUniformSurfaceBufferGenerator(
    format="RGBA", width=W, height=H,
    max_batch_size=32, pool_size=2, gpu_id=0,
)
config = TransformConfig()
batch = batched_gen.acquire_batched_surface(config)
batch.fill_slot(src_buf, id=0)
batch.finalize()
gst_buffer = batch.as_gst_buffer()
```

## Nonuniform batching pattern

Nonuniform batching aggregates source buffer pointers without GPU
transform/copy.  After calling `as_gst_buffer()`, delete all source
references so the gst_buffer has refcount 1 (writable).

```python
src_gen = DsNvSurfaceBufferGenerator(
    format="RGBA", width=W, height=H, gpu_id=0, pool_size=1,
)
src_buf, data_ptr, pitch = src_gen.acquire_surface_with_ptr(0)
_upload_canvas_to_gpu(canvas, data_ptr, pitch)

batch = DsNvNonUniformSurfaceBuffer(max_batch_size=32, gpu_id=0)
batch.add(src_buf, id=0)
batch.finalize()
gst_buffer = batch.as_gst_buffer()
del batch, src_buf, src_gen  # ensure gst_buffer refcount == 1
```
