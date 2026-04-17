"""E2E tests for savant_rs.nvinfer — uniform and nonuniform batching."""

from __future__ import annotations

import ctypes
import json
import os
import random
import threading
import time
from typing import Dict, List, Tuple

import pytest

try:
    import cupy
    import numpy as np
    from savant_rs.nvinfer import (
        NvInfer,
        NvInferConfig,
        NvInferBatchingOperator,
        NvInferBatchingOperatorConfig,
        NvInferOutput,
        BatchFormationResult,
        Roi,
        RoiKind,
        SealedDeliveries,
    )
    from savant_rs.deepstream import (
        BufferGenerator,
        NonUniformBatch,
        SavantIdMetaKind,
        UniformBatchGenerator,
        SurfaceView,
        TransformConfig,
        VideoFormat,
        gpu_platform_tag,
        init_cuda,
        nvbuf_as_gpu_mat,
    )
    from savant_rs.primitives import (
        VideoFrame,
        VideoFrameContent,
        VideoFrameTranscodingMethod,
    )
    from savant_rs.primitives.geometry import RBBox

    HAS_DS = True
except ImportError:
    HAS_DS = False

pytestmark = pytest.mark.skipif(not HAS_DS, reason="DeepStream runtime not available")

ASSETS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "savant_deepstream",
    "nvinfer",
    "assets",
)

FRAME_W = 1920
FRAME_H = 1080
FACE_SZ = 112
AGE_TOLERANCE = 15.0


def _has_assets() -> bool:
    return os.path.isdir(os.path.join(ASSETS_DIR, "age_gender"))


def _has_model() -> bool:
    return os.path.isfile(
        os.path.join(ASSETS_DIR, "age_gender_mobilenet_v2_dynBatch.onnx")
    )


def age_gender_properties() -> Dict[str, str]:
    d = ASSETS_DIR
    return {
        "gpu-id": "0",
        "net-scale-factor": "0.007843137254902",
        "offsets": "127.5;127.5;127.5",
        "onnx-file": os.path.join(d, "age_gender_mobilenet_v2_dynBatch.onnx"),
        "model-engine-file": os.path.join(
            d,
            "age_gender_mobilenet_v2_dynBatch.onnx_b32_gpu0_fp16.engine",
        ),
        "batch-size": "32",
        "network-mode": "2",
    }


def load_face_images() -> List[Tuple[str, np.ndarray]]:
    from PIL import Image

    images_dir = os.path.join(ASSETS_DIR, "age_gender")
    entries = sorted(
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    result = []
    for fname in entries:
        img = Image.open(os.path.join(images_dir, fname)).convert("RGBA")
        arr = np.array(img)
        assert arr.shape[:2] == (FACE_SZ, FACE_SZ), f"{fname}: unexpected size"
        result.append((fname, arr))
    return result


ALIGN = 2


def place_non_overlapping(
    faces: List[np.ndarray],
    canvas_w: int,
    canvas_h: int,
    rng: random.Random,
) -> List[Tuple[int, int, int, int]]:
    """Return (left, top, width, height) for each face, snapped to ALIGN-pixel grid.

    gstnvinfer.cpp applies GST_ROUND_UP_2 to crop_rect_params left/top,
    shifting odd coordinates by 1 pixel and misaligning the crop window.
    Snapping placements to even pixels avoids this.
    """
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
                raise RuntimeError(
                    f"Failed to place face {len(placements)} without overlap"
                )
    return placements


def decode_age(tensor_data: np.ndarray) -> float:
    """Expected-value decoding: model already outputs softmax probabilities."""
    probs = tensor_data.astype(np.float32)
    return float(np.dot(probs, np.arange(101)))


def decode_gender(tensor_data: np.ndarray) -> str:
    data = tensor_data.astype(np.float32)
    idx = int(np.argmax(data))
    return "male" if idx == 0 else "female"


def tensor_to_numpy(tv) -> np.ndarray:
    """Zero-copy NumPy view of a TensorView's host memory."""
    if tv.host_ptr == 0 or tv.byte_length == 0:
        return np.empty(0, dtype=tv.numpy_dtype)
    buf = (ctypes.c_char * tv.byte_length).from_address(tv.host_ptr)
    return np.frombuffer(buf, dtype=tv.numpy_dtype)


def tensor_to_cupy(tv) -> cupy.ndarray:
    """Zero-copy CuPy view of a TensorView's device memory."""
    if tv.device_ptr == 0 or tv.byte_length == 0:
        return cupy.empty(0, dtype=tv.numpy_dtype)
    mem = cupy.cuda.UnownedMemory(tv.device_ptr, tv.byte_length, owner=tv)
    ptr = cupy.cuda.MemoryPointer(mem, 0)
    return cupy.ndarray(tv.dims.num_elements, dtype=tv.numpy_dtype, memptr=ptr)


def recv_batch_inference_output(engine, nvinfer_config, context: str):
    """Poll :meth:`NvInfer.recv_timeout` until ``BatchInferenceOutput`` or fail."""
    deadline = time.monotonic() + nvinfer_config.operation_timeout_ms / 1000.0
    while time.monotonic() < deadline:
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
        tick = min(100, remaining_ms)
        item = engine.recv_timeout(tick)
        if item is None:
            continue
        if item.is_inference:
            out = item.as_inference()
            assert out is not None
            return out
        if item.is_error:
            pytest.fail(f"{context} pipeline error: {item.error_message}")
        if item.is_eos:
            pytest.fail(f"{context} unexpected EOS: {item.eos_source_id}")
    pytest.fail(f"{context} operation timeout waiting for inference")


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not _has_assets() or not _has_model(), reason="Model assets missing"
)
def test_age_gender_e2e_real_images():
    """Mirror of test_age_gender.rs::test_age_gender_e2e_real_images."""
    init_cuda(0)

    gt_path = os.path.join(ASSETS_DIR, "age_gender", "ground_truth.json")
    if not os.path.isfile(gt_path):
        pytest.skip("ground_truth.json not found (run generate_age_gender_gt.py)")

    with open(gt_path) as f:
        gt: Dict = json.load(f)

    images = load_face_images()
    num_faces = len(images)
    assert num_faces > 0, "no face images found"
    for fname, _ in images:
        assert fname in gt, f"no GT entry for {fname}"

    rng = random.Random(42)
    placements = place_non_overlapping(
        [arr for _, arr in images], FRAME_W, FRAME_H, rng
    )

    # Build CPU composite canvas (RGBA, 1920x1080)
    canvas = np.zeros((FRAME_H, FRAME_W, 4), dtype=np.uint8)
    for (_, rgba), (x, y, fw, fh) in zip(images, placements):
        canvas[y : y + fh, x : x + fw] = rgba

    # Upload canvas to GPU surface
    src_gen = BufferGenerator(
        format="RGBA",
        width=FRAME_W,
        height=FRAME_H,
        gpu_id=0,
        pool_size=1,
    )
    src_buf = src_gen.acquire(id=0)
    view = SurfaceView.from_buffer(src_buf, cuda_stream=0)
    with nvbuf_as_gpu_mat(view.data_ptr, view.pitch, FRAME_W, FRAME_H) as (
        gpu_mat,
        stream,
    ):
        gpu_mat.upload(np.ascontiguousarray(canvas), stream)

    # Create batched surface with one 1920x1080 slot
    batched_gen = UniformBatchGenerator(
        format="RGBA",
        width=FRAME_W,
        height=FRAME_H,
        max_batch_size=32,
        pool_size=2,
        gpu_id=0,
    )
    config = TransformConfig()
    batch = batched_gen.acquire_batch(config, ids=[(SavantIdMetaKind.FRAME, 0)])
    batch.transform_slot(0, src_buf)
    batch.finalize()
    gst_buffer = batch.shared_buffer()
    del batch, view, src_buf, src_gen

    # Build ROIs
    rois: Dict[int, list] = {
        0: [
            Roi(i, RBBox.ltwh(float(x), float(y), float(fw), float(fh)))
            for i, (x, y, fw, fh) in enumerate(placements)
        ]
    }

    # Engine
    props = age_gender_properties()
    nvinfer_config = NvInferConfig(
        nvinfer_properties=props,
        input_format=VideoFormat.RGBA,
        model_width=FACE_SZ,
        model_height=FACE_SZ,
    )
    engine = NvInfer(nvinfer_config)

    try:
        engine.submit(batch=gst_buffer, rois=rois)
        output = recv_batch_inference_output(
            engine, nvinfer_config, "age_gender uniform"
        )

        assert output.num_elements == num_faces, (
            f"expected {num_faces} elements, got {output.num_elements}"
        )

        pass_count = 0
        for i, elem in enumerate(output.elements):
            fname = images[i][0]
            expected = gt[fname]

            assert elem.roi_id == i, f"{fname}: roi_id mismatch"
            assert len(elem.tensors) >= 2, (
                f"{fname}: expected >= 2 output tensors, got {len(elem.tensors)}"
            )

            age_tensor = next((t for t in elem.tensors if t.name == "age"), None)
            gender_tensor = next((t for t in elem.tensors if t.name == "gender"), None)
            assert age_tensor is not None, f"{fname}: missing 'age' tensor"
            assert gender_tensor is not None, f"{fname}: missing 'gender' tensor"

            age_np = tensor_to_numpy(age_tensor)
            gender_np = tensor_to_numpy(gender_tensor)

            age_cp = tensor_to_cupy(age_tensor).get()
            gender_cp = tensor_to_cupy(gender_tensor).get()
            np.testing.assert_array_equal(
                age_np, age_cp, err_msg=f"{fname}: age CPU/GPU mismatch"
            )
            np.testing.assert_array_equal(
                gender_np, gender_cp, err_msg=f"{fname}: gender CPU/GPU mismatch"
            )

            trt_age = decode_age(age_np)
            trt_gender = decode_gender(gender_np)

            age_diff = abs(trt_age - expected["age"])
            print(
                f"  {fname}: TRT age={trt_age:.1f} gender={trt_gender}  |  "
                f"GT age={expected['age']:.1f} gender={expected['gender']}  |  "
                f"age_diff={age_diff:.1f}"
            )

            assert age_diff < AGE_TOLERANCE, (
                f"{fname}: age diff {age_diff:.1f} exceeds tolerance {AGE_TOLERANCE} "
                f"(TRT={trt_age:.2f}, GT={expected['age']:.2f})"
            )
            assert trt_gender == expected["gender"], (
                f"{fname}: gender mismatch (TRT={trt_gender}, GT={expected['gender']})"
            )
            pass_count += 1

        print(f"\n  All {pass_count}/{num_faces} faces passed age/gender validation.")
    finally:
        engine.shutdown()


@pytest.mark.skipif(
    not _has_assets() or not _has_model(), reason="Model assets missing"
)
def test_age_gender_e2e_nonuniform_callback():
    """Nonuniform batch with submit + recv polling (pull-based API)."""
    init_cuda(0)

    gt_path = os.path.join(ASSETS_DIR, "age_gender", "ground_truth.json")
    if not os.path.isfile(gt_path):
        pytest.skip("ground_truth.json not found (run generate_age_gender_gt.py)")

    with open(gt_path) as f:
        gt: Dict = json.load(f)

    images = load_face_images()
    num_faces = len(images)
    assert num_faces > 0, "no face images found"
    for fname, _ in images:
        assert fname in gt, f"no GT entry for {fname}"

    rng = random.Random(42)
    placements = place_non_overlapping(
        [arr for _, arr in images], FRAME_W, FRAME_H, rng
    )

    # Build CPU composite canvas (RGBA, 1920x1080) — same as uniform test
    canvas = np.zeros((FRAME_H, FRAME_W, 4), dtype=np.uint8)
    for (_, rgba), (x, y, fw, fh) in zip(images, placements):
        canvas[y : y + fh, x : x + fw] = rgba

    # Upload canvas to GPU surface
    src_gen = BufferGenerator(
        format="RGBA",
        width=FRAME_W,
        height=FRAME_H,
        gpu_id=0,
        pool_size=1,
    )
    src_buf = src_gen.acquire(id=0)
    view = SurfaceView.from_buffer(src_buf, cuda_stream=0)
    with nvbuf_as_gpu_mat(view.data_ptr, view.pitch, FRAME_W, FRAME_H) as (
        gpu_mat,
        stream,
    ):
        gpu_mat.upload(np.ascontiguousarray(canvas), stream)

    # Assemble batch via NonUniformBatch (zero-copy add)
    batch = NonUniformBatch(gpu_id=0)
    src_view = SurfaceView.from_buffer(src_buf)
    batch.add(src_view)
    gst_buffer = batch.finalize(ids=[(SavantIdMetaKind.FRAME, 0)])
    del batch, src_view, src_buf, src_gen

    # Build ROIs
    rois: Dict[int, list] = {
        0: [
            Roi(i, RBBox.ltwh(float(x), float(y), float(fw), float(fh)))
            for i, (x, y, fw, fh) in enumerate(placements)
        ]
    }

    props = age_gender_properties()
    nvinfer_config = NvInferConfig(
        nvinfer_properties=props,
        input_format=VideoFormat.RGBA,
        model_width=FACE_SZ,
        model_height=FACE_SZ,
    )
    engine = NvInfer(nvinfer_config)

    try:
        engine.submit(batch=gst_buffer, rois=rois)
        output = recv_batch_inference_output(
            engine, nvinfer_config, "age_gender nonuniform"
        )

        assert output.num_elements == num_faces, (
            f"expected {num_faces} elements, got {output.num_elements}"
        )

        pass_count = 0
        for i, elem in enumerate(output.elements):
            fname = images[i][0]
            expected = gt[fname]

            assert elem.roi_id == i, f"{fname}: roi_id mismatch"
            assert len(elem.tensors) >= 2, (
                f"{fname}: expected >= 2 output tensors, got {len(elem.tensors)}"
            )

            age_tensor = next((t for t in elem.tensors if t.name == "age"), None)
            gender_tensor = next((t for t in elem.tensors if t.name == "gender"), None)
            assert age_tensor is not None, f"{fname}: missing 'age' tensor"
            assert gender_tensor is not None, f"{fname}: missing 'gender' tensor"

            age_np = tensor_to_numpy(age_tensor)
            gender_np = tensor_to_numpy(gender_tensor)

            age_cp = tensor_to_cupy(age_tensor).get()
            gender_cp = tensor_to_cupy(gender_tensor).get()
            np.testing.assert_array_equal(
                age_np, age_cp, err_msg=f"{fname}: age CPU/GPU mismatch"
            )
            np.testing.assert_array_equal(
                gender_np, gender_cp, err_msg=f"{fname}: gender CPU/GPU mismatch"
            )

            trt_age = decode_age(age_np)
            trt_gender = decode_gender(gender_np)

            age_diff = abs(trt_age - expected["age"])
            print(
                f"  {fname}: TRT age={trt_age:.1f} gender={trt_gender}  |  "
                f"GT age={expected['age']:.1f} gender={expected['gender']}  |  "
                f"age_diff={age_diff:.1f}"
            )

            assert age_diff < AGE_TOLERANCE, (
                f"{fname}: age diff {age_diff:.1f} exceeds tolerance {AGE_TOLERANCE} "
                f"(TRT={trt_age:.2f}, GT={expected['age']:.2f})"
            )
            assert trt_gender == expected["gender"], (
                f"{fname}: gender mismatch (TRT={trt_gender}, GT={expected['gender']})"
            )
            pass_count += 1

        print(f"\n  All {pass_count}/{num_faces} faces passed age/gender validation.")
    finally:
        engine.shutdown()


# ── SealedDeliveries guard behavior ──────────────────────────────────────


def _has_identity_onnx() -> bool:
    return os.path.isfile(os.path.join(ASSETS_DIR, "identity.onnx"))


def identity_properties() -> Dict[str, str]:
    d = ASSETS_DIR
    tag = gpu_platform_tag(0)
    engine_dir = os.path.join(d, "engines", tag)
    os.makedirs(engine_dir, exist_ok=True)
    props = {
        "gpu-id": "0",
        "net-scale-factor": "1.0",
        "onnx-file": os.path.join(d, "identity.onnx"),
        "model-engine-file": os.path.join(
            engine_dir, "identity.onnx_b16_gpu0_fp16.engine"
        ),
        "batch-size": "16",
        "network-mode": "2",
    }
    import platform

    if platform.machine() == "aarch64":
        props["scaling-compute-hw"] = "1"
    return props


@pytest.mark.skipif(not _has_identity_onnx(), reason="identity.onnx not found")
def test_sealed_deliveries_guard_behavior():
    """Verify SealedDeliveries blocks until OperatorInferenceOutput is dropped.

    Uses identity.onnx (3x16x16, batch-size 16) via NvInferBatchingOperator.
    The test implicitly verifies GIL release in unseal(): the callback runs on
    a Rust thread that acquires the GIL via Python::attach.  If unseal() held
    the GIL while blocking on the Condvar, the callback thread could not drop
    the OperatorInferenceOutput (which needs GIL for Drop), causing deadlock.
    A successful unseal() return proves GIL is properly released.

    The batching ``result_callback`` receives :class:`OperatorOutput`; this test
    unwraps the inference variant only.
    """
    init_cuda(0)

    W, H = 16, 16

    sealed_holder: List = []
    done = threading.Event()

    def batch_formation_callback(frames):
        ids = [(SavantIdMetaKind.FRAME, i) for i in range(len(frames))]
        rois = [RoiKind.full_frame() for _ in frames]
        return BatchFormationResult(ids=ids, rois=rois)

    def result_callback(op):
        assert op.is_inference, "expected OperatorOutput(Inference(...))"
        output = op.as_operator_inference_output()
        assert output is not None
        sealed = output.take_deliveries()
        assert sealed is not None, (
            "first take_deliveries() must return SealedDeliveries"
        )
        assert len(sealed) == 1
        assert not sealed.is_released(), (
            "seal should not be released while output alive"
        )

        second = output.take_deliveries()
        assert second is None, "second take_deliveries() must return None"

        result = sealed.try_unseal()
        assert result is None, "try_unseal should return None while still sealed"

        sealed_holder.append(sealed)
        done.set()

    nvinfer_config = NvInferConfig(
        nvinfer_properties=identity_properties(),
        input_format=VideoFormat.RGBA,
        model_width=W,
        model_height=H,
    )
    op_config = NvInferBatchingOperatorConfig(
        max_batch_size=1,
        max_batch_wait_ms=5000,
        nvinfer_config=nvinfer_config,
    )

    operator = NvInferBatchingOperator(
        config=op_config,
        batch_formation_callback=batch_formation_callback,
        result_callback=result_callback,
    )

    try:
        gen = BufferGenerator(
            format="RGBA",
            width=W,
            height=H,
            gpu_id=0,
            pool_size=1,
        )
        buf = gen.acquire(id=0)
        view = SurfaceView.from_buffer(buf, cuda_stream=0)
        view.memset(0)
        del view

        frame = VideoFrame(
            source_id="test",
            fps=(30, 1),
            width=W,
            height=H,
            content=VideoFrameContent.none(),
            transcoding_method=VideoFrameTranscodingMethod.Copy,
        )
        operator.add_frame(frame, buf)
        del buf, gen

        assert done.wait(timeout=30), "callback not invoked within 30 s"

        sealed = sealed_holder[0]
        assert sealed.is_released(), (
            "seal should be released after output dropped (callback returned)"
        )

        pairs = sealed.unseal()
        assert len(pairs) == 1
        frame_out, buffer_out = pairs[0]
        assert isinstance(frame_out, VideoFrame)
        print(
            f"  SealedDeliveries: frame source_id={frame_out.source_id}, "
            f"buffer type={type(buffer_out).__name__}"
        )
    finally:
        operator.shutdown()


@pytest.mark.skipif(not _has_identity_onnx(), reason="identity.onnx not found")
def test_nvinfer_graceful_shutdown_py() -> None:
    init_cuda(0)
    W, H = 12, 12
    nvinfer_config = NvInferConfig(
        nvinfer_properties=identity_properties(),
        input_format=VideoFormat.RGBA,
        model_width=W,
        model_height=H,
    )
    engine = NvInfer(nvinfer_config)
    src_gen = BufferGenerator(format="RGBA", width=W, height=H, gpu_id=0, pool_size=1)
    src_buf = src_gen.acquire(id=0)
    batch = NonUniformBatch(gpu_id=0)
    src_view = SurfaceView.from_buffer(src_buf)
    batch.add(src_view)
    gst_buffer = batch.finalize(ids=[(SavantIdMetaKind.FRAME, 0)])
    del batch, src_view, src_buf, src_gen

    engine.submit(batch=gst_buffer, rois=None)
    drained = engine.graceful_shutdown(2_000)
    assert isinstance(drained, list)
    assert any(isinstance(x, NvInferOutput) and x.is_inference for x in drained)
    with pytest.raises(RuntimeError, match="shut down"):
        engine.graceful_shutdown(1_000)
