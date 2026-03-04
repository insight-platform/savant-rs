"""Tests for SurfaceView and __cuda_array_interface__ round-trip."""

from __future__ import annotations

import pytest

_ds = pytest.importorskip("savant_rs.deepstream")
if not hasattr(_ds, "SurfaceView"):
    pytest.skip("savant_rs built without deepstream feature", allow_module_level=True)
ds = _ds


def _ds_runtime_available() -> bool:
    try:
        ds.init_cuda(0)
        gen = ds.NvBufSurfaceGenerator("RGBA", 64, 64, pool_size=1)
        _ = gen.acquire_surface()
        return True
    except Exception:
        return False


_has_runtime = _ds_runtime_available()
skip_no_runtime = pytest.mark.skipif(
    not _has_runtime, reason="CUDA/DeepStream not available"
)


def _has_cv2_cuda() -> bool:
    try:
        import cv2

        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


skip_no_cv2_cuda = pytest.mark.skipif(
    not _has_cv2_cuda(), reason="OpenCV CUDA not available"
)
skip_no_cupy = pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")


# ---------------------------------------------------------------------------
# GpuMatCudaArray wrapper (mirrors python/nvbufsurface/common.py)
# ---------------------------------------------------------------------------


class GpuMatCudaArray:
    """Provides ``__cuda_array_interface__`` for ``cv2.cuda.GpuMat``."""

    __slots__ = ("_mat", "__cuda_array_interface__")

    def __init__(self, mat):
        import cv2

        depth = mat.depth()
        if depth != cv2.CV_8U:
            raise ValueError(f"unsupported depth {depth}")
        channels = mat.channels()
        if channels not in (1, 4):
            raise ValueError(f"unsupported channels {channels}")

        cols, rows = mat.size()
        self._mat = mat
        shape = (rows, cols, channels) if channels > 1 else (rows, cols)
        strides = (mat.step, channels, 1) if channels > 1 else (mat.step, 1)
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": "|u1",
            "descr": [("", "|u1")],
            "data": (mat.cudaPtr(), False),
            "strides": strides,
            "version": 3,
        }


# ===========================================================================
# SurfaceView.from_buffer
# ===========================================================================


@skip_no_runtime
class TestSurfaceViewFromBuffer:
    def test_basic_properties(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 320, 240, pool_size=1)
        buf = gen.acquire_surface()
        view = ds.SurfaceView.from_buffer(buf, 0)
        assert view.width == 320
        assert view.height == 240
        assert view.channels == 4
        assert view.gpu_id == 0
        assert view.data_ptr != 0
        assert view.pitch >= 320 * 4

    def test_cuda_array_interface_roundtrip(self):
        """from_buffer → __cuda_array_interface__ → from_cuda_array round-trip."""
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=1)
        buf = gen.acquire_surface()
        view = ds.SurfaceView.from_buffer(buf, 0)

        iface = view.__cuda_array_interface__
        assert iface["version"] == 3
        assert iface["typestr"] == "|u1"
        assert iface["shape"] == (480, 640, 4)
        data_ptr, _readonly = iface["data"]
        assert data_ptr == view.data_ptr

        view2 = ds.SurfaceView.from_cuda_array(view, gpu_id=0)
        assert view2.width == 640
        assert view2.height == 480
        assert view2.channels == 4
        assert view2.data_ptr == view.data_ptr

    def test_gray8_shape(self):
        gen = ds.NvBufSurfaceGenerator("GRAY8", 256, 128, pool_size=1)
        buf = gen.acquire_surface()
        view = ds.SurfaceView.from_buffer(buf, 0)
        assert view.channels == 1
        iface = view.__cuda_array_interface__
        assert iface["shape"] == (128, 256)


# ===========================================================================
# SurfaceView.from_cuda_array — GpuMat
# ===========================================================================


@skip_no_runtime
@skip_no_cv2_cuda
class TestSurfaceViewFromGpuMat:
    def test_rgba_gpumat(self):
        import cv2

        mat = cv2.cuda.GpuMat(480, 640, cv2.CV_8UC4)
        mat.setTo((10, 20, 30, 255))
        wrapper = GpuMatCudaArray(mat)

        view = ds.SurfaceView.from_cuda_array(wrapper, gpu_id=0)
        assert view.width == 640
        assert view.height == 480
        assert view.channels == 4
        assert view.data_ptr == mat.cudaPtr()
        assert view.pitch == mat.step

    def test_gray8_gpumat(self):
        import cv2

        mat = cv2.cuda.GpuMat(100, 200, cv2.CV_8UC1)
        wrapper = GpuMatCudaArray(mat)

        view = ds.SurfaceView.from_cuda_array(wrapper, gpu_id=0)
        assert view.width == 200
        assert view.height == 100
        assert view.channels == 1

    def test_roundtrip_gpumat(self):
        """GpuMat → from_cuda_array → __cuda_array_interface__ → verify."""
        import cv2

        mat = cv2.cuda.GpuMat(240, 320, cv2.CV_8UC4)
        wrapper = GpuMatCudaArray(mat)
        view = ds.SurfaceView.from_cuda_array(wrapper, gpu_id=0)

        iface = view.__cuda_array_interface__
        assert iface["shape"] == (240, 320, 4)
        data_ptr, _ = iface["data"]
        assert data_ptr == mat.cudaPtr()

    def test_unsupported_channels_rejected(self):
        import cv2

        mat = cv2.cuda.GpuMat(100, 100, cv2.CV_8UC3)
        with pytest.raises(ValueError, match="unsupported channel"):
            GpuMatCudaArray(mat)


# ===========================================================================
# SurfaceView.from_cuda_array — CuPy
# ===========================================================================


@skip_no_runtime
@skip_no_cupy
class TestSurfaceViewFromCuPy:
    def test_rgba_cupy(self):
        import cupy as cp

        arr = cp.zeros((480, 640, 4), dtype=cp.uint8)
        view = ds.SurfaceView.from_cuda_array(arr, gpu_id=0)
        assert view.width == 640
        assert view.height == 480
        assert view.channels == 4
        assert view.data_ptr != 0

    def test_gray_cupy(self):
        import cupy as cp

        arr = cp.zeros((100, 200), dtype=cp.uint8)
        view = ds.SurfaceView.from_cuda_array(arr, gpu_id=0)
        assert view.width == 200
        assert view.height == 100
        assert view.channels == 1

    def test_cupy_roundtrip(self):
        """CuPy → from_cuda_array → __cuda_array_interface__ → verify."""
        import cupy as cp

        arr = cp.zeros((120, 160, 4), dtype=cp.uint8)
        view = ds.SurfaceView.from_cuda_array(arr, gpu_id=0)

        iface = view.__cuda_array_interface__
        assert iface["shape"] == (120, 160, 4)
        data_ptr, _ = iface["data"]
        assert data_ptr == arr.data.ptr

    def test_wrong_dtype_rejected(self):
        import cupy as cp

        arr = cp.zeros((100, 100, 4), dtype=cp.float32)
        with pytest.raises(ValueError, match="unsupported dtype"):
            ds.SurfaceView.from_cuda_array(arr, gpu_id=0)

    def test_wrong_shape_rejected(self):
        import cupy as cp

        arr = cp.zeros((10, 20, 30, 4), dtype=cp.uint8)
        with pytest.raises(ValueError, match="unsupported shape"):
            ds.SurfaceView.from_cuda_array(arr, gpu_id=0)

    def test_3_channels_rejected(self):
        import cupy as cp

        arr = cp.zeros((100, 100, 3), dtype=cp.uint8)
        with pytest.raises(ValueError, match="unsupported channel"):
            ds.SurfaceView.from_cuda_array(arr, gpu_id=0)


# ===========================================================================
# GpuMatCudaArray wrapper validation
# ===========================================================================


@skip_no_cv2_cuda
class TestGpuMatCudaArrayWrapper:
    def test_interface_shape_rgba(self):
        import cv2

        mat = cv2.cuda.GpuMat(480, 640, cv2.CV_8UC4)
        wrapper = GpuMatCudaArray(mat)
        iface = wrapper.__cuda_array_interface__
        assert iface["shape"] == (480, 640, 4)
        assert iface["typestr"] == "|u1"
        assert iface["version"] == 3

    def test_interface_shape_gray(self):
        import cv2

        mat = cv2.cuda.GpuMat(100, 200, cv2.CV_8UC1)
        wrapper = GpuMatCudaArray(mat)
        iface = wrapper.__cuda_array_interface__
        assert iface["shape"] == (100, 200)

    def test_interface_strides(self):
        import cv2

        mat = cv2.cuda.GpuMat(240, 320, cv2.CV_8UC4)
        wrapper = GpuMatCudaArray(mat)
        iface = wrapper.__cuda_array_interface__
        row_stride, pixel_stride, channel_stride = iface["strides"]
        assert row_stride == mat.step
        assert pixel_stride == 4
        assert channel_stride == 1

    def test_data_ptr_matches(self):
        import cv2

        mat = cv2.cuda.GpuMat(64, 64, cv2.CV_8UC4)
        wrapper = GpuMatCudaArray(mat)
        data_ptr, readonly = wrapper.__cuda_array_interface__["data"]
        assert data_ptr == mat.cudaPtr()
        assert readonly is False

    def test_float_depth_rejected(self):
        import cv2

        mat = cv2.cuda.GpuMat(64, 64, cv2.CV_32FC4)
        with pytest.raises(ValueError, match="unsupported.*depth"):
            GpuMatCudaArray(mat)
