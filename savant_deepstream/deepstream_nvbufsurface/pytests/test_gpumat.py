"""Tests for the OpenCV CUDA GpuMat interop API."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deepstream_nvbufsurface import (
    NvBufSurfaceGenerator,
    VideoFormat,
    as_gpu_mat,
    from_gpumat,
    get_nvbufsurface_info,
)


# ---------------------------------------------------------------------------
# Generator property getters
# ---------------------------------------------------------------------------


class TestGeneratorProperties:
    """Verify that width/height/format are exposed as Python properties."""

    def test_width(self, rgba_gen: NvBufSurfaceGenerator) -> None:
        assert rgba_gen.width == 640

    def test_height(self, rgba_gen: NvBufSurfaceGenerator) -> None:
        assert rgba_gen.height == 480

    def test_format(self, rgba_gen: NvBufSurfaceGenerator) -> None:
        assert rgba_gen.format == VideoFormat.RGBA

    def test_different_dimensions(self) -> None:
        gen = NvBufSurfaceGenerator("RGBA", 1920, 1080, pool_size=2)
        assert gen.width == 1920
        assert gen.height == 1080
        assert gen.format == VideoFormat.RGBA


# ---------------------------------------------------------------------------
# get_nvbufsurface_info
# ---------------------------------------------------------------------------


class TestGetNvBufSurfaceInfo:
    """Tests for the get_nvbufsurface_info helper."""

    def test_returns_tuple(self, rgba_gen: NvBufSurfaceGenerator) -> None:
        buf_ptr = rgba_gen.acquire_surface()
        info = get_nvbufsurface_info(buf_ptr)
        assert isinstance(info, tuple)
        assert len(info) == 4

    def test_correct_dimensions(self, rgba_gen: NvBufSurfaceGenerator) -> None:
        buf_ptr = rgba_gen.acquire_surface()
        data_ptr, pitch, width, height = get_nvbufsurface_info(buf_ptr)
        assert width == 640
        assert height == 480
        assert data_ptr != 0
        assert pitch >= 640 * 4  # at least width * 4 bytes for RGBA

    def test_null_ptr_raises(self) -> None:
        with pytest.raises(ValueError, match="null"):
            get_nvbufsurface_info(0)


# ---------------------------------------------------------------------------
# as_gpu_mat context manager
# ---------------------------------------------------------------------------


class TestAsGpuMat:
    """Tests for the as_gpu_mat context manager."""

    def test_yields_gpumat_and_stream(
        self, rgba_gen: NvBufSurfaceGenerator
    ) -> None:
        buf_ptr = rgba_gen.acquire_surface()
        with as_gpu_mat(buf_ptr) as (mat, stream):
            assert isinstance(mat, cv2.cuda.GpuMat)
            assert isinstance(stream, cv2.cuda.Stream)

    def test_gpumat_shape(self, rgba_gen: NvBufSurfaceGenerator) -> None:
        buf_ptr = rgba_gen.acquire_surface()
        with as_gpu_mat(buf_ptr) as (mat, _stream):
            assert mat.size() == (640, 480)  # (cols, rows)
            assert mat.channels() == 4
            assert mat.type() == cv2.CV_8UC4

    def test_writable_and_readable(
        self, rgba_gen: NvBufSurfaceGenerator
    ) -> None:
        """Fill the buffer with green via GpuMat, verify on CPU."""
        buf_ptr = rgba_gen.acquire_surface()
        with as_gpu_mat(buf_ptr) as (mat, stream):
            mat.setTo((0, 255, 0, 255), stream=stream)

        # After the context exits the stream has been synced.
        # Re-open to verify the write persisted.
        with as_gpu_mat(buf_ptr) as (mat, _stream):
            cpu = mat.download()
            assert cpu.shape == (480, 640, 4)
            np.testing.assert_array_equal(cpu[0, 0], [0, 255, 0, 255])
            np.testing.assert_array_equal(cpu[239, 319], [0, 255, 0, 255])

    def test_stream_synced_on_exit(
        self, rgba_gen: NvBufSurfaceGenerator
    ) -> None:
        """Ensure the stream is synchronised when the with block exits."""
        buf_ptr = rgba_gen.acquire_surface()
        with as_gpu_mat(buf_ptr) as (mat, stream):
            mat.setTo((128, 64, 32, 255), stream=stream)
        # If we reach here without error the stream was synced.
        # Verify the data is actually flushed.
        with as_gpu_mat(buf_ptr) as (mat, _):
            px = mat.download()[0, 0]
            np.testing.assert_array_equal(px, [128, 64, 32, 255])


# ---------------------------------------------------------------------------
# from_gpumat
# ---------------------------------------------------------------------------


class TestFromGpuMat:
    """Tests for the from_gpumat function."""

    def test_same_size_copy(self) -> None:
        """Copy a 640x480 GpuMat into a 640x480 buffer."""
        gen = NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=2)
        src = cv2.cuda.GpuMat(480, 640, cv2.CV_8UC4)
        src.setTo((10, 20, 30, 255))

        buf_ptr = from_gpumat(gen, src)
        assert isinstance(buf_ptr, int)
        assert buf_ptr != 0

        with as_gpu_mat(buf_ptr) as (mat, _):
            cpu = mat.download()
            np.testing.assert_array_equal(cpu[0, 0], [10, 20, 30, 255])

    def test_upscale(self) -> None:
        """Scale a 320x240 GpuMat into a 640x480 buffer."""
        gen = NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=2)
        src = cv2.cuda.GpuMat(240, 320, cv2.CV_8UC4)
        src.setTo((255, 0, 0, 255))

        buf_ptr = from_gpumat(gen, src)
        with as_gpu_mat(buf_ptr) as (mat, _):
            assert mat.size() == (640, 480)
            cpu = mat.download()
            # After scaling the uniform red should still be red
            np.testing.assert_array_equal(cpu[0, 0], [255, 0, 0, 255])

    def test_downscale(self) -> None:
        """Scale a 1280x960 GpuMat into a 640x480 buffer."""
        gen = NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=2)
        src = cv2.cuda.GpuMat(960, 1280, cv2.CV_8UC4)
        src.setTo((0, 0, 255, 128))

        buf_ptr = from_gpumat(gen, src)
        with as_gpu_mat(buf_ptr) as (mat, _):
            assert mat.size() == (640, 480)
            cpu = mat.download()
            np.testing.assert_array_equal(cpu[0, 0], [0, 0, 255, 128])

    def test_with_id(self) -> None:
        gen = NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=2)
        src = cv2.cuda.GpuMat(480, 640, cv2.CV_8UC4)
        src.setTo((1, 2, 3, 4))

        buf_ptr = from_gpumat(gen, src, id=42)
        assert buf_ptr != 0

    def test_nearest_interpolation(self) -> None:
        gen = NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=2)
        src = cv2.cuda.GpuMat(240, 320, cv2.CV_8UC4)
        src.setTo((50, 100, 150, 200))

        buf_ptr = from_gpumat(gen, src, interpolation=cv2.INTER_NEAREST)
        with as_gpu_mat(buf_ptr) as (mat, _):
            cpu = mat.download()
            np.testing.assert_array_equal(cpu[0, 0], [50, 100, 150, 200])
