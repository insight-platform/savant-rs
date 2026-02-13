"""Type stubs for deepstream_nvbufsurface.gpumat."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import cv2

from deepstream_nvbufsurface._native import NvBufSurfaceGenerator

@contextmanager
def as_gpu_mat(
    buf_ptr: int,
) -> Generator[tuple[cv2.cuda.GpuMat, cv2.cuda.Stream], None, None]:
    """Expose an NvBufSurface buffer as a ``GpuMat`` + CUDA ``Stream``.

    The stream is synchronised when the ``with`` block exits.

    Yields:
        ``(gpumat, stream)``
    """
    ...

def from_gpumat(
    gen: NvBufSurfaceGenerator,
    gpumat: cv2.cuda.GpuMat,
    *,
    interpolation: int = ...,
    id: int | None = None,
) -> int:
    """Acquire a buffer and fill it from a ``GpuMat`` (with optional scaling).

    Returns:
        Raw ``GstBuffer*`` pointer address.
    """
    ...
