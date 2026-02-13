"""OpenCV CUDA GpuMat helpers for NvBufSurface buffers.

Provides zero-copy interop between NvBufSurface ``GstBuffer`` objects and
:class:`cv2.cuda.GpuMat` via :func:`cv2.cuda.createGpuMatFromCudaMemory`.

Two main entry points:

1. :func:`as_gpu_mat` -- context manager that exposes an existing buffer as
   a ``GpuMat`` + CUDA ``Stream``.  The stream is synchronised automatically
   when the ``with`` block exits.

2. :func:`from_gpumat` -- acquires a new buffer from the generator's pool
   and copies (with optional scaling) a ``GpuMat`` into it.

.. warning::

    Inside the ``as_gpu_mat`` context the ``GpuMat`` wraps memory owned by
    the ``GstBuffer``.  Do **not** push or unref the buffer while the
    context is active.

Example::

    from deepstream_nvbufsurface import (
        NvBufSurfaceGenerator, init_cuda,
        as_gpu_mat, from_gpumat,
    )
    import cv2

    init_cuda()
    gen = NvBufSurfaceGenerator("RGBA", 1920, 1080)

    # ── Read / modify an existing buffer ──────────────────────────
    buf_ptr = gen.acquire_surface()
    with as_gpu_mat(buf_ptr) as (mat, stream):
        mat.setTo((0, 255, 0, 255), stream=stream)
    # stream is synchronised here; buf_ptr is safe to push downstream

    # ── Create a buffer from a GpuMat ─────────────────────────────
    src_gpumat = cv2.cuda.GpuMat(480, 640, cv2.CV_8UC4)
    src_gpumat.setTo((255, 0, 0, 255))
    buf_ptr = from_gpumat(gen, src_gpumat)  # scaled to 1920x1080
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import cv2

from deepstream_nvbufsurface._native import (
    NvBufSurfaceGenerator,
    get_nvbufsurface_info,
)

# RGBA -> 4 channels, 8-bit unsigned
_RGBA_CV_TYPE = cv2.CV_8UC4


@contextmanager
def as_gpu_mat(
    buf_ptr: int,
) -> Generator[tuple[cv2.cuda.GpuMat, cv2.cuda.Stream], None, None]:
    """Expose an NvBufSurface ``GstBuffer`` as an OpenCV CUDA ``GpuMat``.

    Creates a zero-copy ``GpuMat`` wrapping the buffer's CUDA device
    memory together with a CUDA ``Stream``.  When the ``with`` block
    exits the stream is synchronised (``waitForCompletion``).

    Args:
        buf_ptr: Raw ``GstBuffer*`` pointer address.

    Yields:
        ``(gpumat, stream)`` -- the ``GpuMat`` is ``CV_8UC4`` with the
        buffer's native width, height and pitch.
    """
    data_ptr, pitch, width, height = get_nvbufsurface_info(buf_ptr)
    gpumat = cv2.cuda.createGpuMatFromCudaMemory(
        int(height), int(width), _RGBA_CV_TYPE, int(data_ptr), int(pitch),
    )
    stream = cv2.cuda.Stream()
    try:
        yield gpumat, stream
    finally:
        stream.waitForCompletion()


def from_gpumat(
    gen: NvBufSurfaceGenerator,
    gpumat: cv2.cuda.GpuMat,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    id: int | None = None,
) -> int:
    """Acquire a buffer from the pool and fill it from a ``GpuMat``.

    If the source ``GpuMat`` dimensions differ from the generator's
    dimensions the image is scaled using :func:`cv2.cuda.resize` with
    the given *interpolation* method.  When sizes match the data is
    copied directly (zero-overhead ``copyTo``).

    Args:
        gen: Surface generator (determines destination dimensions and
            format).
        gpumat: Source ``GpuMat`` (must be ``CV_8UC4``).
        interpolation: OpenCV interpolation flag (default
            ``cv2.INTER_LINEAR``).  Common choices:
            ``cv2.INTER_NEAREST``, ``cv2.INTER_LINEAR``,
            ``cv2.INTER_CUBIC``, ``cv2.INTER_AREA``.
        id: Optional frame identifier for ``SavantIdMeta``.

    Returns:
        Raw ``GstBuffer*`` pointer address of the newly acquired buffer.
    """
    buf_ptr, data_ptr, pitch = gen.acquire_surface_with_ptr(id=id)
    dst = cv2.cuda.createGpuMatFromCudaMemory(
        gen.height, gen.width, _RGBA_CV_TYPE, data_ptr, pitch,
    )

    src_w, src_h = gpumat.size()  # (cols, rows)
    dst_w, dst_h = dst.size()

    if (src_w, src_h) == (dst_w, dst_h):
        gpumat.copyTo(dst)
    else:
        cv2.cuda.resize(gpumat, (dst_w, dst_h), dst=dst, interpolation=interpolation)

    return buf_ptr
