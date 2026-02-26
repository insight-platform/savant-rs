"""OpenCV CUDA GpuMat helpers for NvBufSurface buffers.

Injected into ``savant_rs.deepstream`` at import time so that
``from savant_rs.deepstream import nvgstbuf_as_gpu_mat`` etc. work.

Two context managers for different call sites:

- :func:`nvgstbuf_as_gpu_mat` — takes a ``GstBuffer*`` pointer, extracts
  NvBufSurface metadata internally.  Use outside callbacks (e.g. pre-filling
  a background before ``send_frame``).

- :func:`nvbuf_as_gpu_mat` — takes raw CUDA params ``(data_ptr, pitch,
  width, height)`` directly.  Use inside the ``on_gpumat`` callback which
  already provides these values.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import cv2

from savant_rs.deepstream import NvBufSurfaceGenerator, get_nvbufsurface_info

_RGBA_CV_TYPE = cv2.CV_8UC4


@contextmanager
def nvgstbuf_as_gpu_mat(
    buf_ptr: int,
) -> Generator[tuple[cv2.cuda.GpuMat, cv2.cuda.Stream], None, None]:
    """Expose an NvBufSurface ``GstBuffer`` as an OpenCV CUDA ``GpuMat``.

    Extracts the CUDA device pointer, pitch, width and height from the
    buffer's NvBufSurface metadata, then creates a zero-copy ``GpuMat``
    together with a CUDA ``Stream``.  When the ``with`` block exits the
    stream is synchronised (``waitForCompletion``).

    Args:
        buf_ptr: Raw ``GstBuffer*`` pointer address.

    Yields:
        ``(gpumat, stream)`` -- the ``GpuMat`` is ``CV_8UC4`` with the
        buffer's native width, height and pitch.
    """
    data_ptr, pitch, width, height = get_nvbufsurface_info(buf_ptr)
    gpumat = cv2.cuda.createGpuMatFromCudaMemory(
        int(height),
        int(width),
        _RGBA_CV_TYPE,
        int(data_ptr),
        int(pitch),
    )
    stream = cv2.cuda.Stream()
    try:
        yield gpumat, stream
    finally:
        stream.waitForCompletion()


@contextmanager
def nvbuf_as_gpu_mat(
    data_ptr: int,
    pitch: int,
    width: int,
    height: int,
) -> Generator[tuple[cv2.cuda.GpuMat, cv2.cuda.Stream], None, None]:
    """Wrap raw CUDA memory as an OpenCV CUDA ``GpuMat``.

    Unlike :func:`nvgstbuf_as_gpu_mat`, this function takes the CUDA
    device pointer and layout directly — no ``GstBuffer`` or
    ``get_nvbufsurface_info`` call involved.  Designed for the Picasso
    ``on_gpumat`` callback which already supplies these values.

    Args:
        data_ptr: CUDA device pointer to the surface data.
        pitch: Row stride in bytes.
        width: Surface width in pixels.
        height: Surface height in pixels.

    Yields:
        ``(gpumat, stream)`` -- the ``GpuMat`` is ``CV_8UC4``.
    """
    gpumat = cv2.cuda.createGpuMatFromCudaMemory(
        int(height),
        int(width),
        _RGBA_CV_TYPE,
        int(data_ptr),
        int(pitch),
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
        gen.height,
        gen.width,
        _RGBA_CV_TYPE,
        data_ptr,
        pitch,
    )

    src_w, src_h = gpumat.size()
    dst_w, dst_h = dst.size()

    if (src_w, src_h) == (dst_w, dst_h):
        gpumat.copyTo(dst)
    else:
        cv2.cuda.resize(gpumat, (dst_w, dst_h), dst=dst, interpolation=interpolation)

    return buf_ptr
