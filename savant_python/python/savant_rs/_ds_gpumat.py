"""OpenCV CUDA GpuMat helpers for NvBufSurface buffers.

Injected into ``savant_rs.deepstream`` at import time so that
``from savant_rs.deepstream import nvgstbuf_as_gpu_mat`` etc. work.

Two context managers for different call sites:

- :func:`nvgstbuf_as_gpu_mat` — takes a ``SharedBuffer`` guard (or raw ``int``
  pointer), extracts NvBufSurface metadata internally.  Use outside callbacks
  (e.g. pre-filling a background before ``send_frame``).

- :func:`nvbuf_as_gpu_mat` — takes raw CUDA params ``(data_ptr, pitch,
  width, height)`` directly.  Use inside the ``on_gpumat`` callback which
  already provides these values.

- :class:`GpuMatCudaArray` — exposes ``__cuda_array_interface__`` (v3) for a
  ``cv2.cuda.GpuMat``, bridging it to consumers like Picasso ``send_frame``.

- :func:`make_gpu_mat` — allocates a zero-initialised ``GpuMat``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Union

import cv2

from savant_rs.deepstream import (
    BufferGenerator,
    SharedBuffer,
    get_nvbufsurface_info,
)


# ---------------------------------------------------------------------------
# GpuMat ↔ __cuda_array_interface__ wrapper
# ---------------------------------------------------------------------------


class GpuMatCudaArray:
    """Exposes ``__cuda_array_interface__`` (v3) for a ``cv2.cuda.GpuMat``.

    OpenCV's ``GpuMat`` does not implement the protocol natively, so this
    thin wrapper bridges it to any consumer that expects the interface
    (CuPy, ``SurfaceView.from_cuda_array``, Picasso ``send_frame``, etc.).

    Only ``CV_8UC1`` (GRAY8) and ``CV_8UC4`` (RGBA) mats are supported.

    The wrapper keeps a reference to the source mat so the underlying
    device memory stays alive for as long as this object exists.
    """

    __slots__ = ("_mat", "__cuda_array_interface__")

    def __init__(self, mat: cv2.cuda.GpuMat) -> None:
        depth = mat.depth()
        if depth != cv2.CV_8U:
            raise ValueError(
                f"unsupported GpuMat depth {depth}; only CV_8U is supported"
            )
        channels = mat.channels()
        if channels not in (1, 4):
            raise ValueError(
                f"unsupported channel count {channels}; expected 1 (GRAY8) or 4 (RGBA)"
            )

        cols, rows = mat.size()
        self._mat = mat
        shape: tuple[int, ...] = (
            (rows, cols, channels) if channels > 1 else (rows, cols)
        )
        strides: tuple[int, ...] = (
            (mat.step, channels, 1) if channels > 1 else (mat.step, 1)
        )
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": "|u1",
            "descr": [("", "|u1")],
            "data": (mat.cudaPtr(), False),
            "strides": strides,
            "version": 3,
        }


def make_gpu_mat(width: int, height: int, channels: int = 4) -> cv2.cuda.GpuMat:
    """Allocate a ``cv2.cuda.GpuMat`` of the given size.

    Returns:
        A zero-initialised ``GpuMat`` with ``CV_8UC<channels>`` type.
    """
    cv_type = {1: cv2.CV_8UC1, 4: cv2.CV_8UC4}[channels]
    return cv2.cuda.GpuMat(height, width, cv_type)


# ---------------------------------------------------------------------------
# NvBufSurface ↔ GpuMat context managers
# ---------------------------------------------------------------------------

_RGBA_CV_TYPE = cv2.CV_8UC4


@contextmanager
def nvgstbuf_as_gpu_mat(
    buf: Union[SharedBuffer, int],
    stream: cv2.cuda.Stream | None = None,
) -> Generator[tuple[cv2.cuda.GpuMat, cv2.cuda.Stream], None, None]:
    """Expose an NvBufSurface ``SharedBuffer`` as an OpenCV CUDA ``GpuMat``.

    Extracts the CUDA device pointer, pitch, width and height from the
    buffer's NvBufSurface metadata, then creates a zero-copy ``GpuMat``
    together with a CUDA ``Stream``.  When the ``with`` block exits the
    stream is synchronised (``waitForCompletion``).

    Args:
        buf: ``SharedBuffer`` RAII guard or raw ``GstBuffer*`` pointer as ``int``.

    Yields:
        ``(gpumat, stream)`` -- the ``GpuMat`` is ``CV_8UC4`` with the
        buffer's native width, height and pitch.
    """
    data_ptr, pitch, width, height = get_nvbufsurface_info(buf)
    gpumat = cv2.cuda.createGpuMatFromCudaMemory(
        int(height),
        int(width),
        _RGBA_CV_TYPE,
        int(data_ptr),
        int(pitch),
    )
    stream = stream or cv2.cuda.Stream()
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
    stream: cv2.cuda.Stream | None = None,
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
    stream = stream or cv2.cuda.Stream()
    try:
        yield gpumat, stream
    finally:
        stream.waitForCompletion()


def from_gpumat(
    gen: BufferGenerator,
    gpumat: cv2.cuda.GpuMat,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    id: int | None = None,
) -> SharedBuffer:
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
        ``SharedBuffer`` RAII guard owning the newly acquired buffer.
    """
    buf = gen.acquire(id=id)
    data_ptr, pitch, width, height = get_nvbufsurface_info(buf)
    dst = cv2.cuda.createGpuMatFromCudaMemory(
        int(height),
        int(width),
        _RGBA_CV_TYPE,
        int(data_ptr),
        int(pitch),
    )

    src_w, src_h = gpumat.size()
    dst_w, dst_h = dst.size()

    if (src_w, src_h) == (dst_w, dst_h):
        gpumat.copyTo(dst)
    else:
        cv2.cuda.resize(gpumat, (dst_w, dst_h), dst=dst, interpolation=interpolation)

    return buf
