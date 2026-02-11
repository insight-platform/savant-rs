"""Type stubs for deepstream_nvbufsurface.skia_canvas."""

from __future__ import annotations

import skia

from deepstream_nvbufsurface._native import SkiaContext, TransformConfig

class SkiaCanvas:
    """Convenience wrapper: SkiaContext + skia-python in one object.

    Handles creation of the skia ``GrDirectContext`` and ``Surface``
    backed by the ``SkiaContext``'s GPU FBO.
    """

    def __init__(self, ctx: SkiaContext) -> None:
        """Initialize from an existing ``SkiaContext``.

        Args:
            ctx: ``SkiaContext`` from ``deepstream_nvbufsurface``.
        """
        ...
    @classmethod
    def create(cls, width: int, height: int, gpu_id: int = 0) -> SkiaCanvas:
        """Create with an empty (transparent) canvas.

        Args:
            width:  Canvas width in pixels.
            height: Canvas height in pixels.
            gpu_id: GPU device ID (default 0).
        """
        ...
    @classmethod
    def from_nvbuf(cls, buf_ptr: int, gpu_id: int = 0) -> SkiaCanvas:
        """Create with canvas pre-loaded from an NvBufSurface.

        Canvas dimensions match the source buffer.

        Args:
            buf_ptr: Raw pointer of the source ``GstBuffer``.
            gpu_id:  GPU device ID (default 0).
        """
        ...
    @property
    def gr_context(self) -> skia.GrDirectContext:
        """The Skia GPU ``GrDirectContext`` backing this canvas.

        Use to create GPU-resident images via
        ``image.makeTextureImage(canvas.gr_context)``.
        """
        ...
    @property
    def width(self) -> int:
        """Canvas width in pixels."""
        ...
    @property
    def height(self) -> int:
        """Canvas height in pixels."""
        ...
    def canvas(self) -> skia.Canvas:
        """Get the ``skia-python`` Canvas for drawing."""
        ...
    def render_to_nvbuf(
        self,
        buf_ptr: int,
        config: TransformConfig | None = None,
    ) -> None:
        """Flush Skia and copy to destination NvBufSurface.

        Args:
            buf_ptr: Raw pointer of the destination ``GstBuffer``.
            config:  Optional ``TransformConfig`` for scaling / letterboxing.
                     ``None`` means direct 1:1 copy.
        """
        ...
