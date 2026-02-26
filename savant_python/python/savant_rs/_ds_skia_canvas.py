"""Convenience wrapper: SkiaContext + skia-python in one object.

Injected into ``savant_rs.deepstream`` at import time so that
``from savant_rs.deepstream import SkiaCanvas`` works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import skia

if TYPE_CHECKING:
    from savant_rs.deepstream import TransformConfig

GL_RGBA8 = 0x8058


class SkiaCanvas:
    """Convenience wrapper: SkiaContext + skia-python in one object.

    Handles creation of the skia GrDirectContext and Surface backed
    by the SkiaContext's GPU FBO.
    """

    def __init__(self, ctx):
        """Initialize from an existing SkiaContext.

        Args:
            ctx: ``SkiaContext`` from ``savant_rs.deepstream``.
        """
        self._ctx = ctx
        interface = skia.GrGLInterface.MakeEGL()
        self._gr_context = skia.GrDirectContext.MakeGL(interface)
        fb_info = skia.GrGLFramebufferInfo(ctx.fbo_id, GL_RGBA8)
        backend_rt = skia.GrBackendRenderTarget(ctx.width, ctx.height, 0, 8, fb_info)
        self._surface = skia.Surface.MakeFromBackendRenderTarget(
            self._gr_context,
            backend_rt,
            skia.kTopLeft_GrSurfaceOrigin,
            skia.kRGBA_8888_ColorType,
            None,
        )

    @classmethod
    def from_fbo(cls, fbo_id: int, width: int, height: int) -> "SkiaCanvas":
        """Create from an existing OpenGL FBO.

        Used internally by the Picasso ``on_render`` callback to wrap the
        worker's GPU canvas without creating a separate ``SkiaContext``.

        Args:
            fbo_id: OpenGL FBO ID backing the canvas.
            width:  Canvas width in pixels.
            height: Canvas height in pixels.
        """
        obj = cls.__new__(cls)
        obj._ctx = None
        interface = skia.GrGLInterface.MakeEGL()
        obj._gr_context = skia.GrDirectContext.MakeGL(interface)
        fb_info = skia.GrGLFramebufferInfo(fbo_id, GL_RGBA8)
        backend_rt = skia.GrBackendRenderTarget(width, height, 0, 8, fb_info)
        obj._surface = skia.Surface.MakeFromBackendRenderTarget(
            obj._gr_context,
            backend_rt,
            skia.kTopLeft_GrSurfaceOrigin,
            skia.kRGBA_8888_ColorType,
            None,
        )
        return obj

    @classmethod
    def create(cls, width: int, height: int, gpu_id: int = 0):
        """Create with an empty (transparent) canvas.

        Args:
            width:  Canvas width in pixels.
            height: Canvas height in pixels.
            gpu_id: GPU device ID (default 0).
        """
        from savant_rs.deepstream import SkiaContext

        ctx = SkiaContext(width, height, gpu_id)
        return cls(ctx)

    @classmethod
    def from_nvbuf(cls, buf_ptr: int, gpu_id: int = 0):
        """Create with canvas pre-loaded from an NvBufSurface.

        Canvas dimensions match the source buffer.

        Args:
            buf_ptr: Raw pointer of the source GstBuffer.
            gpu_id:  GPU device ID (default 0).
        """
        from savant_rs.deepstream import SkiaContext

        ctx = SkiaContext.from_nvbuf(buf_ptr, gpu_id)
        return cls(ctx)

    @property
    def gr_context(self) -> skia.GrDirectContext:
        """The Skia GPU ``GrDirectContext`` backing this canvas.

        Use this to create GPU-resident images via
        :meth:`skia.Image.makeTextureImage` for efficient repeated
        drawing without per-frame CPU -> GPU transfers::

            raster = skia.Image.MakeFromEncoded(data)
            gpu_img = raster.makeTextureImage(canvas.gr_context)
            # gpu_img now lives in VRAM; drawImage is pure GPU work
        """
        return self._gr_context

    @property
    def width(self) -> int:
        """Canvas width in pixels."""
        return self._ctx.width

    @property
    def height(self) -> int:
        """Canvas height in pixels."""
        return self._ctx.height

    def canvas(self) -> skia.Canvas:
        """Get the skia-python Canvas for drawing."""
        return self._surface.getCanvas()

    def render_to_nvbuf(
        self,
        buf_ptr: int,
        config: TransformConfig | None = None,
    ):
        """Flush Skia and copy to destination NvBufSurface.

        Supports optional scaling + letterboxing when canvas dimensions
        differ from the destination buffer.

        Args:
            buf_ptr: Raw pointer of the destination GstBuffer.
            config:  Optional ``TransformConfig`` for scaling / letterboxing.
                     ``None`` means direct 1:1 copy (canvas and destination
                     must have the same dimensions).
        """
        self._gr_context.flushAndSubmit()
        self._ctx.render_to_nvbuf(buf_ptr, config=config)
