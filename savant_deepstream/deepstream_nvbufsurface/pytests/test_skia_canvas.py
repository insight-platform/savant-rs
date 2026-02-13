"""Tests for SkiaCanvas — gr_context property and GPU image workflow."""

from __future__ import annotations

import pytest
import skia

from deepstream_nvbufsurface import (
    NvBufSurfaceGenerator,
    SkiaCanvas,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def skia_canvas() -> SkiaCanvas:
    """Module-scoped 640x480 SkiaCanvas."""
    return SkiaCanvas.create(640, 480)


@pytest.fixture(scope="module")
def rgba_gen_module() -> NvBufSurfaceGenerator:
    """Module-scoped 640x480 RGBA generator."""
    return NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=4)


# ---------------------------------------------------------------------------
# gr_context property
# ---------------------------------------------------------------------------


class TestGrContextProperty:
    """Tests for the newly exposed gr_context property."""

    def test_returns_gr_direct_context(self, skia_canvas: SkiaCanvas) -> None:
        ctx = skia_canvas.gr_context
        assert isinstance(ctx, skia.GrDirectContext)

    def test_same_object_each_call(self, skia_canvas: SkiaCanvas) -> None:
        """The property should return the same context, not create new ones."""
        ctx1 = skia_canvas.gr_context
        ctx2 = skia_canvas.gr_context
        assert ctx1 is ctx2

    def test_context_is_valid(self, skia_canvas: SkiaCanvas) -> None:
        """The context should not be abandoned/released."""
        ctx = skia_canvas.gr_context
        # GrDirectContext has maxSurfaceSampleCountForColorType; if the
        # context were invalid this would fail.
        count = ctx.maxSurfaceSampleCountForColorType(skia.kRGBA_8888_ColorType)
        assert count >= 1


# ---------------------------------------------------------------------------
# GPU texture image workflow
# ---------------------------------------------------------------------------


def _make_test_raster(width: int = 256, height: int = 256) -> skia.Image:
    """Create a solid red RGBA raster image for testing."""
    info = skia.ImageInfo.MakeN32Premul(width, height)
    surface = skia.Surface.MakeRaster(info)
    canvas = surface.getCanvas()
    canvas.clear(skia.Color(255, 0, 0, 255))
    return surface.makeImageSnapshot()


class TestMakeTextureImage:
    """Tests for uploading raster images to GPU via makeTextureImage."""

    def test_upload_to_gpu(self, skia_canvas: SkiaCanvas) -> None:
        raster = _make_test_raster()
        assert not raster.isTextureBacked()

        gpu_image = raster.makeTextureImage(skia_canvas.gr_context)
        assert gpu_image is not None
        assert gpu_image.isTextureBacked()

    def test_gpu_image_preserves_dimensions(self, skia_canvas: SkiaCanvas) -> None:
        raster = _make_test_raster(512, 300)
        gpu_image = raster.makeTextureImage(skia_canvas.gr_context)
        assert gpu_image.width() == 512
        assert gpu_image.height() == 300

    def test_gpu_image_is_reusable(self, skia_canvas: SkiaCanvas) -> None:
        """Upload once, draw many times — the image stays GPU-resident."""
        raster = _make_test_raster(128, 128)
        gpu_image = raster.makeTextureImage(skia_canvas.gr_context)

        canvas = skia_canvas.canvas()
        paint = skia.Paint(AntiAlias=True)
        # Draw the same GPU image 50 times (all GPU-side)
        for _ in range(50):
            canvas.drawImage(gpu_image, 0, 0, paint=paint)
        skia_canvas.gr_context.flushAndSubmit()
        # If we reach here without errors the image was reused correctly.
        assert gpu_image.isTextureBacked()


class TestGpuImageDrawOnNvBuf:
    """End-to-end: upload image to GPU, draw on canvas, render to NvBuf."""

    def test_draw_and_render(
        self,
        skia_canvas: SkiaCanvas,
        rgba_gen_module: NvBufSurfaceGenerator,
    ) -> None:
        # 1. Create a solid blue GPU texture
        raster = _make_test_raster(200, 200)
        gpu_image = raster.makeTextureImage(skia_canvas.gr_context)
        assert gpu_image is not None
        assert gpu_image.isTextureBacked()

        # 2. Draw on the Skia canvas
        canvas = skia_canvas.canvas()
        canvas.clear(skia.Color(0, 0, 0, 255))  # black background
        canvas.drawImage(gpu_image, 10, 10, paint=skia.Paint(AntiAlias=True))

        # 3. Render to an NvBufSurface buffer
        buf_ptr = rgba_gen_module.acquire_surface()
        skia_canvas.render_to_nvbuf(buf_ptr)
        # Success if no error raised — the buffer contains the rendered image.

    def test_large_image_upload(self, skia_canvas: SkiaCanvas) -> None:
        """Upload a large image (2048x2048 ≈ 16 MB RGBA) to GPU."""
        raster = _make_test_raster(2048, 2048)
        raw_mb = 2048 * 2048 * 4 / (1024 * 1024)
        assert raw_mb == pytest.approx(16.0)

        gpu_image = raster.makeTextureImage(skia_canvas.gr_context)
        assert gpu_image is not None
        assert gpu_image.isTextureBacked()
        assert gpu_image.width() == 2048
        assert gpu_image.height() == 2048

    def test_multiple_gpu_images(self, skia_canvas: SkiaCanvas) -> None:
        """Multiple images can coexist in GPU memory."""
        images = []
        for size in [128, 256, 512, 1024]:
            raster = _make_test_raster(size, size)
            gpu = raster.makeTextureImage(skia_canvas.gr_context)
            assert gpu is not None
            assert gpu.isTextureBacked()
            images.append(gpu)

        # Draw all four on the canvas
        canvas = skia_canvas.canvas()
        canvas.clear(skia.ColorBLACK)
        paint = skia.Paint()
        for i, img in enumerate(images):
            canvas.drawImage(img, float(i * 10), float(i * 10), paint=paint)
        skia_canvas.gr_context.flushAndSubmit()
        # All images are still GPU-resident
        assert all(img.isTextureBacked() for img in images)
