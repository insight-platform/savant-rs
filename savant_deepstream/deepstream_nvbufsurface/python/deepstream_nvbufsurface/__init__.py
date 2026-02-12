"""deepstream_nvbufsurface — GPU buffer generation for DeepStream pipelines.

Re-exports the native Rust module and adds the pure-Python
:class:`SkiaCanvas` helper for GPU-accelerated Skia rendering.

Quick start::

    from deepstream_nvbufsurface import (
        NvBufSurfaceGenerator,
        TransformConfig, Padding, Interpolation, ComputeMode,
        init_cuda,
    )

    init_cuda()
    gen = NvBufSurfaceGenerator("RGBA", 1920, 1080)
    buf = gen.acquire_surface(id=0)
"""

from __future__ import annotations

# ── Native (Rust/PyO3) symbols ───────────────────────────────────────────
from deepstream_nvbufsurface._native import (  # noqa: F401
    ComputeMode,
    Interpolation,
    MemType,
    NvBufSurfaceGenerator,
    Padding,
    TransformConfig,
    VideoFormat,
    bridge_savant_id_meta_py,
    get_nvbufsurface_info,
    get_savant_id_meta,
    init_cuda,
)

# ── SkiaContext (only when built with the "skia" feature) ────────────────
try:
    from deepstream_nvbufsurface._native import SkiaContext  # noqa: F401
except ImportError:
    pass

# ── Pure-Python helpers ──────────────────────────────────────────────────
try:
    from deepstream_nvbufsurface.skia_canvas import SkiaCanvas  # noqa: F401
except ImportError:
    # skia-python not installed
    pass

# ── OpenCV CUDA GpuMat helpers (require opencv-python with CUDA) ────────
try:
    from deepstream_nvbufsurface.gpumat import (  # noqa: F401
        as_gpu_mat,
        from_gpumat,
    )
except ImportError:
    # opencv-python not installed or built without CUDA support
    pass

__all__ = [
    # Enums
    "Padding",
    "Interpolation",
    "ComputeMode",
    "MemType",
    "VideoFormat",
    # Config
    "TransformConfig",
    # Generator
    "NvBufSurfaceGenerator",
    # Functions
    "init_cuda",
    "bridge_savant_id_meta_py",
    "get_savant_id_meta",
    "get_nvbufsurface_info",
    # Skia (optional)
    "SkiaContext",
    "SkiaCanvas",
    # GpuMat (optional, requires opencv-python with CUDA)
    "as_gpu_mat",
    "from_gpumat",
]
