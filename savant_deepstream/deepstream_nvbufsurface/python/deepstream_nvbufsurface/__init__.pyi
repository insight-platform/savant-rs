"""Type stubs for the deepstream_nvbufsurface package."""

from __future__ import annotations

# Re-export all native symbols
from deepstream_nvbufsurface._native import (
    ComputeMode as ComputeMode,
    Interpolation as Interpolation,
    NvBufSurfaceGenerator as NvBufSurfaceGenerator,
    Padding as Padding,
    SkiaContext as SkiaContext,
    TransformConfig as TransformConfig,
    bridge_savant_id_meta_py as bridge_savant_id_meta_py,
    get_nvbufsurface_info as get_nvbufsurface_info,
    get_savant_id_meta as get_savant_id_meta,
    init_cuda as init_cuda,
)
from deepstream_nvbufsurface.gpumat import (
    as_gpu_mat as as_gpu_mat,
    from_gpumat as from_gpumat,
)
from deepstream_nvbufsurface.skia_canvas import SkiaCanvas as SkiaCanvas

__all__: list[str]
