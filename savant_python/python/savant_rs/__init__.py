from .savant_rs import *  # type: ignore

__all__ = savant_rs.__all__  # type: ignore

# ---------------------------------------------------------------------------
# Inject pure-Python helpers into native submodules.
#
# The native .so registers e.g. ``savant_rs.deepstream`` directly in
# ``sys.modules`` which shadows the Python ``deepstream/`` directory.
# We import the helper modules (placed at the package root as ``_ds_*``)
# and attach their public symbols to the native module so that
# ``from savant_rs.deepstream import nvgstbuf_as_gpu_mat`` works at runtime.
# ---------------------------------------------------------------------------
import sys as _sys

_ds = _sys.modules.get("savant_rs.deepstream")
if _ds is not None:
    try:
        from savant_rs._ds_gpumat import (
            GpuMatCudaArray,
            from_gpumat,
            make_gpu_mat,
            nvbuf_as_gpu_mat,
            nvgstbuf_as_gpu_mat,
        )

        _ds.GpuMatCudaArray = GpuMatCudaArray
        _ds.make_gpu_mat = make_gpu_mat
        _ds.nvgstbuf_as_gpu_mat = nvgstbuf_as_gpu_mat
        _ds.nvbuf_as_gpu_mat = nvbuf_as_gpu_mat
        _ds.from_gpumat = from_gpumat
    except ImportError:
        pass

    try:
        from savant_rs._ds_skia_canvas import SkiaCanvas

        _ds.SkiaCanvas = SkiaCanvas
    except ImportError:
        pass
