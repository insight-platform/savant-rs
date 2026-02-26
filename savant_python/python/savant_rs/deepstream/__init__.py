# This directory is shadowed at runtime by the native submodule that
# the .so registers in sys.modules["savant_rs.deepstream"].
# Pure-Python helpers (nvgstbuf_as_gpu_mat, SkiaCanvas, etc.) are injected by
# savant_rs/__init__.py via _ds_gpumat.py / _ds_skia_canvas.py.
#
# The deepstream.pyi file in this directory provides type stubs.
