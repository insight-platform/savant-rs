Picasso GPU Video Pipeline (savant_rs.picasso)
===============================================

The Picasso GPU video processing pipeline bindings are part of the
``savant_rs`` package and are available when ``savant_rs`` is built with the
``deepstream`` Cargo feature (which automatically enables the ``gst``
feature).

Picasso provides a multi-source, GPU-accelerated frame processing engine
with Skia rendering, hardware encoding (H.264, HEVC, JPEG, AV1), and
per-source configuration.

.. note::

   These bindings were previously shipped as the standalone ``picasso``
   Python package (``picasso_py`` crate).  They have been consolidated
   into ``savant_rs.picasso`` as of version 1.15.

.. automodule:: savant_rs.picasso
   :members:
   :undoc-members:
