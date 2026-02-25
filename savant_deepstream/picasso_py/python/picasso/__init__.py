"""Picasso — GPU-accelerated video processing pipeline with Python bindings.

Re-exports all types from the native Rust module and provides the
``_OnRenderAdapter`` that converts raw FBO info into a ``SkiaCanvas``
for end users.

Quick start::

    from picasso import (
        PicassoEngine, Callbacks, GeneralSpec, SourceSpec,
        CodecSpec, EncoderConfig, TransformConfig,
    )

    def on_encoded(output):
        if output.is_video_frame:
            frame = output.as_video_frame()
            ...

    callbacks = Callbacks(on_encoded_frame=on_encoded)
    engine = PicassoEngine(GeneralSpec(), callbacks)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

# ── Native (Rust/PyO3) symbols ───────────────────────────────────────────
from picasso._native import (  # noqa: F401
    # Encoder enums
    Av1DgpuProps,
    BypassOutput,
    Codec,
    CodecSpec,
    ConditionalSpec,
    DgpuPreset,
    EncodedOutput,
    EncoderConfig,
    EncoderProperties,
    EvictionDecision,
    GeneralSpec,
    H264DgpuProps,
    H264JetsonProps,
    H264Profile,
    HevcDgpuProps,
    HevcJetsonProps,
    HevcProfile,
    JetsonPresetLevel,
    JpegProps,
    MemType,
    ObjectDrawSpec,
    PicassoEngine,
    Platform,
    RateControl,
    SourceSpec,
    TuningPreset,
    VideoFormat,
)
from picasso._native import Callbacks as _NativeCallbacks

if TYPE_CHECKING:
    from savant_rs.primitives import VideoFrame  # noqa: F401


# ── OnRender adapter ────────────────────────────────────────────────────


class _OnRenderAdapter:
    """Wraps a user callback ``(source_id, skia_canvas, frame)`` to the
    raw signature ``(source_id, fbo_id, width, height, frame)`` used by
    the native module.  Caches the ``SkiaCanvas`` across calls.
    """

    __slots__ = ("_user_cb", "_canvas", "_fbo_id")

    def __init__(self, user_cb: Callable) -> None:
        self._user_cb = user_cb
        self._canvas = None
        self._fbo_id: int | None = None

    def __call__(
        self,
        source_id: str,
        fbo_id: int,
        width: int,
        height: int,
        frame: object,
    ) -> None:
        if self._canvas is None or self._fbo_id != fbo_id:
            from deepstream_nvbufsurface import SkiaCanvas

            self._canvas = SkiaCanvas.from_fbo(fbo_id, width, height)
            self._fbo_id = fbo_id
        self._user_cb(source_id, self._canvas, frame)


# ── High-level Callbacks wrapper ────────────────────────────────────────


class Callbacks(_NativeCallbacks):
    """Extended ``Callbacks`` that automatically wraps ``on_render`` with
    the ``_OnRenderAdapter`` so that users receive a ``SkiaCanvas`` instead
    of raw FBO info.

    All other callback slots are passed through unchanged.
    """

    def __init__(
        self,
        *,
        on_encoded_frame: Callable | None = None,
        on_bypass_frame: Callable | None = None,
        on_render: Callable | None = None,
        on_object_draw_spec: Callable | None = None,
        on_gpumat: Callable | None = None,
        on_eviction: Callable | None = None,
    ) -> None:
        adapted_render = _OnRenderAdapter(on_render) if on_render is not None else None
        super().__init__(
            on_encoded_frame=on_encoded_frame,
            on_bypass_frame=on_bypass_frame,
            on_render=adapted_render,
            on_object_draw_spec=on_object_draw_spec,
            on_gpumat=on_gpumat,
            on_eviction=on_eviction,
        )

    @_NativeCallbacks.on_render.setter  # type: ignore[attr-defined]
    def on_render(self, cb: Callable | None) -> None:  # type: ignore[override]
        adapted = _OnRenderAdapter(cb) if cb is not None else None
        _NativeCallbacks.on_render.fset(self, adapted)  # type: ignore[union-attr]


__all__ = [
    # Engine
    "PicassoEngine",
    # Callbacks
    "Callbacks",
    # Specs
    "GeneralSpec",
    "EvictionDecision",
    "SourceSpec",
    "CodecSpec",
    "ConditionalSpec",
    "ObjectDrawSpec",
    # Messages
    "EncodedOutput",
    "BypassOutput",
    # Encoder enums
    "Platform",
    "RateControl",
    "H264Profile",
    "HevcProfile",
    "DgpuPreset",
    "TuningPreset",
    "JetsonPresetLevel",
    # Encoder props
    "H264DgpuProps",
    "HevcDgpuProps",
    "H264JetsonProps",
    "HevcJetsonProps",
    "JpegProps",
    "Av1DgpuProps",
    "EncoderProperties",
    "EncoderConfig",
    # Re-exports from other packages
    "Codec",
    "VideoFormat",
    "MemType",
]
