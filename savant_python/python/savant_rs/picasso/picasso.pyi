"""Type stubs for ``savant_rs.picasso`` submodule.

Only available when ``savant_rs`` is built with the ``deepstream`` Cargo feature.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Optional, Union

from savant_rs.deepstream import (
    EncoderConfig,
    Rect,
    SharedBuffer,
    SurfaceView,
    TransformConfig,
)
from savant_rs.draw_spec import ObjectDraw
from savant_rs.gstreamer import Codec
from savant_rs.primitives import EndOfStream, VideoFrame

__all__ = [
    # spec
    "GeneralSpec",
    "CallbackInvocationOrder",
    "EvictionDecision",
    "ConditionalSpec",
    "ObjectDrawSpec",
    "CodecSpec",
    "SourceSpec",
    # messages
    "OutputMessage",
    # callbacks & engine
    "Callbacks",
    "PicassoEngine",
]

# ═══════════════════════════════════════════════════════════════════════════
# Spec types
# ═══════════════════════════════════════════════════════════════════════════

class PtsResetPolicy:
    """Policy for handling non-monotonic (decreasing) PTS values.

    Construct via the factory static methods.
    """

    @staticmethod
    def eos_on_decreasing_pts() -> PtsResetPolicy:
        """Emit a synthetic EOS before recreating the encoder (default).

        Downstream sees a clean EOS boundary between old and new streams.
        """
        ...

    @staticmethod
    def recreate_on_decreasing_pts() -> PtsResetPolicy:
        """Silently recreate the encoder without emitting EOS."""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class StreamResetReason:
    """Reason the worker's encoder was reset.

    Passed to the ``on_stream_reset`` callback.
    """

    @property
    def last_pts_ns(self) -> int:
        """PTS of the last successfully accepted frame (nanoseconds)."""
        ...

    @property
    def new_pts_ns(self) -> int:
        """PTS of the incoming frame that triggered the reset (nanoseconds)."""
        ...

    def __repr__(self) -> str: ...

class GeneralSpec:
    """Global defaults for the Picasso engine."""

    name: str
    idle_timeout_secs: int
    inflight_queue_size: int
    pts_reset_policy: PtsResetPolicy

    def __init__(
        self,
        name: str = "picasso",
        idle_timeout_secs: int = 30,
        inflight_queue_size: int = 8,
        pts_reset_policy: Optional[PtsResetPolicy] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

class CallbackInvocationOrder:
    """Controls when the ``on_gpumat`` callback fires relative to Skia rendering."""

    SkiaGpuMat: ClassVar[CallbackInvocationOrder]
    """Skia render then ``on_gpumat`` (default)."""

    GpuMatSkia: ClassVar[CallbackInvocationOrder]
    """``on_gpumat`` then Skia render."""

    GpuMatSkiaGpuMat: ClassVar[CallbackInvocationOrder]
    """``on_gpumat`` before **and** after Skia render."""

    @staticmethod
    def from_name(name: str) -> CallbackInvocationOrder:
        """Create from string name (``SkiaGpuMat``, ``GpuMatSkia``, ``GpuMatSkiaGpuMat``)."""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class EvictionDecision:
    """Decision returned by the ``on_eviction`` callback."""

    @staticmethod
    def keep_for(secs: int) -> EvictionDecision:
        """Keep the source alive for at least *secs* more seconds."""
        ...

    @staticmethod
    def terminate() -> EvictionDecision:
        """Drain the encoder (send EOS) then terminate the worker."""
        ...

    @staticmethod
    def terminate_immediately() -> EvictionDecision:
        """Terminate the worker immediately without draining."""
        ...

    def __repr__(self) -> str: ...

class ConditionalSpec:
    """Attribute-based gates for conditional processing."""

    encode_attribute: Optional[tuple[str, str]]
    render_attribute: Optional[tuple[str, str]]

    def __init__(
        self,
        encode_attribute: Optional[tuple[str, str]] = None,
        render_attribute: Optional[tuple[str, str]] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

class ObjectDrawSpec:
    """Static per-object draw specifications keyed by ``(namespace, label)``."""

    def __init__(self) -> None: ...
    def insert(self, namespace: str, label: str, draw: ObjectDraw) -> None:
        """Insert a draw specification for the given ``(namespace, label)`` pair."""
        ...

    def lookup(self, namespace: str, label: str) -> Optional[ObjectDraw]:
        """Look up the draw spec for an exact ``(namespace, label)`` match."""
        ...

    def is_empty(self) -> bool: ...
    def len(self) -> int: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class CodecSpec:
    """Describes what to do with each incoming frame for a given source.

    Construct via the factory static methods.
    """

    @staticmethod
    def drop_frames() -> CodecSpec:
        """Discard the frame entirely."""
        ...

    @staticmethod
    def bypass() -> CodecSpec:
        """Pass the frame through without encoding."""
        ...

    @staticmethod
    def encode(transform: TransformConfig, encoder: EncoderConfig) -> CodecSpec:
        """GPU-transform the frame, optionally render Skia overlays, then encode."""
        ...

    @property
    def is_drop(self) -> bool: ...
    @property
    def is_bypass(self) -> bool: ...
    @property
    def is_encode(self) -> bool: ...
    def __repr__(self) -> str: ...

class SourceSpec:
    """Complete per-source configuration combining all spec facets."""

    font_family: str
    idle_timeout_secs: Optional[int]
    use_on_render: bool
    use_on_gpumat: bool
    callback_order: CallbackInvocationOrder

    def __init__(
        self,
        codec: Optional[CodecSpec] = None,
        conditional: Optional[ConditionalSpec] = None,
        draw: Optional[ObjectDrawSpec] = None,
        font_family: str = "sans-serif",
        idle_timeout_secs: Optional[int] = None,
        use_on_render: bool = False,
        use_on_gpumat: bool = False,
        callback_order: CallbackInvocationOrder = ...,
    ) -> None: ...
    @property
    def codec(self) -> CodecSpec: ...
    @codec.setter
    def codec(self, value: CodecSpec) -> None: ...
    @property
    def conditional(self) -> ConditionalSpec: ...
    @conditional.setter
    def conditional(self, value: ConditionalSpec) -> None: ...
    @property
    def draw(self) -> ObjectDrawSpec: ...
    @draw.setter
    def draw(self, value: ObjectDrawSpec) -> None: ...
    def __repr__(self) -> str: ...

# ═══════════════════════════════════════════════════════════════════════════
# Output messages
# ═══════════════════════════════════════════════════════════════════════════

class OutputMessage:
    """Output produced by the encoding pipeline."""

    @property
    def is_video_frame(self) -> bool: ...
    @property
    def is_eos(self) -> bool: ...
    def as_video_frame(self) -> VideoFrame:
        """Extract the encoded ``VideoFrame``.

        Raises:
            RuntimeError: If this is an EOS output.
        """
        ...

    def as_eos(self) -> EndOfStream:
        """Extract the ``EndOfStream`` signal.

        Raises:
            RuntimeError: If this is a video-frame output.
        """
        ...

    def __repr__(self) -> str: ...

# ═══════════════════════════════════════════════════════════════════════════
# Callbacks & Engine
# ═══════════════════════════════════════════════════════════════════════════

class Callbacks:
    """Aggregate holder for all optional Python callbacks."""

    on_encoded_frame: Optional[Callable[[OutputMessage], Any]]
    on_bypass_frame: Optional[Callable[[OutputMessage], Any]]
    on_render: Optional[Callable[[str, int, int, int, VideoFrame], Any]]
    on_object_draw_spec: Optional[Callable[..., Optional[ObjectDraw]]]
    on_gpumat: Optional[Callable[[str, VideoFrame, int, int, int, int, int], Any]]
    on_eviction: Optional[Callable[[str], EvictionDecision]]
    on_stream_reset: Optional[Callable[[str, StreamResetReason], Any]]

    def __init__(
        self,
        on_encoded_frame: Optional[Callable[[OutputMessage], Any]] = None,
        on_bypass_frame: Optional[Callable[[OutputMessage], Any]] = None,
        on_render: Optional[Callable[[str, int, int, int, VideoFrame], Any]] = None,
        on_object_draw_spec: Optional[Callable[..., Optional[ObjectDraw]]] = None,
        on_gpumat: Optional[
            Callable[[str, VideoFrame, int, int, int, int, int], Any]
        ] = None,
        on_eviction: Optional[Callable[[str], EvictionDecision]] = None,
        on_stream_reset: Optional[Callable[[str, StreamResetReason], Any]] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

class PicassoEngine:
    """The main entry point for the Picasso frame-processing pipeline.

    Manages per-source worker threads, a watchdog for idle-source eviction,
    and dispatches frames to the appropriate worker.
    """

    def __init__(self, general: GeneralSpec, callbacks: Callbacks) -> None: ...
    def set_source_spec(self, source_id: str, spec: SourceSpec) -> None:
        """Set or replace the processing spec for a specific source."""
        ...

    def remove_source_spec(self, source_id: str) -> None:
        """Remove the spec for a source.  The worker will be shut down."""
        ...

    def send_frame(
        self,
        source_id: str,
        frame: VideoFrame,
        buf: Union[SurfaceView, SharedBuffer, int, Any],
        src_rect: Optional[Rect] = None,
    ) -> None:
        """Submit a video frame for processing.

        Accepts one of:

        - ``SurfaceView`` — the preferred input type.
        - Any object with ``__cuda_array_interface__`` (CuPy array,
          PyTorch CUDA tensor) — automatically wrapped in a ``SurfaceView``.
        - ``SharedBuffer`` or raw ``int`` pointer (legacy API).

        Args:
            source_id: Source identifier.
            frame: The ``VideoFrame`` proxy.
            buf: Surface data — ``SurfaceView``, ``__cuda_array_interface__``
                object, ``SharedBuffer``, or raw ``int`` pointer.
            src_rect: Optional source crop rectangle (top, left, width, height).
        """
        ...

    def send_eos(self, source_id: str) -> None:
        """Send an end-of-stream signal to a specific source."""
        ...

    def shutdown(self) -> None:
        """Gracefully shut down all workers and the watchdog."""
        ...

    def __repr__(self) -> str: ...
