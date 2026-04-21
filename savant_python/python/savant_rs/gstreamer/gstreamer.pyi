"""Type stubs for ``savant_rs.gstreamer`` submodule.

The ``Mp4Muxer``, ``Mp4Demuxer``, and ``DemuxedPacket`` classes are only
available when ``savant_rs`` is built with the ``gst`` Cargo feature.

``Codec`` is the same class as :class:`savant_rs.primitives.Codec` —
re-exported here for convenience so existing ``from savant_rs.gstreamer
import Codec`` imports keep working.
"""

from __future__ import annotations

from typing import Callable, Optional, Union, final

from savant_rs.primitives import Codec

__all__ = [
    "FlowResult",
    "InvocationReason",
    "Codec",
    "Mp4Muxer",
    "DemuxedPacket",
    "VideoInfo",
    "Mp4DemuxerOutput",
    "Mp4Demuxer",
]

# ── Always available ─────────────────────────────────────────────────────

@final
class FlowResult:
    CustomSuccess2: FlowResult
    CustomSuccess1: FlowResult
    CustomSuccess: FlowResult
    Ok: FlowResult
    NotLinked: FlowResult
    Flushing: FlowResult
    Eos: FlowResult
    NotNegotiated: FlowResult
    Error: FlowResult
    NotSupported: FlowResult
    CustomError: FlowResult
    CustomError1: FlowResult
    CustomError2: FlowResult

@final
class InvocationReason:
    Buffer: InvocationReason
    SinkEvent: InvocationReason
    SourceEvent: InvocationReason
    StateChange: InvocationReason
    IngressMessageTransformer: InvocationReason

# ── Available with gst feature ───────────────────────────────────────────

# ``Codec`` is imported from :mod:`savant_rs.primitives` above; see
# :class:`savant_rs.primitives.Codec` for the full enum definition.

class Mp4Muxer:
    """Minimal GStreamer pipeline: ``appsrc -> parser -> qtmux -> filesink``
    (requires ``gst`` feature).

    Accepts raw encoded frames (H.264, HEVC, JPEG, AV1) and writes them
    into an MP4 (QuickTime) container.

    Args:
        codec: Video codec — a :class:`Codec` enum value or a string name.
        output_path: Filesystem path for the output ``.mp4`` file.
        fps_num: Framerate numerator (default 30).
        fps_den: Framerate denominator (default 1).

    Raises:
        ValueError: If the codec name is not recognized.
        RuntimeError: If the GStreamer pipeline fails to start.
    """

    def __init__(
        self,
        codec: Union[Codec, str],
        output_path: str,
        fps_num: int = 30,
        fps_den: int = 1,
    ) -> None: ...
    def push(
        self,
        data: bytes,
        pts_ns: int,
        dts_ns: Optional[int] = None,
        duration_ns: Optional[int] = None,
    ) -> None:
        """Push an encoded frame into the muxer pipeline.

        Args:
            data: Raw encoded bitstream for a single frame.
            pts_ns: Presentation timestamp in nanoseconds.
            dts_ns: Optional decode timestamp in nanoseconds.
            duration_ns: Optional frame duration in nanoseconds.

        Raises:
            RuntimeError: On push failure or if the muxer has been finalized.
        """
        ...

    def finish(self) -> None:
        """Send EOS and shut down the muxer pipeline.

        Safe to call multiple times. After this call, :meth:`push` will
        raise :class:`RuntimeError`.
        """
        ...

    @property
    def is_finished(self) -> bool:
        """Whether the muxer has been finalized."""
        ...

@final
class DemuxedPacket:
    """A single demuxed elementary stream packet.

    Attributes:
        data: Encoded bitstream payload.
        pts_ns: Presentation timestamp in nanoseconds.
        dts_ns: Decode timestamp in nanoseconds, or ``None``.
        duration_ns: Frame duration in nanoseconds, or ``None``.
        is_keyframe: Whether this packet is a keyframe (sync point).
    """

    @property
    def data(self) -> bytes:
        """Encoded bitstream payload."""
        ...

    @property
    def pts_ns(self) -> int:
        """Presentation timestamp in nanoseconds."""
        ...

    @property
    def dts_ns(self) -> Optional[int]:
        """Decode timestamp in nanoseconds, or ``None``."""
        ...

    @property
    def duration_ns(self) -> Optional[int]:
        """Frame duration in nanoseconds, or ``None``."""
        ...

    @property
    def is_keyframe(self) -> bool:
        """Whether this packet is a keyframe."""
        ...

    def __repr__(self) -> str: ...

@final
class VideoInfo:
    """Video-stream metadata extracted from the MP4 container at demux time.

    Dimensions are the *encoded* values; any QuickTime display-orientation
    transform is not applied here.

    Attributes:
        codec: Detected video codec.
        width: Encoded frame width in pixels.
        height: Encoded frame height in pixels.
        framerate_num: Framerate numerator (``0`` if unknown).
        framerate_den: Framerate denominator (``1`` if unknown).
    """

    @property
    def codec(self) -> Codec: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def framerate_num(self) -> int: ...
    @property
    def framerate_den(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class Mp4DemuxerOutput:
    """Callback payload from :class:`Mp4Demuxer`.

    Use the ``is_*`` properties to determine the variant, then call the
    corresponding ``as_*`` method to get a typed value.

    Variants:
        - **StreamInfo** — container-derived :class:`VideoInfo` metadata.
          Fires once before the first packet.
        - **Packet** — a demuxed :class:`DemuxedPacket`.
        - **Eos** — end of stream; all packets have been delivered.
        - **Error** — a pipeline error message (string).
    """

    @property
    def is_packet(self) -> bool:
        """``True`` if this is a :class:`DemuxedPacket` variant."""
        ...

    @property
    def is_eos(self) -> bool:
        """``True`` if this is an end-of-stream marker."""
        ...

    @property
    def is_error(self) -> bool:
        """``True`` if this is an error variant."""
        ...

    def as_packet(self) -> Optional[DemuxedPacket]:
        """Downcast to :class:`DemuxedPacket`, or ``None``."""
        ...

    def as_error_message(self) -> Optional[str]:
        """Return the error message string, or ``None``."""
        ...

    @property
    def is_stream_info(self) -> bool:
        """``True`` if this is a :class:`VideoInfo` variant."""
        ...

    def as_stream_info(self) -> Optional[VideoInfo]:
        """Downcast to :class:`VideoInfo`, or ``None``."""
        ...

    def __repr__(self) -> str: ...

class Mp4Demuxer:
    """Callback-based GStreamer pipeline: ``filesrc -> qtdemux -> queue -> appsink``
    (requires ``gst`` feature).

    Reads encoded packets from an MP4 (QuickTime) container and delivers them
    through the ``result_callback`` supplied at construction. A single
    ``StreamInfo`` (:class:`VideoInfo`) callback is emitted before the first
    ``Packet`` when stream caps are observed.

    When ``parsed=True`` (the default), codec-specific parsers are inserted
    so that H.264/HEVC output uses byte-stream (Annex-B) format instead of
    container-native AVC/HEV1 length-prefixed NALUs.

    The pipeline starts immediately on construction.  Use :meth:`wait` to
    block until all packets have been delivered (EOS) or an error occurs.

    **Threading**: the callback fires on GStreamer's internal streaming
    thread.  Do **not** call :meth:`finish` from within the callback.

    Args:
        input_path: Filesystem path to the ``.mp4`` file.
        result_callback: ``Callable[[Mp4DemuxerOutput], None]`` invoked for
            stream-info, packet, EOS, or error outputs.
        parsed: If ``True``, insert parsers for byte-stream output.
            Defaults to ``True``.

    Raises:
        RuntimeError: If the file does not exist or the pipeline fails to
            start.
    """

    def __init__(
        self,
        input_path: str,
        result_callback: Callable[[Mp4DemuxerOutput], None],
        parsed: bool = True,
    ) -> None: ...
    def wait(self) -> None:
        """Block until the demuxer reaches EOS, encounters an error, or
        :meth:`finish` is called.

        The GIL is released while waiting so the callback can fire.
        """
        ...

    def wait_timeout(self, timeout_ms: int) -> bool:
        """Block until the demuxer finishes or the timeout expires.

        Args:
            timeout_ms: Timeout in milliseconds.

        Returns:
            ``True`` if finished, ``False`` on timeout.
        """
        ...

    @property
    def detected_codec(self) -> Optional[Codec]:
        """Auto-detected video codec from the container, or ``None``."""
        ...

    @property
    def video_info(self) -> Optional[VideoInfo]:
        """Auto-detected :class:`VideoInfo` from the container, or ``None``."""
        ...

    def wait_for_video_info(self, timeout_ms: int) -> Optional[VideoInfo]:
        """Block until :class:`VideoInfo` is known, the pipeline terminates,
        or the timeout expires.

        Args:
            timeout_ms: Timeout in milliseconds.

        Returns:
            The detected :class:`VideoInfo`, or ``None`` on timeout or if the
            pipeline ended before caps were observed.

        The GIL is released while waiting.
        """
        ...

    def finish(self) -> None:
        """Shut down the demuxer pipeline.

        Safe to call multiple times.  After this call, no more callbacks
        will fire.

        Must **not** be called from within the ``result_callback``.
        """
        ...

    @property
    def is_finished(self) -> bool:
        """Whether the demuxer has been finalized."""
        ...
