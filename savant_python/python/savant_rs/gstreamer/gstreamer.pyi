"""Type stubs for ``savant_rs.gstreamer`` submodule.

The ``Codec``, ``Mp4Muxer``, ``Mp4Demuxer``, and ``DemuxedPacket`` classes
are only available when ``savant_rs`` is built with the ``gst`` Cargo feature.
"""

from __future__ import annotations

from typing import List, Optional, Union, final

__all__ = [
    "FlowResult",
    "InvocationReason",
    "Codec",
    "Mp4Muxer",
    "DemuxedPacket",
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

@final
class Codec:
    """Video codec identifier (requires ``gst`` feature).

    - ``H264``     — H.264 / AVC.
    - ``HEVC``     — H.265 / HEVC.
    - ``JPEG``     — Motion JPEG.
    - ``AV1``      — AV1.
    - ``PNG``      — PNG (CPU-based, lossless).
    - ``VP8``      — VP8.
    - ``VP9``      — VP9.
    - ``RAW_RGBA`` — Raw RGBA pixel data (no encoding).
    - ``RAW_RGB``  — Raw RGB pixel data (no encoding).
    - ``RAW_NV12`` — Raw NV12 pixel data (no encoding).
    """

    H264: Codec
    HEVC: Codec
    JPEG: Codec
    AV1: Codec
    PNG: Codec
    VP8: Codec
    VP9: Codec
    RAW_RGBA: Codec
    RAW_RGB: Codec
    RAW_NV12: Codec

    @staticmethod
    def from_name(name: str) -> Codec:
        """Parse a codec from a string name.

        Accepted names (case-insensitive): ``h264``, ``hevc``, ``h265``,
        ``jpeg``, ``av1``, ``png``, ``vp8``, ``vp9``,
        ``raw_rgba``, ``raw_rgb``, ``raw_nv12``.

        Args:
            name: Codec name.

        Returns:
            The parsed codec.

        Raises:
            ValueError: If the name is not recognized.
        """
        ...

    def name(self) -> str:
        """Return the canonical name of this codec (e.g. ``"hevc"``)."""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

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

class Mp4Demuxer:
    """Minimal GStreamer pipeline: ``filesrc -> qtdemux -> queue -> appsink``
    (requires ``gst`` feature).

    Reads encoded packets from an MP4 (QuickTime) container and exposes them
    as elementary stream payloads with timestamps.

    When ``parsed=True`` (the default), codec-specific parsers are inserted
    so that H.264/HEVC output uses byte-stream (Annex-B) format instead of
    container-native AVC/HEV1 length-prefixed NALUs.

    Args:
        input_path: Filesystem path to the ``.mp4`` file.
        parsed: If ``True``, insert parsers for byte-stream output.
            Defaults to ``True``.

    Raises:
        RuntimeError: If the file does not exist or the pipeline fails to
            start.
    """

    def __init__(self, input_path: str, parsed: bool = True) -> None: ...

    def pull(self) -> Optional[DemuxedPacket]:
        """Pull the next demuxed packet (5 s default timeout).

        Returns:
            The next packet, or ``None`` on EOS.

        Raises:
            RuntimeError: On pipeline error, timeout, or if already finished.
        """
        ...

    def pull_timeout(self, timeout_ms: int) -> Optional[DemuxedPacket]:
        """Pull the next demuxed packet with a custom timeout.

        Args:
            timeout_ms: Timeout in milliseconds.

        Returns:
            The next packet, or ``None`` on EOS.

        Raises:
            RuntimeError: On pipeline error, timeout, or if already finished.
        """
        ...

    def pull_all(self) -> List[DemuxedPacket]:
        """Pull all remaining packets until EOS.

        Returns:
            All remaining packets.

        Raises:
            RuntimeError: On pipeline error or if already finished.
        """
        ...

    @property
    def detected_codec(self) -> Optional[Codec]:
        """Auto-detected video codec from the container, or ``None``."""
        ...

    def finish(self) -> None:
        """Shut down the demuxer pipeline.

        Safe to call multiple times.
        """
        ...

    @property
    def is_finished(self) -> bool:
        """Whether the demuxer has been finalized."""
        ...
