"""Type stubs for savant_rs.gstreamer submodule.

The Codec and Mp4Muxer classes are only available when savant_rs is built
with the ``gst`` Cargo feature.
"""

from __future__ import annotations

from typing import Optional, Union, final

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

    - ``H264`` — H.264 / AVC.
    - ``HEVC`` — H.265 / HEVC.
    - ``JPEG`` — Motion JPEG.
    - ``AV1``  — AV1.
    """

    H264: Codec
    HEVC: Codec
    JPEG: Codec
    AV1: Codec

    @staticmethod
    def from_name(name: str) -> Codec:
        """Parse a codec from a string name.

        Accepted names (case-insensitive): ``h264``, ``hevc``, ``h265``,
        ``jpeg``, ``av1``.

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

    Parameters
    ----------
    codec : Codec | str
        Video codec — a :class:`Codec` enum value or a string name
        (``"h264"``, ``"hevc"`` / ``"h265"``, ``"jpeg"``, ``"av1"``).
    output_path : str
        Filesystem path for the output ``.mp4`` file.
    fps_num : int
        Framerate numerator (default 30).
    fps_den : int
        Framerate denominator (default 1).

    Raises
    ------
    ValueError
        If the codec name is not recognized.
    RuntimeError
        If the GStreamer pipeline fails to start.
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

        Parameters
        ----------
        data : bytes
            Raw encoded bitstream for a single frame.
        pts_ns : int
            Presentation timestamp in nanoseconds.
        dts_ns : int or None
            Optional decode timestamp in nanoseconds.
        duration_ns : int or None
            Optional frame duration in nanoseconds.

        Raises
        ------
        RuntimeError
            On push failure or if the muxer has been finalized.
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
