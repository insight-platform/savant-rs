"""Type stubs for the deepstream_encoders._native Rust extension."""

from __future__ import annotations

from typing import Optional, Union, final

# ── Codec enum ───────────────────────────────────────────────────────────

@final
class Codec:
    """Video codec identifier.

    This is the same ``Codec`` enum as :class:`savant_gstreamer.Codec`,
    but registered under ``deepstream_encoders._native`` so that values
    returned by getters (e.g. ``config.codec``) are the same Python type
    as ``Codec.HEVC`` imported from this package.

    Cross-package interoperability is handled automatically: both
    :class:`~savant_gstreamer.Mp4Muxer` and :class:`EncoderConfig` accept
    a ``Codec`` from either package or a plain ``str``.

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

# ── EncoderConfig ────────────────────────────────────────────────────────

class EncoderConfig:
    """Configuration for creating an :class:`NvEncoder`.

    The internal buffer pools are always configured with exactly 1 buffer.
    This is required because the NVENC hardware encoder may continue
    DMA-reading from GPU memory after releasing the GStreamer buffer
    reference.  A pool of 1 forces serialization that prevents stale-data
    artifacts.

    Args:
        codec: Video codec — a :class:`Codec` enum value or a string name.
        width: Frame width in pixels.
        height: Frame height in pixels.
        format: Video format (default ``"NV12"``).
        fps_num: Framerate numerator (default 30).
        fps_den: Framerate denominator (default 1).
        gpu_id: GPU device ID (default 0).
        mem_type: NvBufSurface memory type (default 0).
        encoder_properties: Encoder-specific GStreamer properties as
            string key/value pairs.  B-frame properties are rejected.

    Raises:
        ValueError: If an encoder property name refers to B-frames.

    Example::

        config = EncoderConfig(
            Codec.HEVC, 1920, 1080,
            encoder_properties={"bitrate": "4000000"},
        )
    """

    def __init__(
        self,
        codec: Union[Codec, str],
        width: int,
        height: int,
        format: str = "NV12",
        fps_num: int = 30,
        fps_den: int = 1,
        gpu_id: int = 0,
        mem_type: int = 0,
        encoder_properties: dict[str, str] | None = None,
    ) -> None: ...

    @property
    def codec(self) -> Codec:
        """Video codec."""
        ...

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        ...

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        ...

    @property
    def format(self) -> str:
        """Video format string (e.g. ``"NV12"``, ``"RGBA"``)."""
        ...

    def __repr__(self) -> str: ...

# ── EncodedFrame ─────────────────────────────────────────────────────────

class EncodedFrame:
    """A single encoded frame returned by :meth:`NvEncoder.pull_encoded`.

    This object is read-only; it is produced by the encoder pipeline.
    """

    @property
    def frame_id(self) -> int:
        """User-defined frame identifier."""
        ...

    @property
    def pts_ns(self) -> int:
        """Presentation timestamp in nanoseconds."""
        ...

    @property
    def dts_ns(self) -> int | None:
        """Decode timestamp in nanoseconds (if set by the encoder).

        For streams without B-frames this is typically equal to PTS
        or ``None``.
        """
        ...

    @property
    def duration_ns(self) -> int | None:
        """Duration in nanoseconds (if known)."""
        ...

    @property
    def data(self) -> bytes:
        """Encoded bitstream data."""
        ...

    @property
    def codec(self) -> Codec:
        """Codec used to produce this frame."""
        ...

    @property
    def size(self) -> int:
        """Size of the encoded data in bytes."""
        ...

    def __repr__(self) -> str: ...

# ── NvEncoder ────────────────────────────────────────────────────────────

class NvEncoder:
    """GPU-accelerated video encoder.

    Creates an internal GStreamer pipeline that encodes NVMM buffers
    using hardware-accelerated NVENC / NVJPEG encoders.

    The encoder:

    - Rejects any property that would enable B-frames.
    - Validates that PTS values are strictly monotonically increasing.
    - Provides access to the internal
      :class:`~deepstream_nvbufsurface.NvBufSurfaceGenerator` for
      acquiring GPU buffers.

    Args:
        config: Encoder configuration.

    Example::

        from deepstream_encoders import NvEncoder, EncoderConfig, Codec
        from deepstream_nvbufsurface import init_cuda

        init_cuda()
        config = EncoderConfig(Codec.HEVC, 1920, 1080)
        encoder = NvEncoder(config)

        for i in range(100):
            buf = encoder.acquire_surface(id=i)
            encoder.submit_frame(buf, frame_id=i,
                                 pts_ns=i * 33_333_333,
                                 duration_ns=33_333_333)

        remaining = encoder.finish()
    """

    def __init__(self, config: EncoderConfig) -> None: ...

    @property
    def codec(self) -> Codec:
        """The codec used by this encoder."""
        ...

    def nvmm_caps_str(self) -> str:
        """Return the NVMM caps string for the internal generator.

        Returns:
            Caps string with ``memory:NVMM`` feature.
        """
        ...

    def acquire_surface(self, id: Optional[int] = None) -> int:
        """Acquire a new NvBufSurface buffer from the internal pool.

        This is a convenience shortcut for
        ``encoder.generator.acquire_surface(id=...)``.

        Args:
            id: Optional frame identifier for SavantIdMeta.

        Returns:
            Raw GstBuffer pointer address.
        """
        ...

    def submit_frame(
        self,
        buffer_ptr: int,
        frame_id: int,
        pts_ns: int,
        duration_ns: Optional[int] = None,
    ) -> None:
        """Submit a filled NVMM buffer to the encoder.

        The buffer must have been acquired from :meth:`acquire_surface` or
        from the generator directly.  PTS values must be strictly
        monotonically increasing.

        Args:
            buffer_ptr: Raw GstBuffer pointer (from :meth:`acquire_surface`).
            frame_id: User-defined frame identifier.
            pts_ns: Presentation timestamp in nanoseconds.
            duration_ns: Optional duration in nanoseconds.

        Raises:
            ValueError: On PTS reordering or null buffer pointer.
            RuntimeError: On pipeline error or if the encoder has been
                finalized.
        """
        ...

    def pull_encoded(self) -> EncodedFrame | None:
        """Pull one encoded frame (non-blocking).

        Returns:
            The encoded frame, or ``None`` if no frame is ready yet.
        """
        ...

    def pull_encoded_timeout(
        self, timeout_ms: int = 100
    ) -> EncodedFrame | None:
        """Pull one encoded frame with a timeout.

        Args:
            timeout_ms: Maximum time to wait in milliseconds (default 100).

        Returns:
            The encoded frame, or ``None`` on timeout.
        """
        ...

    def finish(
        self, drain_timeout_ms: Optional[int] = None
    ) -> list[EncodedFrame]:
        """Send EOS and drain all remaining encoded frames.

        After this call, no more frames can be submitted.

        Args:
            drain_timeout_ms: Per-frame drain timeout in ms (default 2000).

        Returns:
            Remaining encoded frames from the pipeline.
        """
        ...

    def check_error(self) -> None:
        """Check the pipeline bus for errors (non-blocking).

        Raises:
            RuntimeError: If a pipeline error is pending.
        """
        ...
