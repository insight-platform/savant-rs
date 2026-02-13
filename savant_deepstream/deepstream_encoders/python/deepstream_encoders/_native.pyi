"""Type stubs for the deepstream_encoders._native Rust extension."""

from __future__ import annotations

from typing import Optional, Union, final

# ── Codec enum ───────────────────────────────────────────────────────────

@final
class Codec:
    """Video codec identifier.

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
        """Parse a codec from a string name."""
        ...

    def name(self) -> str:
        """Return the canonical name of this codec (e.g. ``"hevc"``)."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...


# ── VideoFormat enum ──────────────────────────────────────────────────────

@final
class VideoFormat:
    """Video pixel format.

    - ``RGBA``  — 8-bit RGBA (4 bytes/pixel).
    - ``BGRx``  — 8-bit BGRx (4 bytes/pixel, alpha ignored).
    - ``NV12``  — YUV 4:2:0 semi-planar (default encoder format).
    - ``NV21``  — YUV 4:2:0 semi-planar (UV swapped).
    - ``I420``  — YUV 4:2:0 planar (JPEG encoder format).
    - ``UYVY``  — YUV 4:2:2 packed.
    - ``GRAY8`` — single-channel grayscale.
    """

    RGBA: VideoFormat
    BGRx: VideoFormat
    NV12: VideoFormat
    NV21: VideoFormat
    I420: VideoFormat
    UYVY: VideoFormat
    GRAY8: VideoFormat

    @staticmethod
    def from_name(name: str) -> VideoFormat:
        """Parse a format from a string name."""
        ...

    def name(self) -> str:
        """Return the canonical name (e.g. ``"NV12"``)."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...


# ── MemType enum ──────────────────────────────────────────────────────────

@final
class MemType:
    """NvBufSurface memory type.

    - ``DEFAULT``       — CUDA Device for dGPU, Surface Array for Jetson.
    - ``CUDA_PINNED``   — CUDA Host (pinned) memory.
    - ``CUDA_DEVICE``   — CUDA Device memory.
    - ``CUDA_UNIFIED``  — CUDA Unified memory.
    - ``SURFACE_ARRAY`` — NVRM Surface Array (Jetson only).
    - ``HANDLE``        — NVRM Handle (Jetson only).
    - ``SYSTEM``        — System memory (malloc).
    """

    DEFAULT: MemType
    CUDA_PINNED: MemType
    CUDA_DEVICE: MemType
    CUDA_UNIFIED: MemType
    SURFACE_ARRAY: MemType
    HANDLE: MemType
    SYSTEM: MemType

    def name(self) -> str:
        """Return the canonical name."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...


# ── Platform enum ─────────────────────────────────────────────────────────

@final
class Platform:
    """Hardware platform.

    - ``DGPU``  — Discrete GPU.
    - ``JETSON`` — NVIDIA Jetson.
    """

    DGPU: Platform
    JETSON: Platform

    @staticmethod
    def from_name(name: str) -> Platform:
        """Parse a platform from a string name."""
        ...

    def name(self) -> str:
        """Return the canonical name."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...


# ── RateControl enum ──────────────────────────────────────────────────────

@final
class RateControl:
    """Rate control mode.

    - ``VBR`` — Variable bitrate.
    - ``CBR`` — Constant bitrate.
    - ``CQP`` — Constant quantization parameter.
    """

    VBR: RateControl
    CBR: RateControl
    CQP: RateControl

    @staticmethod
    def from_name(name: str) -> RateControl:
        """Parse a rate control mode from a string name."""
        ...

    def name(self) -> str:
        """Return the canonical name."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...


# ── H264Profile enum ──────────────────────────────────────────────────────

@final
class H264Profile:
    """H.264 profile.

    - ``BASELINE`` — Baseline profile.
    - ``MAIN``     — Main profile.
    - ``HIGH``     — High profile.
    - ``HIGH444``  — High 4:4:4 profile.
    """

    BASELINE: H264Profile
    MAIN: H264Profile
    HIGH: H264Profile
    HIGH444: H264Profile

    @staticmethod
    def from_name(name: str) -> H264Profile:
        """Parse an H.264 profile from a string name."""
        ...

    def name(self) -> str:
        """Return the canonical name."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...


# ── HevcProfile enum ───────────────────────────────────────────────────────

@final
class HevcProfile:
    """HEVC profile.

    - ``MAIN``   — Main profile.
    - ``MAIN10`` — Main 10 profile.
    - ``FREXT``  — Frext profile.
    """

    MAIN: HevcProfile
    MAIN10: HevcProfile
    FREXT: HevcProfile

    @staticmethod
    def from_name(name: str) -> HevcProfile:
        """Parse an HEVC profile from a string name."""
        ...

    def name(self) -> str:
        """Return the canonical name."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...


# ── DgpuPreset enum ────────────────────────────────────────────────────────

@final
class DgpuPreset:
    """dGPU encoding preset (P1–P7, fastest to slowest).

    - ``P1`` — Fastest.
    - ``P2``
    - ``P3``
    - ``P4``
    - ``P5``
    - ``P6``
    - ``P7`` — Slowest (highest quality).
    """

    P1: DgpuPreset
    P2: DgpuPreset
    P3: DgpuPreset
    P4: DgpuPreset
    P5: DgpuPreset
    P6: DgpuPreset
    P7: DgpuPreset

    @staticmethod
    def from_name(name: str) -> DgpuPreset:
        """Parse a dGPU preset from a string name."""
        ...

    def name(self) -> str:
        """Return the canonical name."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...


# ── TuningPreset enum ──────────────────────────────────────────────────────

@final
class TuningPreset:
    """Tuning preset for dGPU encoders.

    - ``HIGH_QUALITY``       — Optimize for quality.
    - ``LOW_LATENCY``        — Optimize for low latency.
    - ``ULTRA_LOW_LATENCY``  — Optimize for ultra-low latency.
    - ``LOSSLESS``           — Lossless encoding.
    """

    HIGH_QUALITY: TuningPreset
    LOW_LATENCY: TuningPreset
    ULTRA_LOW_LATENCY: TuningPreset
    LOSSLESS: TuningPreset

    @staticmethod
    def from_name(name: str) -> TuningPreset:
        """Parse a tuning preset from a string name."""
        ...

    def name(self) -> str:
        """Return the canonical name."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...


# ── JetsonPresetLevel enum ─────────────────────────────────────────────────

@final
class JetsonPresetLevel:
    """Jetson encoding preset level.

    - ``DISABLED``      — Preset disabled.
    - ``ULTRA_FAST``    — Ultra-fast encoding.
    - ``FAST``          — Fast encoding.
    - ``MEDIUM``        — Medium encoding.
    - ``SLOW``          — Slow encoding (highest quality).
    """

    DISABLED: JetsonPresetLevel
    ULTRA_FAST: JetsonPresetLevel
    FAST: JetsonPresetLevel
    MEDIUM: JetsonPresetLevel
    SLOW: JetsonPresetLevel

    @staticmethod
    def from_name(name: str) -> JetsonPresetLevel:
        """Parse a Jetson preset level from a string name."""
        ...

    def name(self) -> str:
        """Return the canonical name."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...


# ── Property structs ───────────────────────────────────────────────────────

class H264DgpuProps:
    """H.264 encoder properties for dGPU."""

    def __init__(
        self,
        *,
        bitrate: Optional[int] = None,
        control_rate: Optional[Union[RateControl, str]] = None,
        profile: Optional[Union[H264Profile, str]] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset: Optional[Union[DgpuPreset, str]] = None,
        tuning_info: Optional[Union[TuningPreset, str]] = None,
        qp_range: Optional[str] = None,
        const_qp: Optional[str] = None,
        init_qp: Optional[str] = None,
        max_bitrate: Optional[int] = None,
        vbv_buf_size: Optional[int] = None,
        vbv_init: Optional[int] = None,
        cq: Optional[int] = None,
        aq: Optional[int] = None,
        temporal_aq: Optional[bool] = None,
        extended_colorformat: Optional[bool] = None,
    ) -> None: ...

    @staticmethod
    def from_pairs(pairs: dict[str, str]) -> H264DgpuProps:
        """Create from key-value pairs."""
        ...


class HevcDgpuProps:
    """HEVC encoder properties for dGPU."""

    def __init__(
        self,
        *,
        bitrate: Optional[int] = None,
        control_rate: Optional[Union[RateControl, str]] = None,
        profile: Optional[Union[HevcProfile, str]] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset: Optional[Union[DgpuPreset, str]] = None,
        tuning_info: Optional[Union[TuningPreset, str]] = None,
        qp_range: Optional[str] = None,
        const_qp: Optional[str] = None,
        init_qp: Optional[str] = None,
        max_bitrate: Optional[int] = None,
        vbv_buf_size: Optional[int] = None,
        vbv_init: Optional[int] = None,
        cq: Optional[int] = None,
        aq: Optional[int] = None,
        temporal_aq: Optional[bool] = None,
        extended_colorformat: Optional[bool] = None,
    ) -> None: ...

    @staticmethod
    def from_pairs(pairs: dict[str, str]) -> HevcDgpuProps:
        """Create from key-value pairs."""
        ...


class H264JetsonProps:
    """H.264 encoder properties for Jetson."""

    def __init__(
        self,
        *,
        bitrate: Optional[int] = None,
        control_rate: Optional[Union[RateControl, str]] = None,
        profile: Optional[Union[H264Profile, str]] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset_level: Optional[Union[JetsonPresetLevel, str]] = None,
        peak_bitrate: Optional[int] = None,
        vbv_size: Optional[int] = None,
        qp_range: Optional[str] = None,
        quant_i_frames: Optional[int] = None,
        quant_p_frames: Optional[int] = None,
        ratecontrol_enable: Optional[bool] = None,
        maxperf_enable: Optional[bool] = None,
        two_pass_cbr: Optional[bool] = None,
        num_ref_frames: Optional[int] = None,
        insert_sps_pps: Optional[bool] = None,
        insert_aud: Optional[bool] = None,
        insert_vui: Optional[bool] = None,
        disable_cabac: Optional[bool] = None,
    ) -> None: ...

    @staticmethod
    def from_pairs(pairs: dict[str, str]) -> H264JetsonProps:
        """Create from key-value pairs."""
        ...


class HevcJetsonProps:
    """HEVC encoder properties for Jetson."""

    def __init__(
        self,
        *,
        bitrate: Optional[int] = None,
        control_rate: Optional[Union[RateControl, str]] = None,
        profile: Optional[Union[HevcProfile, str]] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset_level: Optional[Union[JetsonPresetLevel, str]] = None,
        peak_bitrate: Optional[int] = None,
        vbv_size: Optional[int] = None,
        qp_range: Optional[str] = None,
        quant_i_frames: Optional[int] = None,
        quant_p_frames: Optional[int] = None,
        ratecontrol_enable: Optional[bool] = None,
        maxperf_enable: Optional[bool] = None,
        two_pass_cbr: Optional[bool] = None,
        num_ref_frames: Optional[int] = None,
        enable_lossless: Optional[bool] = None,
    ) -> None: ...

    @staticmethod
    def from_pairs(pairs: dict[str, str]) -> HevcJetsonProps:
        """Create from key-value pairs."""
        ...


class JpegProps:
    """JPEG encoder properties."""

    def __init__(
        self,
        *,
        quality: Optional[int] = None,
    ) -> None: ...

    @staticmethod
    def from_pairs(pairs: dict[str, str]) -> JpegProps:
        """Create from key-value pairs."""
        ...


class Av1DgpuProps:
    """AV1 encoder properties for dGPU."""

    def __init__(
        self,
        *,
        bitrate: Optional[int] = None,
        control_rate: Optional[Union[RateControl, str]] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset: Optional[Union[DgpuPreset, str]] = None,
        tuning_info: Optional[Union[TuningPreset, str]] = None,
        qp_range: Optional[str] = None,
        max_bitrate: Optional[int] = None,
        vbv_buf_size: Optional[int] = None,
        vbv_init: Optional[int] = None,
        cq: Optional[int] = None,
        aq: Optional[int] = None,
        temporal_aq: Optional[bool] = None,
    ) -> None: ...

    @staticmethod
    def from_pairs(pairs: dict[str, str]) -> Av1DgpuProps:
        """Create from key-value pairs."""
        ...


# ── EncoderConfig ─────────────────────────────────────────────────────────

class EncoderConfig:
    """Configuration for creating an :class:`NvEncoder`."""

    def __init__(
        self,
        codec: Union[Codec, str],
        width: int,
        height: int,
        format: Union[VideoFormat, str] | None = None,
        fps_num: int = 30,
        fps_den: int = 1,
        gpu_id: int = 0,
        mem_type: Union[MemType, int] | None = None,
        properties: Union[
            H264DgpuProps,
            HevcDgpuProps,
            H264JetsonProps,
            HevcJetsonProps,
            JpegProps,
            Av1DgpuProps,
            None,
        ] = None,
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
    def format(self) -> VideoFormat:
        """Video format (e.g. ``VideoFormat.NV12``, ``VideoFormat.RGBA``)."""
        ...

    def __repr__(self) -> str: ...


# ── EncodedFrame ───────────────────────────────────────────────────────────

class EncodedFrame:
    """A single encoded frame returned by :meth:`NvEncoder.pull_encoded`."""

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
        """Decode timestamp in nanoseconds (if set by the encoder)."""
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


# ── NvEncoder ──────────────────────────────────────────────────────────────

class NvEncoder:
    """GPU-accelerated video encoder."""

    def __init__(self, config: EncoderConfig) -> None: ...

    @property
    def codec(self) -> Codec:
        """The codec used by this encoder."""
        ...

    def nvmm_caps_str(self) -> str:
        """Return the NVMM caps string for the internal generator."""
        ...

    def acquire_surface(self, id: Optional[int] = None) -> int:
        """Acquire a new NvBufSurface buffer from the internal pool."""
        ...

    def submit_frame(
        self,
        buffer_ptr: int,
        frame_id: int,
        pts_ns: int,
        duration_ns: Optional[int] = None,
    ) -> None:
        """Submit a filled NVMM buffer to the encoder."""
        ...

    def pull_encoded(self) -> EncodedFrame | None:
        """Pull one encoded frame (non-blocking)."""
        ...

    def pull_encoded_timeout(
        self, timeout_ms: int = 100
    ) -> EncodedFrame | None:
        """Pull one encoded frame with a timeout."""
        ...

    def finish(
        self, drain_timeout_ms: Optional[int] = None
    ) -> list[EncodedFrame]:
        """Send EOS and drain all remaining encoded frames."""
        ...

    def check_error(self) -> None:
        """Check the pipeline bus for errors (non-blocking)."""
        ...
