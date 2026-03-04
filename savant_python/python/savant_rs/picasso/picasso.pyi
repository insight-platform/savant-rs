"""Type stubs for ``savant_rs.picasso`` submodule.

Only available when ``savant_rs`` is built with the ``deepstream`` Cargo feature.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union, final

from savant_rs.deepstream import (
    GstBuffer,
    MemType,
    Rect,
    SurfaceView,
    TransformConfig,
    VideoFormat,
)
from savant_rs.draw_spec import ObjectDraw
from savant_rs.gstreamer import Codec
from savant_rs.primitives import EndOfStream, VideoFrame

__all__ = [
    # encoder enums
    "Platform",
    "RateControl",
    "H264Profile",
    "HevcProfile",
    "DgpuPreset",
    "TuningPreset",
    "JetsonPresetLevel",
    # encoder property structs
    "H264DgpuProps",
    "HevcDgpuProps",
    "H264JetsonProps",
    "HevcJetsonProps",
    "JpegProps",
    "PngProps",
    "Av1DgpuProps",
    "EncoderProperties",
    "EncoderConfig",
    # spec
    "GeneralSpec",
    "EvictionDecision",
    "ConditionalSpec",
    "ObjectDrawSpec",
    "CodecSpec",
    "SourceSpec",
    # messages
    "EncodedOutput",
    "BypassOutput",
    # callbacks & engine
    "Callbacks",
    "PicassoEngine",
]

# ═══════════════════════════════════════════════════════════════════════════
# Encoder enums
# ═══════════════════════════════════════════════════════════════════════════

@final
class Platform:
    DGPU: Platform
    JETSON: Platform

    @staticmethod
    def from_name(name: str) -> Platform: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class RateControl:
    VARIABLE_BITRATE: RateControl
    CONSTANT_BITRATE: RateControl
    CONSTANT_QP: RateControl

    @staticmethod
    def from_name(name: str) -> RateControl: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class H264Profile:
    BASELINE: H264Profile
    MAIN: H264Profile
    HIGH: H264Profile
    HIGH444: H264Profile

    @staticmethod
    def from_name(name: str) -> H264Profile: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class HevcProfile:
    MAIN: HevcProfile
    MAIN10: HevcProfile
    FREXT: HevcProfile

    @staticmethod
    def from_name(name: str) -> HevcProfile: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class DgpuPreset:
    P1: DgpuPreset
    P2: DgpuPreset
    P3: DgpuPreset
    P4: DgpuPreset
    P5: DgpuPreset
    P6: DgpuPreset
    P7: DgpuPreset

    @staticmethod
    def from_name(name: str) -> DgpuPreset: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class TuningPreset:
    HIGH_QUALITY: TuningPreset
    LOW_LATENCY: TuningPreset
    ULTRA_LOW_LATENCY: TuningPreset
    LOSSLESS: TuningPreset

    @staticmethod
    def from_name(name: str) -> TuningPreset: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class JetsonPresetLevel:
    DISABLED: JetsonPresetLevel
    ULTRA_FAST: JetsonPresetLevel
    FAST: JetsonPresetLevel
    MEDIUM: JetsonPresetLevel
    SLOW: JetsonPresetLevel

    @staticmethod
    def from_name(name: str) -> JetsonPresetLevel: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

# ═══════════════════════════════════════════════════════════════════════════
# Per-codec encoder property structs
# ═══════════════════════════════════════════════════════════════════════════

class H264DgpuProps:
    bitrate: Optional[int]
    control_rate: Optional[RateControl]
    profile: Optional[H264Profile]
    iframeinterval: Optional[int]
    idrinterval: Optional[int]
    preset: Optional[DgpuPreset]
    tuning_info: Optional[TuningPreset]
    qp_range: Optional[str]
    const_qp: Optional[str]
    init_qp: Optional[str]
    max_bitrate: Optional[int]
    vbv_buf_size: Optional[int]
    vbv_init: Optional[int]
    cq: Optional[int]
    aq: Optional[int]
    temporal_aq: Optional[bool]
    extended_colorformat: Optional[bool]

    def __init__(
        self,
        bitrate: Optional[int] = None,
        control_rate: Optional[RateControl] = None,
        profile: Optional[H264Profile] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset: Optional[DgpuPreset] = None,
        tuning_info: Optional[TuningPreset] = None,
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
    def __repr__(self) -> str: ...

class HevcDgpuProps:
    bitrate: Optional[int]
    control_rate: Optional[RateControl]
    profile: Optional[HevcProfile]
    iframeinterval: Optional[int]
    idrinterval: Optional[int]
    preset: Optional[DgpuPreset]
    tuning_info: Optional[TuningPreset]
    qp_range: Optional[str]
    const_qp: Optional[str]
    init_qp: Optional[str]
    max_bitrate: Optional[int]
    vbv_buf_size: Optional[int]
    vbv_init: Optional[int]
    cq: Optional[int]
    aq: Optional[int]
    temporal_aq: Optional[bool]
    extended_colorformat: Optional[bool]

    def __init__(
        self,
        bitrate: Optional[int] = None,
        control_rate: Optional[RateControl] = None,
        profile: Optional[HevcProfile] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset: Optional[DgpuPreset] = None,
        tuning_info: Optional[TuningPreset] = None,
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
    def __repr__(self) -> str: ...

class H264JetsonProps:
    bitrate: Optional[int]
    control_rate: Optional[RateControl]
    profile: Optional[H264Profile]
    iframeinterval: Optional[int]
    idrinterval: Optional[int]
    preset_level: Optional[JetsonPresetLevel]
    peak_bitrate: Optional[int]
    vbv_size: Optional[int]
    qp_range: Optional[str]
    quant_i_frames: Optional[int]
    quant_p_frames: Optional[int]
    ratecontrol_enable: Optional[bool]
    maxperf_enable: Optional[bool]
    two_pass_cbr: Optional[bool]
    num_ref_frames: Optional[int]
    insert_sps_pps: Optional[bool]
    insert_aud: Optional[bool]
    insert_vui: Optional[bool]
    disable_cabac: Optional[bool]

    def __init__(
        self,
        bitrate: Optional[int] = None,
        control_rate: Optional[RateControl] = None,
        profile: Optional[H264Profile] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset_level: Optional[JetsonPresetLevel] = None,
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
    def __repr__(self) -> str: ...

class HevcJetsonProps:
    bitrate: Optional[int]
    control_rate: Optional[RateControl]
    profile: Optional[HevcProfile]
    iframeinterval: Optional[int]
    idrinterval: Optional[int]
    preset_level: Optional[JetsonPresetLevel]
    peak_bitrate: Optional[int]
    vbv_size: Optional[int]
    qp_range: Optional[str]
    quant_i_frames: Optional[int]
    quant_p_frames: Optional[int]
    ratecontrol_enable: Optional[bool]
    maxperf_enable: Optional[bool]
    two_pass_cbr: Optional[bool]
    num_ref_frames: Optional[int]
    enable_lossless: Optional[bool]

    def __init__(
        self,
        bitrate: Optional[int] = None,
        control_rate: Optional[RateControl] = None,
        profile: Optional[HevcProfile] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset_level: Optional[JetsonPresetLevel] = None,
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
    def __repr__(self) -> str: ...

class JpegProps:
    quality: Optional[int]

    def __init__(self, quality: Optional[int] = None) -> None: ...
    def __repr__(self) -> str: ...

class PngProps:
    """PNG encoder properties (CPU-based, ``pngenc`` from gst-plugins-good)."""

    compression_level: Optional[int]

    def __init__(self, compression_level: Optional[int] = None) -> None: ...
    def __repr__(self) -> str: ...

class Av1DgpuProps:
    bitrate: Optional[int]
    control_rate: Optional[RateControl]
    iframeinterval: Optional[int]
    idrinterval: Optional[int]
    preset: Optional[DgpuPreset]
    tuning_info: Optional[TuningPreset]
    qp_range: Optional[str]
    max_bitrate: Optional[int]
    vbv_buf_size: Optional[int]
    vbv_init: Optional[int]
    cq: Optional[int]
    aq: Optional[int]
    temporal_aq: Optional[bool]

    def __init__(
        self,
        bitrate: Optional[int] = None,
        control_rate: Optional[RateControl] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset: Optional[DgpuPreset] = None,
        tuning_info: Optional[TuningPreset] = None,
        qp_range: Optional[str] = None,
        max_bitrate: Optional[int] = None,
        vbv_buf_size: Optional[int] = None,
        vbv_init: Optional[int] = None,
        cq: Optional[int] = None,
        aq: Optional[int] = None,
        temporal_aq: Optional[bool] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

# ═══════════════════════════════════════════════════════════════════════════
# EncoderProperties (tagged union via factory statics)
# ═══════════════════════════════════════════════════════════════════════════

class EncoderProperties:
    @staticmethod
    def h264_dgpu(props: H264DgpuProps) -> EncoderProperties: ...
    @staticmethod
    def h264_jetson(props: H264JetsonProps) -> EncoderProperties: ...
    @staticmethod
    def hevc_dgpu(props: HevcDgpuProps) -> EncoderProperties: ...
    @staticmethod
    def hevc_jetson(props: HevcJetsonProps) -> EncoderProperties: ...
    @staticmethod
    def jpeg(props: JpegProps) -> EncoderProperties: ...
    @staticmethod
    def av1_dgpu(props: Av1DgpuProps) -> EncoderProperties: ...
    @staticmethod
    def png(props: PngProps) -> EncoderProperties: ...
    def __repr__(self) -> str: ...

# ═══════════════════════════════════════════════════════════════════════════
# EncoderConfig
# ═══════════════════════════════════════════════════════════════════════════

class EncoderConfig:
    """Encoder configuration.

    Uses a builder pattern — chain ``.format(...).fps(...).gpu_id(...)``
    etc. after construction.

    **Important:** The builder methods ``format()``, ``gpu_id()`` shadow
    the underlying property setters.  At runtime, property assignment
    (``cfg.gpu_id = 0``) raises ``AttributeError: read-only``.
    Always use the builder method call form::

        cfg = EncoderConfig(Codec.H264, 1280, 720)
        cfg.format(VideoFormat.RGBA)   # builder call — OK
        cfg.gpu_id(0)                  # builder call — OK
        cfg.fps(30, 1)
        cfg.properties(props)
    """

    def __init__(self, codec: Codec, width: int, height: int) -> None: ...

    # ── read-only property getters ──
    # Note: ``format`` and ``gpu_id`` have builder methods with the same
    # name that shadow the property setter; use the builder call form.
    @property
    def fps_num(self) -> int: ...
    @property
    def fps_den(self) -> int: ...
    @property
    def mem_type(self) -> MemType: ...
    @property
    def encoder_params(self) -> Optional[EncoderProperties]: ...

    # ── builder methods (return self for chaining) ──
    def format(self, fmt: VideoFormat) -> EncoderConfig: ...
    def fps(self, num: int, den: int) -> EncoderConfig: ...
    def gpu_id(self, id: int) -> EncoderConfig: ...
    def properties(self, props: EncoderProperties) -> EncoderConfig: ...
    def __repr__(self) -> str: ...

# ═══════════════════════════════════════════════════════════════════════════
# Spec types
# ═══════════════════════════════════════════════════════════════════════════

class GeneralSpec:
    """Global defaults for the Picasso engine."""

    idle_timeout_secs: int

    def __init__(self, idle_timeout_secs: int = 30) -> None: ...
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

    def __init__(
        self,
        codec: Optional[CodecSpec] = None,
        conditional: Optional[ConditionalSpec] = None,
        draw: Optional[ObjectDrawSpec] = None,
        font_family: str = "sans-serif",
        idle_timeout_secs: Optional[int] = None,
        use_on_render: bool = False,
        use_on_gpumat: bool = False,
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

class EncodedOutput:
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

class BypassOutput:
    """Output for bypass mode — frame with bboxes transformed back to
    initial coordinates.  The GstBuffer is dropped on the Rust side."""

    @property
    def source_id(self) -> str: ...
    @property
    def frame(self) -> VideoFrame: ...
    def __repr__(self) -> str: ...

# ═══════════════════════════════════════════════════════════════════════════
# Callbacks & Engine
# ═══════════════════════════════════════════════════════════════════════════

class Callbacks:
    """Aggregate holder for all optional Python callbacks."""

    on_encoded_frame: Optional[Callable[[EncodedOutput], Any]]
    on_bypass_frame: Optional[Callable[[BypassOutput], Any]]
    on_render: Optional[Callable[[str, int, int, int, VideoFrame], Any]]
    on_object_draw_spec: Optional[Callable[..., Optional[ObjectDraw]]]
    on_gpumat: Optional[Callable[[str, VideoFrame, int, int, int, int], Any]]
    on_eviction: Optional[Callable[[str], EvictionDecision]]

    def __init__(
        self,
        on_encoded_frame: Optional[Callable[[EncodedOutput], Any]] = None,
        on_bypass_frame: Optional[Callable[[BypassOutput], Any]] = None,
        on_render: Optional[Callable[[str, int, int, int, VideoFrame], Any]] = None,
        on_object_draw_spec: Optional[Callable[..., Optional[ObjectDraw]]] = None,
        on_gpumat: Optional[
            Callable[[str, VideoFrame, int, int, int, int], Any]
        ] = None,
        on_eviction: Optional[Callable[[str], EvictionDecision]] = None,
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
        buf: Union[SurfaceView, GstBuffer, int, Any],
        src_rect: Optional[Rect] = None,
    ) -> None:
        """Submit a video frame for processing.

        Accepts one of:

        - ``SurfaceView`` — the preferred input type.
        - Any object with ``__cuda_array_interface__`` (CuPy array,
          PyTorch CUDA tensor) — automatically wrapped in a ``SurfaceView``.
        - ``GstBuffer`` or raw ``int`` pointer (legacy API).

        Args:
            source_id: Source identifier.
            frame: The ``VideoFrame`` proxy.
            buf: Surface data — ``SurfaceView``, ``__cuda_array_interface__``
                object, ``GstBuffer``, or raw ``int`` pointer.
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
