"""Type stubs for picasso._native (Rust/PyO3 module)."""

from __future__ import annotations

from typing import Callable

from savant_rs.draw_spec import ObjectDraw
from savant_rs.primitives import EndOfStream, VideoFrame

# ── Re-exported from other native modules ────────────────────────────────

from deepstream_nvbufsurface import (
    MemType as MemType,
    TransformConfig as TransformConfig,
    VideoFormat as VideoFormat,
)
from savant_gstreamer import Codec as Codec

# ── Encoder enums ────────────────────────────────────────────────────────

class Platform:
    DGPU: Platform
    JETSON: Platform
    @staticmethod
    def from_name(name: str) -> Platform: ...

class RateControl:
    VARIABLE_BITRATE: RateControl
    CONSTANT_BITRATE: RateControl
    CONSTANT_QP: RateControl
    @staticmethod
    def from_name(name: str) -> RateControl: ...

class H264Profile:
    BASELINE: H264Profile
    MAIN: H264Profile
    HIGH: H264Profile
    HIGH444: H264Profile
    @staticmethod
    def from_name(name: str) -> H264Profile: ...

class HevcProfile:
    MAIN: HevcProfile
    MAIN10: HevcProfile
    FREXT: HevcProfile
    @staticmethod
    def from_name(name: str) -> HevcProfile: ...

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

class TuningPreset:
    HIGH_QUALITY: TuningPreset
    LOW_LATENCY: TuningPreset
    ULTRA_LOW_LATENCY: TuningPreset
    LOSSLESS: TuningPreset
    @staticmethod
    def from_name(name: str) -> TuningPreset: ...

class JetsonPresetLevel:
    DISABLED: JetsonPresetLevel
    ULTRA_FAST: JetsonPresetLevel
    FAST: JetsonPresetLevel
    MEDIUM: JetsonPresetLevel
    SLOW: JetsonPresetLevel
    @staticmethod
    def from_name(name: str) -> JetsonPresetLevel: ...

# ── Encoder property structs ─────────────────────────────────────────────

class H264DgpuProps:
    bitrate: int | None
    control_rate: RateControl | None
    profile: H264Profile | None
    iframeinterval: int | None
    idrinterval: int | None
    preset: DgpuPreset | None
    tuning_info: TuningPreset | None
    qp_range: str | None
    const_qp: str | None
    init_qp: str | None
    max_bitrate: int | None
    vbv_buf_size: int | None
    vbv_init: int | None
    cq: int | None
    aq: int | None
    temporal_aq: bool | None
    extended_colorformat: bool | None
    def __init__(self, **kwargs: object) -> None: ...

class HevcDgpuProps:
    bitrate: int | None
    control_rate: RateControl | None
    profile: HevcProfile | None
    iframeinterval: int | None
    idrinterval: int | None
    preset: DgpuPreset | None
    tuning_info: TuningPreset | None
    qp_range: str | None
    const_qp: str | None
    init_qp: str | None
    max_bitrate: int | None
    vbv_buf_size: int | None
    vbv_init: int | None
    cq: int | None
    aq: int | None
    temporal_aq: bool | None
    extended_colorformat: bool | None
    def __init__(self, **kwargs: object) -> None: ...

class H264JetsonProps:
    bitrate: int | None
    control_rate: RateControl | None
    profile: H264Profile | None
    iframeinterval: int | None
    idrinterval: int | None
    preset_level: JetsonPresetLevel | None
    peak_bitrate: int | None
    vbv_size: int | None
    qp_range: str | None
    quant_i_frames: int | None
    quant_p_frames: int | None
    ratecontrol_enable: bool | None
    maxperf_enable: bool | None
    two_pass_cbr: bool | None
    num_ref_frames: int | None
    insert_sps_pps: bool | None
    insert_aud: bool | None
    insert_vui: bool | None
    disable_cabac: bool | None
    def __init__(self, **kwargs: object) -> None: ...

class HevcJetsonProps:
    bitrate: int | None
    control_rate: RateControl | None
    profile: HevcProfile | None
    iframeinterval: int | None
    idrinterval: int | None
    preset_level: JetsonPresetLevel | None
    peak_bitrate: int | None
    vbv_size: int | None
    qp_range: str | None
    quant_i_frames: int | None
    quant_p_frames: int | None
    ratecontrol_enable: bool | None
    maxperf_enable: bool | None
    two_pass_cbr: bool | None
    num_ref_frames: int | None
    enable_lossless: bool | None
    def __init__(self, **kwargs: object) -> None: ...

class JpegProps:
    quality: int | None
    def __init__(self, quality: int | None = None) -> None: ...

class Av1DgpuProps:
    bitrate: int | None
    control_rate: RateControl | None
    iframeinterval: int | None
    idrinterval: int | None
    preset: DgpuPreset | None
    tuning_info: TuningPreset | None
    qp_range: str | None
    max_bitrate: int | None
    vbv_buf_size: int | None
    vbv_init: int | None
    cq: int | None
    aq: int | None
    temporal_aq: bool | None
    def __init__(self, **kwargs: object) -> None: ...

# ── EncoderProperties ────────────────────────────────────────────────────

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

# ── EncoderConfig ────────────────────────────────────────────────────────

class EncoderConfig:
    format: VideoFormat
    fps_num: int
    fps_den: int
    gpu_id: int
    mem_type: MemType
    encoder_params: EncoderProperties | None
    def __init__(self, codec: Codec, width: int, height: int) -> None: ...

# ── Spec types ───────────────────────────────────────────────────────────

class GeneralSpec:
    idle_timeout_secs: int
    def __init__(self, idle_timeout_secs: int = 30) -> None: ...

class EvictionDecision:
    @staticmethod
    def keep_for(secs: int) -> EvictionDecision: ...
    @staticmethod
    def terminate() -> EvictionDecision: ...
    @staticmethod
    def terminate_immediately() -> EvictionDecision: ...

class ConditionalSpec:
    encode_attribute: tuple[str, str] | None
    render_attribute: tuple[str, str] | None
    def __init__(
        self,
        encode_attribute: tuple[str, str] | None = None,
        render_attribute: tuple[str, str] | None = None,
    ) -> None: ...

class ObjectDrawSpec:
    def __init__(self) -> None: ...
    def insert(self, namespace: str, label: str, draw: ObjectDraw) -> None: ...
    def lookup(self, namespace: str, label: str) -> ObjectDraw | None: ...
    def is_empty(self) -> bool: ...
    def len(self) -> int: ...
    def __len__(self) -> int: ...

class CodecSpec:
    @staticmethod
    def drop_frames() -> CodecSpec: ...
    @staticmethod
    def bypass() -> CodecSpec: ...
    @staticmethod
    def encode(transform: TransformConfig, encoder: EncoderConfig) -> CodecSpec: ...
    @property
    def is_drop(self) -> bool: ...
    @property
    def is_bypass(self) -> bool: ...
    @property
    def is_encode(self) -> bool: ...

class SourceSpec:
    codec: CodecSpec
    conditional: ConditionalSpec
    draw: ObjectDrawSpec
    font_family: str
    idle_timeout_secs: int | None
    use_on_render: bool
    use_on_gpumat: bool
    def __init__(
        self,
        codec: CodecSpec | None = None,
        conditional: ConditionalSpec | None = None,
        draw: ObjectDrawSpec | None = None,
        font_family: str = "sans-serif",
        idle_timeout_secs: int | None = None,
        use_on_render: bool = False,
        use_on_gpumat: bool = False,
    ) -> None: ...

# ── Message types ────────────────────────────────────────────────────────

class EncodedOutput:
    @property
    def is_video_frame(self) -> bool: ...
    @property
    def is_eos(self) -> bool: ...
    def as_video_frame(self) -> VideoFrame: ...
    def as_eos(self) -> EndOfStream: ...

class BypassOutput:
    @property
    def source_id(self) -> str: ...
    @property
    def frame(self) -> VideoFrame: ...
    @property
    def buffer_ptr(self) -> int: ...

# ── Callbacks ────────────────────────────────────────────────────────────

class Callbacks:
    on_encoded_frame: Callable | None
    on_bypass_frame: Callable | None
    on_render: Callable | None
    on_object_draw_spec: Callable | None
    on_gpumat: Callable | None
    on_eviction: Callable | None
    def __init__(
        self,
        on_encoded_frame: Callable | None = None,
        on_bypass_frame: Callable | None = None,
        on_render: Callable | None = None,
        on_object_draw_spec: Callable | None = None,
        on_gpumat: Callable | None = None,
        on_eviction: Callable | None = None,
    ) -> None: ...

# ── Engine ───────────────────────────────────────────────────────────────

class PicassoEngine:
    def __init__(self, general: GeneralSpec, callbacks: Callbacks) -> None: ...
    def set_source_spec(self, source_id: str, spec: SourceSpec) -> None: ...
    def remove_source_spec(self, source_id: str) -> None: ...
    def send_frame(self, source_id: str, frame: VideoFrame, buf_ptr: int) -> None: ...
    def send_eos(self, source_id: str) -> None: ...
    def shutdown(self) -> None: ...
