from typing import Dict, List, Optional, Tuple, Any
from savant_rs.primitives import VideoFrame, VideoFrameBatch, VideoFrameUpdate
from savant_rs.utils import TelemetrySpan

class StageFunction:
    """A pipeline stage function."""
    
    @classmethod
    def none(cls) -> "StageFunction": ...

def handle_psf(f: StageFunction) -> None: ...

def load_stage_function_plugin(
    libname: str,
    init_name: str,
    plugin_name: str,
    params: Dict[str, Any]
) -> StageFunction: ...

class VideoPipelineStagePayloadType:
    """Defines which type of payload a stage handles."""
    Frame: VideoPipelineStagePayloadType
    Batch: VideoPipelineStagePayloadType

class FrameProcessingStatRecordType:
    """Type of frame processing stat record."""
    Initial: FrameProcessingStatRecordType
    Frame: FrameProcessingStatRecordType
    Timestamp: FrameProcessingStatRecordType

class StageLatencyMeasurements:
    """Measurements of stage latency."""
    @property
    def source_stage_name(self) -> Optional[str]: ...
    @property
    def min_latency_micros(self) -> int: ...
    @property
    def max_latency_micros(self) -> int: ...
    @property
    def accumulated_latency_millis(self) -> int: ...
    @property
    def count(self) -> int: ...
    @property
    def avg_latency_micros(self) -> int: ...

class StageLatencyStat:
    """Statistics about stage latency."""
    @property
    def stage_name(self) -> str: ...
    @property
    def latencies(self) -> List[StageLatencyMeasurements]: ...

class StageProcessingStat:
    """Statistics about stage processing."""
    @property
    def stage_name(self) -> str: ...
    @property
    def queue_length(self) -> int: ...
    @property
    def frame_counter(self) -> int: ...
    @property
    def object_counter(self) -> int: ...
    @property
    def batch_counter(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class FrameProcessingStatRecord:
    """Record of frame processing statistics."""
    @property
    def id(self) -> int: ...
    @property
    def ts(self) -> int: ...
    @property
    def frame_no(self) -> int: ...
    @property
    def record_type(self) -> FrameProcessingStatRecordType: ...
    @property
    def object_counter(self) -> int: ...
    @property
    def stage_stats(self) -> List[Tuple[StageProcessingStat, StageLatencyStat]]: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class VideoPipelineConfiguration:
    """Configuration for a video pipeline."""
    def __init__(self) -> None: ...
    @property
    def keyframe_history(self) -> int: ...
    @keyframe_history.setter
    def keyframe_history(self, v: int) -> None: ...
    @property
    def append_frame_meta_to_otlp_span(self) -> bool: ...
    @append_frame_meta_to_otlp_span.setter
    def append_frame_meta_to_otlp_span(self, v: bool) -> None: ...
    @property
    def timestamp_period(self) -> Optional[int]: ...
    @timestamp_period.setter
    def timestamp_period(self, v: Optional[int]) -> None: ...
    @property
    def frame_period(self) -> Optional[int]: ...
    @frame_period.setter
    def frame_period(self, v: Optional[int]) -> None: ...
    @property
    def collection_history(self) -> int: ...
    @collection_history.setter
    def collection_history(self, v: int) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class VideoPipeline:
    """A video pipeline."""
    def __init__(
        self,
        name: str,
        stages: List[Tuple[str, VideoPipelineStagePayloadType, StageFunction, StageFunction]],
        configuration: VideoPipelineConfiguration,
    ) -> None: ...
    
    def get_keyframe_history(self, f: VideoFrame) -> Optional[List[Tuple[int, int]]]: ...
    def get_stat_records(self, max_n: int) -> List[FrameProcessingStatRecord]: ...
    def get_stat_records_newer_than(self, id: int) -> List[FrameProcessingStatRecord]: ...
    def log_final_fps(self) -> None: ...
    def clear_source_ordering(self, source_id: str) -> None: ...
    
    @property
    def memory_handle(self) -> int: ...
    @property
    def get_root_span_name(self) -> str: ...
    
    @property
    def sampling_period(self) -> int: ...
    @sampling_period.setter
    def set_sampling_period(self, period: int) -> None: ...
    
    def get_stage_type(self, name: str) -> VideoPipelineStagePayloadType: ...
    def add_frame_update(self, frame_id: int, update: VideoFrameUpdate) -> None: ...
    def add_batched_frame_update(self, batch_id: int, frame_id: int, update: VideoFrameUpdate) -> None: ...
    def add_frame(self, stage_name: str, frame: VideoFrame) -> int: ...
    def add_frame_with_telemetry(self, stage_name: str, frame: VideoFrame, parent_span: TelemetrySpan) -> int: ...
    def delete(self, id: int) -> Dict[int, TelemetrySpan]: ...
    def get_stage_queue_len(self, stage_name: str) -> int: ...
    def get_independent_frame(self, frame_id: int) -> Tuple[VideoFrame, TelemetrySpan]: ...
    def get_batched_frame(self, batch_id: int, frame_id: int) -> Tuple[VideoFrame, TelemetrySpan]: ...
    def get_batch(self, batch_id: int) -> Tuple[VideoFrameBatch, Dict[int, TelemetrySpan]]: ...
    def apply_updates(self, id: int, no_gil: bool = True) -> None: ... 