"""Type stubs for ``savant_rs.nvtracker`` (deepstream feature)."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, final

from savant_rs.deepstream import SavantIdMetaKind, SharedBuffer, VideoFormat
from savant_rs.nvinfer import Roi
from savant_rs.primitives import VideoFrame

__all__ = [
    "TrackingIdResetMode",
    "TrackState",
    "NvTrackerConfig",
    "TrackedFrame",
    "TrackedObject",
    "MiscTrackFrame",
    "MiscTrackData",
    "TrackerOutput",
    "NvTracker",
    "NvTrackerBatchingOperatorConfig",
    "TrackerBatchFormationResult",
    "TrackerOperatorFrameOutput",
    "SealedDeliveries",
    "TrackerOperatorOutput",
    "NvTrackerBatchingOperator",
]

@final
class TrackingIdResetMode:
    NONE: TrackingIdResetMode
    ON_STREAM_RESET: TrackingIdResetMode
    ON_EOS: TrackingIdResetMode
    ON_STREAM_RESET_AND_EOS: TrackingIdResetMode

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...

@final
class TrackState:
    EMPTY: TrackState
    ACTIVE: TrackState
    INACTIVE: TrackState
    TENTATIVE: TrackState
    PROJECTED: TrackState

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...

@final
class NvTrackerConfig:
    def __init__(
        self,
        ll_lib_file: str,
        ll_config_file: str,
        input_format: VideoFormat,
        *,
        name: str = ...,
        tracker_width: int = ...,
        tracker_height: int = ...,
        max_batch_size: int = ...,
        gpu_id: int = ...,
        element_properties: Optional[dict[str, str]] = ...,
        tracking_id_reset_mode: TrackingIdResetMode = ...,
        queue_depth: int = ...,
    ) -> None: ...

    @property
    def ll_lib_file(self) -> str: ...

    @property
    def ll_config_file(self) -> str: ...

    @property
    def queue_depth(self) -> int: ...

@final
class TrackedFrame:
    """One frame to track: source name, single-surface NVMM buffer, and
    detections keyed by ``class_id``.

    Args:
        source: Stream name (e.g. ``"cam-1"``).
        buffer: Single-surface NVMM buffer (consumed on construction).
        rois: Detections grouped by class_id.
    """

    def __init__(
        self,
        source: str,
        buffer: SharedBuffer,
        rois: Dict[int, List[Roi]],
    ) -> None: ...

    @property
    def source(self) -> str: ...

    def __repr__(self) -> str: ...

@final
class TrackedObject:
    @property
    def object_id(self) -> int: ...
    @property
    def class_id(self) -> int: ...
    @property
    def bbox_left(self) -> float: ...
    @property
    def bbox_top(self) -> float: ...
    @property
    def bbox_width(self) -> float: ...
    @property
    def bbox_height(self) -> float: ...
    @property
    def confidence(self) -> float: ...
    @property
    def tracker_confidence(self) -> float: ...
    @property
    def label(self) -> Optional[str]: ...
    @property
    def slot_number(self) -> int: ...
    @property
    def source_id(self) -> str: ...

@final
class MiscTrackFrame:
    @property
    def frame_num(self) -> int: ...
    @property
    def bbox_left(self) -> float: ...
    @property
    def bbox_top(self) -> float: ...
    @property
    def bbox_width(self) -> float: ...
    @property
    def bbox_height(self) -> float: ...
    @property
    def confidence(self) -> float: ...
    @property
    def age(self) -> int: ...
    @property
    def state(self) -> TrackState: ...
    @property
    def visibility(self) -> float: ...

@final
class MiscTrackData:
    @property
    def object_id(self) -> int: ...
    @property
    def class_id(self) -> int: ...
    @property
    def label(self) -> Optional[str]: ...
    @property
    def source_id(self) -> str: ...
    @property
    def frames(self) -> List[MiscTrackFrame]: ...

@final
class TrackerOutput:
    @property
    def current_tracks(self) -> List[TrackedObject]: ...
    @property
    def shadow_tracks(self) -> List[MiscTrackData]: ...
    @property
    def terminated_tracks(self) -> List[MiscTrackData]: ...
    @property
    def past_frame_data(self) -> List[MiscTrackData]: ...

    def buffer(self) -> SharedBuffer: ...
    def __repr__(self) -> str: ...

@final
class NvTracker:
    def __init__(
        self,
        config: NvTrackerConfig,
        callback: Callable[[TrackerOutput], None],
    ) -> None: ...

    def track(
        self,
        frames: List[TrackedFrame],
        ids: List[Tuple[SavantIdMetaKind, int]],
    ) -> None: ...

    def track_sync(
        self,
        frames: List[TrackedFrame],
        ids: List[Tuple[SavantIdMetaKind, int]],
    ) -> TrackerOutput: ...

    def reset_stream(self, source_id: str) -> None: ...
    def shutdown(self) -> None: ...
    def __repr__(self) -> str: ...

@final
class NvTrackerBatchingOperatorConfig:
    def __init__(
        self,
        max_batch_size: int,
        max_batch_wait_ms: int,
        nvtracker_config: NvTrackerConfig,
    ) -> None: ...

    @property
    def max_batch_size(self) -> int: ...

    @property
    def max_batch_wait_ms(self) -> int: ...

    @property
    def nvtracker_config(self) -> NvTrackerConfig: ...

@final
class TrackerBatchFormationResult:
    def __init__(
        self,
        ids: List[Tuple[SavantIdMetaKind, int]],
        rois: List[Dict[int, List[Roi]]],
    ) -> None: ...

    @property
    def ids(self) -> List[Tuple[SavantIdMetaKind, int]]: ...

@final
class TrackerOperatorFrameOutput:
    @property
    def frame(self) -> "VideoFrame": ...
    @property
    def tracked_objects(self) -> List[TrackedObject]: ...
    @property
    def shadow_tracks(self) -> List[MiscTrackData]: ...
    @property
    def terminated_tracks(self) -> List[MiscTrackData]: ...
    @property
    def past_frame_data(self) -> List[MiscTrackData]: ...

@final
class SealedDeliveries:
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    def is_released(self) -> bool: ...
    def unseal(
        self,
        timeout_ms: Optional[int] = ...,
    ) -> List[Tuple["VideoFrame", SharedBuffer]]: ...
    def try_unseal(self) -> Optional[List[Tuple["VideoFrame", SharedBuffer]]]: ...

@final
class TrackerOperatorOutput:
    @property
    def frames(self) -> List[TrackerOperatorFrameOutput]: ...
    @property
    def num_frames(self) -> int: ...
    def take_deliveries(self) -> Optional[SealedDeliveries]: ...

@final
class NvTrackerBatchingOperator:
    def __init__(
        self,
        config: NvTrackerBatchingOperatorConfig,
        batch_formation_callback: Callable[[List["VideoFrame"]], TrackerBatchFormationResult],
        result_callback: Callable[[TrackerOperatorOutput], None],
    ) -> None: ...

    def add_frame(self, frame: "VideoFrame", buffer: SharedBuffer | int) -> None: ...
    def flush(self) -> None: ...
    def reset_stream(self, source_id: str) -> None: ...
    def shutdown(self) -> None: ...
