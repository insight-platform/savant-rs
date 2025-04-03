from enum import Enum
from typing import Any, Optional, Union

from savant_rs.primitives import VideoFrame

__all__ = [
    "eval_expr",
    "gen_frame",
    "gen_empty_frame",
    "round_2_digits",
    "estimate_gil_contention",
    "enable_dl_detection",
    "incremental_uuid_v7",
    "relative_time_uuid_v7",
    "TelemetrySpan",
    "MaybeTelemetrySpan",
    "PropagatedContext",
    "ByteBuffer",
    "VideoObjectBBoxType",
    "VideoObjectBBoxTransformation",
    "BBoxMetricType",
]

def eval_expr(
    expr: str, ttl: int, no_gil: bool = True
) -> Union[int, float, str, bool, None, list[Any]]: ...
def gen_frame() -> VideoFrame: ...
def gen_empty_frame() -> VideoFrame: ...
def round_2_digits(num: float) -> float: ...
def estimate_gil_contention(): ...
def enable_dl_detection(): ...
def incremental_uuid_v7() -> str: ...
def relative_time_uuid_v7(uuid: str, offset_millis: int) -> str: ...

class TelemetrySpan:
    @classmethod
    def current(cls) -> TelemetrySpan: ...
    @classmethod
    def context_depth(cls) -> int: ...
    def __init__(self, name: str): ...
    @classmethod
    def default(cls) -> TelemetrySpan: ...
    def nested_span(self, name: str) -> TelemetrySpan: ...
    def nested_span_when(self, name: str, condition: bool) -> MaybeTelemetrySpan: ...
    def propagate(self) -> PropagatedContext: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    def trace_id(self) -> str: ...
    @property
    def is_valid(self) -> bool: ...
    def span_id(self) -> str: ...
    def set_string_attribute(self, key: str, value: str): ...
    def set_string_vec_attribute(self, key: str, value: list[str]): ...
    def set_bool_attribute(self, key: str, value: bool): ...
    def set_bool_vec_attribute(self, key: str, value: list[bool]): ...
    def set_int_attribute(self, key: str, value: int): ...
    def set_int_vec_attribute(self, key: str, value: list[int]): ...
    def set_float_attribute(self, key: str, value: float): ...
    def set_float_vec_attribute(self, key: str, value: list[float]): ...
    def add_event(self, name: str, attributes: dict[str, str]): ...
    def set_status_error(self, message: str): ...
    def set_status_ok(self): ...
    def set_status_unset(self): ...

class MaybeTelemetrySpan:
    def __init__(self, span: Optional[TelemetrySpan]): ...
    def nested_span(self, name: str) -> MaybeTelemetrySpan: ...
    def nested_span_when(self, name: str, condition: bool) -> MaybeTelemetrySpan: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    @classmethod
    def current(cls) -> TelemetrySpan: ...
    @property
    def is_span(self) -> bool: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def trace_id(self) -> Optional[str]: ...

class PropagatedContext:
    def nested_span(self, name: str) -> TelemetrySpan: ...
    def nested_span_when(self, name: str, condition: bool) -> MaybeTelemetrySpan: ...
    def as_dict(self) -> dict[str, str]: ...

class ByteBuffer:
    def __init__(self, data: bytes, checksum: Optional[int]): ...
    def len(self) -> int: ...
    def __len__(self): ...
    def is_empty(self) -> bool: ...
    def checksum(self) -> Optional[int]: ...
    @property
    def bytes(self) -> bytes: ...

class VideoObjectBBoxType(Enum):
    Detection: ...
    TrackingInfo: ...

class VideoObjectBBoxTransformation:
    @classmethod
    def scale(cls, x: float, y: float) -> VideoObjectBBoxTransformation: ...
    @classmethod
    def shift(cls, dx: float, dy: float) -> VideoObjectBBoxTransformation: ...

class BBoxMetricType:
    IoU: ...
    IoS: ...
    IoOther: ...
