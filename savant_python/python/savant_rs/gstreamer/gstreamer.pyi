from enum import Enum
from typing import Optional, List
from savant_rs.primitives import Attribute


class GstBuffer:
    @property
    def raw_pointer(self) -> int: ...

    @property
    def pts(self) -> Optional[int]: ...

    @property
    def dts(self) -> Optional[int]: ...

    @property
    def dts_or_pts(self) -> Optional[int]: ...

    @property
    def duration(self) -> Optional[int]: ...

    @property
    def flags(self) -> int: ...

    def unset_flags(self, flags: int) -> int: ...

    @property
    def maxsize(self) -> int: ...

    @property
    def n_memory(self) -> int: ...

    @property
    def offset(self) -> int: ...

    @property
    def offset_end(self) -> int: ...

    @property
    def size(self) -> int: ...

    @property
    def is_writable(self) -> bool: ...

    def copy(self) -> GstBuffer: ...

    def copy_deep(self) -> GstBuffer: ...

    def append(self, buf: GstBuffer) -> GstBuffer: ...

    # fn get_savant_meta(&self) -> Vec<Attribute>
    @property
    def id_meta(self) -> List[Attribute]: ...

    # pub fn replace_id_meta(&self, ids: Vec<i64>) -> PyResult<Option<Vec<i64>>>
    def replace_id_meta(self, ids: List[int]) -> Optional[List[int]]: ...

    # pub fn clear_id_meta(&self) -> PyResult<Option<Vec<i64>>>
    def clear_id_meta(self) -> Optional[List[int]]: ...


class FlowResult(Enum):
    CustomSuccess2: ...
    CustomSuccess1: ...
    CustomSuccess: ...
    Ok: ...
    NotLinked: ...
    Flushing: ...
    Eos: ...
    NotNegotiated: ...
    Error: ...
    NotSupported: ...
    CustomError: ...
    CustomError1: ...
    CustomError2: ...


class InvocationReason(Enum):
    Buffer: ...
    SinkEvent: ...
    SourceEvent: ...
