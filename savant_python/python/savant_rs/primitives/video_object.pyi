from enum import Enum
from typing import Optional, List, Tuple

from savant_rs.draw_spec import SetDrawLabelKind
from savant_rs.match_query import MatchQuery
from savant_rs.primitives.geometry import RBBox
from savant_rs.utils import VideoObjectBBoxTransformation
from .attribute import Attribute
from .attribute_value import AttributeValue

__all__ = [
    'VideoObject',
    'VideoObjectBBoxType',
    'BorrowedVideoObject',
    'IdCollisionResolutionPolicy',
]

class IdCollisionResolutionPolicy(Enum):
    GenerateNewId: ...
    Overwrite: ...
    Error: ...


class VideoObjectBBoxType(Enum):
    Detection: ...
    TrackingInfo: ...


class BorrowedVideoObject:
    confidence: Optional[float]
    namespace: str
    label: str
    draw_label: str
    detection_box: RBBox
    track_id: Optional[int]
    track_box: Optional[RBBox]

    @property
    def memory_handle(self) -> int: ...

    @property
    def attributes(self) -> List[Tuple[str, str]]: ...

    def clear_attributes(self): ...

    @property
    def id(self) -> int: ...

    def delete_attribute(self, namespace: str, name: str) -> Optional[Attribute]: ...

    def delete_attributes_with_ns(self, namespace: str): ...

    def delete_attributes_with_names(self, names: List[str]): ...

    def delete_attributes_with_hints(self, hints: List[Optional[str]]): ...

    def detached_copy(self) -> 'VideoObject': ...

    def find_attributes_with_ns(self,
                                namespace: str) -> List[Tuple[str, str]]: ...

    def find_attributes_with_names(self,
                                   names: List[str]) -> List[Tuple[str, str]]: ...

    def find_attributes_with_hints(self,
                                   hints: List[Optional[str]]) -> List[Tuple[str, str]]: ...

    def get_attribute(self, namespace: str, name: str) -> Optional[Attribute]: ...

    def set_attribute(self, attribute: Attribute) -> Optional[Attribute]: ...

    def set_persistent_attribute(self,
                                 namespace: str,
                                 name: str,
                                 is_hidden: bool,
                                 hint: Optional[str],
                                 values: Optional[List[AttributeValue]]): ...

    def set_temporary_attribute(self,
                                namespace: str,
                                name: str,
                                is_hidden: bool,
                                hint: Optional[str],
                                values: Optional[List[AttributeValue]]): ...

    def set_track_info(self, track_id: int, track_box: RBBox): ...

    def clear_track_info(self): ...

    def transform_geometry(self,
                           ops: List[VideoObjectBBoxTransformation]): ...

    def to_protobuf(self, no_gil: bool = True) -> bytes: ...


class VideoObject:
    id: int
    namespace: str
    label: str
    confidence: Optional[float]
    detection_box: RBBox
    track_box: Optional[RBBox]
    track_id: Optional[int]
    draw_label: str

    def __init__(self,
                 id: int,
                 namespace: str,
                 label: str,
                 detection_box: RBBox,
                 attributes: List[Attribute],
                 confidence: Optional[float],
                 track_id: Optional[int],
                 track_box: Optional[RBBox],
                 ): ...

    def to_protobuf(self, no_gil: bool = True) -> bytes: ...

    @classmethod
    def from_protobuf(cls,
                      protobuf: bytes,
                      no_gil: bool = True) -> 'VideoObject': ...

    @property
    def namespace(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def id(self) -> int: ...

    @property
    def detection_box(self) -> RBBox: ...

    @property
    def track_id(self) -> Optional[int]: ...

    @property
    def track_box(self) -> Optional[RBBox]: ...

    @property
    def confidence(self) -> Optional[float]: ...

    @property
    def attributes(self) -> List[Tuple[str, str]]: ...

    def get_attribute(self, namespace: str, name: str) -> Optional[Attribute]: ...

    def set_attribute(self, attribute: Attribute) -> Optional[Attribute]: ...

    def set_persistent_attribute(self,
                                 namespace: str,
                                 name: str,
                                 is_hidden: bool,
                                 hint: Optional[str],
                                 values: Optional[List[AttributeValue]]): ...

    def set_temporary_attribute(self,
                                namespace: str,
                                name: str,
                                is_hidden: bool,
                                hint: Optional[str],
                                values: Optional[List[AttributeValue]]): ...


class VideoObjectsView:
    def __len__(self) -> int: ...

    def __getitem__(self, item) -> BorrowedVideoObject: ...

    def memory_handle(self) -> int: ...

    @property
    def ids(self) -> List[int]: ...

    @property
    def track_ids(self) -> List[int]: ...

    @property
    def sorted_by_id(self) -> List[BorrowedVideoObject]: ...


