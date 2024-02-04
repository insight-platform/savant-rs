from enum import Enum
from typing import Optional

from savant_rs.draw_spec import SetDrawLabelKind
from savant_rs.match_query import MatchQuery
from savant_rs.primitives.geometry import Intersection, RBBox, Point, PolygonalArea
from savant_rs.utils import VideoObjectBBoxTransformation
from savant_rs.utils.serialization import Message


class AttributeValueType(Enum):
    Bytes: ...
    String: ...
    StringList: ...
    Integer: ...
    IntegerList: ...
    Float: ...
    FloatList: ...
    Boolean: ...
    BooleanList: ...
    BBox: ...
    BBoxList: ...
    Point: ...
    PointList: ...
    Polygon: ...
    PolygonList: ...
    Intersection: ...
    TemporaryValue: ...
    None_: ...


class AttributeValue:
    confidence: Optional[float]

    def get_value_type(self) -> AttributeValueType: ...

    @classmethod
    def intersection(cls,
                     intersection: Intersection,
                     confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def none(cls) -> AttributeValue: ...

    @classmethod
    def temporary_python_object(cls,
                                python_object: object,
                                confidence: Optional[float] = None) -> AttributeValue: ...

    # pub fn bytes_from_list(dims: Vec<i64>, blob: Vec<u8>, confidence: Option<f32>) -> Self
    @classmethod
    def bytes_from_list(cls,
                        dims: list[int],
                        blob: list[int],
                        confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def string(cls,
               string: str,
               confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def strings(cls,
                strings: list[str],
                confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def integer(cls,
                integer: int,
                confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def integers(cls,
                 integers: list[int],
                 confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def floats(cls,
               floats: list[float],
               confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def boolean(cls,
                boolean: bool,
                confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def booleans(cls,
                 booleans: list[bool],
                 confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def bbox(cls,
             bbox: RBBox,
             confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def bboxes(cls,
               bboxes: list[RBBox],
               confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def point(cls,
              point: Point,
              confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def points(cls,
               points: list[Point],
               confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def polygon(cls,
                polygon: PolygonalArea,
                confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def polygons(cls,
                 polygons: list[PolygonalArea],
                 confidence: Optional[float] = None) -> AttributeValue: ...

    def is_none(self) -> bool: ...

    def as_bytes(self) -> Optional[list[int], bytes]: ...

    def as_intersection(self) -> Optional[Intersection]: ...

    def as_string(self) -> Optional[str]: ...

    def as_strings(self) -> Optional[list[str]]: ...

    def as_temporary_python_object(self) -> Optional[object]: ...

    def as_integer(self) -> Optional[int]: ...

    def as_integers(self) -> Optional[list[int]]: ...

    def as_float(self) -> Optional[float]: ...

    def as_floats(self) -> Optional[list[float]]: ...

    def as_boolean(self) -> Optional[bool]: ...

    def as_booleans(self) -> Optional[list[bool]]: ...

    def as_bbox(self) -> Optional[RBBox]: ...

    def as_bboxes(self) -> Optional[list[RBBox]]: ...

    def as_point(self) -> Optional[Point]: ...

    def as_points(self) -> Optional[list[Point]]: ...

    def as_polygon(self) -> Optional[PolygonalArea]: ...

    def as_polygons(self) -> Optional[list[PolygonalArea]]: ...

    # pub fn bytes(dims: Vec<i64>, blob: &PyBytes, confidence: Option<f32>)
    @classmethod
    def bytes(cls,
              dims: list[int],
              blob: bytes,
              confidence: Optional[float] = None) -> AttributeValue: ...

    @classmethod
    def float(cls,
              float: float,
              confidence: Optional[float] = None) -> AttributeValue: ...

    @property
    def json(self) -> str: ...

    @classmethod
    def from_json(cls, json: str) -> AttributeValue: ...


class AttributeValueView:
    def __getitem__(self, item): ...

    @property
    def memory_handle(self) -> int: ...

    def __len__(self) -> int: ...


class Attribute:
    values: list[AttributeValue]

    def __init__(self,
                 namespace: str,
                 name: str,
                 values: list[AttributeValue],
                 hint: Optional[str],
                 is_persistent: bool = True,
                 is_hidden: bool = False): ...

    @classmethod
    def persistent(cls,
                   namespace: str,
                   name: str,
                   values: list[AttributeValue],
                   hint: Optional[str] = None,
                   is_hidden: bool = False): ...

    @classmethod
    def temporary(cls,
                  namespace: str,
                  name: str,
                  values: list[AttributeValue],
                  hint: Optional[str] = None,
                  is_hidden: bool = False): ...

    def is_temporary(self) -> bool: ...

    def is_hidden(self) -> bool: ...

    def make_peristent(self): ...

    def make_temporary(self): ...

    @property
    def namespace(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def values_view(self) -> AttributeValueView: ...

    @property
    def hint(self) -> Optional[str]: ...

    @property
    def json(self) -> str: ...

    @classmethod
    def from_json(cls, json: str) -> Attribute: ...


class AttributeUpdatePolicy(Enum):
    ReplaceWithForeignWhenDuplicate: ...
    KeepOwnWhenDuplicate: ...
    ErrorWhenDuplicate: ...


class ObjectUpdatePolicy(Enum):
    AddForeignObjects: ...
    ErrorIfLabelsCollide: ...
    ReplaceSameLabelObjects: ...


class EndOfStream:
    def __init__(self, source_id: str): ...

    @property
    def source_id(self) -> str: ...

    @property
    def json(self) -> str: ...

    def to_message(self) -> Message: ...


class Shutdown:
    def __init__(self, auth: str): ...

    @property
    def auth(self) -> str: ...

    @property
    def json(self) -> str: ...

    def to_message(self) -> Message: ...


class UserData:
    def __init__(self, source_id: str): ...

    @property
    def source_id(self) -> str: ...

    @property
    def json(self) -> str: ...

    def to_message(self) -> Message: ...

    @property
    def attributes(self) -> list[(str, str)]: ...

    def get_attribute(self,
                      namespace: str,
                      name: str) -> Optional[Attribute]: ...

    def find_attributes_with_ns(self,
                                namespace: str) -> list[(str, str)]: ...

    def find_attributes_with_names(self,
                                   names: list[str]) -> list[(str, str)]: ...

    def find_attributes_with_hints(self,
                                   hints: list[Optional[str]]) -> list[(str, str)]: ...

    def delete_attributes_with_ns(self, namespace: str): ...

    def delete_attributes_with_names(self, names: list[str]): ...

    def delete_attributes_with_hints(self,
                                     hints: list[Optional[str]]): ...

    def delete_attribute(self, namespace: str, name: str) -> Optional[Attribute]: ...

    def set_attribute(self, attribute: Attribute) -> Optional[Attribute]: ...

    def set_persistent_attribute(self,
                                 namespace: str,
                                 name: str,
                                 is_hidden: bool,
                                 hint: Optional[str],
                                 values: Optional[list[AttributeValue]]): ...

    def set_temporary_attribute(self,
                                namespace: str,
                                name: str,
                                is_hidden: bool,
                                hint: Optional[str],
                                values: Optional[list[AttributeValue]]): ...

    def clear_attributes(self): ...

    @property
    def json(self) -> str: ...

    @property
    def json_pretty(self) -> str: ...

    def to_protobuf(self, no_gil: bool = True) -> bytes: ...

    @classmethod
    def from_protobuf(cls,
                      protobuf: bytes,
                      no_gil: bool = True) -> UserData: ...


class ExternalFrame:
    method: str
    location: Optional[str]

    def __init__(self,
                 method: str,
                 location: Optional[str]): ...


class VideoFrameContent:
    @classmethod
    def external(cls, method: str, location: Optional[str]) -> VideoFrameContent: ...

    @classmethod
    def internal(cls, data: bytes) -> VideoFrameContent: ...

    @classmethod
    def none(cls) -> VideoFrameContent: ...

    def is_external(self) -> bool: ...

    def is_internal(self) -> bool: ...

    def is_none(self) -> bool: ...

    def get_data(self) -> bytes: ...

    def get_method(self) -> str: ...

    def get_location(self) -> Optional[str]: ...


class VideoFrameTranscodingMethod(Enum):
    Copy: ...
    Encoded: ...


class VideoFrameTransformation:
    @classmethod
    def initial_size(cls, width: int, height: int) -> VideoFrameTransformation: ...

    @classmethod
    def resulting_size(cls, width: int, height: int) -> VideoFrameTransformation: ...

    @classmethod
    def scale(cls, width: int, height: int) -> VideoFrameTransformation: ...

    @classmethod
    def padding(cls, left: int, top: int, right: int, bottom: int) -> VideoFrameTransformation: ...

    @property
    def is_initial_size(self) -> bool: ...

    @property
    def is_scale(self) -> bool: ...

    @property
    def is_padding(self) -> bool: ...

    @property
    def is_resulting_size(self) -> bool: ...

    @property
    def as_initial_size(self) -> Optional[tuple[int, int]]: ...

    @property
    def as_resulting_size(self) -> Optional[tuple[int, int]]: ...

    @property
    def as_scale(self) -> Optional[tuple[int, int]]: ...

    @property
    def as_padding(self) -> Optional[tuple[int, int, int, int]]: ...


class VideoFrame:
    source_id: str
    time_base: tuple[int, int]
    pts: int
    dts: Optional[int]
    uuid: str
    creation_timestamp_ns: int
    framerate: str
    width: int
    height: int
    duration: Optional[int]
    transcoding_method: VideoFrameTranscodingMethod
    codec: Optional[str]
    content: VideoFrameContent

    @classmethod
    def transform_geometry(cls,
                           ops: list[VideoObjectBBoxTransformation],
                           no_gil: bool = True): ...

    @property
    def memory_handle(self) -> int: ...

    def __init__(self,
                 source_id: str,
                 framerate: str,
                 width: int,
                 height: int,
                 content: VideoFrameContent,
                 transcoding_method: VideoFrameTranscodingMethod,
                 codec: Optional[str],
                 keyframe: Optional[bool],
                 time_base: tuple[int, int],
                 pts: int,
                 dts: Optional[int],
                 duration: Optional[int],
                 ): ...

    def to_message(self) -> Message: ...

    @property
    def previous_frame_seq_id(self) -> Optional[int]: ...

    @property
    def json(self) -> str: ...

    @property
    def json_pretty(self) -> str: ...

    def clear_transformations(self): ...

    def add_transformation(self, transformation: VideoFrameTransformation): ...

    @property
    def transformations(self) -> list[VideoFrameTransformation]: ...

    @property
    def attributes(self) -> list[(str, str)]: ...

    def get_attribute(self,
                      namespace: str,
                      name: str) -> Optional[Attribute]: ...

    def find_attributes_with_ns(self,
                                namespace: str) -> list[(str, str)]: ...

    def find_attributes_with_names(self,
                                   names: list[str]) -> list[(str, str)]: ...

    def find_attributes_with_hints(self,
                                   hints: list[Optional[str]]) -> list[(str, str)]: ...

    def delete_attribute(self, namespace: str, name: str) -> Optional[Attribute]: ...

    def clear_attributes(self): ...

    def delete_attributes_with_ns(self, namespace: str): ...

    def delete_attributes_with_names(self, names: list[str]): ...

    def delete_attributes_with_hints(self,
                                     hints: list[Optional[str]]): ...

    def set_attribute(self, attribute: Attribute) -> Optional[Attribute]: ...

    def set_persistent_attribute(self,
                                 namespace: str,
                                 name: str,
                                 is_hidden: bool,
                                 hint: Optional[str],
                                 values: Optional[list[AttributeValue]]): ...

    def set_temporary_attribute(self,
                                namespace: str,
                                name: str,
                                is_hidden: bool,
                                hint: Optional[str],
                                values: Optional[list[AttributeValue]]): ...

    def set_draw_label(self,
                       q: MatchQuery,
                       draw_label: SetDrawLabelKind,
                       no_gil: bool = False): ...

    def add_object(self, object: VideoObject, policy: IdCollisionResolutionPolicy): ...

    def create_object(self,
                      namespace: str,
                      label: str,
                      parent_id: Optional[int],
                      confidence: Optional[float],
                      detection_box: Optional[RBBox],
                      track_id: Optional[int],
                      track_box: Optional[RBBox],
                      attributes: Optional[list[Attribute]]) -> BorrowedVideoObject: ...

    def get_object(self, id: int) -> Optional[BorrowedVideoObject]: ...

    def get_all_objects(self) -> VideoObjectsView: ...

    def access_objects(self,
                       q: MatchQuery,
                       no_gil: bool = True) -> VideoObjectsView: ...

    def access_objects_with_ids(self,
                              ids: list[int],
                              no_gil: bool = True) -> VideoObjectsView: ...

    def delete_objects(self, q: MatchQuery, no_gil: bool = True) -> VideoObjectsView: ...

    def delete_objects_with_ids(self, ids: list[int]) -> VideoObjectsView: ...

    def set_parent(self,
                   q: MatchQuery,
                   parent: VideoObject,
                   no_gil: bool = True) -> VideoObjectsView: ...

    def set_parent_by_id(self,
                         object_id: int,
                         parent_id: int): ...

    def clear_parent(self,
                     q: MatchQuery,
                     no_gil: bool = True) -> VideoObjectsView: ...

    def clear_objects(self): ...

    def get_children(self, id: int) -> VideoObjectsView: ...

    def copy(self, no_gil: bool = True) -> VideoFrame: ...

    def update(self, update: VideoFrameUpdate, no_gil: bool = True): ...

    def to_protobuf(self, no_gil: bool = True) -> bytes: ...

    @classmethod
    def from_protobuf(cls,
                      protobuf: bytes,
                      no_gil: bool = True) -> VideoFrame: ...


class VideoFrameBatch:
    def __init__(self): ...

    def add(self, id: int, frame: VideoFrame): ...

    def get(self, id: int) -> Optional[VideoFrame]: ...

    def del_(self, id: int) -> Optional[VideoFrame]: ...

    def access_objects(self,
                       q: MatchQuery,
                       no_gil: bool = True) -> dict[tuple[int, VideoObjectsView]]: ...

    def delete_objects(self, q: MatchQuery, no_gil: bool = True): ...

    def to_protobuf(self, no_gil: bool = True) -> bytes: ...

    @classmethod
    def from_protobuf(cls,
                      protobuf: bytes,
                      no_gil: bool = True) -> VideoFrameBatch: ...


class VideoFrameUpdate:
    frame_attribute_policy: AttributeUpdatePolicy
    object_attribute_policy: AttributeUpdatePolicy
    object_policy: ObjectUpdatePolicy

    def __init__(self): ...

    def add_frame_attribute(self, attribute: Attribute): ...

    def add_object_attribute(self, object_id: int, attribute: Attribute): ...

    def add_object(self, object: VideoObject, parent_id: Optional[int]): ...

    def get_objects(self) -> list[tuple[VideoObject, Optional[int]]]: ...

    @property
    def json(self) -> str: ...

    @property
    def json_pretty(self) -> str: ...

    def to_protobuf(self, no_gil: bool = True) -> bytes: ...

    @classmethod
    def from_protobuf(cls,
                      protobuf: bytes,
                      no_gil: bool = True) -> VideoFrameUpdate: ...


class IdCollisionResolutionPolicy(Enum):
    GenerateNewId: ...
    Overwrite: ...
    Error: ...


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
    def attributes(self) -> list[tuple[str, str]]: ...

    def clear_attributes(self): ...

    @property
    def id(self) -> int: ...

    def delete_attribute(self, namespace: str, name: str) -> Optional[Attribute]: ...

    def delete_attributes_with_ns(self, namespace: str): ...

    def delete_attributes_with_names(self, names: list[str]): ...

    def delete_attributes_with_hints(self, hints: list[Optional[str]]): ...

    def detached_copy(self) -> VideoObject: ...

    def find_attributes_with_ns(self,
                                namespace: str) -> list[(str, str)]: ...

    def find_attributes_with_names(self,
                                   names: list[str]) -> list[(str, str)]: ...

    def find_attributes_with_hints(self,
                                   hints: list[Optional[str]]) -> list[(str, str)]: ...

    def get_attribute(self, namespace: str, name: str) -> Optional[Attribute]: ...

    def set_attribute(self, attribute: Attribute) -> Optional[Attribute]: ...

    def set_persistent_attribute(self,
                                 namespace: str,
                                 name: str,
                                 is_hidden: bool,
                                 hint: Optional[str],
                                 values: Optional[list[AttributeValue]]): ...

    def set_temporary_attribute(self,
                                namespace: str,
                                name: str,
                                is_hidden: bool,
                                hint: Optional[str],
                                values: Optional[list[AttributeValue]]): ...

    def set_track_info(self, track_id: int, track_box: RBBox): ...

    def clear_track_info(self): ...

    def transform_geometry(self,
                           ops: list[VideoObjectBBoxTransformation]): ...

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
                 attributes: list[Attribute],
                 confidence: Optional[float],
                 track_id: Optional[int],
                 track_box: Optional[RBBox],
                 ): ...

    def to_protobuf(self, no_gil: bool = True) -> bytes: ...

    @classmethod
    def from_protobuf(cls,
                      protobuf: bytes,
                      no_gil: bool = True) -> VideoObject: ...

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
    def attributes(self) -> list[tuple[str, str]]: ...

    def get_attribute(self, namespace: str, name: str) -> Optional[Attribute]: ...

    def set_attribute(self, attribute: Attribute) -> Optional[Attribute]: ...

    def set_persistent_attribute(self,
                                 namespace: str,
                                 name: str,
                                 is_hidden: bool,
                                 hint: Optional[str],
                                 values: Optional[list[AttributeValue]]): ...

    def set_temporary_attribute(self,
                                namespace: str,
                                name: str,
                                is_hidden: bool,
                                hint: Optional[str],
                                values: Optional[list[AttributeValue]]): ...


class VideoObjectBBoxType(Enum):
    Detection: ...
    TrackingInfo: ...


class VideoObjectsView:
    def __len__(self) -> int: ...

    def __getitem__(self, item) -> BorrowedVideoObject: ...

    def memory_handle(self) -> int: ...

    @property
    def ids(self) -> list[int]: ...

    @property
    def track_ids(self) -> list[int]: ...

    @property
    def sorted_by_id(self) -> list[BorrowedVideoObject]: ...


class QueryFunctions:
    @classmethod
    def filter(cls,
               v: VideoObjectsView,
               q: MatchQuery,
               no_gil: bool = True) -> VideoObjectsView: ...

    @classmethod
    def partition(cls,
                  v: VideoObjectsView,
                  q: MatchQuery,
                  no_gil: bool = True) -> tuple[VideoObjectsView, VideoObjectsView]: ...

