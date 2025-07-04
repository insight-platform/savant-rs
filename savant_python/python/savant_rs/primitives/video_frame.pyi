from enum import Enum
from typing import Dict, List, Optional, Tuple

from savant_rs.draw_spec import SetDrawLabelKind
from savant_rs.match_query import MatchQuery
from savant_rs.primitives import *
from savant_rs.primitives.geometry import RBBox
from savant_rs.primitives.video_object import (BorrowedVideoObject,
                                               VideoObjectTree)
from savant_rs.utils import VideoObjectBBoxTransformation
from savant_rs.utils.serialization import Message

__all__ = [
    "ObjectUpdatePolicy",
    "ExternalFrame",
    "VideoFrameContent",
    "VideoFrameTranscodingMethod",
    "VideoFrameTransformation",
    "VideoFrame",
    "VideoFrameUpdate",
    "VideoFrameBatch",
]

class ObjectUpdatePolicy(Enum):
    AddForeignObjects: ...
    ErrorIfLabelsCollide: ...
    ReplaceSameLabelObjects: ...

class ExternalFrame:
    method: str
    location: Optional[str]

    def __init__(self, method: str, location: Optional[str]): ...

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

class VideoFrame:
    source_id: str
    time_base: Tuple[int, int]
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
    def transform_geometry(
        cls, ops: List[VideoObjectBBoxTransformation], no_gil: bool = True
    ): ...
    @property
    def memory_handle(self) -> int: ...
    def __init__(
        self,
        source_id: str,
        framerate: str,
        width: int,
        height: int,
        content: VideoFrameContent,
        transcoding_method: VideoFrameTranscodingMethod,
        codec: Optional[str],
        keyframe: Optional[bool],
        time_base: Tuple[int, int],
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
    def add_transformation(self, transformation: VideoObjectBBoxTransformation): ...
    @property
    def transformations(self) -> List[VideoObjectBBoxTransformation]: ...
    @property
    def attributes(self) -> List[Tuple[str, str]]: ...
    def get_attribute(self, namespace: str, name: str) -> Optional[Attribute]: ...
    def find_attributes_with_ns(self, namespace: str) -> List[Tuple[str, str]]: ...
    def find_attributes_with_names(self, names: List[str]) -> List[Tuple[str, str]]: ...
    def find_attributes_with_hints(
        self, hints: List[Optional[str]]
    ) -> List[Tuple[str, str]]: ...
    def delete_attribute(self, namespace: str, name: str) -> Optional[Attribute]: ...
    def clear_attributes(self): ...
    def delete_attributes_with_ns(self, namespace: str): ...
    def delete_attributes_with_names(self, names: List[str]): ...
    def delete_attributes_with_hints(self, hints: List[Optional[str]]): ...
    def set_attribute(self, attribute: Attribute) -> Optional[Attribute]: ...
    def set_persistent_attribute(
        self,
        namespace: str,
        name: str,
        is_hidden: bool,
        hint: Optional[str],
        values: Optional[List[AttributeValue]],
    ): ...
    def set_temporary_attribute(
        self,
        namespace: str,
        name: str,
        is_hidden: bool,
        hint: Optional[str],
        values: Optional[List[AttributeValue]],
    ): ...
    def set_draw_label(
        self, q: MatchQuery, draw_label: SetDrawLabelKind, no_gil: bool = False
    ): ...
    def add_object(
        self, object: "VideoObject", policy: IdCollisionResolutionPolicy
    ): ...
    def create_object(
        self,
        namespace: str,
        label: str,
        parent_id: Optional[int],
        confidence: Optional[float],
        detection_box: Optional[RBBox],
        track_id: Optional[int],
        track_box: Optional[RBBox],
        attributes: Optional[List[Attribute]],
    ) -> BorrowedVideoObject: ...
    def get_object(self, id: int) -> Optional[BorrowedVideoObject]: ...
    def get_all_objects(self) -> VideoObjectsView: ...
    def has_objects(self) -> bool: ...
    def export_complete_object_trees(
        self, q: MatchQuery, delete_exported: bool
    ) -> List[VideoObjectTree]: ...
    def import_object_trees(self, trees: List[VideoObjectTree]) -> None: ...

class VideoFrameUpdate:
    frame_attribute_policy: AttributeUpdatePolicy
    object_attribute_policy: AttributeUpdatePolicy
    object_policy: ObjectUpdatePolicy

    def __init__(self): ...
    def add_frame_attribute(self, attribute: Attribute): ...
    def add_object_attribute(self, object_id: int, attribute: Attribute): ...
    def add_object(self, object: VideoObject, parent_id: Optional[int]): ...
    def get_objects(self) -> List[Tuple[VideoObject, Optional[int]]]: ...
    @property
    def json(self) -> str: ...
    @property
    def json_pretty(self) -> str: ...
    def to_protobuf(self, no_gil: bool = True) -> bytes: ...
    @classmethod
    def from_protobuf(
        cls, protobuf: bytes, no_gil: bool = True
    ) -> VideoFrameUpdate: ...

class VideoFrameTransformation:
    @staticmethod
    def initial_size(width: int, height: int) -> VideoFrameTransformation: ...
    @staticmethod
    def scale(width: int, height: int) -> VideoFrameTransformation: ...
    @staticmethod
    def padding(
        left: int, top: int, right: int, bottom: int
    ) -> VideoFrameTransformation: ...
    @staticmethod
    def resulting_size(width: int, height: int) -> VideoFrameTransformation: ...
    @property
    def is_initial_size(self) -> bool: ...
    @property
    def is_scale(self) -> bool: ...
    @property
    def is_padding(self) -> bool: ...
    @property
    def is_resulting_size(self) -> bool: ...
    @property
    def as_initial_size(self) -> Optional[Tuple[int, int]]: ...
    @property
    def as_scale(self) -> Optional[Tuple[int, int]]: ...
    @property
    def as_padding(self) -> Optional[Tuple[int, int, int, int]]: ...
    @property
    def as_resulting_size(self) -> Optional[Tuple[int, int]]: ...

class VideoFrameBatch:
    """A batch of video frames that can be processed together."""

    @property
    def frames(self) -> Dict[int, VideoFrame]:
        """Get all frames in the batch.

        Returns
        -------
        Dict[int, VideoFrame]
            Dictionary mapping frame IDs to VideoFrame objects
        """
        ...

    @property
    def frame_ids(self) -> List[int]:
        """Get IDs of all frames in the batch.

        Returns
        -------
        List[int]
            List of frame IDs in the batch
        """
        ...

    def add_frame(self, frame: VideoFrame) -> int:
        """Add a frame to the batch.

        Parameters
        ----------
        frame : VideoFrame
            The frame to add

        Returns
        -------
        int
            ID assigned to the frame in the batch
        """
        ...

    def get_frame(self, frame_id: int) -> Optional[VideoFrame]:
        """Get a frame from the batch by its ID.

        Parameters
        ----------
        frame_id : int
            ID of the frame to retrieve

        Returns
        -------
        Optional[VideoFrame]
            The frame if found, None otherwise
        """
        ...

    def to_message(self) -> Message:
        """Convert the batch to a message for serialization.

        Returns
        -------
        Message
            Message containing the batch data
        """
        ...

    @classmethod
    def from_frames(cls, frames: List[VideoFrame]) -> VideoFrameBatch:
        """Create a new batch from a list of frames.

        Parameters
        ----------
        frames : List[VideoFrame]
            List of frames to include in the batch

        Returns
        -------
        VideoFrameBatch
            New batch containing the provided frames
        """
        ...
