from enum import Enum
from typing import List, Optional, Tuple

from savant_rs.draw_spec import PaddingDraw

__all__ = [
    'Point',
    'Segment',
    'IntersectionKind',
    'Intersection',
    'PolygonalArea',
    'RBBox',
    'BBox',
    'solely_owned_areas',
    'associate_bboxes',
]

class Point:
    x: float
    y: float

    def __init__(self, x: float, y: float): ...

class Segment:
    def __init__(self, begin: Point, end: Point): ...
    @property
    def begin(self) -> Point: ...
    @property
    def end(self) -> Point: ...

class IntersectionKind(Enum):
    Enter: ...
    Inside: ...
    Leave: ...
    Cross: ...
    Outside: ...

class Intersection:
    def __init__(
        self, kind: IntersectionKind, edges: List[Tuple[int, Optional[str]]]
    ): ...
    @property
    def kind(self) -> IntersectionKind: ...
    @property
    def edges(self) -> List[Tuple[int, Optional[str]]]: ...

class PolygonalArea:
    @classmethod
    def contains_many_points(cls, points: List[Point]) -> List[bool]: ...
    @classmethod
    def crossed_by_segments(cls, segments: List[Segment]) -> List[Intersection]: ...
    def is_self_intersecting(self) -> bool: ...
    def crossed_by_segment(self, segment: Segment) -> Intersection: ...
    def contains(self, point: Point) -> bool: ...
    def build_polygon(self): ...
    def get_tag(self, edge: int) -> Optional[str]: ...
    @classmethod
    def points_positions(
        cls, polys: List[PolygonalArea], points: List[Point], no_gil: bool = False
    ) -> List[List[int]]: ...
    @classmethod
    def segments_intersections(
        cls, polys: List[PolygonalArea], segments: List[Segment], no_gil: bool = False
    ) -> List[List[Intersection]]: ...
    def __init__(
        self, vertices: List[Point], tags: Optional[List[Optional[str]]] = None
    ): ...

class RBBox:
    xc: float
    yc: float
    width: float
    height: float
    angle: Optional[float]
    top: float
    left: float
    @property
    def area(self) -> float: ...
    def eq(self, other: RBBox) -> bool: ...
    def almost_eq(self, other: RBBox, eps: float) -> bool: ...
    def __richcmp__(self, other: RBBox, op: int) -> bool: ...
    @property
    def width_to_height_ratio(self) -> float: ...
    def is_modified(self) -> bool: ...
    def set_modifications(self, value: bool): ...
    def __init__(
        self,
        xc: float,
        yc: float,
        width: float,
        height: float,
        angle: Optional[float] = None,
    ): ...
    def scale(self, scale_x: float, scale_y: float): ...
    @property
    def vertices(self) -> List[Tuple[float, float]]: ...
    @property
    def vertices_rounded(self) -> List[Tuple[float, float]]: ...
    @property
    def vertices_int(self) -> List[Tuple[int, int]]: ...
    def as_polygonal_area(self) -> PolygonalArea: ...
    @property
    def wrapping_box(self) -> BBox: ...
    def get_visual_box(self, padding: PaddingDraw, border_width: int) -> RBBox: ...
    def new_padded(self, padding: PaddingDraw) -> RBBox: ...
    def copy(self) -> RBBox: ...
    def iou(self, other: RBBox) -> float: ...
    def ios(self, other: RBBox) -> float: ...
    def ioo(self, other: RBBox) -> float: ...
    def shift(self, dx: float, dy: float) -> RBBox: ...
    @classmethod
    def ltrb(cls, left: float, top: float, right: float, bottom: float) -> RBBox: ...
    @classmethod
    def ltwh(cls, left: float, top: float, width: float, height: float) -> RBBox: ...
    @property
    def right(self) -> float: ...
    @property
    def bottom(self) -> float: ...
    def as_ltrb(self) -> Tuple[float, float, float, float]: ...
    def as_ltrb_int(self) -> Tuple[int, int, int, int]: ...
    def as_ltwh(self) -> Tuple[float, float, float, float]: ...
    def as_ltwh_int(self) -> Tuple[int, int, int, int]: ...
    def as_xcycwh(self) -> Tuple[float, float, float, float]: ...
    def as_xcycwh_int(self) -> Tuple[int, int, int, int]: ...

class BBox:
    xc: float
    yc: float
    width: float
    height: float
    top: float
    left: float
    def __init__(self, xc: float, yc: float, width: float, height: float): ...
    def eq(self, other: BBox) -> bool: ...
    def almost_eq(self, other: BBox, eps: float) -> bool: ...
    def __richcmp__(self, other: BBox, op: int) -> bool: ...
    def iou(self, other: BBox) -> float: ...
    def ios(self, other: BBox) -> float: ...
    def ioo(self, other: BBox) -> float: ...
    def is_modified(self) -> bool: ...
    @classmethod
    def ltrb(cls, left: float, top: float, right: float, bottom: float) -> BBox: ...
    @classmethod
    def ltwh(cls, left: float, top: float, width: float, height: float) -> BBox: ...
    @property
    def right(self) -> float: ...
    @property
    def bottom(self) -> float: ...
    @property
    def vertices(self) -> List[Tuple[float, float]]: ...
    @property
    def vertices_rounded(self) -> List[Tuple[float, float]]: ...
    @property
    def vertices_int(self) -> List[Tuple[int, int]]: ...
    @property
    def wrapping_box(self) -> BBox: ...
    def get_visual_box(self, padding: PaddingDraw, border_width: int) -> BBox: ...
    def as_ltrb(self) -> Tuple[float, float, float, float]: ...
    def as_ltrb_int(self) -> Tuple[int, int, int, int]: ...
    def as_ltwh(self) -> Tuple[float, float, float, float]: ...
    def as_ltwh_int(self) -> Tuple[int, int, int, int]: ...
    def as_xcycwh(self) -> Tuple[float, float, float, float]: ...
    def as_xcycwh_int(self) -> Tuple[int, int, int, int]: ...
    def as_rbbox(self) -> RBBox: ...
    def scale(self, scale_x: float, scale_y: float) -> BBox: ...
    def shift(self, dx: float, dy: float) -> BBox: ...
    def as_polygonal_area(self) -> PolygonalArea: ...
    def copy(self) -> BBox: ...
    def new_padded(self, padding: PaddingDraw) -> BBox: ...

def solely_owned_areas(bboxes: List[RBBox], parallel: bool) -> List[float]: ...
def associate_bboxes(
    candidates: List[RBBox], owners: List[RBBox], metric: str, threshold: float
) -> dict[int, list[tuple[int, float]]]: ...
