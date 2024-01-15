from enum import Enum
from typing import Tuple, Optional


class ColorDraw:
    def copy(self) -> ColorDraw: ...

    def __init__(self,
                 red: int = 0,
                 green: int = 255,
                 blue: int = 0,
                 alpha: int = 255): ...

    @property
    def bgra(self) -> Tuple[int, int, int, int]: ...

    @property
    def rgba(self) -> Tuple[int, int, int, int]: ...

    @property
    def red(self) -> int: ...

    @property
    def green(self) -> int: ...

    @property
    def blue(self) -> int: ...

    @property
    def alpha(self) -> int: ...

    @classmethod
    def transparent(cls) -> ColorDraw: ...


class PaddingDraw:
    def copy(self) -> PaddingDraw: ...

    def __init__(self,
                 left: int = 0,
                 top: int = 0,
                 right: int = 0,
                 bottom: int = 0): ...

    @property
    def padding(self) -> Tuple[int, int, int, int]: ...

    @property
    def top(self) -> int: ...

    @property
    def right(self) -> int: ...

    @property
    def bottom(self) -> int: ...

    @property
    def left(self) -> int: ...

    @classmethod
    def default_padding(cls) -> PaddingDraw: ...


class BoundingBoxDraw:
    def copy(self) -> BoundingBoxDraw: ...

    def __init__(self,
                 border_color: ColorDraw = ColorDraw.transparent(),
                 background_color: ColorDraw = ColorDraw.transparent(),
                 thickness: int = 2,
                 padding: PaddingDraw = PaddingDraw.default_padding()): ...

    @property
    def border_color(self) -> ColorDraw: ...

    @property
    def background_color(self) -> ColorDraw: ...

    @property
    def thickness(self) -> int: ...

    @property
    def padding(self) -> PaddingDraw: ...


class DotDraw:
    def copy(self) -> DotDraw: ...

    def __init__(self,
                 color: ColorDraw,
                 radius: int = 2): ...

    @property
    def color(self) -> ColorDraw: ...

    @property
    def radius(self) -> int: ...


class LabelPositionKind(Enum):
    TopLeftInside: ...
    TopLeftOutside: ...
    Center: ...


class LabelPosition:
    def copy(self) -> LabelPosition: ...

    def __init__(self,
                 position: LabelPositionKind.TopLeftOutside,
                 margin_x: int = 0,
                 margin_y: int = -10): ...

    @classmethod
    def default_position(cls) -> LabelPosition: ...

    @property
    def position(self) -> LabelPositionKind: ...

    @property
    def margin_x(self) -> int: ...

    @property
    def margin_y(self) -> int: ...


class LabelDraw:
    def copy(self) -> LabelDraw: ...

    def __init__(self,
                 font_color: ColorDraw,
                 background_color: ColorDraw = ColorDraw.transparent(),
                 border_color: ColorDraw = ColorDraw.transparent(),
                 font_scale: float = 1.0,
                 thickness: int = 1,
                 position: LabelPosition = LabelPosition.default_position(),
                 padding: PaddingDraw = PaddingDraw.default_padding(),
                 format: list[str] = ["{label}"]): ...

    @property
    def font_color(self) -> ColorDraw: ...

    @property
    def background_color(self) -> ColorDraw: ...

    @property
    def border_color(self) -> ColorDraw: ...

    @property
    def font_scale(self) -> float: ...

    @property
    def thickness(self) -> int: ...

    @property
    def position(self) -> LabelPosition: ...

    @property
    def padding(self) -> PaddingDraw: ...

    @property
    def format(self) -> list[str]: ...


class ObjectDraw:
    def copy(self) -> ObjectDraw: ...

    def __init__(self,
                 bounding_box: Optional[BoundingBoxDraw] = None,
                 central_dot: Optional[DotDraw] = None,
                 label: Optional[LabelDraw] = None,
                 blur: bool = False,
                 ): ...

    @property
    def bounding_box(self) -> Optional[BoundingBoxDraw]: ...

    @property
    def central_dot(self) -> Optional[DotDraw]: ...

    @property
    def label(self) -> Optional[LabelDraw]: ...

    @property
    def blur(self) -> bool: ...


class SetDrawLabelKind:
    @classmethod
    def own(cls, label: str) -> SetDrawLabelKind: ...

    @classmethod
    def parent(cls, label: str) -> SetDrawLabelKind: ...

    def is_own_label(self) -> bool: ...

    def is_parent_label(self) -> bool: ...

    def get_label(self) -> str: ...