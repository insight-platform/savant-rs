from savant_rs.draw_spec import (BoundingBoxDraw, ColorDraw, DotDraw,
                                 LabelDraw, LabelPosition, LabelPositionKind,
                                 ObjectDraw, PaddingDraw)

spec = ObjectDraw(
    bounding_box=BoundingBoxDraw(
        border_color=ColorDraw(red=100, blue=50, green=50, alpha=100),
        background_color=ColorDraw(red=0, blue=50, green=50, alpha=100),
        thickness=2,
        padding=PaddingDraw(left=5, top=5, right=5, bottom=5),
    ),
    label=LabelDraw(
        font_color=ColorDraw(red=100, blue=50, green=50, alpha=100),
        border_color=ColorDraw(red=100, blue=50, green=50, alpha=100),
        background_color=ColorDraw(red=0, blue=50, green=50, alpha=100),
        padding=PaddingDraw(left=5, top=5, right=5, bottom=5),
        position=LabelPosition(
            position=LabelPositionKind.TopLeftOutside, margin_x=0, margin_y=-20
        ),
        font_scale=2.5,
        thickness=2,
        format=["{model}", "{label}", "{confidence}", "{track_id}"],
    ),
    central_dot=DotDraw(
        color=ColorDraw(red=100, blue=50, green=50, alpha=100), radius=2
    ),
    blur=False,
)

print(spec.bounding_box.border_color.rgba)
print(spec.bounding_box.background_color.rgba)
print(spec)

spec = ObjectDraw(
    bounding_box=None,
    label=None,
    central_dot=DotDraw(
        color=ColorDraw(red=100, blue=50, green=50, alpha=100), radius=2
    ),
    blur=True,
)

print(spec)

spec = ObjectDraw(
    bounding_box=BoundingBoxDraw(
        border_color=ColorDraw(red=100, blue=50, green=50, alpha=100),
    )
)

new_spec = ObjectDraw(
    bounding_box=spec.bounding_box,
    label=spec.label,
    central_dot=spec.central_dot,
    blur=spec.blur,
)

print(new_spec)

spec = ObjectDraw(blur=True)

new_spec = spec.copy()
print(new_spec)
