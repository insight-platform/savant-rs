from savant_rs.primitives import BoundingBoxDraw, ColorDraw, LabelDraw, DotDraw, PaddingDraw, ObjectDraw

spec = ObjectDraw(
    bounding_box=BoundingBoxDraw(
        color=ColorDraw(
            red=100, blue=50, green=50, alpha=100),
        thickness=2,
        padding=PaddingDraw(left=5, top=5, right=5, bottom=5)),
    label=LabelDraw(
        color=ColorDraw(
            red=100, blue=50, green=50, alpha=100),
        font_scale=2.5,
        thickness=2,
        format=["{model}", "{label}", "{confidence}", "{track_id}"]),
    central_dot=DotDraw(
        color=ColorDraw(
            red=100, blue=50, green=50, alpha=100),
        radius=2),
    blur=False
)

print(spec.bounding_box.color.bgra)
print(spec.bounding_box.color.red)
print(spec)

spec = ObjectDraw(
    bounding_box=None,
    label=None,
    central_dot=DotDraw(
        color=ColorDraw(
            red=100, blue=50, green=50, alpha=100),
        radius=2),
    blur=True
)

print(spec)

spec = ObjectDraw(
    bounding_box=BoundingBoxDraw(
        color=ColorDraw(
            red=100, blue=50, green=50, alpha=100),
    )
)

print(spec)

spec = ObjectDraw(
    blur=True
)

print(spec)
