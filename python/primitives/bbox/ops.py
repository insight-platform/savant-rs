from savant_rs.draw_spec import PaddingDraw
from savant_rs.logging import LogLevel, set_log_level
from savant_rs.primitives.geometry import BBox, RBBox

from savant_rs.primitives.geometry import associate_bboxes

from savant_rs.utils import BBoxMetricType

set_log_level(LogLevel.Trace)

box = BBox(50, 50, 50, 50)
print("Original bbox:", box)
print("Vertices:", box.vertices)
print(box.left, box.top, box.bottom, box.right, box.width, box.height)
box.width = 120
box.left = 70
print(box.xc)
print("Modified bbox:", box)

copy = box.copy()
copy.xc = 100
assert box.xc != copy.xc

vertices = box.vertices
print("Vertices:", vertices)

vertices = box.vertices_rounded
print("Vertices rounded to .2", vertices)

vertices = box.vertices_int
print("Integer vertices:", vertices)

box = box.wrapping_box
print("Wrapping box:", box)

# max_x is limited to 180, max_y is limited to 500
box = box.get_visual_box(
    padding=PaddingDraw(5, 5, 5, 5), border_width=2, max_x=180, max_y=500
)
print("Visual box:", box)

ltrb = box.as_ltrb()
print("Left, top, right, bottom:", ltrb)

ltrb_int = box.as_ltrb_int()
print("Left, top, right, bottom (int):", ltrb_int)

ltwh = box.as_ltwh()
print("Left, top, width, height:", ltwh)

ltwh_int = box.as_ltwh_int()
print("Left, top, width, height (int):", ltwh_int)

box.scale(scale_x=2, scale_y=3)
print("Scaled box:", box)

area = box.as_polygonal_area()
vertices = box.vertices
print("Polygonal area:", area)
print("Vertices:", vertices)

box = box.as_rbbox()
box.angle = 45
print(box)

box = RBBox(50, 50, 50, 50, 45)
print(box)

try:
    box.into_bbox()
    assert False
except Exception as e:
    print(e)

scale2 = box.scale(scale_x=2, scale_y=2)
print(scale2)

vertices = box.vertices
print("Vertices:", vertices)

area = box.as_polygonal_area()
print("Polygonal area:", area)

vertices_rounded = box.vertices_rounded
print("Vertices rounded to .2", vertices_rounded)

vertices_int = box.vertices_int
print("Integer vertices:", vertices_int)

box = box.wrapping_box
print("Wrapping box:", box)

# max_x is limited to 100, max_y is limited to 500
box = box.get_visual_box(
    padding=PaddingDraw(5, 5, 5, 5), border_width=2, max_x=100, max_y=500
)
print("Graphical wrapping box:", box)

print(BBox.ltwh(0, 0, 100, 100))
print(BBox.ltrb(0, 0, 100, 100))

b1 = BBox(0, 0, 100, 100)
b2 = BBox(0, 0, 200, 200)
assert b1.inside(b2)

face_bb1 = BBox(0, 0, 10, 10)
face_bb2 = BBox(20, 20, 10, 10)

body_bb1 = BBox(0, 0, 20, 100)
body_bb2 = BBox(20, 20, 20, 100)

associations = associate_bboxes([face_bb1.as_rbbox(), face_bb2.as_rbbox()], [body_bb1.as_rbbox(), body_bb2.as_rbbox()], BBoxMetricType.IoSelf, -1.0)
print(associations)