from savant_rs.primitives import BBox, RBBox
from savant_rs.utils import *

box = BBox(50, 50, 50, 50)
print("Original bbox:", box)
print(box.left, box.top, box.bottom, box.right, box.width, box.height)
box.width = 120
box.left = 70
print(box.xc)
print("Modified bbox:", box)

vertices = box.vertices
print("Vertices:", vertices)

vertices = box.vertices_rounded
print("Vertices rounded to .2", vertices)

vertices = box.vertices_int
print("Integer vertices:", vertices)

box = box.wrapping_box
print("Wrapping box:", box)

# max_x is limited to 180, max_y is limited to 500
box = box.as_graphical_wrapping_box(padding=5, border_width=2, max_x=180, max_y=500)
print("Graphical wrapping box:", box)

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
box = box.as_graphical_wrapping_box(padding=5, border_width=2, max_x=100, max_y=500)
print("Graphical wrapping box:", box)

print(BBox.ltwh(0, 0, 100, 100))
print(BBox.ltrb(0, 0, 100, 100))

arr = bboxes_to_ndarray([BBox(50.0, 50.0, 30.0, 50.0), BBox(70.0, 70.0, 50.0, 50.0)], BBoxFormat.LeftTopRightBottom, 'float64')
print("BBoxes to f64", arr)

boxes = ndarray_to_bboxes(arr, BBoxFormat.LeftTopRightBottom)
print(boxes)

arr = bboxes_to_ndarray([BBox(50.0, 50.0, 30.0, 50.0), BBox(70.0, 70.0, 50.0, 50.0)], BBoxFormat.LeftTopRightBottom, 'float32')
print("BBoxes to f32", arr)

boxes = ndarray_to_bboxes(arr, BBoxFormat.LeftTopRightBottom)
print(boxes)
