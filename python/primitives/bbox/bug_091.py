from savant_rs.draw_spec import PaddingDraw
from savant_rs.primitives.geometry import BBox

padding = PaddingDraw(0, 0, 0, 0)

max_col = 1279
max_row = 719

bbox = BBox(1279, 719, 2, 2)

vis_bbox = bbox.get_visual_box(padding, 0, max_col, max_row)
print(vis_bbox)
left, top, right, bottom = vis_bbox.as_ltrb_int()
print(left, top, right, bottom)

assert left >= 0
assert right <= max_col
assert top >= 0
assert bottom <= max_row
