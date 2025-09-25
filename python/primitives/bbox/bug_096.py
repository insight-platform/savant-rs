from savant_rs.draw_spec import PaddingDraw
from savant_rs.primitives.geometry import BBox

padding = PaddingDraw(0,0,0,0)

max_col = 1279
max_row = 719

bboxes = [
    BBox(380.201, 603.9101, 2.3868103, 4.445815),
    BBox(490.4524, 603.80145, 2.9029846, 3.4778595)
]

for bbox in bboxes:
    vis_bbox = bbox.get_visual_box(padding, 0, max_col, max_row)
    left, top, width, height = vis_bbox.as_ltwh_int()

    assert width >= 1
    assert height >= 1
