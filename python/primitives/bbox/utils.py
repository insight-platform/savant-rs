from savant_rs.primitives.geometry import (RBBox, associate_bboxes,
                                           solely_owned_areas)
from savant_rs.utils import BBoxMetricType

red = RBBox.ltrb(0.0, 2.0, 2.0, 4.0)
green = RBBox.ltrb(1.0, 3.0, 5.0, 5.0)
yellow = RBBox.ltrb(1.0, 1.0, 3.0, 6.0)
purple = RBBox.ltrb(4.0, 0.0, 7.0, 2.0)

areas = solely_owned_areas([red, green, yellow, purple], parallel=True)

red = areas[0]
green = areas[1]
yellow = areas[2]
purple = areas[3]

assert red == 2.0
assert green == 4.0
assert yellow == 5.0
assert purple == 6.0

lp1 = RBBox.ltrb(0.0, 1.0, 2.0, 2.0)
lp2 = RBBox.ltrb(5.0, 2.0, 8.0, 3.0)
lp3 = RBBox.ltrb(100.0, 0.0, 106.0, 3.0)
owner1 = RBBox.ltrb(1.0, 0.0, 6.0, 3.0)
owner2 = RBBox.ltrb(6.0, 1.0, 9.0, 4.0)

associations_iou = associate_bboxes(
    [lp1, lp2, lp3], [owner1, owner2], BBoxMetricType.IoU, 0.01
)

lp1_associations = associations_iou[0]
lp2_associations = associations_iou[1]
lp3_associations = associations_iou[2]

assert list(map(lambda t: t[0], lp1_associations)) == [0]
assert list(map(lambda t: t[0], lp2_associations)) == [1, 0]
assert lp3_associations == []
