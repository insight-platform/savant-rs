from savant_rs.primitives.geometry import BBox, RBBox

box1 = BBox(50, 50, 50, 50)
box2 = BBox(50, 50, 50, 50)
box3 = BBox(50, 50, 50, 60)

assert box1 == box2
assert box1.eq(box2)
assert box1.almost_eq(box2, 0.1)

assert box1 != box3
assert not box1.eq(box3)

box1 = RBBox(50, 50, 50, 50, 30)
box2 = RBBox(50, 50, 50, 50, 30)
box3 = RBBox(50, 50, 50, 50, 30.001)

assert box1 == box2
assert box1.eq(box2)
assert box1.almost_eq(box2, 0.1)

assert box1 != box3
assert box1.almost_eq(box3, 0.1)
iou = box1.iou(box2)
assert iou == 1.0

iou = box1.iou(box3)
assert iou > 0.9

for f in [
    lambda: box1 > box2,
    lambda: box1 < box2,
    lambda: box1 >= box2,
    lambda: box1 <= box2,
]:
    try:
        f()
        assert False
    except NotImplementedError:
        pass

empty_box = RBBox(0, 0, 0, 0)
try:
    iou = box1.iou(empty_box)
    assert False
except ValueError:
    pass

empty_box = BBox(0, 0, 0, 0)
box1 = BBox(50, 50, 50, 50)
try:
    iou = box1.iou(empty_box)
    assert False
except ValueError:
    pass
