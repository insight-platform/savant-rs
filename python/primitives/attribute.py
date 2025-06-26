from typing import List

from savant_rs.primitives import Attribute, AttributeValue
from savant_rs.primitives.geometry import BBox, RBBox

attr = Attribute(
    namespace="some",
    name="attr",
    hint="x",
    values=[
        AttributeValue.bytes(dims=[8, 3, 8, 8], blob=bytes(3 * 8 * 8), confidence=None),
        AttributeValue.bytes_from_list(dims=[4, 1], blob=[0, 1, 2, 3], confidence=None),
        AttributeValue.integer(1, confidence=0.5),
        AttributeValue.float(1.0, confidence=0.5),
        AttributeValue.bbox(BBox(xc=1, yc=2, width=3, height=4), confidence=0.5),
        AttributeValue.rbbox(
            RBBox(xc=1, yc=2, width=3, height=4, angle=30), confidence=0.5
        ),
        AttributeValue.rbboxes(
            [
                RBBox(xc=1, yc=2, width=3, height=4, angle=30),
                RBBox(xc=5, yc=6, width=7, height=8, angle=45),
            ],
            confidence=0.5,
        ),
        AttributeValue.bboxes(
            [
                BBox(xc=1, yc=2, width=3, height=4),
                BBox(xc=5, yc=6, width=7, height=8),
            ],
            confidence=0.5,
        ),
    ],
)
print(attr.json)

vals = attr.values

view = attr.values_view
print(len(view))
print(view[2])

for i in range(4, 6):
    try:
        view[i].as_bbox()
        assert False
    except Exception as e:
        print(e)

for i in range(6, 8):
    try:
        view[i].as_bboxes()
        assert False
    except Exception as e:
        print(e)

v4: BBox = view[4].as_rbbox().into_bbox()

try:
    v5: BBox = view[5].as_rbbox().into_bbox()
    assert False
except Exception as e:
    print(e)

try:
    v6: List[BBox] = [b.into_bbox() for b in view[6].as_rbboxes()]
    assert False
except Exception as e:
    print(e)

v7: List[BBox] = [b.into_bbox() for b in view[7].as_rbboxes()]

attr2 = Attribute.from_json(attr.json)
print(attr2.json)

x = dict(x=5)
temp_py_attr = Attribute(
    namespace="some",
    name="attr",
    hint="x",
    values=[AttributeValue.temporary_python_object(x)],
)

x["y"] = 6

o = temp_py_attr.values[0].as_temporary_python_object()
print(o)
