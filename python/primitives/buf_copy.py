from savant_rs.primitives import Object, Value, Attribute, ParentObject, BBox
from timeit import default_timer as timer

o = Object(
    id=1,
    creator="some",
    label="person",
    bbox=BBox(0.1, 0.2, 0.3, 0.4),
    confidence=0.5,
    attributes={},
    parent=ParentObject(id=2, creator="some", label="car"),
    track_id=None,
)


t = timer()

bts = bytes(256)

a = Attribute(creator="other", name="attr", values=[
    # Value.bytes(dims=[8, 3, 8, 8], blob=bts, confidence=None),
    Value.integer(1, confidence=0.5),
    Value.integer(2, confidence=0.5),
    Value.integer(3, confidence=0.5),
    Value.integer(4, confidence=0.5),
])

for _ in range(1_000):
    o.set_attribute(a)
    a = o.get_attribute(creator="other", name="attr")
    # x = a.name

print(timer() - t)

