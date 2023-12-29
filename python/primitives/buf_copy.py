from timeit import default_timer as timer

from savant_rs.primitives import VideoObject, AttributeValue, Attribute
from savant_rs.primitives.geometry import RBBox

o = VideoObject(
    id=1,
    namespace="some",
    label="person",
    detection_box=RBBox(0.1, 0.2, 0.3, 0.4, None),
    confidence=0.5,
    attributes={},
    track_id=None,
    track_box=None
)

t = timer()

bts = bytes(256)

a = Attribute(namespace="other", name="attr", values=[
    # Value.bytes(dims=[8, 3, 8, 8], blob=bts, confidence=None),
    AttributeValue.integer(1, confidence=0.5),
    AttributeValue.integer(2, confidence=0.5),
    AttributeValue.integer(3, confidence=0.5),
    AttributeValue.integer(4, confidence=0.5),
])

for _ in range(1_000):
    o.set_attribute(a)
    a = o.get_attribute(namespace="other", name="attr")
    # x = a.name

print(timer() - t)
