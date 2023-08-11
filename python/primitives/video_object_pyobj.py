from savant_rs.primitives import VideoObject
from savant_rs.primitives.geometry import RBBox
import random

from savant_rs.logging import LogLevel, set_log_level
set_log_level(LogLevel.Trace)

obj = VideoObject(
        id=1,
        namespace="created_by_1",
        label="person_1",
        detection_box=RBBox(0.1, 0.2, 0.3, 0.4, 30.0),
        confidence=random.random(),
        attributes={},
        track_id=None,
        track_box=None
    )

x = []
obj.set_pyobject("ns","x", x)

new_x = obj.get_pyobject("ns", "x")
assert new_x == x

x.append(1)
assert new_x == x

new_x2 = obj.get_pyobject("ns", "x")
assert new_x2 == x
assert new_x2 == new_x

attaches = obj.list_pyobjects()
assert attaches == [('ns', 'x')]

new_x = obj.delete_pyobject("ns", "x")
assert new_x == x

obj.clear_pyobjects()
