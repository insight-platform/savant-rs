from savant_rs.logging import LogLevel, set_log_level
from savant_rs.primitives import VideoObject
from savant_rs.primitives.geometry import BBox

set_log_level(LogLevel.Trace)

obj = VideoObject(
    id=1,
    namespace="some",
    label="person",
    detection_box=BBox(0.1, 0.2, 0.3, 0.4).as_rbbox(),
    confidence=0.5,
    attributes=[],
    track_id=2**66,  # must be cropped
    track_box=BBox(0.1, 0.2, 0.3, 0.4).as_rbbox(),
)
