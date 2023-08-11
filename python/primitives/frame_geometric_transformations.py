from savant_rs.utils import gen_frame, VideoObjectBBoxTransformation

from savant_rs.logging import LogLevel, set_log_level
set_log_level(LogLevel.Trace)

f = gen_frame()
obj = f.get_object(0)
obj.detection_box.xc = 25
obj.detection_box.yc = 50
obj.detection_box.height = 10
obj.detection_box.width = 20
print(obj)

transformations = [
    VideoObjectBBoxTransformation.scale(2.0, 2.0),
    VideoObjectBBoxTransformation.shift(10, 10)
]

f.transform_geometry(transformations)
obj = f.get_object(0)
print(obj)
