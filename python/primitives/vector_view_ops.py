from savant_rs.utils import gen_frame, VideoObjectBBoxType
from savant_rs.utils.numpy import BBoxFormat
from savant_rs.utils.udf_api import register_plugin_function, is_plugin_function_registered, UserFunctionType
from savant_rs.video_object_query import MatchQuery as Q, IntExpression as IE, QueryFunctions as QF

from savant_rs.logging import LogLevel, set_log_level
set_log_level(LogLevel.Trace)

register_plugin_function("../../target/debug/libsavant_rs.so", "map_modifier", UserFunctionType.ObjectMapModifier,
                         "sample.map_modifier")

assert is_plugin_function_registered("sample.map_modifier")

register_plugin_function("../../target/debug/libsavant_rs.so", "inplace_modifier",
                         UserFunctionType.ObjectInplaceModifier,
                         "sample.inplace_modifier")

assert is_plugin_function_registered("sample.inplace_modifier")

f = gen_frame()

objects_x = QF.filter(f.access_objects(Q.idle()), Q.eval("id % 2 == 1")).sorted_by_id()

objects = QF.filter(f.access_objects(Q.idle()), Q.id(IE.one_of(1, 2))).sorted_by_id()

new_objects = QF.map_udf(objects, "sample.map_modifier")
print(new_objects[0].label)
assert new_objects[0].label == "modified_test"
assert objects[0].label == "test"

QF.foreach_udf(objects, "sample.inplace_modifier")
assert objects[0].label == "modified_test"

ids = objects.ids
boxes = objects.rotated_boxes_as_numpy(VideoObjectBBoxType.Detection)
print("Ids:", ids)
print("Detections:", boxes)

track_ids = objects.track_ids
print("Track ids:", track_ids)
tr_boxes = objects.rotated_boxes_as_numpy(VideoObjectBBoxType.TrackingInfo)
print("Tracking:", tr_boxes)

objects.update_from_numpy_boxes(boxes, BBoxFormat.XcYcWidthHeight, VideoObjectBBoxType.Detection)
objects.update_from_numpy_rotated_boxes(boxes, VideoObjectBBoxType.Detection)

# tracking_boxes = objects.tracking_boxes_as_numpy()
