from savant_rs.utils import gen_frame, VideoObjectBBoxKind
from savant_rs.utils.numpy import BBoxFormat
from savant_rs.utils.udf_api import register_plugin_function, is_plugin_function_registered, UserFunctionType
from savant_rs.video_object_query import Query as Q, IntExpression as IE

register_plugin_function("../../target/debug/libsample_plugin.so", "map_modifier", UserFunctionType.ObjectMapModifier,
                         "sample.map_modifier")

assert is_plugin_function_registered("sample.map_modifier")

register_plugin_function("../../target/debug/libsample_plugin.so", "inplace_modifier",
                         UserFunctionType.ObjectInplaceModifier,
                         "sample.inplace_modifier")

assert is_plugin_function_registered("sample.inplace_modifier")

f = gen_frame()

objects_x = f.access_objects(Q.idle()).filter(Q.eval("id % 2 == 1", [])).sorted_by_id()

objects = f.access_objects(Q.idle()).filter(Q.id(IE.one_of(1, 2))).sorted_by_id()

new_objects = objects.map_udf("sample.map_modifier")
print(new_objects[0].label)
assert new_objects[0].label == "modified_test"
assert objects[0].label == "test"

objects.foreach_udf("sample.inplace_modifier")
assert objects[0].label == "modified_test"

ids = objects.ids
boxes = objects.rotated_boxes_as_numpy(VideoObjectBBoxKind.Detection)
print("Ids:", ids)
print("Detections:", boxes)

track_ids = objects.track_ids
print("Track ids:", track_ids)
tr_boxes = objects.rotated_boxes_as_numpy(VideoObjectBBoxKind.TrackingInfo)
print("Tracking:", tr_boxes)

objects.update_from_numpy_boxes(boxes, BBoxFormat.XcYcWidthHeight, VideoObjectBBoxKind.Detection)
objects.update_from_numpy_rotated_boxes(boxes, VideoObjectBBoxKind.Detection)

# tracking_boxes = objects.tracking_boxes_as_numpy()
