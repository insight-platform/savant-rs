from savant_rs.match_query import IntExpression as IE
from savant_rs.match_query import MatchQuery as Q
from savant_rs.match_query import QueryFunctions as QF
from savant_rs.utils import gen_frame

f = gen_frame()

objects_x = QF.filter(f.access_objects(Q.idle()), Q.eval("id % 2 == 1")).sorted_by_id

objects = QF.filter(f.access_objects(Q.idle()), Q.id(IE.one_of(1, 2))).sorted_by_id

ids = objects.ids
print("Ids:", ids)
track_ids = objects.track_ids
print("Track ids:", track_ids)

# boxes = objects.rotated_boxes_as_numpy(VideoObjectBBoxType.Detection)
# print("Detections:", boxes)
#
# tr_boxes = objects.rotated_boxes_as_numpy(VideoObjectBBoxType.TrackingInfo)
# print("Tracking:", tr_boxes)

# objects.update_from_numpy_boxes(boxes, BBoxFormat.XcYcWidthHeight, VideoObjectBBoxType.Detection)
# objects.update_from_numpy_rotated_boxes(boxes, VideoObjectBBoxType.Detection)

# tracking_boxes = objects.tracking_boxes_as_numpy()
