from savant_rs.utils import gen_frame
from savant_rs.utils.serialization import save_message, load_message, Message
from savant_rs.primitives import VideoObject, AttributeValue, \
    Attribute, VideoFrame, VideoFrameUpdate, ObjectUpdateCollisionResolutionPolicy, \
    AttributeUpdateCollisionResolutionPolicy
from savant_rs.video_object_query import Query as Q

frame = gen_frame()
update = VideoFrameUpdate()

update.object_collision_resolution_policy = ObjectUpdateCollisionResolutionPolicy.add_foreign_objects()
update.attribute_collision_resolution_policy = AttributeUpdateCollisionResolutionPolicy.replace_with_foreign()
objects = frame.access_objects(Q.idle())

for o in objects:
    update.add_object(o)

attributes = frame.attributes

for (creator, label) in attributes:
    attr = frame.get_attribute(creator, label)
    update.add_attribute(attr)

m = Message.video_frame_update(update)
binary = save_message(m)
m2 = load_message(binary)

