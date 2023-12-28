from savant_rs.primitives import VideoFrameUpdate, ObjectUpdatePolicy, \
    AttributeUpdatePolicy
from savant_rs.utils import gen_frame
from savant_rs.utils.serialization import save_message, load_message, Message
from savant_rs.video_object_query import MatchQuery as Q

frame = gen_frame()
update = VideoFrameUpdate()

update.object_policy = ObjectUpdatePolicy.AddForeignObjects
update.frame_attribute_policy = AttributeUpdatePolicy.ReplaceWithForeignWhenDuplicate
update.object_attribute_policy = AttributeUpdatePolicy.ReplaceWithForeignWhenDuplicate

objects = frame.access_objects(Q.idle())

for o in objects:
    update.add_object(o.detached_copy(), None)

attributes = frame.attributes

for (namespace, label) in attributes:
    attr = frame.get_attribute(namespace, label)
    update.add_frame_attribute(attr)

print(update.json)
print(update.json_pretty)

pb = update.to_protobuf()
restored = VideoFrameUpdate.from_protobuf(pb)
assert update.json == restored.json

m = Message.video_frame_update(update)
binary = save_message(m)
m2 = load_message(binary)
