import json
from timeit import default_timer as timer
from typing import Optional

from savant_rs.draw_spec import SetDrawLabelKind
from savant_rs.logging import LogLevel, set_log_level
from savant_rs.match_query import IntExpression as IE
from savant_rs.match_query import MatchQuery as Q
from savant_rs.match_query import QueryFunctions as QF
from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    IdCollisionResolutionPolicy,
    VideoFrame,
    VideoFrameContent,
    VideoFrameTransformation,
    VideoObject,
)
from savant_rs.primitives.geometry import BBox, Point, PolygonalArea
from savant_rs.utils import gen_frame
from savant_rs.utils.serialization import Message, load_message, save_message

set_log_level(LogLevel.Info)

f = gen_frame()
print(f.json_pretty)
f.creation_timestamp_ns = 1_000_000_000

assert len(f.get_children(0)) == 2

print(f.uuid)
print(f.creation_timestamp_ns)

pb = f.to_protobuf()
restored = VideoFrame.from_protobuf(pb)
assert f.uuid == restored.uuid

t = timer()
for _ in range(1_00):
    r = f.json
print(timer() - t)

r = json.loads(f.json)
print(type(r) is dict)

frame = VideoFrame(
    source_id="Test",
    framerate="30/1",
    width=1920,
    height=1080,
    content=VideoFrameContent.external("s3", "s3://some-bucket/some-key.jpeg"),
    codec="jpeg",
    keyframe=True,
    pts=0,
    dts=None,
    duration=None,
)
print(frame.json)

frame.width = 3840
frame.height = 2160
frame.dts = 100
frame.duration = 100

print(frame.json)

frame.add_transformation(VideoFrameTransformation.initial_size(1920, 1080))
frame.add_transformation(VideoFrameTransformation.scale(3840, 2160))
frame.add_transformation(
    VideoFrameTransformation.padding(left=0, top=120, right=0, bottom=0)
)

print(frame.transformations)

print(frame.transformations[0].is_initial_size)
print(frame.transformations[0].as_initial_size)

print(frame.transformations[0].is_scale)

frame.clear_transformations()

frame.set_persistent_attribute(
    namespace="some",
    name="attr",
    hint="x",
    values=[
        AttributeValue.none(),
        AttributeValue.bytes(dims=[8, 3, 8, 8], blob=bytes(3 * 8 * 8), confidence=None),
        AttributeValue.bytes_from_list(dims=[4, 1], blob=[0, 1, 2, 3], confidence=None),
        AttributeValue.integer(1, confidence=0.5),
        AttributeValue.float(1.0, confidence=0.5),
        AttributeValue.string("hello", confidence=0.5),
        AttributeValue.boolean(True, confidence=0.5),
        AttributeValue.strings(["hello", "world"], confidence=0.5),
        AttributeValue.integers([1, 2, 3], confidence=0.5),
        AttributeValue.floats([1.0, 2.0, 3.0], confidence=0.5),
        AttributeValue.booleans([True, False, True], confidence=0.5),
        AttributeValue.rbbox(BBox(0.1, 0.2, 0.3, 0.4).as_rbbox(), confidence=0.5),
        AttributeValue.rbboxes(
            [BBox(0.1, 0.2, 0.3, 0.4).as_rbbox(), BBox(0.1, 0.2, 0.3, 0.4).as_rbbox()],
            confidence=0.5,
        ),
        AttributeValue.point(Point(0.1, 0.2), confidence=0.5),
        AttributeValue.points([Point(0.1, 0.2), Point(0.1, 0.2)], confidence=0.5),
        AttributeValue.polygon(
            PolygonalArea(
                [Point(-1, 1), Point(1, 1), Point(1, -1), Point(-1, -1)],
                ["up", None, "down", None],
            ),
            confidence=0.5,
        ),
        AttributeValue.polygons(
            [
                PolygonalArea(
                    [Point(-1, 1), Point(1, 1), Point(1, -1), Point(-1, -1)],
                    ["up", None, "down", None],
                ),
                PolygonalArea(
                    [Point(-1, 1), Point(1, 1), Point(1, -1), Point(-1, -1)],
                    ["up", None, "down", None],
                ),
            ],
            confidence=0.5,
        ),
    ],
)

frame.set_persistent_attribute(
    namespace="other",
    name="attr",
    values=[
        AttributeValue.integer(1, confidence=0.5),
    ],
)

frame.set_temporary_attribute(
    "hidden",
    "attribute",
    values=[AttributeValue.temporary_python_object(dict(x=5), confidence=0.5)],
    is_hidden=True,
)

print("All public attributes", frame.attributes)  # hidden is not there
# but we can access it directly
print("Hidden attribute", frame.get_attribute(namespace="hidden", name="attribute"))

print(frame.find_attributes_with_names(["attr"]))
print(frame.find_attributes_with_ns("other"))
print(frame.find_attributes_with_hints(["x"]))

print(frame.get_attribute(namespace="other", name="attr"))
deleted = frame.delete_attribute(namespace="some", name="attr")
print(deleted)

obj = VideoObject(
    id=1,
    namespace="some",
    label="person",
    detection_box=BBox(0.1, 0.2, 0.3, 0.4).as_rbbox(),
    confidence=0.5,
    attributes=[],
    track_id=1,
    track_box=BBox(0.1, 0.2, 0.3, 0.4).as_rbbox(),
)

# demonstrates protobuf serialization
#
pb = obj.to_protobuf()
restored = VideoObject.from_protobuf(pb)
assert obj.id == restored.id

frame.add_object(obj, IdCollisionResolutionPolicy.Error)

f = gen_frame()
print("Raw address to pass to C-funcs: ", f.memory_handle)
vec = f.access_objects(Q.with_children(Q.idle(), IE.eq(2)))

# demonstrates ObjectsView len() op
print("ObjectsView len() op", len(vec))

print("Object with two children:", vec[0])


# demonstrates ObjectsView index access operation
vec = vec[0]
print("Object", vec)

parent_chain = f.get_parent_chain(vec)
print(f"Parent chain for object {vec.id}: {parent_chain}")

print(f"Parent id for object {vec.id}: {vec.parent_id}")

vec.set_attribute(
    Attribute(
        namespace="other",
        name="attr",
        values=[
            AttributeValue.integer(1, confidence=0.5),
        ],
    )
)

vec.set_attribute(
    Attribute(
        namespace="some",
        name="attr",
        values=[
            AttributeValue.integers([1, 2, 3], confidence=0.5),
        ],
    )
)

# demonstrates chained filtering on ObjectsView object
#
f = gen_frame()
one, two = QF.partition(
    QF.filter(f.access_objects(Q.idle()), Q.id(IE.one_of(1, 2))), Q.id(IE.eq(1))
)

print("One", one)
print("Two", two)


message = Message.video_frame(frame)

t = timer()

frame_message = None
for _ in range(1_000):
    bytes = save_message(message)
frame_message = load_message(bytes)

print("1K ser/des for frame took:", timer() - t)

print(frame_message.is_video_frame())
frame = frame_message.as_video_frame()

# print(frame)
objects = frame.access_objects(Q.idle())
assert len(objects) == 1

frame.set_draw_label(Q.idle(), SetDrawLabelKind.own("person"))
frame.set_draw_label(Q.idle(), SetDrawLabelKind.parent("also_person"))

before_len = len(f.get_all_objects())
trees = f.export_complete_object_trees(Q.idle(), delete_exported=True)

print(trees)


def walker(obj: VideoObject, parent: Optional[VideoObject], result: int):
    print(obj.namespace, obj.label, parent.id if parent else None, result)
    if result is None:
        return 0
    return result + 1


trees[0].walk_objects(walker)

f.import_object_trees(trees)

after_len = len(f.get_all_objects())
assert after_len == before_len
frame.delete_objects(Q.idle())

objects = frame.access_objects(Q.idle())
assert len(objects) == 0
