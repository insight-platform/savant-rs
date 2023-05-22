from savant_rs.utils import gen_frame, save_message, load_message
from savant_rs.primitives import Message, ParentObject, Object, PolygonalArea, Point, BBox, Value, Attribute, VideoFrame, PyVideoFrameContent, PyFrameTransformation
import json
from timeit import default_timer as timer

f = gen_frame()
t = timer()
for _ in range(1_000):
    r = f.json
print(timer() - t)

r = json.loads(f.json)
print(type(r) is dict)

frame = VideoFrame(
    source_id="Test",
    framerate="30/1",
    width=1920,
    height=1080,
    content=PyVideoFrameContent.external("s3", "s3://some-bucket/some-key.jpeg"),
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

frame.add_transformation(PyFrameTransformation.initial_size(1920, 1080))
frame.add_transformation(PyFrameTransformation.scale(3840, 2160))
frame.add_transformation(PyFrameTransformation.padding(left=0, top=120, right=0, bottom=0))

print(frame.transformations)

print(frame.transformations[0].is_initial_size)
print(frame.transformations[0].as_initial_size)

print(frame.transformations[0].is_scale)

frame.clear_transformations()

frame.set_attribute(Attribute(creator="some", name="attr", hint="x", values=[
    Value.none(),
    Value.bytes(dims=[8, 3, 8, 8], blob=bytes(3 * 8 * 8), confidence=None),
    Value.integer(1, confidence=0.5),
    Value.float(1.0, confidence=0.5),
    Value.string("hello", confidence=0.5),
    Value.boolean(True, confidence=0.5),
    Value.strings(["hello", "world"], confidence=0.5),
    Value.integers([1, 2, 3], confidence=0.5),
    Value.floats([1.0, 2.0, 3.0], confidence=0.5),
    Value.booleans([True, False, True], confidence=0.5),
    Value.bbox(BBox(0.1, 0.2, 0.3, 0.4).as_rbbox(), confidence=0.5),
    Value.bboxes([BBox(0.1, 0.2, 0.3, 0.4).as_rbbox(), BBox(0.1, 0.2, 0.3, 0.4).as_rbbox()], confidence=0.5),
    Value.point(Point(0.1, 0.2), confidence=0.5),
    Value.points([Point(0.1, 0.2), Point(0.1, 0.2)], confidence=0.5),
    Value.polygon(PolygonalArea([Point(-1, 1), Point(1, 1), Point(1, -1), Point(-1, -1)], ["up", None, "down", None]), confidence=0.5),
    Value.polygons([
        PolygonalArea([Point(-1, 1), Point(1, 1), Point(1, -1), Point(-1, -1)], ["up", None, "down", None]),
        PolygonalArea([Point(-1, 1), Point(1, 1), Point(1, -1), Point(-1, -1)], ["up", None, "down", None])], confidence=0.5),
]))

frame.set_attribute(Attribute(creator="other", name="attr", values=[
    Value.integer(1, confidence=0.5),
]))

print(frame.attributes)

print(frame.find_attributes(name="attr"))
print(frame.find_attributes(creator="other"))
print(frame.find_attributes(creator="other", name="attr"))
print(frame.find_attributes(hint="x"))
print(frame.find_attributes(creator="some", hint="x"))

print(frame.get_attribute(creator="other", name="attr"))
deleted = frame.delete_attribute(creator="some", name="attr")
print(deleted)

frame.add_object(Object(
    id=1,
    creator="some",
    label="person",
    bbox=BBox(0.1, 0.2, 0.3, 0.4).as_rbbox(),
    confidence=0.5,
    attributes={},
    parent=ParentObject(id=2, creator="some", label="car"),
    track_id=None,
))

o = frame.access_objects_by_id([1])
o = o[0]
print(o)

o.set_attribute(Attribute(creator="other", name="attr", values=[
    Value.integer(1, confidence=0.5),
]))

o.set_attribute(Attribute(creator="some", name="attr", values=[
    Value.integers([1, 2, 3], confidence=0.5),
]))

message = Message.video_frame(frame)

t = timer()

frame_message = None
for _ in range(1_000):
    bytes = save_message(message)
    frame_message = load_message(bytes)

print("1K ser/des for frame took:", timer() - t)

print(frame_message.is_video_frame)
frame = frame_message.as_video_frame

# print(frame)


