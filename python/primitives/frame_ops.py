import ctypes

from savant_rs.utils import gen_frame, save_message, load_message
from savant_rs.primitives import SetDrawLabelKind, Message, ParentObject, Object, PolygonalArea, Point, BBox, Value, \
    Attribute, VideoFrame, PyVideoFrameContent, PyFrameTransformation
from savant_rs.video_object_query import Query as Q, \
    IntExpression as IE
import json
from timeit import default_timer as timer
from ctypes import *

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
    Value.polygon(PolygonalArea([Point(-1, 1), Point(1, 1), Point(1, -1), Point(-1, -1)], ["up", None, "down", None]),
                  confidence=0.5),
    Value.polygons([
        PolygonalArea([Point(-1, 1), Point(1, 1), Point(1, -1), Point(-1, -1)], ["up", None, "down", None]),
        PolygonalArea([Point(-1, 1), Point(1, 1), Point(1, -1), Point(-1, -1)], ["up", None, "down", None])],
        confidence=0.5),
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
    parent=None,
    track=None,
))

f = gen_frame()
print("Raw address to pass to C-funcs: ", f.memory_handle)
o = f.access_objects(Q.with_children(Q.idle(), IE.eq(2)))
print("Object with two children:", o[0])

# demonstrates chained filtering on VectorView object
#
f = gen_frame()
one, two = f.access_objects(Q.idle()) \
    .filter(Q.id(IE.one_of(1, 2))) \
    .partition(Q.id(IE.eq(1)))

print("One", one)
print("Two", two)

# demonstrates Rust/Python/C interoperability with descriptor passing between Rust to C through Python
#
lib = cdll.LoadLibrary("../../target/release/libsavant_rs.so")
lib.object_vector_len.argtypes = [c_uint64]
lib.object_vector_len.rettype = c_uint64
print("Length:", lib.object_vector_len(o.memory_handle))


# Demonstrates Rust/Python/C interoperability with descriptor passing between Rust to C through Python
# Return complex object from C-compatible Rust-function
#
#     pub id: i64,
#     pub creator_id: i64,
#     pub label_id: i64,
#     pub confidence: f64,
#     pub parent_id: i64,
#     pub box_xc: f64,
#     pub box_yx: f64,
#     pub box_width: f64,
#     pub box_height: f64,
#     pub box_angle: f64,
#     pub track_id: i64,
#     pub track_box_xc: f64,
#     pub track_box_yx: f64,
#     pub track_box_width: f64,
#     pub track_box_height: f64,
#     pub track_box_angle: f64,

class InferenceMeta(Structure):
    _fields_ = [
        ("id", c_int64),
        ("creator_id", c_int64),
        ("label_id", c_int64),
        ("confidence", c_double),
        ("parent_id", c_int64),
        ("box_xc", c_double),
        ("box_yx", c_double),
        ("box_width", c_double),
        ("box_height", c_double),
        ("box_angle", c_double),
        ("track_id", c_int64),
        ("track_box_xc", c_double),
        ("track_box_yx", c_double),
        ("track_box_width", c_double),
        ("track_box_height", c_double),
        ("track_box_angle", c_double),
    ]


lib.get_inference_meta.argtypes = [c_uint64, c_uint64]
lib.get_inference_meta.restype = InferenceMeta
meta = lib.get_inference_meta(o.memory_handle, 0)

print("C-struct: ", meta)
for field_name, field_type in meta._fields_:
    print("\t", field_name, getattr(meta, field_name))

# demonstrates VectorView len() op
print("Vector View len() op", len(o))

# demonstrates VectorView index access operation
o = o[0]
print("Object", o)

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
objects = frame.access_objects(Q.idle())
assert len(objects) == 1

frame.set_draw_label(Q.idle(), SetDrawLabelKind.own("person"))
frame.set_draw_label(Q.idle(), SetDrawLabelKind.parent("also_person"))

frame.delete_objects(Q.idle())

objects = frame.access_objects(Q.idle())
assert len(objects) == 0
