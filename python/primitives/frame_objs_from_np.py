import numpy as np
from savant_rs.primitives import VideoFrame, VideoFrameContent
from savant_rs.utils.symbol_mapper import register_model_objects, RegistrationPolicy
from savant_rs.video_object_query import MatchQuery as Q

register_model_objects("detector", { 1: "person" }, RegistrationPolicy.ErrorIfNonUnique)

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

objs = np.array([
    [1, 0.75, 0.0, 0.0, 10.0, 20.0]
])

frame.create_objects_from_numpy("detector", objs)
objs = frame.access_objects(Q.idle())
assert len(objs) == 1

incorrect_objs = np.array([
    [1, 0.75, 0.0, 0.0, 10.0]
])

try:
    frame.create_objects_from_numpy("detector", incorrect_objs)
    assert False
except:
    assert True

try:
    # not registered model
    frame.create_objects_from_numpy("detector2", objs)
    assert False
except:
    assert True

unregistered_objs = np.array([
    [2, 0.75, 0.0, 0.0, 10.0, 20.0]
])

try:
    # not registered class
    frame.create_objects_from_numpy("detector", objs)
    assert False
except:
    assert True

objs_with_angle = np.array([
    [1, 0.75, 0.0, 0.0, 10.0, 20.0, 35]
])

frame.create_objects_from_numpy("detector", objs_with_angle)
objs = frame.access_objects(Q.idle())
assert len(objs) == 2

