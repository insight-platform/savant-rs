from timeit import timeit

from savant_rs.primitives import VideoObject, VideoFrame, VideoFrameContent, IdCollisionResolutionPolicy
from savant_rs.primitives.geometry import BBox

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

def add_object_fn(frame: VideoFrame):
    obj = VideoObject(
        id=0,
        namespace="some",
        label="person",
        detection_box=BBox(0.1, 0.2, 0.3, 0.4).as_rbbox(),
        confidence=0.5,
        attributes=[],
        track_id=None,
        track_box=None
    )
    frame.add_object(obj, IdCollisionResolutionPolicy.GenerateNewId)


print(timeit(lambda: add_object_fn(frame), number=10000))


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

def create_object_fn(frame: VideoFrame):
    frame.create_object(namespace="some",
                        label="person",
                        detection_box=BBox(0.1, 0.2, 0.3, 0.4).as_rbbox(),
                        confidence=0.5,
                        attributes=[],
                        track_id=None,
                        track_box=None)

print(timeit(lambda: create_object_fn(frame), number=10000))
