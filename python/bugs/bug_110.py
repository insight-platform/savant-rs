from savant_rs.primitives import (IdCollisionResolutionPolicy, VideoFrame,
                                  VideoFrameContent, VideoObject)
from savant_rs.primitives.geometry import BBox
from savant_rs.utils.serialization import Message, load_message, save_message

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

frame.add_object(
    VideoObject(
        id=-1401514819,
        namespace="yolov8n",
        label="Car",
        detection_box=BBox(485, 675, 886, 690).as_rbbox(),
        confidence=0.933,
        attributes=[],
        track_id=None,
        track_box=None,
    ),
    IdCollisionResolutionPolicy.Error,
)

frame.add_object(
    VideoObject(
        id=537435614,
        namespace="LPDNet",
        label="lpd",
        detection_box=BBox(557.58374, 883.9291, 298.5735, 84.460144).as_rbbox(),
        confidence=0.39770508,
        attributes=[],
        track_id=None,
        track_box=None,
    ),
    IdCollisionResolutionPolicy.Error,
)

frame.set_parent_by_id(537435614, -1401514819)

print(frame.get_object(-1401514819))
print(frame.get_object(537435614))

assert len(frame.get_children(-1401514819)) == 1

m = Message.video_frame(frame)

s = save_message(m)
new_m = load_message(s)

frame = new_m.as_video_frame()
frame.source_id = "Test2"

print(frame.get_object(-1401514819))
print(frame.get_object(537435614))

assert len(frame.get_children(-1401514819)) == 1

frame2 = frame.copy()
frame2.source_id = "Test3"

print(frame2.get_object(-1401514819))
print(frame2.get_object(537435614))

assert len(frame2.get_children(-1401514819)) == 1
