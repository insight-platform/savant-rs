import savant_rs

from savant_rs.logging import log, LogLevel, set_log_level

set_log_level(LogLevel.Trace)

from savant_rs.pipeline import VideoPipelineStagePayloadType, VideoPipeline, VideoPipelineConfiguration, StageFunction
from savant_rs.primitives import VideoFrame, VideoFrameContent, VideoFrameTranscodingMethod, VideoFrameTransformation, \
    Attribute, AttributeValue
from savant_rs.utils import gen_frame, TelemetrySpan, enable_dl_detection
from savant_rs import init_jaeger_tracer

if __name__ == "__main__":
    savant_rs.version()
    enable_dl_detection()  # enables internal DL detection (checks every 5 secs)
    log(LogLevel.Info, "root", "Begin operation", dict(savant_rs_version=savant_rs.version()))
    init_jaeger_tracer("demo-pipeline", "localhost:6831")

    conf = VideoPipelineConfiguration()
    conf.append_frame_meta_to_otlp_span = True

    frame = VideoFrame(source_id="test", framerate="30/1", width=1400, height=720,
                       content=VideoFrameContent.internal(bytes("this is it", 'utf-8')),
                       transcoding_method=VideoFrameTranscodingMethod.Encoded, codec="h264", keyframe=True,
                       time_base=(1, 1000000), pts=10000, dts=10000, duration=10)
    frame.add_transformation(VideoFrameTransformation.initial_size(1920, 1080))
    frame.add_transformation(VideoFrameTransformation.scale(1280, 720))
    frame.add_transformation(VideoFrameTransformation.padding(120, 0, 0, 0))
    frame.add_transformation(VideoFrameTransformation.resulting_size(1400, 720))

    frame.set_attribute(Attribute("Configuration", "CamMode",
                                  [AttributeValue.string("fisheye"), AttributeValue.integers([180, 180])],
                                  "PlatformConfig", True))

    print(frame.json_pretty)

    p = VideoPipeline("video-pipeline-root", [
        ("input", VideoPipelineStagePayloadType.Frame, StageFunction.none(), StageFunction.none()),
        ("proc1", VideoPipelineStagePayloadType.Batch, StageFunction.none(), StageFunction.none()),
        ("proc2", VideoPipelineStagePayloadType.Batch, StageFunction.none(), StageFunction.none()),
        ("output", VideoPipelineStagePayloadType.Frame, StageFunction.none(), StageFunction.none()),
    ], conf)
    p.sampling_period = 10

    root_span = TelemetrySpan("new-telemetry")
    frame1 = gen_frame()
    frame1.source_id = "test1"
    frame_id1 = p.add_frame_with_telemetry("input", frame1, root_span)
    del root_span

    batch_id = p.move_and_pack_frames("proc1", [frame_id1])
    p.move_as_is("proc2", [batch_id])
    frame_map = p.move_and_unpack_batch("output", batch_id)
    assert frame_map == [frame_id1]
    root_spans_1 = p.delete(frame_id1)
