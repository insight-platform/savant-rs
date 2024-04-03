import time
from threading import Thread, current_thread

import savant_plugin_sample
import savant_rs
from savant_rs.logging import log, LogLevel, log_level_enabled
from savant_rs.logging import set_log_level
from savant_rs.pipeline import VideoPipelineStagePayloadType, VideoPipeline, VideoPipelineConfiguration, StageFunction
from savant_rs.primitives import AttributeValue

set_log_level(LogLevel.Trace)

from savant_rs.utils import gen_frame, TelemetrySpan, enable_dl_detection

if __name__ == "__main__":
    savant_rs.savant_rs.version()
    enable_dl_detection()  # enables internal DL detection (checks every 5 secs)
    log(LogLevel.Info, "root", "Begin operation", dict(savant_rs_version=savant_rs.version()))

    # from savant_rs import init_jaeger_tracer
    # init_jaeger_tracer("demo-pipeline", "localhost:6831")

    conf = VideoPipelineConfiguration()
    conf.append_frame_meta_to_otlp_span = True
    conf.frame_period = 1  # every single frame, insane
    conf.timestamp_period = 1000  # every sec

    p = VideoPipeline("video-pipeline-root", [
        ("input", VideoPipelineStagePayloadType.Frame, StageFunction.none(), StageFunction.none()),
    ], conf)
    p.sampling_period = 10

    assert p.get_stage_type("input") == VideoPipelineStagePayloadType.Frame
    frame1 = gen_frame()
    frame1.keyframe = True
    frame1.source_id = "test1"
    frame_id1 = p.add_frame("input", frame1)
    frame1, _ = p.get_independent_frame(frame_id1)
    assert frame1.previous_keyframe_uuid is None
    uuid = frame1.uuid

    frame2 = gen_frame()
    frame2.keyframe = False
    frame2.source_id = "test1"
    frame_id2 = p.add_frame("input", frame2)
    frame2, _ = p.get_independent_frame(frame_id2)
    assert frame2.previous_keyframe_uuid == uuid

    del p
