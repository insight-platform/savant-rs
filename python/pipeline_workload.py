import time
from threading import Thread, current_thread

import savant_plugin_sample
import savant_rs
from savant_rs.logging import log, LogLevel, log_level_enabled
from savant_rs.logging import set_log_level
from savant_rs.pipeline import VideoPipelineStagePayloadType, VideoPipeline, VideoPipelineConfiguration, StageFunction
from savant_rs.primitives import AttributeValue

print(savant_rs.version())
set_log_level(LogLevel.Info)

plugin_function_1 = savant_plugin_sample.get_instance("doesnotmatter", {})
plugin_function_2 = savant_plugin_sample.get_instance("doesnotmatter", dict(attr=AttributeValue.integer(1)))

from savant_rs.utils import gen_frame, TelemetrySpan, enable_dl_detection
from savant_rs.primitives import VideoFrameUpdate, ObjectUpdatePolicy, \
    AttributeUpdatePolicy
from savant_rs.match_query import MatchQuery as Q

# LOGLEVEL=info,a=error,a.b=debug python python/pipeline.py


if __name__ == "__main__":
    conf = VideoPipelineConfiguration()
    conf.append_frame_meta_to_otlp_span = True
    conf.frame_period = 1000  # every single frame, insane
    conf.timestamp_period = 1000  # every sec

    p = VideoPipeline("video-pipeline-root", [
        ("input", VideoPipelineStagePayloadType.Frame, plugin_function_1, plugin_function_2),
        ("proc1", VideoPipelineStagePayloadType.Batch, StageFunction.none(), StageFunction.none()),
        ("proc2", VideoPipelineStagePayloadType.Batch, StageFunction.none(), StageFunction.none()),
        ("output", VideoPipelineStagePayloadType.Frame, StageFunction.none(), StageFunction.none()),
    ], conf)
    p.sampling_period = 10

    for it in range(1_000_000):
        if it % 1000 == 0:
            log(LogLevel.Info, "root", "Iteration {}".format(it), None)
        frame1 = gen_frame()
        frame1.source_id = "test1" + str(it)
        frame_id1 = p.add_frame("input", frame1)
        frame2 = gen_frame()
        frame2.source_id = "test2" + str(it)
        frame_id2 = p.add_frame("input", frame2)
        update = VideoFrameUpdate()
        update.object_policy = ObjectUpdatePolicy.AddForeignObjects
        update.frame_attribute_policy = AttributeUpdatePolicy.ReplaceWithForeignWhenDuplicate
        update.object_attribute_policy = AttributeUpdatePolicy.ReplaceWithForeignWhenDuplicate
        p.add_frame_update(frame_id1, update)
        frame1, ctxt1 = p.get_independent_frame(frame_id1)
        frame1, ctxt1 = p.get_independent_frame(frame_id1)
        with ctxt1.nested_span("print"):
            ctxt1.set_float_attribute("seconds", 0.01)
            log(LogLevel.Trace, "root", "Context 1: {}".format(ctxt1.propagate().as_dict()), None)
        frame2, ctxt2 = p.get_independent_frame(frame_id2)
        batch_id = p.move_and_pack_frames("proc1", [frame_id1, frame_id2])
        p.apply_updates(batch_id)
        p.clear_updates(batch_id)
        p.move_as_is("proc2", [batch_id])
        objects = p.access_objects(batch_id, Q.idle())
        frame_map = p.move_and_unpack_batch("output", batch_id)
        frame1, ctxt1 = p.get_independent_frame(frame_id1)
        frame2, ctxt2 = p.get_independent_frame(frame_id2)
        root_spans_1 = p.delete(frame_id1)
        root_spans_2 = p.delete(frame_id2)

        # todo: remove when testing versus memory leak
        exit(0)

    p.log_final_fps()
    # del root_spans_1
    del p
