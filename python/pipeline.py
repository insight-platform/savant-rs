import time
from threading import Thread, current_thread

import savant_plugin_sample
import savant_rs
from savant_rs.logging import log, LogLevel, log_level_enabled
from savant_rs.logging import set_log_level
from savant_rs.pipeline import (
    VideoPipelineStagePayloadType,
    VideoPipeline,
    VideoPipelineConfiguration,
    StageFunction,
)
from savant_rs.primitives import AttributeValue

set_log_level(LogLevel.Trace)

# plugin_function_1 = savant_plugin_sample.get_instance("doesnotmatter", {})
# plugin_function_2 = savant_plugin_sample.get_instance(
#    "doesnotmatter", dict(attr=AttributeValue.integer(1))
# )

from savant_rs.utils import gen_frame, TelemetrySpan, enable_dl_detection
from savant_rs.primitives import (
    VideoFrameUpdate,
    ObjectUpdatePolicy,
    AttributeUpdatePolicy,
)
from savant_rs.match_query import MatchQuery as Q

# LOGLEVEL=info,a=error,a.b=debug python python/pipeline.py

if __name__ == "__main__":
    savant_rs.savant_rs.version()
    enable_dl_detection()  # enables internal DL detection (checks every 5 secs)
    log(
        LogLevel.Info,
        "root",
        "Begin operation",
        dict(savant_rs_version=savant_rs.version()),
    )

    # from savant_rs.telemetry import init, shutdown, Protocol, TelemetryConfiguration, TracerConfiguration
    # tracer_conf = TracerConfiguration("demo-pipeline", Protocol.Grpc, "http://localhost:4317")
    # telemetry_conf = TelemetryConfiguration(tracer=tracer_conf)
    # init(telemetry_conf)

    conf = VideoPipelineConfiguration()
    conf.append_frame_meta_to_otlp_span = True
    conf.frame_period = 1  # every single frame, insane
    conf.timestamp_period = 1000  # every sec

    p = VideoPipeline(
        "video-pipeline-root",
        [
            (
                "input",
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                "proc1",
                VideoPipelineStagePayloadType.Batch,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                "proc2",
                VideoPipelineStagePayloadType.Batch,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                "output",
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
        ],
        conf,
    )
    p.sampling_period = 10

    assert p.get_stage_type("input") == VideoPipelineStagePayloadType.Frame
    assert p.get_stage_type("proc1") == VideoPipelineStagePayloadType.Batch

    root_span = TelemetrySpan("new-telemetry")
    log(
        LogLevel.Info,
        target="root",
        message="TraceID={}".format(root_span.trace_id()),
        params=None,
    )
    external_span_propagation = root_span.propagate()

    frame1 = gen_frame()
    frame1.source_id = "test1"
    frame_id1 = p.add_frame_with_telemetry("input", frame1, root_span)
    del root_span
    assert frame_id1 == 1

    frame2 = gen_frame()
    frame2.source_id = "test2"

    frame_id2 = p.add_frame("input", frame2)
    assert frame_id2 == 2

    update = VideoFrameUpdate()

    update.object_policy = ObjectUpdatePolicy.AddForeignObjects
    update.frame_attribute_policy = (
        AttributeUpdatePolicy.ReplaceWithForeignWhenDuplicate
    )
    update.object_attribute_policy = (
        AttributeUpdatePolicy.ReplaceWithForeignWhenDuplicate
    )

    p.add_frame_update(frame_id1, update)

    frame1, ctxt1 = p.get_independent_frame(frame_id1)
    log(
        LogLevel.Info, "root", "Context 1: {}".format(ctxt1.propagate().as_dict()), None
    )

    frame2, ctxt2 = p.get_independent_frame(frame_id2)
    log(
        LogLevel.Info, "root", "Context 2: {}".format(ctxt2.propagate().as_dict()), None
    )

    history1 = p.get_keyframe_history(frame1)
    history2 = p.get_keyframe_history(frame2)
    print(history1, history2)

    batch_id = p.move_and_pack_frames("proc1", [frame_id1, frame_id2])
    assert batch_id == 3
    assert p.get_stage_queue_len("input") == 0
    assert p.get_stage_queue_len("proc1") == 1

    p.apply_updates(batch_id)
    p.clear_updates(batch_id)

    p.move_as_is("proc2", [batch_id])
    objects = p.access_objects(batch_id, Q.idle())

    frame_map = p.move_and_unpack_batch("output", batch_id)
    frame_map.sort()
    assert len(frame_map) == 2
    assert frame_map == [frame_id1, frame_id2]

    frame1, ctxt1 = p.get_independent_frame(frame_id1)
    with ctxt1.nested_span("print"):
        log(
            LogLevel.Info,
            "root",
            "Context 1: {}".format(ctxt1.propagate().as_dict()),
            None,
        )

    frame2, ctxt2 = p.get_independent_frame(frame_id2)
    log(
        LogLevel.Info, "root", "Context 2: {}".format(ctxt2.propagate().as_dict()), None
    )

    root_spans_1 = p.delete(frame_id1)
    root_spans_1 = root_spans_1[1]
    root_spans_1_propagated = root_spans_1.propagate()
    del root_spans_1

    root_spans_2 = p.delete(frame_id2)
    del root_spans_2

    with root_spans_1_propagated.nested_span("queue_len") as ns:
        assert p.get_stage_queue_len("input") == 0
        assert p.get_stage_queue_len("proc1") == 0
        assert p.get_stage_queue_len("proc2") == 0
        assert p.get_stage_queue_len("output") == 0
        time.sleep(0.1)
        with ns.nested_span("sleep") as root_span:
            root_span.set_float_attribute("seconds", 0.01)
            time.sleep(0.01)


    # shutdown()

    def f(span):
        with span.nested_span("func") as s:
            log(
                LogLevel.Error,
                "a",
                "Context Depth: {}".format(TelemetrySpan.context_depth()),
                dict(context_depth=TelemetrySpan.context_depth()),
            )
            s.set_float_attribute("seconds", 0.1)
            s.set_string_attribute("thread_name", current_thread().name)
            for i in range(10):
                with s.nested_span("loop") as s1:
                    log(
                        LogLevel.Warning,
                        "a::b",
                        "Context Depth: {}".format(TelemetrySpan.context_depth()),
                    )
                    s1.set_status_ok()
                    s1.set_int_attribute("i", i)
                    s1.add_event("Begin computation", {"res": str(1)})
                    res = 1
                    for i in range(1, 1000):
                        res += i
                    s1.set_string_attribute("res", str(res))
                    s1.add_event("End computation", {"res": str(res)})
                    time.sleep(0.1)
                log(
                    LogLevel.Warning,
                    "a::b",
                    "Context Depth: {}".format(TelemetrySpan.context_depth()),
                )
        log(
            LogLevel.Warning,
            "c",
            "Context Depth: {}".format(TelemetrySpan.context_depth()),
        )


    thr1 = Thread(target=f, args=(root_spans_1_propagated,))
    thr2 = Thread(target=f, args=(root_spans_1_propagated,))
    t = time.time()
    thr1.start()
    thr2.start()

    thr1.join()
    thr2.join()
    delta = time.time() - t
    log(LogLevel.Info, "root", "Time: {}".format(delta), params=dict(time_spent=delta))

    try:
        with root_spans_1_propagated.nested_span("sleep-1") as root_span:
            with root_span.nested_span_when(
                    "sleep-debugging", log_level_enabled(LogLevel.Debug)
            ) as sds:
                log(LogLevel.Info, "a::b::c", "Always seen when Info")
            if log_level_enabled(LogLevel.Debug):
                log(LogLevel.Debug, "a::b", "I'm debugging: {}".format(1))
            root_span.set_float_attribute("seconds", 0.2)
            time.sleep(0.2)
            if log_level_enabled(LogLevel.Debug):
                log(LogLevel.Debug, "a::b", "I'm debugging: {}".format(2))
            if log_level_enabled(LogLevel.Warning):
                log(LogLevel.Warning, "a::b", "I'm warning: {}".format(1))
            raise Exception("test")
    except Exception as e:
        log(LogLevel.Error, "root", "Exception: {}".format(e))

    time.sleep(0.3)
    recs = p.get_stat_records(10)
    # print(recs)
    recs = p.get_stat_records_newer_than(recs[1].id)
    print(recs)
    p.log_final_fps()
    recs = p.get_stat_records(1)
    # del root_spans_1

del p
