import time
from threading import Thread, current_thread

from savant_rs.pipeline import VideoPipelineStagePayloadType, VideoPipeline

from savant_rs.utils import gen_frame, OTLPSpan
from savant_rs.primitives import VideoFrameUpdate, VideoObjectUpdateCollisionResolutionPolicy, \
    AttributeUpdateCollisionResolutionPolicy
from savant_rs import init_jaeger_tracer

if __name__ == "__main__":
    init_jaeger_tracer("demo-pipeline", "localhost:6831")
    p = VideoPipeline("demo-pipeline")

    p.add_stage("input", VideoPipelineStagePayloadType.Frame)
    p.add_stage("proc1", VideoPipelineStagePayloadType.Batch)
    p.add_stage("proc2", VideoPipelineStagePayloadType.Batch)
    p.add_stage("output", VideoPipelineStagePayloadType.Frame)

    assert p.get_stage_type("input") == VideoPipelineStagePayloadType.Frame
    assert p.get_stage_type("proc1") == VideoPipelineStagePayloadType.Batch

    external_span = OTLPSpan("new-telemetry")
    print(external_span.trace_id())
    external_span_propagation = external_span.propagate()
    del external_span

    frame1 = gen_frame()
    frame1.source_id = "test1"
    frame_id1 = p.add_frame_with_remote_telemetry("input", frame1, external_span_propagation)
    assert frame_id1 == 1

    frame2 = gen_frame()
    frame2.source_id = "test2"

    frame_id2 = p.add_frame("input", frame2)
    assert frame_id2 == 2

    update = VideoFrameUpdate()

    update.object_collision_resolution_policy = VideoObjectUpdateCollisionResolutionPolicy.add_foreign_objects()
    update.attribute_collision_resolution_policy = AttributeUpdateCollisionResolutionPolicy.replace_with_foreign()

    p.add_frame_update("input", frame_id1, update)

    frame1, ctxt1 = p.get_independent_frame("input", frame_id1)
    print("ctx1", ctxt1.as_dict())

    frame2, ctxt2 = p.get_independent_frame("input", frame_id2)
    print("ctx2", ctxt2.as_dict())

    batch_id = p.move_and_pack_frames("input", "proc1", [frame_id1, frame_id2])
    assert batch_id == 3
    assert p.get_stage_queue_len("input") == 0
    assert p.get_stage_queue_len("proc1") == 1

    p.apply_updates("proc1", batch_id)

    p.move_as_is("proc1", "proc2", [batch_id])

    frame_map = p.move_and_unpack_batch("proc2", "output", batch_id)
    assert len(frame_map) == 2
    assert frame_map == {"test1": frame_id1, "test2": frame_id2}

    frame1, ctxt1 = p.get_independent_frame("output", frame_id1)
    with ctxt1.nested_span("print"):
        print("ctx1", ctxt1.as_dict())

    frame2, ctxt2 = p.get_independent_frame("output", frame_id2)
    print("ctx2", ctxt2.as_dict())

    root_spans_1 = p.delete("output", frame_id1)
    root_spans_1 = root_spans_1[1]
    print("root_spans 1", root_spans_1.trace_id())

    root_spans_2 = p.delete("output", frame_id2)
    root_spans_2 = root_spans_2[2]
    print("root_spans 2", root_spans_2.trace_id())

    with root_spans_1.nested_span("queue_len") as ns:
        assert p.get_stage_queue_len("input") == 0
        assert p.get_stage_queue_len("proc1") == 0
        assert p.get_stage_queue_len("proc2") == 0
        assert p.get_stage_queue_len("output") == 0
        time.sleep(0.1)
        with ns.nested_span("sleep") as external_span:
            external_span.set_float_attribute("seconds", 0.01)
            time.sleep(0.01)

    def f(span):
        with span.nested_span("func") as s:
            s.set_float_attribute("seconds", 0.1)
            s.set_string_attribute("thread_name", current_thread().name)
            for i in range(10):
                with s.nested_span("loop") as s1:
                    s1.set_status_ok()
                    s1.set_int_attribute("i", i)
                    s1.add_event("Begin computation", {"res": str(1)})
                    res = 1
                    for i in range(1, 1000):
                        res += i
                    s1.set_string_attribute("res", str(res))
                    s1.add_event("End computation", {"res": str(res)})
                    time.sleep(0.1)

    thr1 = Thread(target=f, args=(root_spans_1,))
    thr2 = Thread(target=f, args=(root_spans_1,))
    t = time.time()
    thr1.start()
    thr2.start()

    thr1.join()
    thr2.join()
    print("time", time.time() - t)

    with root_spans_1.nested_span("sleep-1") as external_span:
        external_span.set_float_attribute("seconds", 0.2)
        time.sleep(0.2)

    time.sleep(0.3)



    del root_spans_1


