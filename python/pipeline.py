from savant_rs.pipeline import VideoPipelineStagePayloadType, \
    add_stage, \
    get_stage_type, \
    add_frame_update, add_batched_frame_update, \
    add_frame, \
    delete, \
    get_independent_frame, \
    get_batched_frame, get_batch, apply_updates, \
    move_as_is, move_and_pack_frames, move_and_unpack_batch, get_stage_queue_len

from savant_rs.utils import gen_frame
from savant_rs.primitives import VideoFrameUpdate, VideoObjectUpdateCollisionResolutionPolicy, \
    AttributeUpdateCollisionResolutionPolicy
from savant_rs import init_jaeger_tracer

if __name__ == "__main__":
    init_jaeger_tracer("demo-pipeline", "localhost:6831")
    add_stage("input", VideoPipelineStagePayloadType.Frame)
    add_stage("proc1", VideoPipelineStagePayloadType.Batch)
    add_stage("proc2", VideoPipelineStagePayloadType.Batch)
    add_stage("output", VideoPipelineStagePayloadType.Frame)

    assert get_stage_type("input") == VideoPipelineStagePayloadType.Frame
    assert get_stage_type("proc1") == VideoPipelineStagePayloadType.Batch

    frame1 = gen_frame()
    frame1.source_id = "test1"
    frame_id1 = add_frame("input", frame1)
    assert frame_id1 == 1

    frame2 = gen_frame()
    frame2.source_id = "test2"

    frame_id2 = add_frame("input", frame2)
    assert frame_id2 == 2

    update = VideoFrameUpdate()

    update.object_collision_resolution_policy = VideoObjectUpdateCollisionResolutionPolicy.add_foreign_objects()
    update.attribute_collision_resolution_policy = AttributeUpdateCollisionResolutionPolicy.replace_with_foreign()

    add_frame_update("input", frame_id1, update)

    frame1, ctxt1 = get_independent_frame("input", frame_id1)
    print("ctx1", ctxt1.as_dict())

    frame2, ctxt2 = get_independent_frame("input", frame_id2)
    print("ctx2", ctxt2.as_dict())

    batch_id = move_and_pack_frames("input", "proc1", [frame_id1, frame_id2])
    assert batch_id == 3
    assert get_stage_queue_len("input") == 0
    assert get_stage_queue_len("proc1") == 1

    apply_updates("proc1", batch_id)

    move_as_is("proc1", "proc2", [batch_id])

    frame_map = move_and_unpack_batch("proc2", "output", batch_id)
    assert len(frame_map) == 2
    assert frame_map == {"test1": frame_id1, "test2": frame_id2}

    frame1, ctxt1 = get_independent_frame("output", frame_id1)
    print("ctx1", ctxt1.as_dict())

    frame2, ctxt2 = get_independent_frame("output", frame_id2)
    print("ctx2", ctxt2.as_dict())

    delete("output", frame_id1)
    delete("output", frame_id2)

    assert get_stage_queue_len("input") == 0
    assert get_stage_queue_len("proc1") == 0
    assert get_stage_queue_len("proc2") == 0
    assert get_stage_queue_len("output") == 0
