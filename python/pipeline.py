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

if __name__ == "__main__":
    add_stage("input", VideoPipelineStagePayloadType.Frame)
    add_stage("proc1", VideoPipelineStagePayloadType.Batch)
    add_stage("proc2", VideoPipelineStagePayloadType.Batch)
    add_stage("output", VideoPipelineStagePayloadType.Frame)

    assert get_stage_type("input") == VideoPipelineStagePayloadType.Frame
    assert get_stage_type("proc1") == VideoPipelineStagePayloadType.Batch

    frame = gen_frame()
    frame_id = add_frame("input", frame)
    assert frame_id == 1

    update = VideoFrameUpdate()

    update.object_collision_resolution_policy = VideoObjectUpdateCollisionResolutionPolicy.add_foreign_objects()
    update.attribute_collision_resolution_policy = AttributeUpdateCollisionResolutionPolicy.replace_with_foreign()

    add_frame_update("input", frame_id, update)

    frame = get_independent_frame("input", frame_id)

    batch_id = move_and_pack_frames("input", "proc1", [frame_id])
    assert batch_id == 2
    assert get_stage_queue_len("input") == 0
    assert get_stage_queue_len("proc1") == 1

    apply_updates("proc1", batch_id)

    move_as_is("proc1", "proc2", [batch_id])

    frame_map = move_and_unpack_batch("proc2", "output", batch_id)
    assert len(frame_map) == 1
    assert frame_map == {frame.source_id: frame_id}

    delete("output", frame_id)

    assert get_stage_queue_len("input") == 0 and get_stage_queue_len("proc1") == 0 and get_stage_queue_len(
        "proc2") == 0 and get_stage_queue_len("output") == 0
