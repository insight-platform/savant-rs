from savant_rs.pipeline import VideoPipelineStagePayloadType, \
    add_stage, \
    retrieve_telemetry, \
    get_stage_type, \
    add_frame_update, add_batched_frame_update, \
    add_frame, \
    delete, \
    get_independent_frame, \
    get_batched_frame, get_batch, apply_updates, \
    move_as_is, move_and_pack_frames, move_and_unpack_batch, get_stage_queue_len, add_user_telemetry

from savant_rs.utils import gen_frame
from savant_rs.primitives import VideoFrameUpdate, VideoObjectUpdateCollisionResolutionPolicy, \
    AttributeUpdateCollisionResolutionPolicy

if __name__ == "__main__":
    add_stage("input", VideoPipelineStagePayloadType.Frame)
    add_stage("proc1", VideoPipelineStagePayloadType.Batch)
    add_stage("proc2", VideoPipelineStagePayloadType.Batch)
    add_stage("output", VideoPipelineStagePayloadType.Frame)

    telemetry = retrieve_telemetry()  # list(VideoPipelineTelemetryMessage, ...)
    assert len(telemetry) == 0

    assert get_stage_type("input") == VideoPipelineStagePayloadType.Frame
    assert get_stage_type("proc1") == VideoPipelineStagePayloadType.Batch

    frame = gen_frame()
    frame.trace_id = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f]
    frame_id = add_frame("input", frame)
    assert frame_id == 1

    telemetry = retrieve_telemetry()
    assert len(telemetry) == 1
    print(telemetry[0])

    update = VideoFrameUpdate()

    update.object_collision_resolution_policy = VideoObjectUpdateCollisionResolutionPolicy.add_foreign_objects()
    update.attribute_collision_resolution_policy = AttributeUpdateCollisionResolutionPolicy.replace_with_foreign()

    add_frame_update("input", frame_id, update)

    frame = get_independent_frame("input", frame_id)
    print("Frame trace_id: ", "".join([f"{x:02x}" for x in frame.trace_id]))

    batch_id = move_and_pack_frames("input", "proc1", [frame_id])
    assert batch_id == 2
    assert get_stage_queue_len("input") == 0
    assert get_stage_queue_len("proc1") == 1

    telemetry = retrieve_telemetry()
    assert len(telemetry) == 1
    print(telemetry[0])

    apply_updates("proc1", batch_id)

    move_as_is("proc1", "proc2", [batch_id])

    telemetry = retrieve_telemetry()
    assert len(telemetry) == 1
    print(telemetry[0])

    frame_map = move_and_unpack_batch("proc2", "output", batch_id)
    assert len(frame_map) == 1
    assert frame_map == {frame.source_id: frame_id}

    telemetry = retrieve_telemetry()
    assert len(telemetry) == 1
    print(telemetry[0])

    delete("output", frame_id)

    telemetry = retrieve_telemetry()
    assert len(telemetry) == 1
    print(telemetry[0])

    assert get_stage_queue_len("input") == 0 and get_stage_queue_len("proc1") == 0 and get_stage_queue_len(
        "proc2") == 0 and get_stage_queue_len("output") == 0

    add_user_telemetry(frame.trace_id, "custom.stage", '{"value": 10}')
    telemetry = retrieve_telemetry()
    assert len(telemetry) == 1
    print(telemetry[0])
