"""Tests for savant_rs.pipeline – Pipeline, PipelineConfiguration,
StageFunction, and related types."""

from __future__ import annotations

import pytest

from savant_rs.pipeline import (
    FrameProcessingStatRecordType,
    StageFunction,
    VideoPipeline,
    VideoPipelineConfiguration,
    VideoPipelineStagePayloadType,
)
from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    VideoFrame,
    VideoFrameContent,
    VideoFrameUpdate,
)
from savant_rs.utils import TelemetrySpan


# ── VideoPipelineStagePayloadType ────────────────────────────────────────


class TestVideoPipelineStagePayloadType:
    def test_variants(self):
        assert VideoPipelineStagePayloadType.Frame is not None
        assert VideoPipelineStagePayloadType.Batch is not None


# ── FrameProcessingStatRecordType ────────────────────────────────────────


class TestFrameProcessingStatRecordType:
    def test_variants(self):
        assert FrameProcessingStatRecordType.Initial is not None
        assert FrameProcessingStatRecordType.Frame is not None
        assert FrameProcessingStatRecordType.Timestamp is not None


# ── StageFunction ────────────────────────────────────────────────────────


class TestStageFunction:
    def test_none(self):
        sf = StageFunction.none()
        assert sf is not None


# ── PipelineConfiguration ────────────────────────────────────────────────


class TestPipelineConfiguration:
    def test_create(self):
        cfg = VideoPipelineConfiguration()
        assert cfg is not None

    def test_set_keyframe_history(self):
        cfg = VideoPipelineConfiguration()
        cfg.keyframe_history = 10
        s = repr(cfg)
        assert "keyframe_history: 10" in s

    def test_set_append_frame_meta(self):
        cfg = VideoPipelineConfiguration()
        cfg.append_frame_meta_to_otlp_span = True
        s = repr(cfg)
        assert "true" in s

    def test_set_timestamp_period(self):
        cfg = VideoPipelineConfiguration()
        cfg.timestamp_period = 5000
        s = repr(cfg)
        assert "5000" in s

    def test_set_frame_period(self):
        cfg = VideoPipelineConfiguration()
        cfg.frame_period = 100
        s = repr(cfg)
        assert "100" in s

    def test_set_collection_history(self):
        cfg = VideoPipelineConfiguration()
        cfg.collection_history = 50
        s = repr(cfg)
        assert "50" in s

    def test_repr(self):
        cfg = VideoPipelineConfiguration()
        assert isinstance(repr(cfg), str)
        assert isinstance(str(cfg), str)


# ── Pipeline ─────────────────────────────────────────────────────────────


class TestPipeline:
    @pytest.fixture()
    def pipeline(self):
        cfg = VideoPipelineConfiguration()
        stages = [
            (
                "input",
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                "proc",
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
        ]
        return VideoPipeline("test-pipeline", stages, cfg)

    def test_create(self, pipeline):
        assert pipeline is not None

    def test_memory_handle(self, pipeline):
        assert isinstance(pipeline.memory_handle, int)

    def test_root_span_name(self, pipeline):
        name = pipeline.root_span_name
        assert isinstance(name, str)

    def test_sampling_period(self, pipeline):
        sp = pipeline.sampling_period
        assert isinstance(sp, int)

    def test_get_stage_type(self, pipeline):
        st = pipeline.get_stage_type("input")
        assert st is not None

    def test_get_stage_queue_len(self, pipeline):
        qlen = pipeline.get_stage_queue_len("input")
        assert isinstance(qlen, int)
        assert qlen == 0

    def test_add_frame_and_retrieve(self, pipeline):
        f = VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=640,
            height=480,
            content=VideoFrameContent.none(),
        )
        fid = pipeline.add_frame("input", f)
        assert isinstance(fid, int)

        retrieved_frame, span = pipeline.get_independent_frame(fid)
        assert retrieved_frame is not None
        assert retrieved_frame.source_id == "cam"

    def test_delete(self, pipeline):
        f = VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=640,
            height=480,
            content=VideoFrameContent.none(),
        )
        fid = pipeline.add_frame("input", f)
        result = pipeline.delete(fid)
        assert isinstance(result, dict)

    def test_add_frame_update_and_apply(self, pipeline):
        f = VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=640,
            height=480,
            content=VideoFrameContent.none(),
        )
        fid = pipeline.add_frame("input", f)

        update = VideoFrameUpdate()
        update.add_frame_attribute(
            Attribute.persistent("ns", "key", [AttributeValue.string("value")])
        )
        pipeline.add_frame_update(fid, update)
        pipeline.apply_updates(fid)

        retrieved, _ = pipeline.get_independent_frame(fid)
        assert retrieved.get_attribute("ns", "key") is not None

    def test_get_stat_records(self, pipeline):
        records = pipeline.get_stat_records(10)
        assert isinstance(records, list)

    def test_add_frame_with_telemetry(self, pipeline):
        f = VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=640,
            height=480,
            content=VideoFrameContent.none(),
        )
        span = TelemetrySpan.default()
        fid = pipeline.add_frame_with_telemetry("input", f, span)
        assert isinstance(fid, int)
