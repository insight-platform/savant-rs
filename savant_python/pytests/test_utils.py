"""Tests for savant_rs.utils – eval_expr, round_2_digits, UUID functions,
ByteBuffer, BBoxMetricType, VideoObjectBBoxType, VideoObjectBBoxTransformation,
TelemetrySpan, MaybeTelemetrySpan, PropagatedContext."""

from __future__ import annotations

import pytest

from savant_rs.utils import (
    BBoxMetricType,
    ByteBuffer,
    MaybeTelemetrySpan,
    PropagatedContext,
    TelemetrySpan,
    VideoObjectBBoxTransformation,
    VideoObjectBBoxType,
    eval_expr,
    gen_empty_frame,
    gen_frame,
    incremental_uuid_v7,
    relative_time_uuid_v7,
    round_2_digits,
)
from savant_rs.primitives import VideoFrame


# ── eval_expr ─────────────────────────────────────────────────────────────


class TestEvalExpr:
    def test_math(self):
        result = eval_expr("2 + 3", ttl=100)
        assert result == 5

    def test_float_math(self):
        result = eval_expr("2.5 * 4.0", ttl=100)
        assert result == pytest.approx(10.0)

    def test_string(self):
        result = eval_expr('"hello"', ttl=100)
        assert result == "hello"

    def test_boolean(self):
        result = eval_expr("true", ttl=100)
        assert result is True


# ── round_2_digits ────────────────────────────────────────────────────────


class TestRound2Digits:
    def test_round(self):
        assert round_2_digits(3.14159) == pytest.approx(3.14)

    def test_round_no_change(self):
        assert round_2_digits(1.0) == pytest.approx(1.0)

    def test_round_negative(self):
        assert round_2_digits(-2.567) == pytest.approx(-2.57)


# ── gen_frame / gen_empty_frame ───────────────────────────────────────────


class TestGenFrame:
    def test_gen_frame(self):
        f = gen_frame()
        assert isinstance(f, VideoFrame)
        assert f.source_id == "test"
        assert f.width == 1280
        assert f.height == 720
        assert f.has_objects()

    def test_gen_empty_frame(self):
        f = gen_empty_frame()
        assert isinstance(f, VideoFrame)
        assert f.source_id == "test"
        assert f.width == 0
        assert f.height == 0
        assert not f.has_objects()


# ── UUID functions ────────────────────────────────────────────────────────


class TestUuidFunctions:
    def test_incremental_uuid_v7(self):
        u1 = incremental_uuid_v7()
        u2 = incremental_uuid_v7()
        assert isinstance(u1, str)
        assert len(u1) > 0
        assert u1 != u2

    def test_relative_time_uuid_v7(self):
        base = incremental_uuid_v7()
        rel = relative_time_uuid_v7(base, 1000)
        assert isinstance(rel, str)
        assert rel != base


# ── ByteBuffer ────────────────────────────────────────────────────────────


class TestByteBuffer:
    def test_create(self):
        bb = ByteBuffer(b"\x01\x02\x03", checksum=None)
        assert bb.len() == 3
        assert len(bb) == 3
        assert not bb.is_empty()
        assert bb.checksum() is None
        assert bb.bytes == b"\x01\x02\x03"

    def test_empty(self):
        bb = ByteBuffer(b"", checksum=None)
        assert bb.is_empty()
        assert bb.len() == 0

    def test_with_checksum(self):
        bb = ByteBuffer(b"\xff", checksum=12345)
        assert bb.checksum() == 12345


# ── VideoObjectBBoxType ──────────────────────────────────────────────────


class TestVideoObjectBBoxType:
    def test_variants(self):
        assert VideoObjectBBoxType.Detection is not None
        assert VideoObjectBBoxType.TrackingInfo is not None


# ── VideoObjectBBoxTransformation ────────────────────────────────────────


class TestVideoObjectBBoxTransformation:
    def test_scale(self):
        t = VideoObjectBBoxTransformation.scale(2.0, 3.0)
        assert t is not None

    def test_shift(self):
        t = VideoObjectBBoxTransformation.shift(10.0, 20.0)
        assert t is not None


# ── BBoxMetricType ───────────────────────────────────────────────────────


class TestBBoxMetricType:
    def test_variants(self):
        assert BBoxMetricType.IoU is not None
        assert BBoxMetricType.IoSelf is not None
        assert BBoxMetricType.IoOther is not None


# ── TelemetrySpan ────────────────────────────────────────────────────────


class TestTelemetrySpan:
    def test_default(self):
        span = TelemetrySpan.default()
        assert span is not None

    def test_create(self):
        span = TelemetrySpan("test-span")
        assert span is not None

    def test_nested_span(self):
        span = TelemetrySpan.default()
        child = span.nested_span("child")
        assert child is not None

    def test_context_manager(self):
        span = TelemetrySpan.default()
        with span:
            pass  # Should not raise

    def test_propagate(self):
        span = TelemetrySpan.default()
        ctx = span.propagate()
        assert isinstance(ctx, PropagatedContext)

    def test_set_string_attribute(self):
        span = TelemetrySpan.default()
        span.set_string_attribute("key", "value")

    def test_set_bool_attribute(self):
        span = TelemetrySpan.default()
        span.set_bool_attribute("flag", True)

    def test_set_int_attribute(self):
        span = TelemetrySpan.default()
        span.set_int_attribute("count", 42)

    def test_set_float_attribute(self):
        span = TelemetrySpan.default()
        span.set_float_attribute("ratio", 3.14)

    def test_set_vec_attributes(self):
        span = TelemetrySpan.default()
        span.set_string_vec_attribute("tags", ["a", "b"])
        span.set_bool_vec_attribute("flags", [True, False])
        span.set_int_vec_attribute("ids", [1, 2, 3])
        span.set_float_vec_attribute("scores", [0.1, 0.2])

    def test_add_event(self):
        span = TelemetrySpan.default()
        span.add_event("test-event", {"k": "v"})

    def test_set_status(self):
        span = TelemetrySpan.default()
        span.set_status_ok()
        span.set_status_unset()
        span.set_status_error("err msg")

    def test_trace_id_and_span_id(self):
        span = TelemetrySpan.default()
        tid = span.trace_id()
        sid = span.span_id()
        assert isinstance(tid, str)
        assert isinstance(sid, str)

    def test_nested_span_when(self):
        span = TelemetrySpan.default()
        maybe = span.nested_span_when("maybe-child", True)
        assert isinstance(maybe, MaybeTelemetrySpan)

    def test_context_depth(self):
        depth = TelemetrySpan.context_depth()
        assert isinstance(depth, int)


# ── MaybeTelemetrySpan ──────────────────────────────────────────────────


class TestMaybeTelemetrySpan:
    def test_create_with_span(self):
        span = TelemetrySpan.default()
        maybe = MaybeTelemetrySpan(span)
        assert maybe is not None

    def test_create_with_none(self):
        maybe = MaybeTelemetrySpan(None)
        assert maybe is not None

    def test_context_manager(self):
        maybe = MaybeTelemetrySpan(None)
        with maybe:
            pass

    def test_nested_span(self):
        maybe = MaybeTelemetrySpan(TelemetrySpan.default())
        child = maybe.nested_span("child")
        assert isinstance(child, MaybeTelemetrySpan)


# ── PropagatedContext ────────────────────────────────────────────────────


class TestPropagatedContext:
    def test_as_dict(self):
        span = TelemetrySpan.default()
        ctx = span.propagate()
        d = ctx.as_dict()
        assert isinstance(d, dict)

    def test_nested_span(self):
        span = TelemetrySpan.default()
        ctx = span.propagate()
        child = ctx.nested_span("propagated-child")
        assert isinstance(child, TelemetrySpan)

    def test_nested_span_when(self):
        span = TelemetrySpan.default()
        ctx = span.propagate()
        maybe = ctx.nested_span_when("maybe", False)
        assert isinstance(maybe, MaybeTelemetrySpan)
