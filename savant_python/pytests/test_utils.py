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
    forget_video_id,
    gen_empty_frame,
    gen_frame,
    incremental_uuid_v7,
    mint_video_id,
    relative_time_uuid_v7,
    relative_time_video_id,
    round_2_digits,
    video_id_lower_bound,
    video_id_upper_bound,
)
from savant_rs.primitives import VideoFrame


# ── eval_expr ─────────────────────────────────────────────────────────────


class TestEvalExpr:
    def test_math(self):
        result, _cached = eval_expr("2 + 3", ttl=100)
        assert result == 5

    def test_float_math(self):
        result, _cached = eval_expr("2.5 * 4.0", ttl=100)
        assert result == pytest.approx(10.0)

    def test_string(self):
        result, _cached = eval_expr('"hello"', ttl=100)
        assert result == "hello"

    def test_boolean(self):
        result, _cached = eval_expr("true", ttl=100)
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

    def test_relative_time_uuid_v7_invalid_string(self):
        with pytest.raises(ValueError, match="invalid"):
            relative_time_uuid_v7("not-a-uuid", 0)

    def test_relative_time_uuid_v7_non_timestamp_uuid(self):
        v4 = "550e8400-e29b-41d4-a716-446655440000"
        with pytest.raises(ValueError, match="no embedded timestamp"):
            relative_time_uuid_v7(v4, 0)


# ── VideoId functions (UUIDv8 composite) ──────────────────────────────────


class TestVideoIdFunctions:
    """Smoke tests for the video_id subsystem exposed to Python.

    The Rust module is exhaustively tested already; these tests cover
    the Python boundary: type coercion, the singleton state machine,
    the optional ``wall_clock_ns`` argument, error paths, and that
    minted ids are real RFC 9562 UUIDv8 values.

    Layout reminder (high → low bits):
      [127:80]  ts_ns_hi : 48
      [79:76]   version  :  4 (= 8)
      [75:64]   ts_ns_mid: 12
      [63:62]   variant  :  2 (= 0b10)
      [61:58]   ts_ns_lo :  4
      [57:50]   epoch    :  8
      [49:0]    pts      : 50
    """

    SOURCE = "pytest-video-id-source"
    # 2024-01-01T00:00:00Z in ns -- comfortably inside u64 range.
    BASE_NS = 1_700_000_000_000_000_000

    def setup_method(self):
        forget_video_id(self.SOURCE)

    def teardown_method(self):
        forget_video_id(self.SOURCE)

    @staticmethod
    def _as_int(s: str) -> int:
        return int(s.replace("-", ""), 16)

    @classmethod
    def _epoch(cls, s: str) -> int:
        # `epoch` lives at bits [57:50] of the u128.
        return (cls._as_int(s) >> 50) & 0xFF

    def test_mint_returns_uuid_string(self):
        s = mint_video_id(self.SOURCE, 0, True, self.BASE_NS)
        assert isinstance(s, str)
        import uuid

        u = uuid.UUID(s)
        assert u.version == 8
        assert u.variant == uuid.RFC_4122

    def test_within_gop_sorts_by_pts(self):
        i_frame = mint_video_id(self.SOURCE, 0, True, self.BASE_NS)
        p_frame = mint_video_id(self.SOURCE, 100, False, self.BASE_NS + 1)
        b_frame = mint_video_id(self.SOURCE, 50, False, self.BASE_NS + 2)
        a = self._as_int
        assert a(i_frame) < a(b_frame) < a(p_frame)

    def test_cross_gop_sorts_by_keyframe_arrival(self):
        early = mint_video_id(self.SOURCE, 0, True, self.BASE_NS)
        _ = mint_video_id(self.SOURCE, 100, False, self.BASE_NS + 1)
        late = mint_video_id(self.SOURCE, 0, True, self.BASE_NS + 500_000_000)
        assert self._as_int(early) < self._as_int(late)

    def test_pts_reset_bumps_epoch(self):
        pre = mint_video_id(self.SOURCE, 10_000_000, True, self.BASE_NS)
        post = mint_video_id(self.SOURCE, 0, True, self.BASE_NS + 500_000_000)
        assert self._epoch(pre) == 0
        assert self._epoch(post) == 1

    def test_default_wall_clock_uses_system_time(self):
        s = mint_video_id(self.SOURCE, 0, True)
        import uuid

        assert uuid.UUID(s).version == 8

    def test_forget_resets_state(self):
        a = mint_video_id(self.SOURCE, 0, True, self.BASE_NS)
        forget_video_id(self.SOURCE)
        # Same wall_clock_ns after forget -- strict-monotonic only
        # kicks in while state survives, so the same ts_ns is reused.
        b = mint_video_id(self.SOURCE, 0, True, self.BASE_NS)
        assert a == b

    def test_relative_time_video_id_round_trip(self):
        base = mint_video_id(self.SOURCE, 50, True, self.BASE_NS)
        later = relative_time_video_id(base, 250)
        assert isinstance(later, str)
        assert later != base
        again = relative_time_video_id(base, 250)
        assert again == later
        earlier = relative_time_video_id(base, -250)
        a = self._as_int
        assert a(earlier) < a(base) < a(later)

    def test_relative_time_video_id_invalid_string(self):
        with pytest.raises(ValueError):
            relative_time_video_id("not-a-uuid", 0)

    def test_relative_time_video_id_underflow(self):
        early = video_id_lower_bound(100_000)  # 100 µs
        with pytest.raises(ValueError, match="underflow"):
            relative_time_video_id(early, -10_000)  # -10 s

    def test_bounds_bracket_minted_ids(self):
        lo = video_id_lower_bound(self.BASE_NS)
        hi = video_id_upper_bound(self.BASE_NS)
        mid = mint_video_id(self.SOURCE, 50, True, self.BASE_NS)
        a = self._as_int
        assert a(lo) <= a(mid) <= a(hi)

    def test_bounds_are_uuidv8(self):
        import uuid

        lo = video_id_lower_bound(self.BASE_NS)
        hi = video_id_upper_bound(self.BASE_NS)
        assert uuid.UUID(lo).version == 8
        assert uuid.UUID(hi).version == 8

    def test_distinct_sources_with_same_inputs_collide(self):
        # Source identity is not encoded in the id; identical
        # (ts_ns, epoch, pts) collide by design.
        forget_video_id("source-a")
        forget_video_id("source-b")
        a = mint_video_id("source-a", 42, True, self.BASE_NS)
        b = mint_video_id("source-b", 42, True, self.BASE_NS)
        assert a == b
        forget_video_id("source-a")
        forget_video_id("source-b")


# ── ByteBuffer ────────────────────────────────────────────────────────────


class TestByteBuffer:
    def test_create(self):
        bb = ByteBuffer(b"\x01\x02\x03", checksum=None)
        assert bb.len() == 3
        assert not bb.is_empty()
        assert bb.checksum is None
        assert bb.bytes == b"\x01\x02\x03"

    def test_empty(self):
        bb = ByteBuffer(b"", checksum=None)
        assert bb.is_empty()
        assert bb.len() == 0

    def test_with_checksum(self):
        bb = ByteBuffer(b"\xff", checksum=12345)
        assert bb.checksum == 12345


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
