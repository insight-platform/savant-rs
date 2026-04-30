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
    the optional ``wall_clock_ms`` argument, error paths, and that
    minted ids are real RFC 9562 UUIDv8 values.
    """

    SOURCE = "pytest-video-id-source"

    def setup_method(self):
        # Each test starts with a clean per-source state.
        forget_video_id(self.SOURCE)

    def teardown_method(self):
        forget_video_id(self.SOURCE)

    def test_mint_returns_uuid_string(self):
        s = mint_video_id(self.SOURCE, 0, True, 1_700_000_000_000)
        assert isinstance(s, str)
        # Must parse as a real UUID and be UUIDv8.
        import uuid

        u = uuid.UUID(s)
        assert u.version == 8
        # RFC 4122 variant maps to uuid.RFC_4122 in the stdlib.
        assert u.variant == uuid.RFC_4122

    def test_within_gop_sorts_by_pts(self):
        ts = 1_700_000_000_000
        i_frame = mint_video_id(self.SOURCE, 0, True, ts)
        p_frame = mint_video_id(self.SOURCE, 100, False, ts + 1)
        b_frame = mint_video_id(self.SOURCE, 50, False, ts + 2)
        # u128 ordering = lexicographic UUID hex ordering (no hyphens).
        as_int = lambda s: int(s.replace("-", ""), 16)
        assert as_int(i_frame) < as_int(b_frame) < as_int(p_frame)

    def test_cross_gop_sorts_by_keyframe_arrival(self):
        early = mint_video_id(self.SOURCE, 0, True, 1_700_000_000_000)
        _ = mint_video_id(self.SOURCE, 100, False, 1_700_000_000_001)
        late = mint_video_id(self.SOURCE, 0, True, 1_700_000_000_500)
        as_int = lambda s: int(s.replace("-", ""), 16)
        assert as_int(early) < as_int(late)

    def test_pts_reset_bumps_epoch(self):
        # PTS regresses past the threshold (default 1_000_000) -> epoch++.
        pre = mint_video_id(self.SOURCE, 10_000_000, True, 1_700_000_000_000)
        post = mint_video_id(self.SOURCE, 0, True, 1_700_000_000_500)
        as_int = lambda s: int(s.replace("-", ""), 16)
        # Epoch byte sits at bits [47:40]. Extract it from the u128.
        epoch = lambda s: (as_int(s) >> 40) & 0xFF
        assert epoch(pre) == 0
        assert epoch(post) == 1

    def test_default_wall_clock_uses_system_time(self):
        # Omitting wall_clock_ms should still produce a valid id.
        s = mint_video_id(self.SOURCE, 0, True)
        import uuid

        assert uuid.UUID(s).version == 8

    def test_forget_resets_state(self):
        a = mint_video_id(self.SOURCE, 0, True, 1_700_000_000_000)
        forget_video_id(self.SOURCE)
        # Same wall_clock_ms after forget — strict-monotonic only kicks
        # in when state survives, so the same ts_ms is reused.
        b = mint_video_id(self.SOURCE, 0, True, 1_700_000_000_000)
        assert a == b

    def test_relative_time_video_id_round_trip(self):
        base = mint_video_id(self.SOURCE, 50, True, 1_700_000_000_000)
        later = relative_time_video_id(base, 250)
        assert isinstance(later, str)
        assert later != base
        # Deterministic: same input -> same output.
        again = relative_time_video_id(base, 250)
        assert again == later
        # Earlier than `base` should produce a different id sorting before.
        earlier = relative_time_video_id(base, -250)
        as_int = lambda s: int(s.replace("-", ""), 16)
        assert as_int(earlier) < as_int(base) < as_int(later)

    def test_relative_time_video_id_invalid_string(self):
        with pytest.raises(ValueError):
            relative_time_video_id("not-a-uuid", 0)

    def test_relative_time_video_id_underflow(self):
        early = video_id_lower_bound(self.SOURCE, 100)
        with pytest.raises(ValueError, match="underflow"):
            relative_time_video_id(early, -10_000)

    def test_bounds_bracket_minted_ids(self):
        ts = 1_700_000_000_000
        lo = video_id_lower_bound(self.SOURCE, ts)
        hi = video_id_upper_bound(self.SOURCE, ts)
        mid = mint_video_id(self.SOURCE, 50, True, ts)
        as_int = lambda s: int(s.replace("-", ""), 16)
        assert as_int(lo) <= as_int(mid) <= as_int(hi)

    def test_bounds_are_uuidv8(self):
        import uuid

        lo = video_id_lower_bound(self.SOURCE, 1_700_000_000_000)
        hi = video_id_upper_bound(self.SOURCE, 1_700_000_000_000)
        assert uuid.UUID(lo).version == 8
        assert uuid.UUID(hi).version == 8

    def test_distinct_sources_have_distinct_prefixes(self):
        forget_video_id("source-a")
        forget_video_id("source-b")
        a = mint_video_id("source-a", 0, True, 1_700_000_000_000)
        b = mint_video_id("source-b", 0, True, 1_700_000_000_000)
        # Top 32 bits = crc32(source_id). They must differ for different sources.
        as_int = lambda s: int(s.replace("-", ""), 16)
        assert (as_int(a) >> 96) != (as_int(b) >> 96)
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
