"""Tests for misc-track types and VideoFrame misc-track API."""

from __future__ import annotations

import pytest

from savant_rs.primitives import (
    IdCollisionResolutionPolicy,
    MiscTrackCategory,
    MiscTrackData,
    MiscTrackFrame,
    TrackState,
    TrackUpdate,
    VideoFrame,
    VideoFrameContent,
    VideoFrameTranscodingMethod,
    VideoObject,
)
from savant_rs.primitives.geometry import RBBox


def _make_frame() -> VideoFrame:
    return VideoFrame(
        source_id="test-cam",
        fps=(30, 1),
        width=1920,
        height=1080,
        content=VideoFrameContent.none(),
        transcoding_method=VideoFrameTranscodingMethod.Copy,
        codec=None,
        keyframe=None,
        time_base=(1, 1_000_000),
        pts=0,
        dts=None,
        duration=None,
    )


def _make_misc_frame(frame_num: int = 0) -> MiscTrackFrame:
    return MiscTrackFrame(
        frame_num=frame_num,
        bbox_left=10.0,
        bbox_top=20.0,
        bbox_width=30.0,
        bbox_height=40.0,
        confidence=0.95,
        age=3,
        state=TrackState.ACTIVE,
        visibility=0.8,
    )


def _make_misc_track(category: MiscTrackCategory) -> MiscTrackData:
    return MiscTrackData(
        object_id=42,
        class_id=1,
        source_id="test-cam",
        category=category,
        label="person",
        frames=[_make_misc_frame(0), _make_misc_frame(1)],
    )


# ── TrackState ─────────────────────────────────────────────────────────


class TestTrackState:
    def test_values_exist(self):
        assert TrackState.EMPTY is not None
        assert TrackState.ACTIVE is not None
        assert TrackState.INACTIVE is not None
        assert TrackState.TENTATIVE is not None
        assert TrackState.PROJECTED is not None

    def test_equality(self):
        assert TrackState.ACTIVE == TrackState.ACTIVE
        assert TrackState.ACTIVE != TrackState.EMPTY

    def test_int_conversion(self):
        assert int(TrackState.EMPTY) == 0
        assert int(TrackState.ACTIVE) == 1
        assert int(TrackState.INACTIVE) == 2
        assert int(TrackState.TENTATIVE) == 3
        assert int(TrackState.PROJECTED) == 4


# ── MiscTrackCategory ─────────────────────────────────────────────────


class TestMiscTrackCategory:
    def test_values_exist(self):
        assert MiscTrackCategory.SHADOW is not None
        assert MiscTrackCategory.TERMINATED is not None
        assert MiscTrackCategory.PAST_FRAME is not None

    def test_equality(self):
        assert MiscTrackCategory.SHADOW == MiscTrackCategory.SHADOW
        assert MiscTrackCategory.SHADOW != MiscTrackCategory.TERMINATED


# ── MiscTrackFrame ────────────────────────────────────────────────────


class TestMiscTrackFrame:
    def test_constructor_and_properties(self):
        f = _make_misc_frame(42)
        assert f.frame_num == 42
        assert f.bbox_left == pytest.approx(10.0)
        assert f.bbox_top == pytest.approx(20.0)
        assert f.bbox_width == pytest.approx(30.0)
        assert f.bbox_height == pytest.approx(40.0)
        assert f.confidence == pytest.approx(0.95)
        assert f.age == 3
        assert f.state == TrackState.ACTIVE
        assert f.visibility == pytest.approx(0.8)

    def test_repr(self):
        f = _make_misc_frame()
        r = repr(f)
        assert "MiscTrackFrame" in r
        assert "frame_num=0" in r


# ── MiscTrackData ─────────────────────────────────────────────────────


class TestMiscTrackData:
    def test_constructor_and_properties(self):
        t = _make_misc_track(MiscTrackCategory.SHADOW)
        assert t.object_id == 42
        assert t.class_id == 1
        assert t.label == "person"
        assert t.source_id == "test-cam"
        assert t.category == MiscTrackCategory.SHADOW
        assert len(t.frames) == 2

    def test_no_frames(self):
        t = MiscTrackData(
            object_id=1,
            class_id=0,
            source_id="s",
            category=MiscTrackCategory.TERMINATED,
        )
        assert len(t.frames) == 0
        assert t.label is None

    def test_repr(self):
        t = _make_misc_track(MiscTrackCategory.PAST_FRAME)
        r = repr(t)
        assert "MiscTrackData" in r
        assert "object_id=42" in r


# ── TrackUpdate ────────────────────────────────────────────────────────


class TestTrackUpdate:
    def test_constructor_and_properties(self):
        box_ = RBBox(100.0, 200.0, 50.0, 60.0, None)
        u = TrackUpdate(object_id=7, track_id=999, track_box=box_)
        assert u.object_id == 7
        assert u.track_id == 999
        assert u.track_box.xc == pytest.approx(100.0)

    def test_repr(self):
        u = TrackUpdate(0, 1, RBBox(0.0, 0.0, 1.0, 1.0, None))
        assert "TrackUpdate" in repr(u)


# ── VideoFrame misc-track API ─────────────────────────────────────────


class TestVideoFrameMiscTracks:
    def test_empty_by_default(self):
        f = _make_frame()
        assert f.get_misc_tracks() == []

    def test_add_single(self):
        f = _make_frame()
        f.add_misc_track(_make_misc_track(MiscTrackCategory.SHADOW))
        assert len(f.get_misc_tracks()) == 1

    def test_add_multiple(self):
        f = _make_frame()
        f.add_misc_tracks(
            [
                _make_misc_track(MiscTrackCategory.SHADOW),
                _make_misc_track(MiscTrackCategory.TERMINATED),
                _make_misc_track(MiscTrackCategory.PAST_FRAME),
            ]
        )
        assert len(f.get_misc_tracks()) == 3

    def test_filter_by_category(self):
        f = _make_frame()
        f.add_misc_tracks(
            [
                _make_misc_track(MiscTrackCategory.SHADOW),
                _make_misc_track(MiscTrackCategory.SHADOW),
                _make_misc_track(MiscTrackCategory.TERMINATED),
            ]
        )
        assert len(f.get_misc_tracks_by_category(MiscTrackCategory.SHADOW)) == 2
        assert len(f.get_misc_tracks_by_category(MiscTrackCategory.TERMINATED)) == 1
        assert len(f.get_misc_tracks_by_category(MiscTrackCategory.PAST_FRAME)) == 0

    def test_clear_all(self):
        f = _make_frame()
        f.add_misc_track(_make_misc_track(MiscTrackCategory.SHADOW))
        f.clear_misc_tracks()
        assert f.get_misc_tracks() == []

    def test_clear_by_category(self):
        f = _make_frame()
        f.add_misc_tracks(
            [
                _make_misc_track(MiscTrackCategory.SHADOW),
                _make_misc_track(MiscTrackCategory.TERMINATED),
            ]
        )
        f.clear_misc_tracks_by_category(MiscTrackCategory.SHADOW)
        tracks = f.get_misc_tracks()
        assert len(tracks) == 1
        assert tracks[0].category == MiscTrackCategory.TERMINATED

    def test_data_integrity(self):
        f = _make_frame()
        f.add_misc_track(_make_misc_track(MiscTrackCategory.PAST_FRAME))
        t = f.get_misc_tracks()[0]
        assert t.object_id == 42
        assert t.class_id == 1
        assert t.label == "person"
        assert t.category == MiscTrackCategory.PAST_FRAME
        assert len(t.frames) == 2
        assert t.frames[0].state == TrackState.ACTIVE


# ── apply_tracking_info ───────────────────────────────────────────────


class TestApplyTrackingInfo:
    def test_basic(self):
        f = _make_frame()
        obj = f.create_object(
            "detector", "car", detection_box=RBBox(50.0, 50.0, 20.0, 20.0, None)
        )
        obj_id = obj.id

        box_ = RBBox(100.0, 200.0, 50.0, 60.0, None)
        unmatched = f.apply_tracking_info(
            track_updates=[TrackUpdate(obj_id, 777, box_)],
            misc_tracks=[_make_misc_track(MiscTrackCategory.SHADOW)],
        )
        assert unmatched == []

        updated = f.get_object(obj_id)
        assert updated.track_id == 777
        assert updated.track_box is not None
        assert len(f.get_misc_tracks()) == 1

    def test_replaces_previous_misc_tracks(self):
        f = _make_frame()
        f.add_misc_track(_make_misc_track(MiscTrackCategory.PAST_FRAME))
        assert len(f.get_misc_tracks()) == 1

        unmatched = f.apply_tracking_info(
            track_updates=[],
            misc_tracks=[
                _make_misc_track(MiscTrackCategory.SHADOW),
                _make_misc_track(MiscTrackCategory.TERMINATED),
            ],
        )
        assert unmatched == []
        assert len(f.get_misc_tracks()) == 2
        assert f.get_misc_tracks_by_category(MiscTrackCategory.PAST_FRAME) == []

    def test_missing_object_returns_unmatched(self):
        f = _make_frame()
        update = TrackUpdate(999, 1, RBBox(0.0, 0.0, 1.0, 1.0, None))
        unmatched = f.apply_tracking_info(
            track_updates=[update],
            misc_tracks=[],
        )
        # Missing update returned verbatim to the caller, not raised.
        assert len(unmatched) == 1
        assert unmatched[0].object_id == 999
        assert unmatched[0].track_id == 1
        # Display round-trip (Rust std::fmt::Display backs __str__ / __repr__).
        assert "object_id=999" in str(unmatched[0])
        assert "track_id=1" in repr(unmatched[0])


# ── Protobuf round-trip ──────────────────────────────────────────────


class TestMiscTracksProtobuf:
    def test_round_trip_empty(self):
        f = _make_frame()
        pb = f.to_protobuf()
        f2 = VideoFrame.from_protobuf(pb)
        assert f2.get_misc_tracks() == []

    def test_round_trip_with_tracks(self):
        f = _make_frame()
        f.add_misc_tracks(
            [
                _make_misc_track(MiscTrackCategory.SHADOW),
                _make_misc_track(MiscTrackCategory.TERMINATED),
            ]
        )
        pb = f.to_protobuf()
        f2 = VideoFrame.from_protobuf(pb)
        tracks = f2.get_misc_tracks()
        assert len(tracks) == 2
        assert tracks[0].object_id == 42
        assert tracks[0].category == MiscTrackCategory.SHADOW
        assert len(tracks[0].frames) == 2
        assert tracks[0].frames[0].state == TrackState.ACTIVE
        assert tracks[1].category == MiscTrackCategory.TERMINATED

    def test_round_trip_preserves_all_fields(self):
        f = _make_frame()
        f.add_misc_track(
            MiscTrackData(
                object_id=99,
                class_id=3,
                source_id="cam-2",
                category=MiscTrackCategory.PAST_FRAME,
                label="bicycle",
                frames=[
                    MiscTrackFrame(
                        frame_num=7,
                        bbox_left=1.5,
                        bbox_top=2.5,
                        bbox_width=3.5,
                        bbox_height=4.5,
                        confidence=0.77,
                        age=10,
                        state=TrackState.TENTATIVE,
                        visibility=0.6,
                    )
                ],
            )
        )
        pb = f.to_protobuf()
        f2 = VideoFrame.from_protobuf(pb)
        t = f2.get_misc_tracks()[0]
        assert t.object_id == 99
        assert t.class_id == 3
        assert t.label == "bicycle"
        assert t.source_id == "cam-2"
        assert t.category == MiscTrackCategory.PAST_FRAME
        fr = t.frames[0]
        assert fr.frame_num == 7
        assert fr.bbox_left == pytest.approx(1.5)
        assert fr.bbox_top == pytest.approx(2.5)
        assert fr.bbox_width == pytest.approx(3.5)
        assert fr.bbox_height == pytest.approx(4.5)
        assert fr.confidence == pytest.approx(0.77)
        assert fr.age == 10
        assert fr.state == TrackState.TENTATIVE
        assert fr.visibility == pytest.approx(0.6)


# ── JSON output ───────────────────────────────────────────────────────


class TestMiscTracksJson:
    def test_json_includes_misc_tracks(self):
        import json

        f = _make_frame()
        f.add_misc_track(_make_misc_track(MiscTrackCategory.TERMINATED))
        j = json.loads(f.json)
        assert "misc_tracks" in j
        assert len(j["misc_tracks"]) == 1
        assert j["misc_tracks"][0]["category"] == "Terminated"
        assert j["misc_tracks"][0]["object_id"] == 42
