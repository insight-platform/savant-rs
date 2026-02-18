"""Tests for VideoFrame, VideoFrameContent, VideoFrameTranscodingMethod,
VideoFrameTransformation, VideoFrameUpdate, VideoFrameBatch."""

from __future__ import annotations

import json

import pytest

from savant_rs.primitives import (
    VideoFrame,
    VideoFrameBatch,
    VideoFrameContent,
    VideoFrameTranscodingMethod,
    VideoFrameUpdate,
)
from savant_rs.primitives.attribute import Attribute, AttributeUpdatePolicy
from savant_rs.primitives.attribute_value import AttributeValue
from savant_rs.primitives.geometry import RBBox
from savant_rs.primitives.video_frame import (
    ObjectUpdatePolicy,
    VideoFrameTransformation,
)
from savant_rs.primitives.video_object import IdCollisionResolutionPolicy, VideoObject
from savant_rs.match_query import MatchQuery, StringExpression


# ── VideoFrameContent ─────────────────────────────────────────────────────


class TestVideoFrameContent:
    def test_none(self):
        c = VideoFrameContent.none()
        assert c.is_none()
        assert not c.is_internal()
        assert not c.is_external()

    def test_internal(self):
        c = VideoFrameContent.internal(b"\x00\x01\x02")
        assert c.is_internal()
        assert c.get_data() == b"\x00\x01\x02"

    def test_external(self):
        c = VideoFrameContent.external("s3", "s3://bucket/key")
        assert c.is_external()
        assert c.get_method() == "s3"
        assert c.get_location() == "s3://bucket/key"

    def test_external_no_location(self):
        c = VideoFrameContent.external("method", None)
        assert c.is_external()
        assert c.get_location() is None


# ── VideoFrameTranscodingMethod ───────────────────────────────────────────


class TestVideoFrameTranscodingMethod:
    def test_variants(self):
        assert VideoFrameTranscodingMethod.Copy is not None
        assert VideoFrameTranscodingMethod.Encoded is not None


# ── VideoFrameTransformation ─────────────────────────────────────────────


class TestVideoFrameTransformation:
    def test_initial_size(self):
        t = VideoFrameTransformation.initial_size(1920, 1080)
        assert t.is_initial_size
        assert t.as_initial_size == (1920, 1080)
        assert not t.is_scale

    def test_scale(self):
        t = VideoFrameTransformation.scale(640, 480)
        assert t.is_scale
        assert t.as_scale == (640, 480)

    def test_padding(self):
        t = VideoFrameTransformation.padding(10, 20, 10, 20)
        assert t.is_padding
        assert t.as_padding == (10, 20, 10, 20)

    def test_resulting_size(self):
        t = VideoFrameTransformation.resulting_size(800, 600)
        assert t.is_resulting_size
        assert t.as_resulting_size == (800, 600)

    def test_wrong_type_returns_none(self):
        t = VideoFrameTransformation.initial_size(100, 100)
        assert t.as_scale is None
        assert t.as_padding is None
        assert t.as_resulting_size is None


# ── VideoFrame construction & properties ──────────────────────────────────


class TestVideoFrameConstruction:
    def test_basic(self):
        f = VideoFrame(
            source_id="cam-1",
            framerate="30/1",
            width=1920,
            height=1080,
            content=VideoFrameContent.none(),
        )
        assert f.source_id == "cam-1"
        assert f.framerate == "30/1"
        assert f.width == 1920
        assert f.height == 1080
        assert f.pts == 0
        assert f.dts is None
        assert f.duration is None
        assert f.codec is None
        assert f.keyframe is None
        assert f.transcoding_method == VideoFrameTranscodingMethod.Copy

    def test_full_params(self):
        f = VideoFrame(
            source_id="src",
            framerate="25/1",
            width=640,
            height=480,
            content=VideoFrameContent.internal(b"\x00"),
            transcoding_method=VideoFrameTranscodingMethod.Encoded,
            codec="h264",
            keyframe=True,
            time_base=(1, 90000),
            pts=12345,
            dts=12340,
            duration=3600,
        )
        assert f.codec == "h264"
        assert f.keyframe is True
        assert f.time_base == (1, 90000)
        assert f.pts == 12345
        assert f.dts == 12340
        assert f.duration == 3600


class TestVideoFrameProperties:
    @pytest.fixture()
    def frame(self):
        return VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=1280,
            height=720,
            content=VideoFrameContent.none(),
        )

    def test_uuid(self, frame):
        assert isinstance(frame.uuid, str)
        assert len(frame.uuid) > 0

    def test_creation_timestamp_ns(self, frame):
        ts = frame.creation_timestamp_ns
        assert isinstance(ts, int)
        frame.creation_timestamp_ns = 123456
        assert frame.creation_timestamp_ns == 123456

    def test_memory_handle(self, frame):
        assert isinstance(frame.memory_handle, int)

    def test_setters(self, frame):
        frame.source_id = "new-cam"
        assert frame.source_id == "new-cam"

        frame.framerate = "60/1"
        assert frame.framerate == "60/1"

        frame.width = 3840
        assert frame.width == 3840

        frame.height = 2160
        assert frame.height == 2160

        frame.pts = 999
        assert frame.pts == 999

        frame.dts = 998
        assert frame.dts == 998

        frame.duration = 33333
        assert frame.duration == 33333

        frame.time_base = (1, 90000)
        assert frame.time_base == (1, 90000)

        frame.transcoding_method = VideoFrameTranscodingMethod.Encoded
        assert frame.transcoding_method == VideoFrameTranscodingMethod.Encoded

        frame.codec = "hevc"
        assert frame.codec == "hevc"

        frame.keyframe = True
        assert frame.keyframe is True

        frame.content = VideoFrameContent.internal(b"\xff")
        assert frame.content.is_internal()

    def test_json(self, frame):
        j = frame.json
        parsed = json.loads(j)
        assert parsed is not None

    def test_json_pretty(self, frame):
        jp = frame.json_pretty
        assert len(jp) > len(frame.json)


# ── VideoFrame attributes ────────────────────────────────────────────────


class TestVideoFrameAttributes:
    @pytest.fixture()
    def frame(self):
        return VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=100,
            height=100,
            content=VideoFrameContent.none(),
        )

    def test_set_get_attribute(self, frame):
        attr = Attribute.persistent("ns", "k", [AttributeValue.string("v")])
        frame.set_attribute(attr)
        fetched = frame.get_attribute("ns", "k")
        assert fetched is not None
        assert fetched.namespace == "ns"

    def test_set_attributes_bulk(self, frame):
        attrs = [
            Attribute.persistent("ns", "a", [AttributeValue.integer(1)]),
            Attribute.persistent("ns", "b", [AttributeValue.integer(2)]),
        ]
        frame.set_attributes(attrs)
        assert frame.get_attribute("ns", "a") is not None
        assert frame.get_attribute("ns", "b") is not None

    def test_delete_attribute(self, frame):
        frame.set_attribute(
            Attribute.persistent("ns", "k", [AttributeValue.integer(1)])
        )
        deleted = frame.delete_attribute("ns", "k")
        assert deleted is not None
        assert frame.get_attribute("ns", "k") is None

    def test_clear_attributes(self, frame):
        frame.set_attribute(
            Attribute.persistent("ns", "k", [AttributeValue.integer(1)])
        )
        frame.clear_attributes()
        assert len(frame.attributes) == 0

    def test_find_attributes_with_ns(self, frame):
        frame.set_attribute(
            Attribute.persistent("ns1", "a", [AttributeValue.integer(1)])
        )
        frame.set_attribute(
            Attribute.persistent("ns2", "b", [AttributeValue.integer(2)])
        )
        found = frame.find_attributes_with_ns("ns1")
        assert len(found) == 1

    def test_set_persistent_attribute(self, frame):
        frame.set_persistent_attribute("ns", "key", False, "hint", [])
        assert frame.get_attribute("ns", "key") is not None

    def test_set_temporary_attribute(self, frame):
        frame.set_temporary_attribute("ns", "key", False, None, [])
        attr = frame.get_attribute("ns", "key")
        assert attr is not None
        assert attr.is_temporary()


# ── VideoFrame transformations ────────────────────────────────────────────


class TestVideoFrameTransformations:
    @pytest.fixture()
    def frame(self):
        return VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=100,
            height=100,
            content=VideoFrameContent.none(),
        )

    def test_add_and_get_transformations(self, frame):
        frame.add_transformation(VideoFrameTransformation.initial_size(1920, 1080))
        frame.add_transformation(VideoFrameTransformation.scale(640, 480))
        transforms = frame.transformations
        assert len(transforms) == 2
        assert transforms[0].is_initial_size
        assert transforms[1].is_scale

    def test_clear_transformations(self, frame):
        frame.add_transformation(VideoFrameTransformation.scale(640, 480))
        frame.clear_transformations()
        assert len(frame.transformations) == 0


# ── VideoFrame objects ────────────────────────────────────────────────────


class TestVideoFrameObjects:
    @pytest.fixture()
    def frame(self):
        return VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=1280,
            height=720,
            content=VideoFrameContent.none(),
        )

    def test_create_object(self, frame):
        obj = frame.create_object(
            namespace="detector",
            label="person",
            detection_box=RBBox(100.0, 200.0, 50.0, 100.0),
            confidence=0.95,
        )
        assert obj.namespace == "detector"
        assert obj.label == "person"
        assert obj.confidence == pytest.approx(0.95)

    def test_has_objects(self, frame):
        assert not frame.has_objects()
        frame.create_object("ns", "lbl", detection_box=RBBox(0, 0, 10, 10))
        assert frame.has_objects()

    def test_get_object(self, frame):
        obj = frame.create_object("ns", "lbl", detection_box=RBBox(0, 0, 10, 10))
        fetched = frame.get_object(obj.id)
        assert fetched is not None
        assert fetched.id == obj.id

    def test_get_object_not_found(self, frame):
        result = frame.get_object(99999)
        assert result is None

    def test_get_all_objects(self, frame):
        frame.create_object("ns", "a", detection_box=RBBox(0, 0, 10, 10))
        frame.create_object("ns", "b", detection_box=RBBox(0, 0, 10, 10))
        objs = frame.get_all_objects()
        assert len(objs) == 2

    def test_add_object(self, frame):
        vo = VideoObject(
            id=100,
            namespace="ns",
            label="lbl",
            detection_box=RBBox(0.0, 0.0, 10.0, 10.0),
            attributes=[],
            confidence=0.5,
            track_id=None,
            track_box=None,
        )
        borrowed = frame.add_object(vo, IdCollisionResolutionPolicy.GenerateNewId)
        assert borrowed is not None

    def test_access_objects(self, frame):
        frame.create_object("ns1", "cat", detection_box=RBBox(0, 0, 10, 10))
        frame.create_object("ns2", "dog", detection_box=RBBox(0, 0, 10, 10))
        q = MatchQuery.namespace(StringExpression.eq("ns1"))
        view = frame.access_objects(q)
        assert len(view) == 1

    def test_delete_objects(self, frame):
        frame.create_object("ns", "a", detection_box=RBBox(0, 0, 10, 10))
        frame.create_object("ns", "b", detection_box=RBBox(0, 0, 10, 10))
        q = MatchQuery.label(StringExpression.eq("a"))
        deleted = frame.delete_objects(q)
        assert len(deleted) == 1
        assert len(frame.get_all_objects()) == 1

    def test_clear_objects(self, frame):
        frame.create_object("ns", "a", detection_box=RBBox(0, 0, 10, 10))
        frame.clear_objects()
        assert not frame.has_objects()

    def test_access_objects_with_ids(self, frame):
        o1 = frame.create_object("ns", "a", detection_box=RBBox(0, 0, 10, 10))
        frame.create_object("ns", "b", detection_box=RBBox(0, 0, 10, 10))
        view = frame.access_objects_with_ids([o1.id])
        assert len(view) == 1

    def test_delete_objects_with_ids(self, frame):
        o1 = frame.create_object("ns", "a", detection_box=RBBox(0, 0, 10, 10))
        frame.create_object("ns", "b", detection_box=RBBox(0, 0, 10, 10))
        deleted = frame.delete_objects_with_ids([o1.id])
        assert len(deleted) == 1
        assert len(frame.get_all_objects()) == 1

    def test_set_parent_by_id(self, frame):
        parent = frame.create_object("ns", "parent", detection_box=RBBox(0, 0, 10, 10))
        child = frame.create_object("ns", "child", detection_box=RBBox(0, 0, 5, 5))
        frame.set_parent_by_id(child.id, parent.id)
        children = frame.get_children(parent.id)
        assert len(children) == 1

    def test_get_children(self, frame):
        parent = frame.create_object("ns", "parent", detection_box=RBBox(0, 0, 10, 10))
        frame.create_object(
            "ns",
            "child",
            parent_id=parent.id,
            detection_box=RBBox(0, 0, 5, 5),
        )
        children = frame.get_children(parent.id)
        assert len(children) == 1


# ── VideoFrame copy & protobuf ────────────────────────────────────────────


class TestVideoFrameCopyAndProtobuf:
    @pytest.fixture()
    def frame(self):
        f = VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=640,
            height=480,
            content=VideoFrameContent.none(),
        )
        f.create_object("ns", "obj", detection_box=RBBox(0, 0, 10, 10))
        return f

    def test_copy(self, frame):
        f2 = frame.copy()
        assert f2.source_id == frame.source_id
        assert len(f2.get_all_objects()) == 1
        # Copies are independent
        f2.source_id = "other"
        assert frame.source_id == "cam"

    def test_protobuf_roundtrip(self, frame):
        data = frame.to_protobuf()
        assert isinstance(data, bytes)
        f2 = VideoFrame.from_protobuf(data)
        assert f2.source_id == "cam"
        assert f2.width == 640
        assert len(f2.get_all_objects()) == 1

    def test_to_message(self, frame):
        msg = frame.to_message()
        assert msg.is_video_frame()


# ── VideoFrameUpdate ─────────────────────────────────────────────────────


class TestVideoFrameUpdate:
    def test_create(self):
        u = VideoFrameUpdate()
        assert u.frame_attribute_policy is not None
        assert u.object_attribute_policy is not None
        assert u.object_policy is not None

    def test_add_frame_attribute(self):
        u = VideoFrameUpdate()
        u.add_frame_attribute(
            Attribute.persistent("ns", "k", [AttributeValue.string("v")])
        )
        j = u.json
        assert len(j) > 0

    def test_add_object(self):
        u = VideoFrameUpdate()
        vo = VideoObject(
            id=1,
            namespace="ns",
            label="lbl",
            detection_box=RBBox(0, 0, 10, 10),
            attributes=[],
            confidence=None,
            track_id=None,
            track_box=None,
        )
        u.add_object(vo, parent_id=None)
        objs = u.get_objects()
        assert len(objs) == 1

    def test_policies(self):
        u = VideoFrameUpdate()
        u.frame_attribute_policy = (
            AttributeUpdatePolicy.ReplaceWithForeignWhenDuplicate
        )
        u.object_attribute_policy = AttributeUpdatePolicy.KeepOwnWhenDuplicate
        u.object_policy = ObjectUpdatePolicy.AddForeignObjects
        assert (
            u.frame_attribute_policy
            == AttributeUpdatePolicy.ReplaceWithForeignWhenDuplicate
        )

    def test_protobuf_roundtrip(self):
        u = VideoFrameUpdate()
        u.add_frame_attribute(
            Attribute.persistent("ns", "k", [AttributeValue.integer(42)])
        )
        data = u.to_protobuf()
        u2 = VideoFrameUpdate.from_protobuf(data)
        assert u2.json is not None

    def test_json(self):
        u = VideoFrameUpdate()
        j = u.json
        jp = u.json_pretty
        assert len(jp) >= len(j)


# ── VideoFrame.update ─────────────────────────────────────────────────────


class TestVideoFrameApplyUpdate:
    def test_update_adds_attribute(self):
        frame = VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=100,
            height=100,
            content=VideoFrameContent.none(),
        )
        u = VideoFrameUpdate()
        u.add_frame_attribute(
            Attribute.persistent("ns", "new_attr", [AttributeValue.string("hello")])
        )
        frame.update(u)
        assert frame.get_attribute("ns", "new_attr") is not None


# ── VideoFrameBatch ──────────────────────────────────────────────────────


class TestVideoFrameBatch:
    def _make_frame(self, source_id="cam"):
        return VideoFrame(
            source_id=source_id,
            framerate="30/1",
            width=100,
            height=100,
            content=VideoFrameContent.none(),
        )

    def test_from_frames(self):
        frames = [self._make_frame("cam1"), self._make_frame("cam2")]
        batch = VideoFrameBatch.from_frames(frames)
        assert len(batch.frame_ids) == 2

    def test_add_frame(self):
        batch = VideoFrameBatch.from_frames([])
        f = self._make_frame()
        fid = batch.add_frame(f)
        assert isinstance(fid, int)

    def test_get_frame(self):
        f = self._make_frame("test")
        batch = VideoFrameBatch.from_frames([f])
        ids = batch.frame_ids
        assert len(ids) == 1
        retrieved = batch.get_frame(ids[0])
        assert retrieved is not None

    def test_frames_dict(self):
        frames = [self._make_frame("a"), self._make_frame("b")]
        batch = VideoFrameBatch.from_frames(frames)
        d = batch.frames
        assert isinstance(d, dict)
        assert len(d) == 2

    def test_add_get_del(self):
        batch = VideoFrameBatch.from_frames([])
        f = self._make_frame()
        batch.add(42, f)
        retrieved = batch.get(42)
        assert retrieved is not None
        deleted = batch.del_(42)
        assert deleted is not None
        assert batch.get(42) is None

    def test_to_message(self):
        batch = VideoFrameBatch.from_frames([self._make_frame()])
        msg = batch.to_message()
        assert msg.is_video_frame_batch()

    def test_access_objects(self):
        f = self._make_frame()
        f.create_object("ns", "lbl", detection_box=RBBox(0, 0, 10, 10))
        batch = VideoFrameBatch.from_frames([f])
        q = MatchQuery.idle()
        result = batch.access_objects(q)
        assert isinstance(result, dict)

    def test_delete_objects(self):
        f = self._make_frame()
        f.create_object("ns", "lbl", detection_box=RBBox(0, 0, 10, 10))
        batch = VideoFrameBatch.from_frames([f])
        q = MatchQuery.idle()
        result = batch.delete_objects(q)
        assert isinstance(result, dict)
