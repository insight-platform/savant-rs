"""Tests for VideoFrame, VideoFrameContent, VideoFrameTranscodingMethod,
VideoFrameTransformation, VideoFrameUpdate, VideoFrameBatch."""

from __future__ import annotations

import json

import pytest

from savant_rs.primitives import (
    Attribute,
    AttributeUpdatePolicy,
    AttributeValue,
    IdCollisionResolutionPolicy,
    ObjectUpdatePolicy,
    VideoFrame,
    VideoFrameBatch,
    VideoFrameCodec,
    VideoFrameContent,
    VideoFrameTranscodingMethod,
    VideoFrameTransformation,
    VideoFrameUpdate,
    VideoObject,
)
from savant_rs.primitives.geometry import RBBox
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
        assert not t.is_letter_box

    def test_letter_box(self):
        t = VideoFrameTransformation.letter_box(660, 500, 10, 10, 10, 10)
        assert t.is_letter_box
        assert t.as_letter_box == (660, 500, 10, 10, 10, 10)

    def test_letter_box_no_padding(self):
        t = VideoFrameTransformation.letter_box(640, 480, 0, 0, 0, 0)
        assert t.is_letter_box
        assert t.as_letter_box == (640, 480, 0, 0, 0, 0)

    def test_padding(self):
        t = VideoFrameTransformation.padding(10, 20, 10, 20)
        assert t.is_padding
        assert t.as_padding == (10, 20, 10, 20)

    def test_crop(self):
        t = VideoFrameTransformation.crop(50, 25, 50, 25)
        assert t.is_crop
        assert t.as_crop == (50, 25, 50, 25)

    def test_wrong_type_returns_none(self):
        t = VideoFrameTransformation.initial_size(100, 100)
        assert t.as_letter_box is None
        assert t.as_padding is None
        assert t.as_crop is None

    def test_letter_box_validation(self):
        with pytest.raises(ValueError):
            VideoFrameTransformation.letter_box(0, 100, 0, 0, 0, 0)
        with pytest.raises(ValueError):
            VideoFrameTransformation.letter_box(100, 100, -1, 0, 0, 0)
        with pytest.raises(ValueError):
            VideoFrameTransformation.letter_box(100, 100, 50, 0, 50, 0)

    def test_crop_validation(self):
        with pytest.raises(ValueError):
            VideoFrameTransformation.crop(-1, 0, 0, 0)

    def test_repr(self):
        t = VideoFrameTransformation.initial_size(1920, 1080)
        assert "InitialSize" in repr(t)
        t2 = VideoFrameTransformation.letter_box(640, 480, 0, 0, 0, 0)
        assert "LetterBox" in repr(t2)
        t3 = VideoFrameTransformation.crop(10, 20, 10, 20)
        assert "Crop" in repr(t3)


# ── VideoFrame construction & properties ──────────────────────────────────


class TestVideoFrameConstruction:
    def test_basic(self):
        f = VideoFrame(
            source_id="cam-1",
            fps=(30, 1),
            width=1920,
            height=1080,
            content=VideoFrameContent.none(),
        )
        assert f.source_id == "cam-1"
        assert f.fps == (30, 1)
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
            fps=(25, 1),
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
        assert f.codec == VideoFrameCodec.H264
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
            fps=(30, 1),
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

        frame.fps = (60, 1)
        assert frame.fps == (60, 1)

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

        frame.codec = VideoFrameCodec.Hevc
        assert frame.codec == VideoFrameCodec.Hevc

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
            fps=(30, 1),
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
            fps=(30, 1),
            width=100,
            height=100,
            content=VideoFrameContent.none(),
        )

    def test_add_and_get_transformations(self, frame):
        # InitialSize(100,100) already added by constructor
        frame.add_transformation(
            VideoFrameTransformation.letter_box(660, 500, 10, 10, 10, 10)
        )
        transforms = frame.transformations
        assert len(transforms) == 2
        assert transforms[0].is_initial_size
        assert transforms[1].is_letter_box

    def test_clear_transformations(self, frame):
        frame.add_transformation(
            VideoFrameTransformation.letter_box(640, 480, 0, 0, 0, 0)
        )
        frame.clear_transformations()
        assert len(frame.transformations) == 0

    def test_add_crop_transformation(self, frame):
        frame.add_transformation(VideoFrameTransformation.crop(100, 50, 100, 50))
        transforms = frame.transformations
        assert len(transforms) == 2
        assert transforms[1].is_crop
        assert transforms[1].as_crop == (100, 50, 100, 50)

    def test_complex_chain(self, frame):
        frame.add_transformation(VideoFrameTransformation.crop(160, 40, 160, 40))
        frame.add_transformation(
            VideoFrameTransformation.letter_box(800, 500, 0, 0, 0, 0)
        )
        frame.add_transformation(VideoFrameTransformation.padding(5, 5, 5, 5))
        transforms = frame.transformations
        assert len(transforms) == 4
        assert transforms[0].is_initial_size
        assert transforms[1].is_crop
        assert transforms[2].is_letter_box
        assert transforms[3].is_padding


# ── VideoFrame transform_backward / transform_forward ─────────────────


class TestVideoFrameTransformToInitial:
    def test_letterbox_to_initial(self):
        f = VideoFrame(
            source_id="cam",
            fps=(30, 1),
            width=1920,
            height=1080,
            content=VideoFrameContent.none(),
        )
        f.add_transformation(
            VideoFrameTransformation.letter_box(660, 500, 10, 10, 10, 10)
        )
        obj = f.create_object(
            "det", "car", detection_box=RBBox(100.0, 100.0, 50.0, 50.0)
        )
        oid = obj.id

        f.transform_backward()

        assert f.width == 1920
        assert f.height == 1080
        transforms = f.transformations
        assert len(transforms) == 1
        assert transforms[0].is_initial_size
        assert transforms[0].as_initial_size == (1920, 1080)

        obj = f.get_object(oid)
        det = obj.detection_box
        sx = 640.0 / 1920.0
        sy = 480.0 / 1080.0
        expected_x = (100.0 - 10.0) / sx
        expected_y = (100.0 - 10.0) / sy
        assert det.xc == pytest.approx(expected_x, abs=0.5)
        assert det.yc == pytest.approx(expected_y, abs=0.5)

    def test_identity_is_noop(self):
        f = VideoFrame(
            source_id="cam",
            fps=(30, 1),
            width=800,
            height=600,
            content=VideoFrameContent.none(),
        )
        obj = f.create_object(
            "det", "car", detection_box=RBBox(400.0, 300.0, 100.0, 80.0)
        )
        oid = obj.id

        f.transform_backward()

        obj = f.get_object(oid)
        det = obj.detection_box
        assert det.xc == pytest.approx(400.0, abs=0.01)
        assert det.yc == pytest.approx(300.0, abs=0.01)
        assert f.width == 800
        assert f.height == 600

    def test_no_initial_size_raises(self):
        f = VideoFrame(
            source_id="cam",
            fps=(30, 1),
            width=100,
            height=100,
            content=VideoFrameContent.none(),
        )
        f.clear_transformations()
        f.add_transformation(VideoFrameTransformation.padding(10, 10, 10, 10))
        with pytest.raises(RuntimeError, match="InitialSize"):
            f.transform_backward()

    def test_crop_then_letterbox_to_initial(self):
        f = VideoFrame(
            source_id="cam",
            fps=(30, 1),
            width=1920,
            height=1080,
            content=VideoFrameContent.none(),
        )
        f.add_transformation(VideoFrameTransformation.crop(160, 40, 160, 40))
        f.add_transformation(VideoFrameTransformation.letter_box(800, 500, 0, 0, 0, 0))
        obj = f.create_object(
            "det", "car", detection_box=RBBox(400.0, 250.0, 100.0, 100.0)
        )
        oid = obj.id

        f.transform_backward()

        assert f.width == 1920
        assert f.height == 1080
        obj = f.get_object(oid)
        det = obj.detection_box
        expected_x = 400.0 * 2.0 + 160.0
        expected_y = 250.0 * 2.0 + 40.0
        assert det.xc == pytest.approx(expected_x, abs=1.0)
        assert det.yc == pytest.approx(expected_y, abs=1.0)


class TestVideoFrameTransformToTarget:
    def test_letterbox_to_target(self):
        """Object in 1920x1080 (InitialSize) → forward through LetterBox → 660x500."""
        f = VideoFrame(
            source_id="cam",
            fps=(30, 1),
            width=1920,
            height=1080,
            content=VideoFrameContent.none(),
        )
        f.add_transformation(
            VideoFrameTransformation.letter_box(660, 500, 10, 10, 10, 10)
        )
        obj = f.create_object(
            "det", "car", detection_box=RBBox(330.0, 250.0, 100.0, 100.0)
        )
        oid = obj.id

        f.transform_forward()

        assert f.width == 660
        assert f.height == 500
        transforms = f.transformations
        assert len(transforms) == 1
        assert transforms[0].as_initial_size == (660, 500)

        obj = f.get_object(oid)
        det = obj.detection_box
        sx = 640.0 / 1920.0
        sy = 480.0 / 1080.0
        expected_x = 330.0 * sx + 10.0
        expected_y = 250.0 * sy + 10.0
        assert det.xc == pytest.approx(expected_x, abs=0.5)
        assert det.yc == pytest.approx(expected_y, abs=0.5)

    def test_no_initial_size_raises(self):
        f = VideoFrame(
            source_id="cam",
            fps=(30, 1),
            width=100,
            height=100,
            content=VideoFrameContent.none(),
        )
        f.clear_transformations()
        f.add_transformation(VideoFrameTransformation.padding(10, 10, 10, 10))
        with pytest.raises(RuntimeError, match="InitialSize"):
            f.transform_forward()

    def test_crop_then_letterbox_to_target(self):
        """Object in 1920x1080 → Crop(160,40,160,40) → LetterBox(800,500,0,0,0,0) → 800x500."""
        f = VideoFrame(
            source_id="cam",
            fps=(30, 1),
            width=1920,
            height=1080,
            content=VideoFrameContent.none(),
        )
        f.add_transformation(VideoFrameTransformation.crop(160, 40, 160, 40))
        f.add_transformation(VideoFrameTransformation.letter_box(800, 500, 0, 0, 0, 0))
        obj = f.create_object(
            "det", "car", detection_box=RBBox(400.0, 250.0, 100.0, 100.0)
        )
        oid = obj.id

        f.transform_forward()

        assert f.width == 800
        assert f.height == 500
        obj = f.get_object(oid)
        det = obj.detection_box
        # forward: sx=0.5, sy=0.5, tx=-80, ty=-20
        expected_x = 400.0 * 0.5 - 80.0
        expected_y = 250.0 * 0.5 - 20.0
        assert det.xc == pytest.approx(expected_x, abs=1.0)
        assert det.yc == pytest.approx(expected_y, abs=1.0)

    def test_identity_is_noop(self):
        """Only InitialSize in chain — no-op."""
        f = VideoFrame(
            source_id="cam",
            fps=(30, 1),
            width=800,
            height=600,
            content=VideoFrameContent.none(),
        )
        obj = f.create_object(
            "det", "car", detection_box=RBBox(400.0, 300.0, 100.0, 80.0)
        )
        oid = obj.id

        f.transform_forward()

        obj = f.get_object(oid)
        det = obj.detection_box
        assert det.xc == pytest.approx(400.0, abs=0.01)
        assert det.yc == pytest.approx(300.0, abs=0.01)
        assert f.width == 800
        assert f.height == 600


# ── VideoFrame objects ────────────────────────────────────────────────────


class TestVideoFrameObjects:
    @pytest.fixture()
    def frame(self):
        return VideoFrame(
            source_id="cam",
            fps=(30, 1),
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
            fps=(30, 1),
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
        objs = u.objects
        assert len(objs) == 1

    def test_policies(self):
        u = VideoFrameUpdate()
        u.frame_attribute_policy = AttributeUpdatePolicy.ReplaceWithForeignWhenDuplicate
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
            fps=(30, 1),
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
            fps=(30, 1),
            width=100,
            height=100,
            content=VideoFrameContent.none(),
        )

    def _make_batch(self, frames=None):
        batch = VideoFrameBatch()
        if frames:
            for i, f in enumerate(frames):
                batch.add(i, f)
        return batch

    def test_create_empty(self):
        batch = VideoFrameBatch()
        assert len(batch.ids) == 0

    def test_add_and_ids(self):
        batch = self._make_batch([self._make_frame("cam1"), self._make_frame("cam2")])
        assert len(batch.ids) == 2

    def test_get_frame(self):
        f = self._make_frame("test")
        batch = self._make_batch([f])
        ids = batch.ids
        assert len(ids) == 1
        retrieved = batch.get(ids[0])
        assert retrieved is not None

    def test_frames_list(self):
        frames = [self._make_frame("a"), self._make_frame("b")]
        batch = self._make_batch(frames)
        d = batch.frames
        assert isinstance(d, list)
        assert len(d) == 2

    def test_add_get_del(self):
        batch = VideoFrameBatch()
        f = self._make_frame()
        batch.add(42, f)
        retrieved = batch.get(42)
        assert retrieved is not None
        deleted = batch.del_(42)
        assert deleted is not None
        assert batch.get(42) is None

    def test_to_protobuf_roundtrip(self):
        batch = self._make_batch([self._make_frame()])
        data = batch.to_protobuf()
        assert isinstance(data, bytes)
        batch2 = VideoFrameBatch.from_protobuf(data)
        assert len(batch2.ids) == 1

    def test_access_objects(self):
        f = self._make_frame()
        f.create_object("ns", "lbl", detection_box=RBBox(0, 0, 10, 10))
        batch = self._make_batch([f])
        q = MatchQuery.idle()
        result = batch.access_objects(q)
        assert isinstance(result, dict)

    def test_delete_objects(self):
        f = self._make_frame()
        f.create_object("ns", "lbl", detection_box=RBBox(0, 0, 10, 10))
        batch = self._make_batch([f])
        q = MatchQuery.idle()
        # delete_objects may return None or dict
        batch.delete_objects(q)
