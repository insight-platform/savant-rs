"""Tests for VideoObject, BorrowedVideoObject, VideoObjectsView,
VideoObjectTree, IdCollisionResolutionPolicy, ObjectUpdatePolicy."""

from __future__ import annotations

import pytest

from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    BorrowedVideoObject,
    IdCollisionResolutionPolicy,
    ObjectUpdatePolicy,
    VideoFrame,
    VideoFrameContent,
    VideoObject,
    VideoObjectsView,
)
from savant_rs.primitives.geometry import RBBox
from savant_rs.utils import VideoObjectBBoxTransformation
from savant_rs.match_query import MatchQuery, StringExpression


# ── IdCollisionResolutionPolicy ───────────────────────────────────────────


class TestIdCollisionResolutionPolicy:
    def test_variants(self):
        assert IdCollisionResolutionPolicy.GenerateNewId is not None
        assert IdCollisionResolutionPolicy.Overwrite is not None
        assert IdCollisionResolutionPolicy.Error is not None


# ── ObjectUpdatePolicy ───────────────────────────────────────────────────


class TestObjectUpdatePolicy:
    def test_variants(self):
        assert ObjectUpdatePolicy.AddForeignObjects is not None
        assert ObjectUpdatePolicy.ErrorIfLabelsCollide is not None
        assert ObjectUpdatePolicy.ReplaceSameLabelObjects is not None


# ── VideoObject ──────────────────────────────────────────────────────────


class TestVideoObject:
    def test_create(self):
        vo = VideoObject(
            id=10,
            namespace="detector",
            label="car",
            detection_box=RBBox(100.0, 200.0, 50.0, 80.0),
            attributes=[],
            confidence=0.9,
            track_id=5,
            track_box=RBBox(101.0, 201.0, 50.0, 80.0),
        )
        assert vo.id == 10
        assert vo.namespace == "detector"
        assert vo.label == "car"
        assert vo.confidence == pytest.approx(0.9)
        assert vo.track_id == 5
        assert vo.track_box is not None
        assert vo.detection_box.xc == pytest.approx(100.0)

    def test_no_optional_fields(self):
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
        assert vo.confidence is None
        assert vo.track_id is None
        assert vo.track_box is None

    def test_attributes(self):
        attr = Attribute.persistent("ns", "key", [AttributeValue.string("val")])
        vo = VideoObject(
            id=1,
            namespace="ns",
            label="lbl",
            detection_box=RBBox(0, 0, 10, 10),
            attributes=[attr],
            confidence=None,
            track_id=None,
            track_box=None,
        )
        assert len(vo.attributes) == 1
        fetched = vo.get_attribute("ns", "key")
        assert fetched is not None

    def test_set_attribute(self):
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
        vo.set_attribute(
            Attribute.persistent("ns", "k", [AttributeValue.integer(42)])
        )
        assert vo.get_attribute("ns", "k") is not None

    def test_set_persistent_attribute(self):
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
        vo.set_persistent_attribute("ns", "key", False, "hint", [])
        assert vo.get_attribute("ns", "key") is not None

    def test_set_temporary_attribute(self):
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
        vo.set_temporary_attribute("ns", "key", False, None, [])
        attr = vo.get_attribute("ns", "key")
        assert attr is not None
        assert attr.is_temporary()

    def test_protobuf_roundtrip(self):
        vo = VideoObject(
            id=1,
            namespace="det",
            label="person",
            detection_box=RBBox(10.0, 20.0, 30.0, 40.0),
            attributes=[],
            confidence=0.85,
            track_id=7,
            track_box=RBBox(11.0, 21.0, 30.0, 40.0),
        )
        data = vo.to_protobuf()
        assert isinstance(data, bytes)
        vo2 = VideoObject.from_protobuf(data)
        assert vo2.namespace == "det"
        assert vo2.label == "person"
        assert vo2.confidence == pytest.approx(0.85)

    def test_has_expected_properties(self):
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
        # Verify basic properties are accessible
        assert vo.namespace == "ns"
        assert vo.label == "lbl"
        assert vo.id == 1


# ── BorrowedVideoObject ──────────────────────────────────────────────────


class TestBorrowedVideoObject:
    @pytest.fixture()
    def frame_with_object(self):
        f = VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=640,
            height=480,
            content=VideoFrameContent.none(),
        )
        obj = f.create_object(
            "det",
            "person",
            detection_box=RBBox(100.0, 200.0, 50.0, 80.0),
            confidence=0.95,
            track_id=10,
            track_box=RBBox(101.0, 201.0, 50.0, 80.0),
        )
        return f, obj

    def test_properties(self, frame_with_object):
        _, obj = frame_with_object
        assert obj.namespace == "det"
        assert obj.label == "person"
        assert obj.confidence == pytest.approx(0.95)
        assert obj.track_id == 10
        assert isinstance(obj.id, int)
        assert isinstance(obj.memory_handle, int)

    def test_detection_box(self, frame_with_object):
        _, obj = frame_with_object
        box = obj.detection_box
        assert box.xc == pytest.approx(100.0)

    def test_track_box(self, frame_with_object):
        _, obj = frame_with_object
        tb = obj.track_box
        assert tb is not None
        assert tb.xc == pytest.approx(101.0)

    def test_set_attribute(self, frame_with_object):
        _, obj = frame_with_object
        obj.set_attribute(
            Attribute.persistent("ns", "k", [AttributeValue.string("v")])
        )
        fetched = obj.get_attribute("ns", "k")
        assert fetched is not None

    def test_delete_attribute(self, frame_with_object):
        _, obj = frame_with_object
        obj.set_attribute(
            Attribute.persistent("ns", "k", [AttributeValue.integer(1)])
        )
        deleted = obj.delete_attribute("ns", "k")
        assert deleted is not None
        assert obj.get_attribute("ns", "k") is None

    def test_clear_attributes(self, frame_with_object):
        _, obj = frame_with_object
        obj.set_attribute(
            Attribute.persistent("ns", "k", [AttributeValue.integer(1)])
        )
        obj.clear_attributes()
        assert len(obj.attributes) == 0

    def test_detached_copy(self, frame_with_object):
        _, obj = frame_with_object
        detached = obj.detached_copy()
        assert isinstance(detached, VideoObject)
        assert detached.namespace == "det"

    def test_set_track_info(self, frame_with_object):
        _, obj = frame_with_object
        new_box = RBBox(500.0, 500.0, 30.0, 30.0)
        obj.set_track_info(99, new_box)
        assert obj.track_id == 99
        assert obj.track_box.xc == pytest.approx(500.0)

    def test_clear_track_info(self, frame_with_object):
        _, obj = frame_with_object
        obj.clear_track_info()
        assert obj.track_id is None
        assert obj.track_box is None

    def test_draw_label(self, frame_with_object):
        _, obj = frame_with_object
        # Default draw_label is the label
        assert isinstance(obj.draw_label, str)
        obj.draw_label = "custom-label"
        assert obj.draw_label == "custom-label"

    def test_transform_geometry(self, frame_with_object):
        _, obj = frame_with_object
        ops = [VideoObjectBBoxTransformation.scale(2.0, 2.0)]
        obj.transform_geometry(ops)
        # Box should be scaled
        box = obj.detection_box
        assert box.width == pytest.approx(100.0)

    def test_find_attributes_with_ns(self, frame_with_object):
        _, obj = frame_with_object
        obj.set_attribute(
            Attribute.persistent("ns1", "a", [AttributeValue.integer(1)])
        )
        obj.set_attribute(
            Attribute.persistent("ns2", "b", [AttributeValue.integer(2)])
        )
        found = obj.find_attributes_with_ns("ns1")
        assert len(found) == 1

    def test_set_persistent_attribute(self, frame_with_object):
        _, obj = frame_with_object
        obj.set_persistent_attribute("ns", "key", False, "hint", [])
        assert obj.get_attribute("ns", "key") is not None

    def test_set_temporary_attribute(self, frame_with_object):
        _, obj = frame_with_object
        obj.set_temporary_attribute("ns", "key", False, None, [])
        attr = obj.get_attribute("ns", "key")
        assert attr is not None
        assert attr.is_temporary()

    def test_to_protobuf(self, frame_with_object):
        _, obj = frame_with_object
        data = obj.to_protobuf()
        assert isinstance(data, bytes)


# ── VideoObjectsView ─────────────────────────────────────────────────────


class TestVideoObjectsView:
    @pytest.fixture()
    def frame_with_objects(self):
        f = VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=640,
            height=480,
            content=VideoFrameContent.none(),
        )
        f.create_object(
            "ns",
            "a",
            detection_box=RBBox(0, 0, 10, 10),
            track_id=100,
        )
        f.create_object(
            "ns",
            "b",
            detection_box=RBBox(0, 0, 10, 10),
            track_id=200,
        )
        return f

    def test_len(self, frame_with_objects):
        view = frame_with_objects.get_all_objects()
        assert len(view) == 2

    def test_getitem(self, frame_with_objects):
        view = frame_with_objects.get_all_objects()
        obj = view[0]
        assert isinstance(obj, BorrowedVideoObject)

    def test_ids(self, frame_with_objects):
        view = frame_with_objects.get_all_objects()
        ids = view.ids
        assert len(ids) == 2

    def test_track_ids(self, frame_with_objects):
        view = frame_with_objects.get_all_objects()
        tids = view.track_ids
        assert set(tids) == {100, 200}

    def test_sorted_by_id(self, frame_with_objects):
        view = frame_with_objects.get_all_objects()
        sorted_objs = view.sorted_by_id
        assert len(sorted_objs) == 2
        assert sorted_objs[0].id <= sorted_objs[1].id


# ── VideoObjectTree ──────────────────────────────────────────────────────


class TestVideoObjectTree:
    def test_export_import_object_trees(self):
        f = VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=640,
            height=480,
            content=VideoFrameContent.none(),
        )
        parent = f.create_object("ns", "parent", detection_box=RBBox(0, 0, 10, 10))
        f.create_object(
            "ns",
            "child",
            parent_id=parent.id,
            detection_box=RBBox(0, 0, 5, 5),
        )
        q = MatchQuery.label(StringExpression.eq("parent"))
        trees = f.export_complete_object_trees(q, delete_exported=False)
        assert len(trees) >= 1

        f2 = VideoFrame(
            source_id="cam2",
            framerate="30/1",
            width=640,
            height=480,
            content=VideoFrameContent.none(),
        )
        f2.import_object_trees(trees)
        assert len(f2.get_all_objects()) >= 2

    def test_walk_objects(self):
        f = VideoFrame(
            source_id="cam",
            framerate="30/1",
            width=640,
            height=480,
            content=VideoFrameContent.none(),
        )
        parent = f.create_object("ns", "parent", detection_box=RBBox(0, 0, 10, 10))
        f.create_object(
            "ns",
            "child",
            parent_id=parent.id,
            detection_box=RBBox(0, 0, 5, 5),
        )
        q = MatchQuery.label(StringExpression.eq("parent"))
        trees = f.export_complete_object_trees(q, delete_exported=False)

        visited = []

        def visitor(obj, parent_obj, _user_data):
            visited.append(obj.label)
            return None

        for tree in trees:
            tree.walk_objects(visitor)

        assert "parent" in visited
        assert "child" in visited
