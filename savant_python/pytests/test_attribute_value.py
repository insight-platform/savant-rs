"""Tests for savant_rs.primitives.attribute_value – AttributeValueType and
AttributeValue with every factory method and getter."""

from __future__ import annotations

import json

import pytest

from savant_rs.primitives import AttributeValue, AttributeValueType
from savant_rs.primitives.geometry import (
    BBox,
    Intersection,
    IntersectionKind,
    Point,
    PolygonalArea,
    RBBox,
)


# ── AttributeValueType enum ──────────────────────────────────────────────


class TestAttributeValueType:
    def test_all_variants(self):
        for name in (
            "Bytes",
            "String",
            "StringList",
            "Integer",
            "IntegerList",
            "Float",
            "FloatList",
            "Boolean",
            "BooleanList",
            "BBox",
            "BBoxList",
            "RBBox",
            "RBBoxList",
            "Point",
            "PointList",
            "Polygon",
            "PolygonList",
            "Intersection",
            "TemporaryValue",
            "None_",
        ):
            assert getattr(AttributeValueType, name) is not None


# ── Factory methods + round-trip getters ──────────────────────────────────


class TestAttributeValueNone:
    def test_none(self):
        v = AttributeValue.none()
        assert v.is_none()
        assert v.value_type == AttributeValueType.None_


class TestAttributeValueString:
    def test_string(self):
        v = AttributeValue.string("hello", confidence=0.9)
        assert v.as_string() == "hello"
        assert v.confidence == pytest.approx(0.9)
        assert v.value_type == AttributeValueType.String

    def test_strings(self):
        v = AttributeValue.strings(["a", "b", "c"])
        assert v.as_strings() == ["a", "b", "c"]
        assert v.value_type == AttributeValueType.StringList


class TestAttributeValueInteger:
    def test_integer(self):
        v = AttributeValue.integer(42, confidence=0.8)
        assert v.as_integer() == 42
        assert v.confidence == pytest.approx(0.8)
        assert v.value_type == AttributeValueType.Integer

    def test_integers(self):
        v = AttributeValue.integers([1, 2, 3])
        assert v.as_integers() == [1, 2, 3]
        assert v.value_type == AttributeValueType.IntegerList


class TestAttributeValueFloat:
    def test_float(self):
        v = AttributeValue.float(3.14, confidence=0.7)
        assert v.as_float() == pytest.approx(3.14)
        assert v.value_type == AttributeValueType.Float

    def test_floats(self):
        v = AttributeValue.floats([1.1, 2.2])
        result = v.as_floats()
        assert len(result) == 2
        assert result[0] == pytest.approx(1.1)
        assert v.value_type == AttributeValueType.FloatList


class TestAttributeValueBoolean:
    def test_boolean(self):
        v = AttributeValue.boolean(True)
        assert v.as_boolean() is True
        assert v.value_type == AttributeValueType.Boolean

    def test_booleans(self):
        v = AttributeValue.booleans([True, False, True])
        assert v.as_booleans() == [True, False, True]
        assert v.value_type == AttributeValueType.BooleanList


class TestAttributeValueBytes:
    def test_bytes(self):
        data = bytes([1, 2, 3, 4])
        v = AttributeValue.bytes([2, 2], data, confidence=0.5)
        raw = v.as_bytes()
        assert raw is not None
        assert v.value_type == AttributeValueType.Bytes

    def test_bytes_from_list(self):
        v = AttributeValue.bytes_from_list([2, 2], [1, 2, 3, 4])
        assert v.as_bytes() is not None
        assert v.value_type == AttributeValueType.Bytes


class TestAttributeValueBBox:
    def test_bbox(self):
        bb = BBox(5.0, 10.0, 20.0, 40.0)
        v = AttributeValue.bbox(bb, confidence=0.6)
        # BBox is stored as RBBox internally, use as_rbbox
        result = v.as_rbbox()
        assert result is not None
        assert result.xc == pytest.approx(5.0)
        assert v.value_type == AttributeValueType.BBox

    def test_bboxes(self):
        bbs = [BBox(1.0, 2.0, 3.0, 4.0), BBox(5.0, 6.0, 7.0, 8.0)]
        v = AttributeValue.bboxes(bbs)
        # BBoxes are stored as RBBoxes internally, use as_rbboxes
        result = v.as_rbboxes()
        assert len(result) == 2
        assert v.value_type == AttributeValueType.BBoxList


class TestAttributeValueRBBox:
    def test_rbbox(self):
        rbb = RBBox(5.0, 10.0, 20.0, 40.0, 45.0)
        v = AttributeValue.rbbox(rbb)
        result = v.as_rbbox()
        assert result is not None
        assert result.angle == pytest.approx(45.0)
        # RBBox is stored as BBox internally
        assert v.value_type == AttributeValueType.BBox

    def test_rbboxes(self):
        rbbs = [RBBox(1.0, 2.0, 3.0, 4.0), RBBox(5.0, 6.0, 7.0, 8.0)]
        v = AttributeValue.rbboxes(rbbs)
        result = v.as_rbboxes()
        assert len(result) == 2
        # RBBox list is stored as BBoxList internally
        assert v.value_type == AttributeValueType.BBoxList


class TestAttributeValuePoint:
    def test_point(self):
        p = Point(1.5, 2.5)
        v = AttributeValue.point(p)
        result = v.as_point()
        assert result is not None
        assert result.x == pytest.approx(1.5)
        assert v.value_type == AttributeValueType.Point

    def test_points(self):
        pts = [Point(1.0, 2.0), Point(3.0, 4.0)]
        v = AttributeValue.points(pts)
        result = v.as_points()
        assert len(result) == 2
        assert v.value_type == AttributeValueType.PointList


class TestAttributeValuePolygon:
    def test_polygon(self):
        poly = PolygonalArea([Point(0.0, 0.0), Point(10.0, 0.0), Point(10.0, 10.0)])
        v = AttributeValue.polygon(poly)
        result = v.as_polygon()
        assert result is not None
        assert v.value_type == AttributeValueType.Polygon

    def test_polygons(self):
        p1 = PolygonalArea([Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0)])
        p2 = PolygonalArea([Point(0.0, 0.0), Point(2.0, 0.0), Point(2.0, 2.0)])
        v = AttributeValue.polygons([p1, p2])
        result = v.as_polygons()
        assert len(result) == 2
        assert v.value_type == AttributeValueType.PolygonList


class TestAttributeValueIntersection:
    def test_intersection(self):
        inter = Intersection(IntersectionKind.Enter, [(0, "edge")])
        v = AttributeValue.intersection(inter, confidence=0.95)
        result = v.as_intersection()
        assert result is not None
        assert result.kind == IntersectionKind.Enter
        assert v.value_type == AttributeValueType.Intersection


class TestAttributeValueTemporary:
    def test_temporary_python_object(self):
        obj = {"key": "value", "num": 42}
        v = AttributeValue.temporary_python_object(obj)
        result = v.as_temporary_python_object()
        assert result == obj
        assert v.value_type == AttributeValueType.TemporaryValue


# ── Cross-type mismatches return None ─────────────────────────────────────


class TestAttributeValueMismatch:
    def test_string_as_integer_returns_none(self):
        v = AttributeValue.string("hello")
        assert v.as_integer() is None

    def test_integer_as_string_returns_none(self):
        v = AttributeValue.integer(42)
        assert v.as_string() is None

    def test_none_as_float_returns_none(self):
        v = AttributeValue.none()
        assert v.as_float() is None


# ── Confidence ────────────────────────────────────────────────────────────


class TestAttributeValueConfidence:
    def test_no_confidence(self):
        v = AttributeValue.string("test")
        assert v.confidence is None

    def test_with_confidence(self):
        v = AttributeValue.string("test", confidence=0.5)
        assert v.confidence == pytest.approx(0.5)

    def test_set_confidence(self):
        v = AttributeValue.string("test")
        v.confidence = 0.99
        assert v.confidence == pytest.approx(0.99)


# ── JSON round-trip ───────────────────────────────────────────────────────


class TestAttributeValueJson:
    def test_string_json_roundtrip(self):
        v = AttributeValue.string("hello", confidence=0.8)
        j = v.json
        parsed = json.loads(j)
        assert parsed is not None
        v2 = AttributeValue.from_json(j)
        assert v2.as_string() == "hello"

    def test_integer_json_roundtrip(self):
        v = AttributeValue.integer(42)
        v2 = AttributeValue.from_json(v.json)
        assert v2.as_integer() == 42

    def test_none_json_roundtrip(self):
        v = AttributeValue.none()
        v2 = AttributeValue.from_json(v.json)
        assert v2.is_none()
