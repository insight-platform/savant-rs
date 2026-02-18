"""Tests for savant_rs.primitives.geometry – Point, Segment, Intersection,
PolygonalArea, RBBox, BBox, and free-standing helper functions."""

from __future__ import annotations

import pytest

from savant_rs.primitives.geometry import (
    BBox,
    Intersection,
    IntersectionKind,
    Point,
    PolygonalArea,
    RBBox,
    Segment,
    associate_bboxes,
    solely_owned_areas,
)
from savant_rs.draw_spec import PaddingDraw
from savant_rs.utils import BBoxMetricType


# ── Point ─────────────────────────────────────────────────────────────────


class TestPoint:
    def test_create(self):
        p = Point(1.5, 2.5)
        assert p.x == 1.5
        assert p.y == 2.5

    def test_modify(self):
        p = Point(0.0, 0.0)
        p.x = 3.0
        p.y = 4.0
        assert p.x == 3.0
        assert p.y == 4.0


# ── Segment ───────────────────────────────────────────────────────────────


class TestSegment:
    def test_create(self):
        s = Segment(Point(0.0, 0.0), Point(10.0, 10.0))
        assert s.begin.x == 0.0
        assert s.end.x == 10.0

    def test_begin_end(self):
        s = Segment(Point(1.0, 2.0), Point(3.0, 4.0))
        assert s.begin.y == 2.0
        assert s.end.y == 4.0


# ── IntersectionKind ──────────────────────────────────────────────────────


class TestIntersectionKind:
    def test_variants_exist(self):
        assert IntersectionKind.Enter is not None
        assert IntersectionKind.Inside is not None
        assert IntersectionKind.Leave is not None
        assert IntersectionKind.Cross is not None
        assert IntersectionKind.Outside is not None


# ── Intersection ──────────────────────────────────────────────────────────


class TestIntersection:
    def test_create(self):
        inter = Intersection(IntersectionKind.Enter, [(0, "edge0")])
        assert inter.kind == IntersectionKind.Enter
        assert inter.edges == [(0, "edge0")]

    def test_empty_edges(self):
        inter = Intersection(IntersectionKind.Outside, [])
        assert inter.edges == []

    def test_multiple_edges(self):
        edges = [(0, "a"), (1, None), (2, "c")]
        inter = Intersection(IntersectionKind.Cross, edges)
        assert len(inter.edges) == 3
        assert inter.edges[1] == (1, None)


# ── PolygonalArea ─────────────────────────────────────────────────────────


class TestPolygonalArea:
    @pytest.fixture()
    def square(self):
        """A 10x10 square centred at origin."""
        pts = [
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 10.0),
        ]
        area = PolygonalArea(pts)
        area.build_polygon()
        return area

    def test_contains_inside(self, square):
        assert square.contains(Point(5.0, 5.0))

    def test_contains_outside(self, square):
        assert not square.contains(Point(20.0, 20.0))

    def test_is_not_self_intersecting(self, square):
        assert not square.is_self_intersecting()

    def test_is_self_intersecting(self):
        pts = [
            Point(0.0, 0.0),
            Point(10.0, 10.0),
            Point(10.0, 0.0),
            Point(0.0, 10.0),
        ]
        area = PolygonalArea(pts)
        assert area.is_self_intersecting()

    def test_crossed_by_segment(self, square):
        seg = Segment(Point(-1.0, 5.0), Point(5.0, 5.0))
        inter = square.crossed_by_segment(seg)
        assert inter.kind is not None

    def test_tags(self):
        pts = [Point(0.0, 0.0), Point(10.0, 0.0), Point(10.0, 10.0)]
        tags = ["a", "b", None]
        area = PolygonalArea(pts, tags)
        assert area.get_tag(0) == "a"
        assert area.get_tag(1) == "b"
        assert area.get_tag(2) is None

    def test_points_positions(self, square):
        polys = [square]
        pts = [Point(5.0, 5.0), Point(20.0, 20.0)]
        result = PolygonalArea.points_positions(polys, pts)
        assert isinstance(result, list)
        # result is [[bool, bool]] - one list per polygon
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_segments_intersections(self, square):
        polys = [square]
        segs = [Segment(Point(-1.0, 5.0), Point(11.0, 5.0))]
        result = PolygonalArea.segments_intersections(polys, segs)
        assert isinstance(result, list)
        assert len(result) == 1


# ── RBBox ─────────────────────────────────────────────────────────────────


class TestRBBox:
    def test_create_no_angle(self):
        bb = RBBox(10.0, 20.0, 100.0, 200.0)
        assert bb.xc == 10.0
        assert bb.yc == 20.0
        assert bb.width == 100.0
        assert bb.height == 200.0
        assert bb.angle is None

    def test_create_with_angle(self):
        bb = RBBox(0.0, 0.0, 50.0, 50.0, 45.0)
        assert bb.angle == 45.0

    def test_area(self):
        bb = RBBox(0.0, 0.0, 10.0, 20.0)
        assert bb.area == pytest.approx(200.0)

    def test_width_to_height_ratio(self):
        bb = RBBox(0.0, 0.0, 10.0, 20.0)
        assert bb.width_to_height_ratio == pytest.approx(0.5)

    def test_top_left_right_bottom(self):
        bb = RBBox(50.0, 50.0, 20.0, 10.0)
        assert bb.left == pytest.approx(40.0)
        assert bb.top == pytest.approx(45.0)
        assert bb.right == pytest.approx(60.0)
        assert bb.bottom == pytest.approx(55.0)

    def test_eq(self):
        a = RBBox(1.0, 2.0, 3.0, 4.0)
        b = RBBox(1.0, 2.0, 3.0, 4.0)
        assert a.eq(b)

    def test_almost_eq(self):
        a = RBBox(1.0, 2.0, 3.0, 4.0)
        b = RBBox(1.001, 2.001, 3.001, 4.001)
        assert a.almost_eq(b, 0.01)
        assert not a.almost_eq(b, 0.0001)

    def test_scale(self):
        bb = RBBox(0.0, 0.0, 10.0, 20.0)
        bb.scale(2.0, 3.0)
        assert bb.width == pytest.approx(20.0)
        assert bb.height == pytest.approx(60.0)

    def test_shift(self):
        bb = RBBox(10.0, 20.0, 5.0, 5.0)
        bb.shift(1.0, 2.0)
        assert bb.xc == pytest.approx(11.0)
        assert bb.yc == pytest.approx(22.0)

    def test_vertices(self):
        bb = RBBox(5.0, 5.0, 10.0, 10.0)
        verts = bb.vertices
        assert len(verts) == 4

    def test_vertices_rounded(self):
        bb = RBBox(5.0, 5.0, 10.0, 10.0)
        verts = bb.vertices_rounded
        assert len(verts) == 4

    def test_vertices_int(self):
        bb = RBBox(5.0, 5.0, 10.0, 10.0)
        verts = bb.vertices_int
        assert len(verts) == 4
        assert all(isinstance(v[0], int) for v in verts)

    def test_as_polygonal_area(self):
        bb = RBBox(5.0, 5.0, 10.0, 10.0)
        pa = bb.as_polygonal_area()
        assert isinstance(pa, PolygonalArea)

    def test_wrapping_box(self):
        bb = RBBox(5.0, 5.0, 10.0, 10.0, 45.0)
        wb = bb.wrapping_box
        assert isinstance(wb, BBox)

    def test_copy(self):
        bb = RBBox(1.0, 2.0, 3.0, 4.0, 5.0)
        c = bb.copy()
        assert c.xc == 1.0

    def test_iou_identical(self):
        a = RBBox(5.0, 5.0, 10.0, 10.0)
        assert a.iou(a) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = RBBox(0.0, 0.0, 2.0, 2.0)
        b = RBBox(100.0, 100.0, 2.0, 2.0)
        assert a.iou(b) == pytest.approx(0.0)

    def test_ios(self):
        a = RBBox(5.0, 5.0, 10.0, 10.0)
        assert a.ios(a) == pytest.approx(1.0)

    def test_ioo(self):
        a = RBBox(5.0, 5.0, 10.0, 10.0)
        assert a.ioo(a) == pytest.approx(1.0)

    def test_ltrb(self):
        bb = RBBox.ltrb(0.0, 0.0, 10.0, 20.0)
        assert bb.xc == pytest.approx(5.0)
        assert bb.yc == pytest.approx(10.0)

    def test_ltwh(self):
        bb = RBBox.ltwh(0.0, 0.0, 10.0, 20.0)
        assert bb.width == pytest.approx(10.0)
        assert bb.height == pytest.approx(20.0)

    def test_as_ltrb(self):
        bb = RBBox(5.0, 10.0, 10.0, 20.0)
        l, t, r, b = bb.as_ltrb()
        assert l == pytest.approx(0.0)
        assert t == pytest.approx(0.0)
        assert r == pytest.approx(10.0)
        assert b == pytest.approx(20.0)

    def test_as_ltrb_int(self):
        bb = RBBox(5.0, 10.0, 10.0, 20.0)
        result = bb.as_ltrb_int()
        assert all(isinstance(v, int) for v in result)

    def test_as_ltwh(self):
        bb = RBBox(5.0, 10.0, 10.0, 20.0)
        result = bb.as_ltwh()
        assert len(result) == 4

    def test_as_ltwh_int(self):
        bb = RBBox(5.0, 10.0, 10.0, 20.0)
        result = bb.as_ltwh_int()
        assert all(isinstance(v, int) for v in result)

    def test_as_xcycwh(self):
        bb = RBBox(5.0, 10.0, 10.0, 20.0)
        xc, yc, w, h = bb.as_xcycwh()
        assert xc == pytest.approx(5.0)
        assert yc == pytest.approx(10.0)

    def test_as_xcycwh_int(self):
        bb = RBBox(5.0, 10.0, 10.0, 20.0)
        result = bb.as_xcycwh_int()
        assert all(isinstance(v, int) for v in result)

    def test_into_bbox(self):
        rbb = RBBox(5.0, 10.0, 10.0, 20.0)
        bbox = rbb.into_bbox()
        assert isinstance(bbox, BBox)
        assert bbox.xc == pytest.approx(5.0)

    def test_is_modified(self):
        bb = RBBox(0.0, 0.0, 1.0, 1.0)
        assert not bb.is_modified()
        bb.xc = 5.0
        assert bb.is_modified()

    def test_set_modifications(self):
        bb = RBBox(0.0, 0.0, 1.0, 1.0)
        bb.xc = 5.0
        assert bb.is_modified()
        bb.set_modifications(False)
        assert not bb.is_modified()

    def test_inside(self):
        inner = RBBox(5.0, 5.0, 2.0, 2.0)
        outer = RBBox(5.0, 5.0, 100.0, 100.0)
        assert inner.inside(outer)
        assert not outer.inside(inner)

    def test_inside_viewport(self):
        bb = RBBox(5.0, 5.0, 8.0, 8.0)
        assert bb.inside_viewport(100.0, 100.0)
        assert not bb.inside_viewport(2.0, 2.0)

    def test_get_visual_box(self):
        bb = RBBox(50.0, 50.0, 20.0, 20.0)
        pad = PaddingDraw(2, 2, 2, 2)
        vb = bb.get_visual_box(pad, 1, 1000.0, 1000.0)
        assert isinstance(vb, RBBox)

    def test_new_padded(self):
        bb = RBBox(50.0, 50.0, 20.0, 20.0)
        pad = PaddingDraw(5, 5, 5, 5)
        padded = bb.new_padded(pad)
        assert padded.width >= bb.width
        assert padded.height >= bb.height


# ── BBox ──────────────────────────────────────────────────────────────────


class TestBBox:
    def test_create(self):
        bb = BBox(5.0, 10.0, 20.0, 40.0)
        assert bb.xc == 5.0
        assert bb.yc == 10.0
        assert bb.width == 20.0
        assert bb.height == 40.0

    def test_top_left_right_bottom(self):
        bb = BBox(50.0, 50.0, 20.0, 10.0)
        assert bb.left == pytest.approx(40.0)
        assert bb.top == pytest.approx(45.0)
        assert bb.right == pytest.approx(60.0)
        assert bb.bottom == pytest.approx(55.0)

    def test_eq(self):
        a = BBox(1.0, 2.0, 3.0, 4.0)
        b = BBox(1.0, 2.0, 3.0, 4.0)
        assert a.eq(b)

    def test_almost_eq(self):
        a = BBox(1.0, 2.0, 3.0, 4.0)
        b = BBox(1.001, 2.001, 3.001, 4.001)
        assert a.almost_eq(b, 0.01)

    def test_iou_identical(self):
        a = BBox(5.0, 5.0, 10.0, 10.0)
        assert a.iou(a) == pytest.approx(1.0)

    def test_ios(self):
        a = BBox(5.0, 5.0, 10.0, 10.0)
        assert a.ios(a) == pytest.approx(1.0)

    def test_ioo(self):
        a = BBox(5.0, 5.0, 10.0, 10.0)
        assert a.ioo(a) == pytest.approx(1.0)

    def test_ltrb(self):
        bb = BBox.ltrb(0.0, 0.0, 10.0, 20.0)
        assert bb.xc == pytest.approx(5.0)
        assert bb.yc == pytest.approx(10.0)

    def test_ltwh(self):
        bb = BBox.ltwh(0.0, 0.0, 10.0, 20.0)
        assert bb.width == pytest.approx(10.0)

    def test_as_ltrb(self):
        bb = BBox(5.0, 10.0, 10.0, 20.0)
        l, t, r, b = bb.as_ltrb()
        assert l == pytest.approx(0.0)
        assert b == pytest.approx(20.0)

    def test_as_ltrb_int(self):
        bb = BBox(5.0, 10.0, 10.0, 20.0)
        result = bb.as_ltrb_int()
        assert all(isinstance(v, int) for v in result)

    def test_as_ltwh(self):
        bb = BBox(5.0, 10.0, 10.0, 20.0)
        result = bb.as_ltwh()
        assert len(result) == 4

    def test_as_ltwh_int(self):
        bb = BBox(5.0, 10.0, 10.0, 20.0)
        result = bb.as_ltwh_int()
        assert all(isinstance(v, int) for v in result)

    def test_as_xcycwh(self):
        bb = BBox(5.0, 10.0, 10.0, 20.0)
        xc, yc, w, h = bb.as_xcycwh()
        assert xc == pytest.approx(5.0)

    def test_as_xcycwh_int(self):
        bb = BBox(5.0, 10.0, 10.0, 20.0)
        result = bb.as_xcycwh_int()
        assert all(isinstance(v, int) for v in result)

    def test_as_rbbox(self):
        bb = BBox(5.0, 10.0, 10.0, 20.0)
        rbb = bb.as_rbbox()
        assert isinstance(rbb, RBBox)
        assert rbb.angle is None

    def test_scale(self):
        bb = BBox(0.0, 0.0, 10.0, 20.0)
        bb.scale(2.0, 3.0)
        assert bb.width == pytest.approx(20.0)
        assert bb.height == pytest.approx(60.0)

    def test_shift(self):
        bb = BBox(10.0, 20.0, 5.0, 5.0)
        bb.shift(1.0, 2.0)
        assert bb.xc == pytest.approx(11.0)
        assert bb.yc == pytest.approx(22.0)

    def test_as_polygonal_area(self):
        bb = BBox(5.0, 5.0, 10.0, 10.0)
        pa = bb.as_polygonal_area()
        assert isinstance(pa, PolygonalArea)

    def test_copy(self):
        bb = BBox(1.0, 2.0, 3.0, 4.0)
        c = bb.copy()
        assert c.xc == 1.0

    def test_vertices(self):
        bb = BBox(5.0, 5.0, 10.0, 10.0)
        assert len(bb.vertices) == 4

    def test_vertices_rounded(self):
        bb = BBox(5.0, 5.0, 10.0, 10.0)
        assert len(bb.vertices_rounded) == 4

    def test_vertices_int(self):
        bb = BBox(5.0, 5.0, 10.0, 10.0)
        verts = bb.vertices_int
        assert len(verts) == 4
        assert all(isinstance(v[0], int) for v in verts)

    def test_is_modified(self):
        bb = BBox(0.0, 0.0, 1.0, 1.0)
        assert not bb.is_modified()

    def test_wrapping_box(self):
        bb = BBox(5.0, 5.0, 10.0, 10.0)
        wb = bb.wrapping_box
        assert isinstance(wb, BBox)

    def test_inside(self):
        inner = BBox(5.0, 5.0, 2.0, 2.0)
        outer = BBox(5.0, 5.0, 100.0, 100.0)
        assert inner.inside(outer)
        assert not outer.inside(inner)

    def test_inside_viewport(self):
        bb = BBox(5.0, 5.0, 8.0, 8.0)
        assert bb.inside_viewport(100.0, 100.0)
        assert not bb.inside_viewport(2.0, 2.0)

    def test_get_visual_box(self):
        bb = BBox(50.0, 50.0, 20.0, 20.0)
        pad = PaddingDraw(2, 2, 2, 2)
        vb = bb.get_visual_box(pad, 1, 1000.0, 1000.0)
        assert isinstance(vb, BBox)

    def test_new_padded(self):
        bb = BBox(50.0, 50.0, 20.0, 20.0)
        pad = PaddingDraw(5, 5, 5, 5)
        padded = bb.new_padded(pad)
        assert padded.width >= bb.width


# ── solely_owned_areas ────────────────────────────────────────────────────


class TestSolelyOwnedAreas:
    def test_non_overlapping(self):
        bboxes = [
            RBBox(0.0, 0.0, 10.0, 10.0),
            RBBox(100.0, 100.0, 10.0, 10.0),
        ]
        areas = solely_owned_areas(bboxes, False)
        assert len(areas) == 2
        assert all(a >= 0.0 for a in areas)

    def test_parallel(self):
        bboxes = [
            RBBox(0.0, 0.0, 10.0, 10.0),
            RBBox(100.0, 100.0, 10.0, 10.0),
        ]
        areas = solely_owned_areas(bboxes, True)
        assert len(areas) == 2


# ── associate_bboxes ─────────────────────────────────────────────────────


class TestAssociateBboxes:
    def test_basic_association(self):
        candidates = [RBBox(5.0, 5.0, 10.0, 10.0)]
        owners = [RBBox(5.0, 5.0, 10.0, 10.0)]
        result = associate_bboxes(candidates, owners, BBoxMetricType.IoU, 0.5)
        assert isinstance(result, dict)

    def test_no_match(self):
        candidates = [RBBox(0.0, 0.0, 2.0, 2.0)]
        owners = [RBBox(100.0, 100.0, 2.0, 2.0)]
        result = associate_bboxes(candidates, owners, BBoxMetricType.IoU, 0.5)
        assert isinstance(result, dict)
