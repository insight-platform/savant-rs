"""Tests for savant_rs.draw_spec – drawing specification classes."""

from __future__ import annotations

from savant_rs.draw_spec import (
    BoundingBoxDraw,
    ColorDraw,
    DotDraw,
    LabelDraw,
    LabelPosition,
    LabelPositionKind,
    ObjectDraw,
    PaddingDraw,
    SetDrawLabelKind,
)


# ── ColorDraw ─────────────────────────────────────────────────────────────


class TestColorDraw:
    def test_default(self):
        c = ColorDraw()
        assert c.red == 0
        assert c.green == 255
        assert c.blue == 0
        assert c.alpha == 255

    def test_custom(self):
        c = ColorDraw(red=10, green=20, blue=30, alpha=40)
        assert c.red == 10
        assert c.green == 20
        assert c.blue == 30
        assert c.alpha == 40

    def test_rgba(self):
        c = ColorDraw(red=1, green=2, blue=3, alpha=4)
        assert c.rgba == (1, 2, 3, 4)

    def test_bgra(self):
        c = ColorDraw(red=1, green=2, blue=3, alpha=4)
        assert c.bgra == (3, 2, 1, 4)

    def test_transparent(self):
        c = ColorDraw.transparent()
        assert c.alpha == 0

    def test_copy(self):
        c = ColorDraw(red=100, green=200, blue=50, alpha=128)
        c2 = c.copy()
        assert c2.red == 100
        assert c2.alpha == 128


# ── PaddingDraw ───────────────────────────────────────────────────────────


class TestPaddingDraw:
    def test_default(self):
        p = PaddingDraw()
        assert p.left == 0
        assert p.top == 0
        assert p.right == 0
        assert p.bottom == 0

    def test_custom(self):
        p = PaddingDraw(left=1, top=2, right=3, bottom=4)
        assert p.left == 1
        assert p.top == 2
        assert p.right == 3
        assert p.bottom == 4

    def test_padding_tuple(self):
        p = PaddingDraw(left=1, top=2, right=3, bottom=4)
        assert p.padding == (1, 2, 3, 4)

    def test_default_padding(self):
        p = PaddingDraw.default_padding()
        assert isinstance(p, PaddingDraw)

    def test_copy(self):
        p = PaddingDraw(left=5, top=6, right=7, bottom=8)
        p2 = p.copy()
        assert p2.left == 5


# ── BoundingBoxDraw ───────────────────────────────────────────────────────


class TestBoundingBoxDraw:
    def test_default(self):
        bb = BoundingBoxDraw()
        assert bb.thickness == 2
        assert isinstance(bb.border_color, ColorDraw)
        assert isinstance(bb.background_color, ColorDraw)
        assert isinstance(bb.padding, PaddingDraw)

    def test_custom(self):
        border = ColorDraw(red=255, green=0, blue=0, alpha=255)
        bg = ColorDraw(red=0, green=0, blue=0, alpha=128)
        bb = BoundingBoxDraw(
            border_color=border,
            background_color=bg,
            thickness=5,
        )
        assert bb.thickness == 5
        assert bb.border_color.red == 255

    def test_copy(self):
        bb = BoundingBoxDraw(thickness=10)
        bb2 = bb.copy()
        assert bb2.thickness == 10


# ── DotDraw ───────────────────────────────────────────────────────────────


class TestDotDraw:
    def test_create(self):
        c = ColorDraw(red=255, green=0, blue=0)
        d = DotDraw(color=c, radius=5)
        assert d.radius == 5
        assert d.color.red == 255

    def test_default_radius(self):
        c = ColorDraw()
        d = DotDraw(color=c)
        assert d.radius == 2

    def test_copy(self):
        d = DotDraw(color=ColorDraw(), radius=10)
        d2 = d.copy()
        assert d2.radius == 10


# ── LabelPositionKind ─────────────────────────────────────────────────────


class TestLabelPositionKind:
    def test_variants(self):
        assert LabelPositionKind.TopLeftInside is not None
        assert LabelPositionKind.TopLeftOutside is not None
        assert LabelPositionKind.Center is not None


# ── LabelPosition ─────────────────────────────────────────────────────────


class TestLabelPosition:
    def test_default_position(self):
        lp = LabelPosition.default_position()
        assert isinstance(lp, LabelPosition)
        assert isinstance(lp.position, LabelPositionKind)

    def test_custom(self):
        lp = LabelPosition(
            position=LabelPositionKind.Center,
            margin_x=5,
            margin_y=10,
        )
        assert lp.position == LabelPositionKind.Center
        assert lp.margin_x == 5
        assert lp.margin_y == 10

    def test_copy(self):
        lp = LabelPosition(position=LabelPositionKind.TopLeftInside)
        lp2 = lp.copy()
        assert lp2.position == LabelPositionKind.TopLeftInside


# ── LabelDraw ─────────────────────────────────────────────────────────────


class TestLabelDraw:
    def test_create(self):
        fc = ColorDraw(red=255, green=255, blue=255)
        ld = LabelDraw(font_color=fc)
        assert ld.font_color.red == 255
        assert ld.font_scale == 1.0
        assert ld.thickness == 1
        assert isinstance(ld.position, LabelPosition)
        assert isinstance(ld.padding, PaddingDraw)
        assert isinstance(ld.format, list)

    def test_custom(self):
        fc = ColorDraw(red=0, green=0, blue=0)
        ld = LabelDraw(
            font_color=fc,
            font_scale=2.0,
            thickness=3,
            format=["{namespace}", "{label}"],
        )
        assert ld.font_scale == 2.0
        assert ld.thickness == 3
        assert ld.format == ["{namespace}", "{label}"]

    def test_copy(self):
        ld = LabelDraw(font_color=ColorDraw(), font_scale=1.5)
        ld2 = ld.copy()
        assert ld2.font_scale == 1.5


# ── ObjectDraw ────────────────────────────────────────────────────────────


class TestObjectDraw:
    def test_defaults(self):
        od = ObjectDraw()
        assert od.bounding_box is None
        assert od.central_dot is None
        assert od.label is None
        assert od.blur is False

    def test_with_bounding_box(self):
        bb = BoundingBoxDraw()
        od = ObjectDraw(bounding_box=bb)
        assert od.bounding_box is not None
        assert od.bounding_box.thickness == 2

    def test_with_all(self):
        bb = BoundingBoxDraw()
        dot = DotDraw(color=ColorDraw())
        lbl = LabelDraw(font_color=ColorDraw())
        od = ObjectDraw(bounding_box=bb, central_dot=dot, label=lbl, blur=True)
        assert od.blur is True
        assert od.central_dot is not None
        assert od.label is not None

    def test_copy(self):
        od = ObjectDraw(blur=True)
        od2 = od.copy()
        assert od2.blur is True


# ── SetDrawLabelKind ──────────────────────────────────────────────────────


class TestSetDrawLabelKind:
    def test_own(self):
        lk = SetDrawLabelKind.own("hello")
        assert lk.is_own_label()
        assert not lk.is_parent_label()
        assert lk.get_label() == "hello"

    def test_parent(self):
        lk = SetDrawLabelKind.parent("world")
        assert lk.is_parent_label()
        assert not lk.is_own_label()
        assert lk.get_label() == "world"
