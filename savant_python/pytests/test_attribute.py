"""Tests for savant_rs.primitives.attribute – Attribute and
AttributeUpdatePolicy."""

from __future__ import annotations

import json

from savant_rs.primitives import Attribute, AttributeUpdatePolicy, AttributeValue


# ── Attribute construction ────────────────────────────────────────────────


class TestAttributeConstruction:
    def test_basic(self):
        vals = [AttributeValue.string("hello")]
        attr = Attribute(
            namespace="ns",
            name="n",
            values=vals,
            hint="h",
            is_persistent=True,
            is_hidden=False,
        )
        assert attr.namespace == "ns"
        assert attr.name == "n"
        assert attr.hint == "h"
        assert not attr.is_temporary()
        assert not attr.is_hidden()

    def test_persistent_classmethod(self):
        vals = [AttributeValue.integer(1)]
        attr = Attribute.persistent("ns", "name", vals, hint="hint")
        assert not attr.is_temporary()
        assert attr.hint == "hint"

    def test_temporary_classmethod(self):
        vals = [AttributeValue.float(1.5)]
        attr = Attribute.temporary("ns", "name", vals)
        assert attr.is_temporary()
        assert attr.hint is None

    def test_hidden(self):
        attr = Attribute.persistent("ns", "name", [], is_hidden=True)
        assert attr.is_hidden()

    def test_make_persistent_temporary(self):
        attr = Attribute.persistent("ns", "name", [])
        assert not attr.is_temporary()
        attr.make_temporary()
        assert attr.is_temporary()
        attr.make_persistent()
        assert not attr.is_temporary()


# ── Attribute values view ─────────────────────────────────────────────────


class TestAttributeValuesView:
    def test_values_view_len(self):
        vals = [AttributeValue.string("a"), AttributeValue.string("b")]
        attr = Attribute.persistent("ns", "name", vals)
        view = attr.values_view
        assert len(view) == 2

    def test_values_view_getitem(self):
        vals = [AttributeValue.string("a"), AttributeValue.integer(42)]
        attr = Attribute.persistent("ns", "name", vals)
        view = attr.values_view
        v0 = view[0]
        assert v0.as_string() == "a"
        v1 = view[1]
        assert v1.as_integer() == 42

    def test_values_view_memory_handle(self):
        attr = Attribute.persistent("ns", "name", [AttributeValue.none()])
        view = attr.values_view
        assert isinstance(view.memory_handle, int)


# ── Attribute JSON ────────────────────────────────────────────────────────


class TestAttributeJson:
    def test_json_roundtrip(self):
        vals = [AttributeValue.string("test", confidence=0.5)]
        attr = Attribute.persistent("myns", "myname", vals, hint="myhint")
        j = attr.json
        parsed = json.loads(j)
        assert parsed is not None
        attr2 = Attribute.from_json(j)
        assert attr2.namespace == "myns"
        assert attr2.name == "myname"
        assert attr2.hint == "myhint"


# ── AttributeUpdatePolicy ────────────────────────────────────────────────


class TestAttributeUpdatePolicy:
    def test_variants(self):
        assert AttributeUpdatePolicy.ReplaceWithForeignWhenDuplicate is not None
        assert AttributeUpdatePolicy.KeepOwnWhenDuplicate is not None
        assert AttributeUpdatePolicy.ErrorWhenDuplicate is not None
