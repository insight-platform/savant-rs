"""Tests for savant_rs.webserver.kvs – key-value store attribute operations."""

from __future__ import annotations

import pytest

from savant_rs.primitives.attribute import Attribute
from savant_rs.primitives.attribute_value import AttributeValue
from savant_rs.webserver.kvs import (
    del_attribute,
    del_attributes,
    deserialize_attributes,
    get_attribute,
    search_attributes,
    search_keys,
    serialize_attributes,
    set_attributes,
)


@pytest.fixture(autouse=True)
def _clean_kvs():
    """Clear all KVS attributes before and after each test."""
    del_attributes(None, None, False)
    yield
    del_attributes(None, None, False)


# ── set / get ─────────────────────────────────────────────────────────────


class TestSetGet:
    def test_set_and_get(self):
        attr = Attribute.persistent("ns", "key", [AttributeValue.string("value")])
        set_attributes([attr], ttl=None)
        result = get_attribute("ns", "key")
        assert result is not None
        assert result.namespace == "ns"
        assert result.name == "key"

    def test_get_nonexistent(self):
        result = get_attribute("nope", "nope")
        assert result is None

    def test_set_multiple(self):
        attrs = [
            Attribute.persistent("ns", "a", [AttributeValue.integer(1)]),
            Attribute.persistent("ns", "b", [AttributeValue.integer(2)]),
        ]
        set_attributes(attrs, ttl=None)
        assert get_attribute("ns", "a") is not None
        assert get_attribute("ns", "b") is not None


# ── search ────────────────────────────────────────────────────────────────


class TestSearch:
    def test_search_attributes_by_ns(self):
        set_attributes(
            [
                Attribute.persistent("ns1", "a", [AttributeValue.integer(1)]),
                Attribute.persistent("ns2", "b", [AttributeValue.integer(2)]),
            ],
            ttl=None,
        )
        results = search_attributes("ns1", None, False)
        assert len(results) == 1

    def test_search_attributes_all(self):
        set_attributes(
            [
                Attribute.persistent("ns1", "a", [AttributeValue.integer(1)]),
                Attribute.persistent("ns2", "b", [AttributeValue.integer(2)]),
            ],
            ttl=None,
        )
        results = search_attributes(None, None, False)
        assert len(results) == 2

    def test_search_keys(self):
        set_attributes(
            [Attribute.persistent("ns", "key", [AttributeValue.integer(1)])],
            ttl=None,
        )
        keys = search_keys(None, None, False)
        assert len(keys) >= 1


# ── delete ────────────────────────────────────────────────────────────────


class TestDelete:
    def test_del_attribute(self):
        set_attributes(
            [Attribute.persistent("ns", "k", [AttributeValue.integer(1)])],
            ttl=None,
        )
        deleted = del_attribute("ns", "k")
        assert deleted is not None
        assert get_attribute("ns", "k") is None

    def test_del_attribute_nonexistent(self):
        deleted = del_attribute("nope", "nope")
        assert deleted is None

    def test_del_attributes_by_ns(self):
        set_attributes(
            [
                Attribute.persistent("ns1", "a", [AttributeValue.integer(1)]),
                Attribute.persistent("ns2", "b", [AttributeValue.integer(2)]),
            ],
            ttl=None,
        )
        del_attributes("ns1", None, False)
        assert get_attribute("ns1", "a") is None
        assert get_attribute("ns2", "b") is not None

    def test_del_attributes_all(self):
        set_attributes(
            [
                Attribute.persistent("ns1", "a", [AttributeValue.integer(1)]),
                Attribute.persistent("ns2", "b", [AttributeValue.integer(2)]),
            ],
            ttl=None,
        )
        del_attributes(None, None, False)
        assert len(search_attributes(None, None, False)) == 0


# ── serialize / deserialize ──────────────────────────────────────────────


class TestSerializeDeserialize:
    def test_roundtrip(self):
        attrs = [
            Attribute.persistent("ns", "a", [AttributeValue.string("x")]),
            Attribute.persistent("ns", "b", [AttributeValue.integer(42)]),
        ]
        data = serialize_attributes(attrs)
        # serialize_attributes returns bytes
        restored = deserialize_attributes(data)
        assert len(restored) == 2
