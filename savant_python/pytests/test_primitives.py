"""Tests for savant_rs.primitives – EndOfStream, Shutdown, UserData."""

from __future__ import annotations

import json


from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    EndOfStream,
    Shutdown,
    UserData,
)


# ── EndOfStream ───────────────────────────────────────────────────────────


class TestEndOfStream:
    def test_create(self):
        eos = EndOfStream("cam-1")
        assert eos.source_id == "cam-1"

    def test_json(self):
        eos = EndOfStream("cam-2")
        j = eos.json
        parsed = json.loads(j)
        assert parsed is not None

    def test_to_message(self):
        eos = EndOfStream("cam-3")
        msg = eos.to_message()
        assert msg.is_end_of_stream()
        assert not msg.is_video_frame()


# ── Shutdown ──────────────────────────────────────────────────────────────


class TestShutdown:
    def test_create(self):
        sd = Shutdown("secret-token")
        assert sd.auth == "secret-token"

    def test_json(self):
        sd = Shutdown("tok")
        j = sd.json
        parsed = json.loads(j)
        assert parsed is not None

    def test_to_message(self):
        sd = Shutdown("tok")
        msg = sd.to_message()
        assert msg.is_shutdown()


# ── UserData ──────────────────────────────────────────────────────────────


class TestUserData:
    def test_create(self):
        ud = UserData("source-1")
        assert ud.source_id == "source-1"

    def test_json(self):
        ud = UserData("s1")
        j = ud.json
        parsed = json.loads(j)
        assert parsed is not None

    def test_json_pretty(self):
        ud = UserData("s1")
        jp = ud.json_pretty
        assert len(jp) > 0

    def test_to_message(self):
        ud = UserData("s1")
        msg = ud.to_message()
        assert msg.is_user_data()

    def test_set_get_attribute(self):
        ud = UserData("s1")
        attr = Attribute.persistent(
            "ns", "key", [AttributeValue.string("val")], hint="h"
        )
        old = ud.set_attribute(attr)
        assert old is None
        fetched = ud.get_attribute("ns", "key")
        assert fetched is not None
        assert fetched.namespace == "ns"

    def test_delete_attribute(self):
        ud = UserData("s1")
        ud.set_attribute(Attribute.persistent("ns", "key", [AttributeValue.integer(1)]))
        deleted = ud.delete_attribute("ns", "key")
        assert deleted is not None
        assert ud.get_attribute("ns", "key") is None

    def test_attributes_list(self):
        ud = UserData("s1")
        ud.set_attribute(Attribute.persistent("ns1", "a", [AttributeValue.string("x")]))
        ud.set_attribute(Attribute.persistent("ns2", "b", [AttributeValue.string("y")]))
        attrs = ud.attributes
        assert len(attrs) == 2

    def test_find_attributes_with_ns(self):
        ud = UserData("s1")
        ud.set_attribute(Attribute.persistent("ns1", "a", [AttributeValue.integer(1)]))
        ud.set_attribute(Attribute.persistent("ns1", "b", [AttributeValue.integer(2)]))
        ud.set_attribute(Attribute.persistent("ns2", "c", [AttributeValue.integer(3)]))
        found = ud.find_attributes_with_ns("ns1")
        assert len(found) == 2

    def test_find_attributes_with_names(self):
        ud = UserData("s1")
        ud.set_attribute(
            Attribute.persistent("ns", "alpha", [AttributeValue.integer(1)])
        )
        ud.set_attribute(
            Attribute.persistent("ns", "beta", [AttributeValue.integer(2)])
        )
        found = ud.find_attributes_with_names(["alpha"])
        assert len(found) >= 1

    def test_find_attributes_with_hints(self):
        ud = UserData("s1")
        ud.set_attribute(
            Attribute.persistent("ns", "a", [AttributeValue.integer(1)], hint="special")
        )
        ud.set_attribute(Attribute.persistent("ns", "b", [AttributeValue.integer(2)]))
        found = ud.find_attributes_with_hints(["special"])
        assert len(found) >= 1

    def test_delete_attributes_with_ns(self):
        ud = UserData("s1")
        ud.set_attribute(Attribute.persistent("ns1", "a", [AttributeValue.integer(1)]))
        ud.set_attribute(Attribute.persistent("ns2", "b", [AttributeValue.integer(2)]))
        ud.delete_attributes_with_ns("ns1")
        assert ud.get_attribute("ns1", "a") is None
        assert ud.get_attribute("ns2", "b") is not None

    def test_delete_attributes_with_names(self):
        ud = UserData("s1")
        ud.set_attribute(Attribute.persistent("ns", "a", [AttributeValue.integer(1)]))
        ud.set_attribute(Attribute.persistent("ns", "b", [AttributeValue.integer(2)]))
        ud.delete_attributes_with_names(["a"])
        assert ud.get_attribute("ns", "a") is None
        assert ud.get_attribute("ns", "b") is not None

    def test_delete_attributes_with_hints(self):
        ud = UserData("s1")
        ud.set_attribute(
            Attribute.persistent(
                "ns", "a", [AttributeValue.integer(1)], hint="remove-me"
            )
        )
        ud.set_attribute(Attribute.persistent("ns", "b", [AttributeValue.integer(2)]))
        ud.delete_attributes_with_hints(["remove-me"])
        assert ud.get_attribute("ns", "a") is None
        assert ud.get_attribute("ns", "b") is not None

    def test_clear_attributes(self):
        ud = UserData("s1")
        ud.set_attribute(Attribute.persistent("ns", "a", [AttributeValue.integer(1)]))
        ud.clear_attributes()
        assert len(ud.attributes) == 0

    def test_set_persistent_attribute(self):
        ud = UserData("s1")
        ud.set_persistent_attribute("ns", "key", False, "hint", [])
        attr = ud.get_attribute("ns", "key")
        assert attr is not None

    def test_set_temporary_attribute(self):
        ud = UserData("s1")
        ud.set_temporary_attribute("ns", "key", False, None, [])
        attr = ud.get_attribute("ns", "key")
        assert attr is not None
        assert attr.is_temporary()

    def test_protobuf_roundtrip(self):
        ud = UserData("s1")
        ud.set_attribute(
            Attribute.persistent("ns", "a", [AttributeValue.string("hello")])
        )
        data = ud.to_protobuf()
        assert isinstance(data, bytes)
        ud2 = UserData.from_protobuf(data)
        assert ud2.source_id == "s1"
        assert ud2.get_attribute("ns", "a") is not None
