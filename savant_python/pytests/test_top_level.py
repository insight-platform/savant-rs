"""Tests for top-level savant_rs functions – version, register_handler,
unregister_handler."""

from __future__ import annotations

import savant_rs


class TestVersion:
    def test_returns_string(self):
        v = savant_rs.version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_looks_like_semver(self):
        v = savant_rs.version()
        parts = v.split(".")
        assert len(parts) >= 2


class TestHandlerRegistration:
    def test_register_and_unregister(self):
        class DummyHandler:
            pass

        handler = DummyHandler()
        savant_rs.register_handler("test_element", handler)
        savant_rs.unregister_handler("test_element")

    def test_unregister_nonexistent(self):
        # Should not raise even if nothing is registered
        savant_rs.unregister_handler("nonexistent_element")
