"""Tests for top-level savant_rs functions – version, register_handler,
unregister_handler."""

from __future__ import annotations

import pytest

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

    def test_unregister_nonexistent_raises(self):
        with pytest.raises(Exception):
            savant_rs.unregister_handler("nonexistent_element")

    def test_clear_all_handlers(self):
        class H:
            pass

        savant_rs.register_handler("clear_test_a", H())
        savant_rs.register_handler("clear_test_b", H())
        savant_rs.clear_all_handlers()
        with pytest.raises(Exception):
            savant_rs.unregister_handler("clear_test_a")
