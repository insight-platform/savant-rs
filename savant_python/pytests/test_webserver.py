"""Tests for savant_rs.webserver – shutdown token and signal helpers."""

from __future__ import annotations

from savant_rs.webserver import is_shutdown_set, set_shutdown_token


# ── Webserver helpers ─────────────────────────────────────────────────────


class TestWebserverHelpers:
    def test_is_shutdown_set_default(self):
        result = is_shutdown_set()
        assert isinstance(result, bool)

    def test_set_shutdown_token(self):
        # Should not raise
        set_shutdown_token("test-token")
