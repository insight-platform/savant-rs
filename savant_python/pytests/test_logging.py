"""Tests for savant_rs.logging – LogLevel, set/get_log_level,
log_level_enabled, log."""

from __future__ import annotations

from savant_rs.logging import LogLevel, get_log_level, log, log_level_enabled, set_log_level


# ── LogLevel enum ─────────────────────────────────────────────────────────


class TestLogLevel:
    def test_variants(self):
        assert LogLevel.Trace is not None
        assert LogLevel.Debug is not None
        assert LogLevel.Info is not None
        assert LogLevel.Warning is not None
        assert LogLevel.Error is not None
        assert LogLevel.Off is not None


# ── set / get / enabled ───────────────────────────────────────────────────


class TestLogLevelControl:
    def test_set_and_get(self):
        old = set_log_level(LogLevel.Info)
        assert isinstance(old, LogLevel)
        current = get_log_level()
        assert current == LogLevel.Info

    def test_log_level_enabled(self):
        set_log_level(LogLevel.Info)
        assert log_level_enabled(LogLevel.Error)

    def test_set_off(self):
        set_log_level(LogLevel.Off)
        assert get_log_level() == LogLevel.Off


# ── log function ──────────────────────────────────────────────────────────


class TestLog:
    def test_log_simple(self):
        # Should not raise
        log(LogLevel.Info, "test_target", "hello from test")

    def test_log_with_params(self):
        log(
            LogLevel.Debug,
            "test_target",
            "param test",
            params={"key": "value"},
        )

    def test_log_no_params(self):
        log(LogLevel.Warning, "test_target", "warning msg", params=None)
