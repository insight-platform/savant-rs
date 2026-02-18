"""Tests for savant_rs.gstreamer – FlowResult and InvocationReason enums."""

from __future__ import annotations

from savant_rs.gstreamer import FlowResult, InvocationReason


# ── FlowResult ───────────────────────────────────────────────────────────


class TestFlowResult:
    def test_all_variants(self):
        for name in (
            "CustomSuccess2",
            "CustomSuccess1",
            "CustomSuccess",
            "Ok",
            "NotLinked",
            "Flushing",
            "Eos",
            "NotNegotiated",
            "Error",
            "NotSupported",
            "CustomError",
            "CustomError1",
            "CustomError2",
        ):
            assert getattr(FlowResult, name) is not None


# ── InvocationReason ─────────────────────────────────────────────────────


class TestInvocationReason:
    def test_all_variants(self):
        for name in (
            "Buffer",
            "SinkEvent",
            "SourceEvent",
            "StateChange",
            "IngressMessageTransformer",
        ):
            assert getattr(InvocationReason, name) is not None
