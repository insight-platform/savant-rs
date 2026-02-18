"""Tests for savant_rs.telemetry – configuration classes, enums, init and
shutdown."""

from __future__ import annotations

from savant_rs.telemetry import (
    ClientTlsConfig,
    ContextPropagationFormat,
    Identity,
    Protocol,
    TelemetryConfiguration,
    TracerConfiguration,
    init,
    shutdown,
)


# ── ContextPropagationFormat ─────────────────────────────────────────────


class TestContextPropagationFormat:
    def test_variants(self):
        assert ContextPropagationFormat.Jaeger is not None
        assert ContextPropagationFormat.W3C is not None


# ── Protocol ─────────────────────────────────────────────────────────────


class TestProtocol:
    def test_variants(self):
        assert Protocol.Grpc is not None
        assert Protocol.HttpBinary is not None
        assert Protocol.HttpJson is not None


# ── Identity ─────────────────────────────────────────────────────────────


class TestIdentity:
    def test_create(self):
        ident = Identity(key="key-data", certificate="cert-data")
        assert ident is not None


# ── ClientTlsConfig ──────────────────────────────────────────────────────


class TestClientTlsConfig:
    def test_no_args(self):
        tls = ClientTlsConfig()
        assert tls is not None

    def test_with_certificate(self):
        tls = ClientTlsConfig(certificate="ca-cert")
        assert tls is not None

    def test_with_identity(self):
        ident = Identity(key="k", certificate="c")
        tls = ClientTlsConfig(identity=ident)
        assert tls is not None


# ── TracerConfiguration ──────────────────────────────────────────────────


class TestTracerConfiguration:
    def test_create(self):
        tc = TracerConfiguration(
            service_name="test-service",
            protocol=Protocol.Grpc,
            endpoint="http://localhost:4317",
        )
        assert tc is not None

    def test_with_tls_and_timeout(self):
        tls = ClientTlsConfig(certificate="cert")
        tc = TracerConfiguration(
            service_name="svc",
            protocol=Protocol.HttpBinary,
            endpoint="https://otel:4318",
            tls=tls,
            timeout=5000,
        )
        assert tc is not None


# ── TelemetryConfiguration ──────────────────────────────────────────────


class TestTelemetryConfiguration:
    def test_no_op(self):
        cfg = TelemetryConfiguration.no_op()
        assert cfg is not None

    def test_custom(self):
        cfg = TelemetryConfiguration(
            context_propagation_format=ContextPropagationFormat.W3C,
        )
        assert cfg is not None

    def test_defaults(self):
        cfg = TelemetryConfiguration()
        assert cfg is not None


# ── init / shutdown ──────────────────────────────────────────────────────


class TestInitShutdown:
    def test_init_noop_and_shutdown(self):
        cfg = TelemetryConfiguration.no_op()
        init(cfg)
        shutdown()
