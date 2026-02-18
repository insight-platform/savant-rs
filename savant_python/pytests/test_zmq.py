"""Tests for savant_rs.zmq – socket types, config builders, TopicPrefixSpec,
and result types."""

from __future__ import annotations

from savant_rs.zmq import (
    ReaderConfig,
    ReaderConfigBuilder,
    ReaderResultBlacklisted,
    ReaderResultMessage,
    ReaderResultPrefixMismatch,
    ReaderResultTimeout,
    ReaderSocketType,
    TopicPrefixSpec,
    WriterConfig,
    WriterConfigBuilder,
    WriterResultAck,
    WriterResultAckTimeout,
    WriterResultSendTimeout,
    WriterResultSuccess,
    WriterSocketType,
)


# ── WriterSocketType ─────────────────────────────────────────────────────


class TestWriterSocketType:
    def test_variants(self):
        assert WriterSocketType.Pub is not None
        assert WriterSocketType.Dealer is not None
        assert WriterSocketType.Req is not None


# ── ReaderSocketType ─────────────────────────────────────────────────────


class TestReaderSocketType:
    def test_variants(self):
        assert ReaderSocketType.Sub is not None
        assert ReaderSocketType.Router is not None
        assert ReaderSocketType.Rep is not None


# ── TopicPrefixSpec ──────────────────────────────────────────────────────


class TestTopicPrefixSpec:
    def test_source_id(self):
        t = TopicPrefixSpec.source_id("cam-1")
        assert t is not None

    def test_prefix(self):
        t = TopicPrefixSpec.prefix("my-prefix")
        assert t is not None

    def test_none(self):
        t = TopicPrefixSpec.none()
        assert t is not None


# ── WriterConfigBuilder ──────────────────────────────────────────────────


class TestWriterConfigBuilder:
    def test_build_default(self):
        b = WriterConfigBuilder("tcp://127.0.0.1:5555")
        cfg = b.build()
        assert isinstance(cfg, WriterConfig)
        assert cfg.endpoint == "tcp://127.0.0.1:5555"

    def test_with_timeouts(self):
        b = WriterConfigBuilder("tcp://127.0.0.1:5555")
        b.with_send_timeout(1000)
        b.with_receive_timeout(2000)
        cfg = b.build()
        assert cfg.send_timeout == 1000
        assert cfg.receive_timeout == 2000

    def test_with_retries(self):
        b = WriterConfigBuilder("tcp://127.0.0.1:5555")
        b.with_send_retries(5)
        b.with_receive_retries(3)
        cfg = b.build()
        assert cfg.send_retries == 5
        assert cfg.receive_retries == 3

    def test_with_hwm(self):
        b = WriterConfigBuilder("tcp://127.0.0.1:5555")
        b.with_send_hwm(100)
        b.with_receive_hwm(200)
        cfg = b.build()
        assert cfg.send_hwm == 100
        assert cfg.receive_hwm == 200

    def test_with_fix_ipc_permissions(self):
        b = WriterConfigBuilder("tcp://127.0.0.1:5555")
        b.with_fix_ipc_permissions(0o777)
        cfg = b.build()
        assert cfg.fix_ipc_permissions == 0o777

    def test_fix_ipc_permissions_none(self):
        b = WriterConfigBuilder("tcp://127.0.0.1:5555")
        b.with_fix_ipc_permissions(None)
        cfg = b.build()
        assert cfg.fix_ipc_permissions is None

    def test_config_properties(self):
        b = WriterConfigBuilder("tcp://127.0.0.1:5555")
        cfg = b.build()
        assert cfg.endpoint == "tcp://127.0.0.1:5555"
        assert cfg.socket_type is not None
        assert isinstance(cfg.bind, bool)


# ── ReaderConfigBuilder ──────────────────────────────────────────────────


class TestReaderConfigBuilder:
    def test_build_default(self):
        b = ReaderConfigBuilder("tcp://127.0.0.1:5556")
        cfg = b.build()
        assert isinstance(cfg, ReaderConfig)
        assert cfg.endpoint == "tcp://127.0.0.1:5556"

    def test_with_receive_timeout(self):
        b = ReaderConfigBuilder("tcp://127.0.0.1:5556")
        b.with_receive_timeout(500)
        cfg = b.build()
        assert cfg.receive_timeout == 500

    def test_with_receive_hwm(self):
        b = ReaderConfigBuilder("tcp://127.0.0.1:5556")
        b.with_receive_hwm(300)
        cfg = b.build()
        assert cfg.receive_hwm == 300

    def test_with_topic_prefix_spec(self):
        b = ReaderConfigBuilder("tcp://127.0.0.1:5556")
        b.with_topic_prefix_spec(TopicPrefixSpec.source_id("cam"))
        cfg = b.build()
        assert cfg.topic_prefix_spec is not None

    def test_with_routing_cache_size(self):
        b = ReaderConfigBuilder("tcp://127.0.0.1:5556")
        b.with_routing_cache_size(512)
        cfg = b.build()
        assert cfg.routing_cache_size == 512

    def test_with_fix_ipc_permissions(self):
        b = ReaderConfigBuilder("tcp://127.0.0.1:5556")
        b.with_fix_ipc_permissions(0o666)
        cfg = b.build()
        assert cfg.fix_ipc_permissions == 0o666


# ── Result types exist ───────────────────────────────────────────────────


class TestResultTypes:
    def test_writer_result_types_exist(self):
        assert WriterResultSendTimeout is not None
        assert WriterResultAckTimeout is not None
        assert WriterResultAck is not None
        assert WriterResultSuccess is not None

    def test_reader_result_types_exist(self):
        assert ReaderResultMessage is not None
        assert ReaderResultBlacklisted is not None
        assert ReaderResultTimeout is not None
        assert ReaderResultPrefixMismatch is not None
