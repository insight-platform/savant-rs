"""Tests for savant_rs.utils.serialization – Message and save/load
functions."""

from __future__ import annotations

import pytest

from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    EndOfStream,
    Shutdown,
    UserData,
    VideoFrame,
    VideoFrameBatch,
    VideoFrameContent,
    VideoFrameUpdate,
    VideoObject,
)
from savant_rs.primitives.geometry import RBBox
from savant_rs.utils import ByteBuffer
from savant_rs.utils.serialization import (
    Message,
    load_message,
    load_message_from_bytebuffer,
    load_message_from_bytes,
    save_message,
    save_message_to_bytebuffer,
    save_message_to_bytes,
)


# ── helpers ───────────────────────────────────────────────────────────────


def _make_frame():
    return VideoFrame(
        source_id="cam",
        framerate="30/1",
        width=640,
        height=480,
        content=VideoFrameContent.none(),
    )


# ── Message factory methods & type checks ────────────────────────────────


class TestMessageTypes:
    def test_unknown(self):
        msg = Message.unknown("hello")
        assert msg.is_unknown()
        assert not msg.is_video_frame()
        assert not msg.is_shutdown()

    def test_shutdown(self):
        msg = Message.shutdown(Shutdown("tok"))
        assert msg.is_shutdown()
        sd = msg.as_shutdown()
        assert sd is not None
        assert sd.auth == "tok"

    def test_end_of_stream(self):
        msg = Message.end_of_stream(EndOfStream("cam"))
        assert msg.is_end_of_stream()
        eos = msg.as_end_of_stream()
        assert eos is not None
        assert eos.source_id == "cam"

    def test_user_data(self):
        ud = UserData("src")
        ud.set_attribute(
            Attribute.persistent("ns", "k", [AttributeValue.string("v")])
        )
        msg = Message.user_data(ud)
        assert msg.is_user_data()
        extracted = msg.as_user_data()
        assert extracted is not None
        assert extracted.source_id == "src"

    def test_video_frame(self):
        f = _make_frame()
        msg = Message.video_frame(f)
        assert msg.is_video_frame()
        extracted = msg.as_video_frame()
        assert extracted is not None
        assert extracted.source_id == "cam"

    def test_video_frame_update(self):
        u = VideoFrameUpdate()
        u.add_frame_attribute(
            Attribute.persistent("ns", "k", [AttributeValue.integer(1)])
        )
        msg = Message.video_frame_update(u)
        assert msg.is_video_frame_update()
        extracted = msg.as_video_frame_update()
        assert extracted is not None

    def test_video_frame_batch(self):
        batch = VideoFrameBatch()
        batch.add(0, _make_frame())
        msg = Message.video_frame_batch(batch)
        assert msg.is_video_frame_batch()
        extracted = msg.as_video_frame_batch()
        assert extracted is not None


# ── Message metadata ─────────────────────────────────────────────────────


class TestMessageMetadata:
    def test_system_id(self):
        msg = Message.unknown("test")
        msg.system_id = "sys-123"
        assert msg.system_id == "sys-123"

    def test_seq_id(self):
        msg = Message.unknown("test")
        msg.seq_id = 42
        assert msg.seq_id == 42

    def test_labels(self):
        msg = Message.unknown("test")
        msg.labels = ["label1", "label2"]
        assert msg.labels == ["label1", "label2"]


# ── Cross-type as_* returns None ─────────────────────────────────────────


class TestMessageMismatch:
    def test_unknown_as_video_frame(self):
        msg = Message.unknown("x")
        assert msg.as_video_frame() is None

    def test_eos_as_shutdown(self):
        msg = Message.end_of_stream(EndOfStream("cam"))
        assert msg.as_shutdown() is None

    def test_shutdown_as_user_data(self):
        msg = Message.shutdown(Shutdown("tok"))
        assert msg.as_user_data() is None


# ── save / load round-trips ──────────────────────────────────────────────


class TestSaveLoadMessage:
    def test_save_load(self):
        msg = Message.end_of_stream(EndOfStream("cam"))
        data = save_message(msg)
        assert isinstance(data, bytes)
        msg2 = load_message(data)
        assert msg2.is_end_of_stream()

    def test_save_load_bytebuffer(self):
        msg = Message.video_frame(_make_frame())
        buf = save_message_to_bytebuffer(msg)
        assert isinstance(buf, ByteBuffer)
        assert buf.len() > 0
        msg2 = load_message_from_bytebuffer(buf)
        assert msg2.is_video_frame()

    def test_save_load_bytes(self):
        msg = Message.shutdown(Shutdown("tok"))
        data = save_message_to_bytes(msg)
        assert isinstance(data, bytes)
        msg2 = load_message_from_bytes(data)
        assert msg2.is_shutdown()

    def test_roundtrip_with_objects(self):
        f = _make_frame()
        f.create_object("ns", "obj", detection_box=RBBox(0, 0, 10, 10))
        msg = Message.video_frame(f)
        data = save_message_to_bytes(msg)
        msg2 = load_message_from_bytes(data)
        f2 = msg2.as_video_frame()
        assert f2 is not None
        assert len(f2.get_all_objects()) == 1
