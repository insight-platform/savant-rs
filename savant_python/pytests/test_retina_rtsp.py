"""Integration test for savant_rs.retina_rtsp — GStreamer backend.

Starts the RetinaRtspService against public RTSP test streams and verifies
that at least 100 video frames per source are received via a ZeroMQ reader.

Requires:
  - savant_rs built with the ``gst`` feature
  - GStreamer runtime with rtspsrc / h264 depay+parse available
  - network access to rtsp://hello.savant.video
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import defaultdict

import pytest

from savant_rs.py.log import init_logging, get_logger

init_logging()

logger = get_logger(__name__)

# ── Feature detection ─────────────────────────────────────────────────

try:
    from savant_rs.retina_rtsp import (
        RetinaRtspService,
        RtspBackend,
        RtspSource,
        RtspSourceGroup,
        SyncConfiguration,
    )

    HAS_RETINA_RTSP = True
except ImportError:
    HAS_RETINA_RTSP = False

skip_no_retina_rtsp = pytest.mark.skipif(
    not HAS_RETINA_RTSP,
    reason="savant_rs built without gst feature (retina_rtsp unavailable)",
)

# ── Constants ─────────────────────────────────────────────────────────

SOURCES = [
    {
        "source_id": "city-traffic",
        "url": "rtsp://hello.savant.video:8554/stream/city-traffic",
    },
    {
        "source_id": "town-centre",
        "url": "rtsp://hello.savant.video:8554/stream/town-centre",
    },
]

MIN_FRAMES_PER_SOURCE = 100

# Use a random high port to avoid conflicts with other tests / services.
ZMQ_PORT = 13399
ZMQ_WRITER_URL = f"pub+connect:tcp://127.0.0.1:{ZMQ_PORT}"
ZMQ_READER_URL = f"sub+bind:tcp://127.0.0.1:{ZMQ_PORT}"


# ── Helpers ───────────────────────────────────────────────────────────


def _write_config(tmp_dir: str) -> str:
    """Write a minimal JSON configuration file and return its path."""
    config = {
        "sink": {
            "url": ZMQ_WRITER_URL,
            "options": {
                "send_timeout": {"secs": 1, "nanos": 0},
                "send_retries": 3,
                "receive_timeout": {"secs": 1, "nanos": 0},
                "receive_retries": 3,
                "send_hwm": 1000,
                "receive_hwm": 1000,
                "inflight_ops": 100,
            },
        },
        "rtsp_sources": {},
        "reconnect_interval": {"secs": 5, "nanos": 0},
    }
    path = os.path.join(tmp_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f)
    return path


# ── Test ──────────────────────────────────────────────────────────────


@skip_no_retina_rtsp
def test_receive_frames_from_rtsp_sources():
    """Start the service, receive >=100 frames per source via ZMQ reader."""
    from savant_rs.zmq import (
        ReaderConfigBuilder,
        ReaderResultMessage,
        ReaderResultTimeout,
        TopicPrefixSpec,
    )

    # 1. Set up the ZMQ reader FIRST (it binds, writer connects).
    reader_cfg = ReaderConfigBuilder(ZMQ_READER_URL)
    reader_cfg.with_receive_timeout(1000)
    reader_cfg.with_receive_hwm(5000)
    reader_cfg.with_topic_prefix_spec(TopicPrefixSpec.none())
    from savant_rs.zmq import BlockingReader

    reader = BlockingReader(reader_cfg.build())
    reader.start()

    # 2. Write config and create the service.
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = _write_config(tmp_dir)
        service = RetinaRtspService(config_path)

        # 3. Build the group programmatically (mirrors the JSON config).
        group = RtspSourceGroup(
            sources=[RtspSource(s["source_id"], s["url"]) for s in SOURCES],
            backend=RtspBackend.Gstreamer,
            rtcp_sr_sync=SyncConfiguration(
                group_window_duration_ms=5000,
                batch_duration_ms=100,
                network_skew_correction=False,
                rtcp_once=False,
            ),
        )

        # 4. Run the group in a background thread.
        group_thread = threading.Thread(
            target=service.run_group,
            args=(group, "test-group"),
            daemon=True,
        )
        group_thread.start()

        # 5. Receive frames until we have enough from each source.
        frame_counts: dict[str, int] = defaultdict(int)
        expected_sources = {s["source_id"] for s in SOURCES}
        deadline = time.monotonic() + 120  # 2-minute timeout

        try:
            while time.monotonic() < deadline:
                result = reader.receive()
                if isinstance(result, ReaderResultTimeout):
                    continue
                if not isinstance(result, ReaderResultMessage):
                    continue

                msg = result.message
                if msg.is_video_frame():
                    vf = msg.as_video_frame()
                    if vf is not None:
                        sid = vf.source_id
                        frame_counts[sid] += 1

                # Check if we have enough from every source.
                if all(
                    frame_counts.get(sid, 0) >= MIN_FRAMES_PER_SOURCE
                    for sid in expected_sources
                ):
                    break
        finally:
            # 6. Tear down.
            service.stop_group("test-group")
            group_thread.join(timeout=10)
            reader.shutdown()

    # 7. Assert.
    for sid in expected_sources:
        count = frame_counts.get(sid, 0)
        assert count >= MIN_FRAMES_PER_SOURCE, (
            f"Source '{sid}' received only {count} frames "
            f"(expected >= {MIN_FRAMES_PER_SOURCE})"
        )
