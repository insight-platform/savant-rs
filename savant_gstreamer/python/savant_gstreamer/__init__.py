"""savant_gstreamer — GStreamer utilities for Savant pipelines.

Provides the :class:`Codec` enum and :class:`Mp4Muxer` for muxing encoded
video bitstreams into MP4 containers.

Quick start::

    from savant_gstreamer import Codec, Mp4Muxer

    muxer = Mp4Muxer(Codec.HEVC, "/tmp/out.mp4", fps_num=30)
    muxer.push(encoded_bytes, pts_ns=0, duration_ns=33_333_333)
    muxer.finish()
"""

from __future__ import annotations

from savant_gstreamer._native import (  # noqa: F401
    Codec,
    Mp4Muxer,
)

__all__ = [
    "Codec",
    "Mp4Muxer",
]
