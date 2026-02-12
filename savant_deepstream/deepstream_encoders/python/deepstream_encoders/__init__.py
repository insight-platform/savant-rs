"""deepstream_encoders — GPU-accelerated video encoders for DeepStream.

Provides a high-level Python API for hardware-accelerated video encoding
(H.264, HEVC, JPEG, AV1) backed by NVIDIA DeepStream's NvBufSurface
buffer pool and NVENC/NVJPEG encoders.

Quick start::

    from deepstream_nvbufsurface import init_cuda
    from deepstream_encoders import NvEncoder, EncoderConfig, Codec

    init_cuda()

    config = EncoderConfig(Codec.HEVC, 1920, 1080)
    encoder = NvEncoder(config)

    for i in range(100):
        buf = encoder.acquire_surface(id=i)
        encoder.submit_frame(buf, frame_id=i,
                             pts_ns=i * 33_333_333,
                             duration_ns=33_333_333)

    remaining = encoder.finish()
    for frame in remaining:
        print(f"frame {frame.frame_id}: {frame.size} bytes")
"""

from __future__ import annotations

# ── Native (Rust/PyO3) symbols ───────────────────────────────────────────
# Codec is registered in deepstream_encoders._native (same underlying Rust
# type as savant_gstreamer.Codec). Importing from _native ensures that
# config.codec == Codec.HEVC works within this package. Cross-package
# Codec instances are handled by the extract_codec() helper in the Rust
# layer (fallback via .name()).
from deepstream_encoders._native import (  # noqa: F401
    Codec,
    EncodedFrame,
    EncoderConfig,
    MemType,
    NvEncoder,
    VideoFormat,
)

__all__ = [
    "Codec",
    "EncoderConfig",
    "EncodedFrame",
    "MemType",
    "NvEncoder",
    "VideoFormat",
]
