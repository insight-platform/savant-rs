"""deepstream_encoders — GPU-accelerated video encoders for DeepStream.

Provides a high-level Python API for hardware-accelerated video encoding
(H.264, HEVC, JPEG, AV1) backed by NVIDIA DeepStream's NvBufSurface
buffer pool and NVENC/NVJPEG encoders.

Quick start::

    from deepstream_nvbufsurface import init_cuda, VideoFormat
    from savant_gstreamer import Codec
    from deepstream_encoders import NvEncoder, EncoderConfig
    from deepstream_encoders import HevcDgpuProps, HevcProfile

    init_cuda()

    props = HevcDgpuProps(bitrate=8_000_000, profile=HevcProfile.MAIN)
    config = EncoderConfig(Codec.HEVC, 1920, 1080, properties=props)
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

from deepstream_encoders._native import (  # noqa: F401
    # Core types
    EncodedFrame,
    EncoderConfig,
    NvEncoder,
    # Property enums
    Platform,
    RateControl,
    H264Profile,
    HevcProfile,
    DgpuPreset,
    TuningPreset,
    JetsonPresetLevel,
    # Property structs
    H264DgpuProps,
    HevcDgpuProps,
    H264JetsonProps,
    HevcJetsonProps,
    JpegProps,
    Av1DgpuProps,
)

__all__ = [
    # Core types
    "EncoderConfig",
    "EncodedFrame",
    "NvEncoder",
    # Property enums
    "Platform",
    "RateControl",
    "H264Profile",
    "HevcProfile",
    "DgpuPreset",
    "TuningPreset",
    "JetsonPresetLevel",
    # Property structs
    "H264DgpuProps",
    "HevcDgpuProps",
    "H264JetsonProps",
    "HevcJetsonProps",
    "JpegProps",
    "Av1DgpuProps",
]
