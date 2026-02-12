"""Type stubs for the deepstream_encoders package."""

from __future__ import annotations

from deepstream_encoders._native import (
    # Core types
    EncodedFrame as EncodedFrame,
    EncoderConfig as EncoderConfig,
    NvEncoder as NvEncoder,
    # Property enums
    Platform as Platform,
    RateControl as RateControl,
    H264Profile as H264Profile,
    HevcProfile as HevcProfile,
    DgpuPreset as DgpuPreset,
    TuningPreset as TuningPreset,
    JetsonPresetLevel as JetsonPresetLevel,
    # Property structs
    H264DgpuProps as H264DgpuProps,
    HevcDgpuProps as HevcDgpuProps,
    H264JetsonProps as H264JetsonProps,
    HevcJetsonProps as HevcJetsonProps,
    JpegProps as JpegProps,
    Av1DgpuProps as Av1DgpuProps,
)

__all__: list[str]
