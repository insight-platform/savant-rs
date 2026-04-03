"""Type aliases for the multi-stream decoder.

At runtime these classes are registered on ``savant_rs.deepstream`` by the
native extension (there is no ``savant_rs.deepstream.inputs`` submodule).
Import from ``savant_rs.deepstream`` in application code.

This file exists so tooling can reference a dedicated path if desired.
"""

from __future__ import annotations

from savant_rs.deepstream import (  # noqa: F401
    DecoderOutput,
    EvictionVerdict,
    MultiStreamDecoder,
    MultiStreamDecoderConfig,
    StopReason,
    SubmitResult,
    UndecodedReason,
)

__all__ = [
    "DecoderOutput",
    "EvictionVerdict",
    "MultiStreamDecoder",
    "MultiStreamDecoderConfig",
    "StopReason",
    "SubmitResult",
    "UndecodedReason",
]
