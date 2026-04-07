"""E2E tests for ``savant_rs.deepstream`` multi-stream decoder bindings.

Submits real frames through ``MultiStreamDecoder`` and asserts callback events.
Requires the deepstream feature and a working CUDA/DeepStream runtime.
"""

from __future__ import annotations

import time
from queue import Empty, Queue

import pytest

from conftest import HAS_DS_RUNTIME

_ds = pytest.importorskip("savant_rs.deepstream", reason="deepstream feature not built")
if not hasattr(_ds, "MultiStreamDecoder"):
    pytest.skip(
        "multistream decoder API not in this savant_rs build; reinstall with deepstream feature",
        allow_module_level=True,
    )

if not HAS_DS_RUNTIME:
    pytest.skip(
        "CUDA/DeepStream runtime not available (see conftest HAS_DS_RUNTIME)",
        allow_module_level=True,
    )


def _drain_until_stream_stopped(
    q: Queue,
    *,
    expect_source_id: str,
    min_decoded: int = 1,
    timeout_sec: float = 120.0,
) -> tuple[int, bool]:
    """Pull ``DecoderOutput`` events until ``StreamStopped`` (EOS). Returns (decoded_count, saw_eos)."""
    deadline = time.monotonic() + timeout_sec
    decoded = 0
    saw_eos = False
    while True:
        wait = deadline - time.monotonic()
        if wait <= 0:
            pytest.fail(
                f"timeout waiting for StreamStopped after {decoded} Decoded, eos={saw_eos}"
            )
        try:
            o = q.get(timeout=min(wait, 1.0))
        except Empty:
            continue
        if o.is_stream_started():
            sid, _codec = o.as_stream_started()
            assert sid == expect_source_id
        elif o.is_decoded():
            decoded += 1
            out_frame, _buf = o.as_decoded()
            assert out_frame.source_id == expect_source_id
        elif o.is_eos():
            saw_eos = True
            assert o.as_eos() == expect_source_id
        elif o.is_stream_stopped():
            sid, reason = o.as_stream_stopped()
            assert sid == expect_source_id
            assert reason.is_eos(), f"expected StopReason Eos, got {reason!r}"
            break
        elif o.is_undecoded():
            uf, _data, reason = o.as_undecoded()
            pytest.fail(f"unexpected Undecoded {reason!r} for frame {uf.source_id!r}")
        elif o.is_pipeline_restarted():
            pytest.fail(f"unexpected PipelineRestarted: {o!r}")
        else:
            pytest.fail(f"unexpected DecoderOutput: {o!r}")
    assert decoded >= min_decoded, f"expected ≥{min_decoded} Decoded, got {decoded}"
    assert saw_eos, "expected an Eos event before StreamStopped"
    return decoded, saw_eos


def test_multistream_decoder_raw_rgba_submit_eos_e2e() -> None:
    """One raw RGBA frame + EOS → Decoded RGBA buffer + Eos + StreamStopped(Eos)."""
    from savant_rs.deepstream import (
        Interpolation,
        MultiStreamDecoder,
        MultiStreamDecoderConfig,
        init_cuda,
    )
    from savant_rs.primitives import (
        EndOfStream,
        VideoFrame,
        VideoFrameCodec,
        VideoFrameContent,
        VideoFrameTranscodingMethod,
    )

    init_cuda(0)

    cfg = MultiStreamDecoderConfig(
        0,
        4,
        interpolation=Interpolation.NEAREST,
        idle_timeout_ms=600_000,
    )
    assert cfg.gpu_id == 0
    assert cfg.output_pool_size == 4
    assert cfg.interpolation == Interpolation.NEAREST

    out_q: Queue = Queue()

    def on_output(o: object) -> None:
        out_q.put(o)

    decoder = MultiStreamDecoder(cfg, on_output)

    w, h = 64, 64
    payload = bytes(w * h * 4)  # black RGBA; CPU→GPU upload path (no compressed codec)

    frame = VideoFrame(
        "e2e_raw_rgba",
        (30, 1),
        w,
        h,
        VideoFrameContent.none(),
        VideoFrameTranscodingMethod.Copy,
        VideoFrameCodec.RawRgba,
        None,
        (1, 1_000_000_000),
        0,
        0,
        33_333_333,
    )

    decoder.submit(frame, payload, timeout_ms=60_000)
    assert decoder.submit_eos(EndOfStream("e2e_raw_rgba"), timeout_ms=60_000) is True

    decoded, _eos = _drain_until_stream_stopped(
        out_q,
        expect_source_id="e2e_raw_rgba",
        min_decoded=1,
        timeout_sec=120.0,
    )
    assert decoded == 1

    decoder.shutdown()
