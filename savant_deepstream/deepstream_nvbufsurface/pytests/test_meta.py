"""Tests for SavantIdMeta Python bindings."""

from __future__ import annotations

import pytest

from deepstream_nvbufsurface import (
    NvBufSurfaceGenerator,
    get_savant_id_meta,
)


def _make_gen(fmt: str = "RGBA", w: int = 320, h: int = 240) -> NvBufSurfaceGenerator:
    return NvBufSurfaceGenerator(fmt, w, h, pool_size=2)


class TestGetSavantIdMeta:
    def test_no_meta_returns_empty(self):
        gen = _make_gen()
        buf = gen.acquire_surface()
        meta = get_savant_id_meta(buf)
        assert meta == []

    def test_frame_meta_present(self):
        gen = _make_gen()
        buf = gen.acquire_surface(id=42)
        meta = get_savant_id_meta(buf)
        assert len(meta) >= 1
        kinds = {kind for kind, _ in meta}
        assert "frame" in kinds
        ids = [v for kind, v in meta if kind == "frame"]
        assert 42 in ids

    def test_different_ids(self):
        gen = _make_gen()
        buf1 = gen.acquire_surface(id=100)
        buf2 = gen.acquire_surface(id=200)
        meta1 = get_savant_id_meta(buf1)
        meta2 = get_savant_id_meta(buf2)
        ids1 = [v for _, v in meta1]
        ids2 = [v for _, v in meta2]
        assert 100 in ids1
        assert 200 in ids2

    def test_null_ptr_raises(self):
        with pytest.raises(ValueError, match="null"):
            get_savant_id_meta(0)
