"""Tests for DsNvNonUniformSurfaceBuffer (zero-copy, nvstreammux2-style) — require CUDA/DeepStream runtime."""

from __future__ import annotations

import gc

import pytest

from conftest import HAS_DS_FEATURE, skip_no_ds_runtime
from deepstream_helpers import (
    LEAK_ITERATIONS,
    WARMUP_ITERATIONS,
    assert_no_leak,
    cpu_rss_kb,
    gpu_mem_used_mb,
    make_gen,
)

if not HAS_DS_FEATURE:
    pytest.skip("savant_rs built without deepstream feature", allow_module_level=True)

import savant_rs.deepstream as _ds

ds = _ds


@skip_no_ds_runtime
class TestHeterogeneousCreate:
    def test_create_with_defaults(self):
        batch = ds.DsNvNonUniformSurfaceBuffer(4)
        assert batch is not None

    def test_create_batch_size_1(self):
        batch = ds.DsNvNonUniformSurfaceBuffer(1)
        assert batch.max_batch_size == 1

    def test_properties(self):
        batch = ds.DsNvNonUniformSurfaceBuffer(4, gpu_id=0)
        assert batch.max_batch_size == 4
        assert batch.num_filled == 0
        assert batch.gpu_id == 0


@skip_no_ds_runtime
class TestHeterogeneousAdd:
    def test_add_single(self):
        gen = make_gen("RGBA", 640, 480)
        batch = ds.DsNvNonUniformSurfaceBuffer(4)
        buf = gen.acquire_surface(id=1)
        batch.add(buf, id=1)
        assert batch.num_filled == 1

    def test_add_multiple_different_sizes(self):
        gen_small = make_gen("RGBA", 320, 240)
        gen_1080p = make_gen("RGBA", 1920, 1080)
        gen_720p = make_gen("RGBA", 1280, 720)
        batch = ds.DsNvNonUniformSurfaceBuffer(8)
        batch.add(gen_small.acquire_surface(), id=1)
        batch.add(gen_1080p.acquire_surface(), id=2)
        batch.add(gen_720p.acquire_surface(), id=3)
        assert batch.num_filled == 3

    def test_add_different_formats(self):
        gen_rgba = make_gen("RGBA", 640, 480)
        gen_nv12 = make_gen("NV12", 640, 480)
        batch = ds.DsNvNonUniformSurfaceBuffer(4)
        batch.add(gen_rgba.acquire_surface(), id=1)
        batch.add(gen_nv12.acquire_surface(), id=2)
        assert batch.num_filled == 2

    def test_add_exceeds_capacity_raises(self):
        gen = make_gen("RGBA", 640, 480)
        batch = ds.DsNvNonUniformSurfaceBuffer(2)
        batch.add(gen.acquire_surface(), id=1)
        batch.add(gen.acquire_surface(), id=2)
        with pytest.raises(RuntimeError, match="overflow|Batch"):
            batch.add(gen.acquire_surface(), id=3)

    def test_add_null_raises(self):
        batch = ds.DsNvNonUniformSurfaceBuffer(4)
        with pytest.raises(ValueError, match="null"):
            batch.add(0, id=1)


@skip_no_ds_runtime
class TestHeterogeneousSlotPtr:
    def test_returns_4_tuple(self):
        gen_small = make_gen("RGBA", 320, 240)
        gen_1080p = make_gen("RGBA", 1920, 1080)
        batch = ds.DsNvNonUniformSurfaceBuffer(4)
        batch.add(gen_small.acquire_surface(), id=1)
        batch.add(gen_1080p.acquire_surface(), id=2)
        batch.finalize()
        result = batch.slot_ptr(0)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_dimensions_match_source(self):
        gen_small = make_gen("RGBA", 320, 240)
        gen_1080p = make_gen("RGBA", 1920, 1080)
        batch = ds.DsNvNonUniformSurfaceBuffer(4)
        batch.add(gen_small.acquire_surface(), id=1)
        batch.add(gen_1080p.acquire_surface(), id=2)
        batch.finalize()
        data_ptr0, pitch0, w0, h0 = batch.slot_ptr(0)
        data_ptr1, pitch1, w1, h1 = batch.slot_ptr(1)
        assert w0 == 320
        assert h0 == 240
        assert w1 == 1920
        assert h1 == 1080
        assert data_ptr0 != 0
        assert data_ptr1 != 0
        assert data_ptr0 != data_ptr1

    def test_slot_ptr_out_of_bounds_raises(self):
        gen = make_gen("RGBA", 640, 480)
        batch = ds.DsNvNonUniformSurfaceBuffer(4)
        batch.add(gen.acquire_surface(), id=1)
        batch.finalize()
        # index 0 is valid (1 filled), index 1 is out of bounds
        _ = batch.slot_ptr(0)
        with pytest.raises(RuntimeError, match="bounds|Slot"):
            batch.slot_ptr(1)
        with pytest.raises(RuntimeError, match="bounds|Slot"):
            batch.slot_ptr(100)


@skip_no_ds_runtime
class TestHeterogeneousFinalize:
    def test_finalize_returns_nonzero(self):
        gen = make_gen("RGBA", 640, 480)
        batch = ds.DsNvNonUniformSurfaceBuffer(4)
        batch.add(gen.acquire_surface(id=1), id=1)
        batch.finalize()
        buf = batch.as_gst_buffer()
        assert buf.ptr != 0

    def test_ids_propagated(self):
        gen = make_gen("RGBA", 640, 480)
        batch = ds.DsNvNonUniformSurfaceBuffer(4)
        for frame_id in [10, 20, 30]:
            buf = gen.acquire_surface(id=frame_id)
            batch.add(buf, id=frame_id)
        batch.finalize()
        result = batch.as_gst_buffer()
        meta = ds.get_savant_id_meta(result)
        meta_ids = [v for _kind, v in meta if _kind == "frame"]
        for frame_id in [10, 20, 30]:
            assert frame_id in meta_ids

    def test_auto_propagate_ids(self):
        gen = make_gen("RGBA", 640, 480)
        buf = gen.acquire_surface(id=42)
        batch = ds.DsNvNonUniformSurfaceBuffer(4)
        batch.add(buf, id=None)
        batch.finalize()
        result = batch.as_gst_buffer()
        meta = ds.get_savant_id_meta(result)
        ids = [v for _kind, v in meta if _kind == "frame"]
        assert 42 in ids

    def test_finalize_empty(self):
        batch = ds.DsNvNonUniformSurfaceBuffer(4)
        batch.finalize()
        buf = batch.as_gst_buffer()
        assert buf.ptr != 0
        meta = ds.get_savant_id_meta(buf)
        assert meta == []

    def test_finalize_twice_raises(self):
        gen = make_gen("RGBA", 640, 480)
        batch = ds.DsNvNonUniformSurfaceBuffer(4)
        batch.add(gen.acquire_surface(), id=1)
        batch.finalize()
        with pytest.raises(RuntimeError, match="finalized"):
            batch.finalize()


# ── Memory-leak smoke tests ─────────────────────────────────────────────


@skip_no_ds_runtime
class TestHeterogeneousMemoryLeak:
    """Smoke tests: loop over heterogeneous batch provisioning / release."""

    def test_hetero_add_finalize_release_no_leak(self):
        """Full cycle: acquire srcs → add to batch → finalize → RAII release."""
        gen = make_gen("RGBA", 1920, 1080)

        for _ in range(WARMUP_ITERATIONS):
            srcs = [gen.acquire_surface(id=j) for j in range(4)]
            batch = ds.DsNvNonUniformSurfaceBuffer(4)
            for j, s in enumerate(srcs):
                batch.add(s, id=j)
            batch.finalize()
            buf = batch.as_gst_buffer()
            del batch, buf, srcs, s, j

        gc.collect()
        cpu_before = cpu_rss_kb()
        gpu_before = gpu_mem_used_mb()

        for i in range(LEAK_ITERATIONS):
            srcs = [gen.acquire_surface(id=j) for j in range(4)]
            batch = ds.DsNvNonUniformSurfaceBuffer(4)
            for j, s in enumerate(srcs):
                batch.add(s, id=j)
            batch.finalize()
            buf = batch.as_gst_buffer()
            del batch, buf, srcs, s, j

        gc.collect()
        assert_no_leak(
            "hetero add/finalize/release",
            cpu_before, cpu_rss_kb(),
            gpu_before, gpu_mem_used_mb(),
        )

    def test_hetero_mixed_sizes_no_leak(self):
        """Different source sizes per iteration, RAII release."""
        gen_small = make_gen("RGBA", 320, 240)
        gen_large = make_gen("RGBA", 1920, 1080)

        for _ in range(WARMUP_ITERATIONS):
            s1 = gen_small.acquire_surface(id=1)
            s2 = gen_large.acquire_surface(id=2)
            batch = ds.DsNvNonUniformSurfaceBuffer(4)
            batch.add(s1, id=1)
            batch.add(s2, id=2)
            batch.finalize()
            buf = batch.as_gst_buffer()
            del batch, buf, s1, s2

        gc.collect()
        cpu_before = cpu_rss_kb()
        gpu_before = gpu_mem_used_mb()

        for i in range(LEAK_ITERATIONS):
            s1 = gen_small.acquire_surface(id=i)
            s2 = gen_large.acquire_surface(id=i + 1000)
            batch = ds.DsNvNonUniformSurfaceBuffer(4)
            batch.add(s1, id=i)
            batch.add(s2, id=i + 1000)
            batch.finalize()
            buf = batch.as_gst_buffer()
            del batch, buf, s1, s2

        gc.collect()
        assert_no_leak(
            "hetero mixed sizes",
            cpu_before, cpu_rss_kb(),
            gpu_before, gpu_mem_used_mb(),
        )

    def test_hetero_drop_without_finalize_no_leak(self):
        """Add sources then drop batch without finalize — no leak."""
        gen = make_gen("RGBA", 640, 480)

        for _ in range(WARMUP_ITERATIONS):
            src = gen.acquire_surface(id=1)
            batch = ds.DsNvNonUniformSurfaceBuffer(4)
            batch.add(src, id=1)
            del batch, src

        gc.collect()
        cpu_before = cpu_rss_kb()
        gpu_before = gpu_mem_used_mb()

        for i in range(LEAK_ITERATIONS):
            src = gen.acquire_surface(id=i)
            batch = ds.DsNvNonUniformSurfaceBuffer(4)
            batch.add(src, id=i)
            del batch, src

        gc.collect()
        assert_no_leak(
            "hetero drop without finalize",
            cpu_before, cpu_rss_kb(),
            gpu_before, gpu_mem_used_mb(),
        )
