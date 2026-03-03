"""Tests for HeterogeneousBatch (zero-copy, nvstreammux2-style) — require CUDA/DeepStream runtime."""

from __future__ import annotations

import gc
import resource

import pytest

_ds = pytest.importorskip("savant_rs.deepstream")
if not hasattr(_ds, "HeterogeneousBatch"):
    pytest.skip("savant_rs built without deepstream feature", allow_module_level=True)
ds = _ds


def _ds_runtime_available() -> bool:
    """Check if DeepStream + CUDA runtime is actually available."""
    try:
        ds.init_cuda(0)
        gen = ds.NvBufSurfaceGenerator("RGBA", 64, 64, pool_size=1)
        _ = gen.acquire_surface()
        return True
    except Exception:
        return False


_has_runtime = _ds_runtime_available()
skip_no_runtime = pytest.mark.skipif(
    not _has_runtime, reason="CUDA/DeepStream not available"
)


def _make_gen(fmt: str, w: int, h: int) -> "ds.NvBufSurfaceGenerator":
    return ds.NvBufSurfaceGenerator(fmt, w, h, pool_size=4)


@skip_no_runtime
class TestHeterogeneousCreate:
    def test_create_with_defaults(self):
        batch = ds.HeterogeneousBatch(4)
        assert batch is not None

    def test_create_batch_size_1(self):
        batch = ds.HeterogeneousBatch(1)
        assert batch.max_batch_size == 1

    def test_properties(self):
        batch = ds.HeterogeneousBatch(4, gpu_id=0)
        assert batch.max_batch_size == 4
        assert batch.num_filled == 0
        assert batch.gpu_id == 0


@skip_no_runtime
class TestHeterogeneousAdd:
    def test_add_single(self):
        gen = _make_gen("RGBA", 640, 480)
        batch = ds.HeterogeneousBatch(4)
        buf = gen.acquire_surface(id=1)
        batch.add(buf, id=1)
        assert batch.num_filled == 1

    def test_add_multiple_different_sizes(self):
        gen_small = _make_gen("RGBA", 320, 240)
        gen_1080p = _make_gen("RGBA", 1920, 1080)
        gen_720p = _make_gen("RGBA", 1280, 720)
        batch = ds.HeterogeneousBatch(8)
        batch.add(gen_small.acquire_surface(), id=1)
        batch.add(gen_1080p.acquire_surface(), id=2)
        batch.add(gen_720p.acquire_surface(), id=3)
        assert batch.num_filled == 3

    def test_add_different_formats(self):
        gen_rgba = _make_gen("RGBA", 640, 480)
        gen_nv12 = _make_gen("NV12", 640, 480)
        batch = ds.HeterogeneousBatch(4)
        batch.add(gen_rgba.acquire_surface(), id=1)
        batch.add(gen_nv12.acquire_surface(), id=2)
        assert batch.num_filled == 2

    def test_add_exceeds_capacity_raises(self):
        gen = _make_gen("RGBA", 640, 480)
        batch = ds.HeterogeneousBatch(2)
        batch.add(gen.acquire_surface(), id=1)
        batch.add(gen.acquire_surface(), id=2)
        with pytest.raises(RuntimeError, match="overflow|Batch"):
            batch.add(gen.acquire_surface(), id=3)

    def test_add_null_raises(self):
        batch = ds.HeterogeneousBatch(4)
        with pytest.raises(ValueError, match="null"):
            batch.add(0, id=1)


@skip_no_runtime
class TestHeterogeneousSlotPtr:
    def test_returns_4_tuple(self):
        gen_small = _make_gen("RGBA", 320, 240)
        gen_1080p = _make_gen("RGBA", 1920, 1080)
        batch = ds.HeterogeneousBatch(4)
        batch.add(gen_small.acquire_surface(), id=1)
        batch.add(gen_1080p.acquire_surface(), id=2)
        result = batch.slot_ptr(0)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_dimensions_match_source(self):
        gen_small = _make_gen("RGBA", 320, 240)
        gen_1080p = _make_gen("RGBA", 1920, 1080)
        batch = ds.HeterogeneousBatch(4)
        batch.add(gen_small.acquire_surface(), id=1)
        batch.add(gen_1080p.acquire_surface(), id=2)
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
        gen = _make_gen("RGBA", 640, 480)
        batch = ds.HeterogeneousBatch(4)
        batch.add(gen.acquire_surface(), id=1)
        # index 0 is valid (1 filled), index 1 is out of bounds
        _ = batch.slot_ptr(0)
        with pytest.raises(RuntimeError, match="bounds|Slot"):
            batch.slot_ptr(1)
        with pytest.raises(RuntimeError, match="bounds|Slot"):
            batch.slot_ptr(100)


@skip_no_runtime
class TestHeterogeneousFinalize:
    def test_finalize_returns_nonzero(self):
        gen = _make_gen("RGBA", 640, 480)
        batch = ds.HeterogeneousBatch(4)
        batch.add(gen.acquire_surface(id=1), id=1)
        buf = batch.finalize()
        assert buf.ptr != 0

    def test_ids_propagated(self):
        gen = _make_gen("RGBA", 640, 480)
        batch = ds.HeterogeneousBatch(4)
        for frame_id in [10, 20, 30]:
            buf = gen.acquire_surface(id=frame_id)
            batch.add(buf, id=frame_id)
        result = batch.finalize()
        meta = ds.get_savant_id_meta(result)
        meta_ids = [v for _kind, v in meta if _kind == "frame"]
        for frame_id in [10, 20, 30]:
            assert frame_id in meta_ids

    def test_auto_propagate_ids(self):
        gen = _make_gen("RGBA", 640, 480)
        buf = gen.acquire_surface(id=42)
        batch = ds.HeterogeneousBatch(4)
        batch.add(buf, id=None)
        result = batch.finalize()
        meta = ds.get_savant_id_meta(result)
        ids = [v for _kind, v in meta if _kind == "frame"]
        assert 42 in ids

    def test_finalize_empty(self):
        batch = ds.HeterogeneousBatch(4)
        buf = batch.finalize()
        assert buf.ptr != 0
        meta = ds.get_savant_id_meta(buf)
        assert meta == []

    def test_finalize_twice_raises(self):
        gen = _make_gen("RGBA", 640, 480)
        batch = ds.HeterogeneousBatch(4)
        batch.add(gen.acquire_surface(), id=1)
        _ = batch.finalize()
        with pytest.raises(RuntimeError, match="finalized"):
            _ = batch.finalize()


# ── Memory-leak smoke tests ─────────────────────────────────────────────

_LEAK_ITERATIONS = 200
_WARMUP_ITERATIONS = 10
_CPU_GROWTH_LIMIT_KB = 10_000  # 10 MB
_GPU_GROWTH_LIMIT_MB = 20


def _gpu_mem_used_mb() -> int:
    """Query current GPU memory usage (MiB)."""
    return ds.gpu_mem_used_mib()


def _cpu_rss_kb() -> int:
    """Current process RSS in KB (Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@skip_no_runtime
class TestHeterogeneousMemoryLeak:
    """Smoke tests: loop over heterogeneous batch provisioning / release."""

    def test_hetero_add_finalize_release_no_leak(self):
        """Full cycle: acquire srcs → add to batch → finalize → RAII release."""
        gen = _make_gen("RGBA", 1920, 1080)

        for _ in range(_WARMUP_ITERATIONS):
            srcs = [gen.acquire_surface(id=j) for j in range(4)]
            batch = ds.HeterogeneousBatch(4)
            for j, s in enumerate(srcs):
                batch.add(s, id=j)
            buf = batch.finalize()
            del batch, buf, srcs, s, j

        gc.collect()
        cpu_before = _cpu_rss_kb()
        gpu_before = _gpu_mem_used_mb()

        for i in range(_LEAK_ITERATIONS):
            srcs = [gen.acquire_surface(id=j) for j in range(4)]
            batch = ds.HeterogeneousBatch(4)
            for j, s in enumerate(srcs):
                batch.add(s, id=j)
            buf = batch.finalize()
            del batch, buf, srcs, s, j

        gc.collect()
        cpu_after = _cpu_rss_kb()
        gpu_after = _gpu_mem_used_mb()

        cpu_growth = cpu_after - cpu_before
        gpu_growth = gpu_after - gpu_before
        assert cpu_growth < _CPU_GROWTH_LIMIT_KB, (
            f"CPU RSS grew by {cpu_growth} KB over {_LEAK_ITERATIONS} iterations"
        )
        assert gpu_growth < _GPU_GROWTH_LIMIT_MB, (
            f"GPU mem grew by {gpu_growth} MB over {_LEAK_ITERATIONS} iterations"
        )

    def test_hetero_mixed_sizes_no_leak(self):
        """Different source sizes per iteration, RAII release."""
        gen_small = _make_gen("RGBA", 320, 240)
        gen_large = _make_gen("RGBA", 1920, 1080)

        for _ in range(_WARMUP_ITERATIONS):
            s1 = gen_small.acquire_surface(id=1)
            s2 = gen_large.acquire_surface(id=2)
            batch = ds.HeterogeneousBatch(4)
            batch.add(s1, id=1)
            batch.add(s2, id=2)
            buf = batch.finalize()
            del batch, buf, s1, s2

        gc.collect()
        cpu_before = _cpu_rss_kb()
        gpu_before = _gpu_mem_used_mb()

        for i in range(_LEAK_ITERATIONS):
            s1 = gen_small.acquire_surface(id=i)
            s2 = gen_large.acquire_surface(id=i + 1000)
            batch = ds.HeterogeneousBatch(4)
            batch.add(s1, id=i)
            batch.add(s2, id=i + 1000)
            buf = batch.finalize()
            del batch, buf, s1, s2

        gc.collect()
        cpu_after = _cpu_rss_kb()
        gpu_after = _gpu_mem_used_mb()

        cpu_growth = cpu_after - cpu_before
        gpu_growth = gpu_after - gpu_before
        assert cpu_growth < _CPU_GROWTH_LIMIT_KB, (
            f"CPU RSS grew by {cpu_growth} KB over {_LEAK_ITERATIONS} iterations"
        )
        assert gpu_growth < _GPU_GROWTH_LIMIT_MB, (
            f"GPU mem grew by {gpu_growth} MB over {_LEAK_ITERATIONS} iterations"
        )

    def test_hetero_drop_without_finalize_no_leak(self):
        """Add sources then drop batch without finalize — no leak."""
        gen = _make_gen("RGBA", 640, 480)

        for _ in range(_WARMUP_ITERATIONS):
            src = gen.acquire_surface(id=1)
            batch = ds.HeterogeneousBatch(4)
            batch.add(src, id=1)
            del batch, src

        gc.collect()
        cpu_before = _cpu_rss_kb()
        gpu_before = _gpu_mem_used_mb()

        for i in range(_LEAK_ITERATIONS):
            src = gen.acquire_surface(id=i)
            batch = ds.HeterogeneousBatch(4)
            batch.add(src, id=i)
            del batch, src

        gc.collect()
        cpu_after = _cpu_rss_kb()
        gpu_after = _gpu_mem_used_mb()

        cpu_growth = cpu_after - cpu_before
        gpu_growth = gpu_after - gpu_before
        assert cpu_growth < _CPU_GROWTH_LIMIT_KB, (
            f"CPU RSS grew by {cpu_growth} KB over {_LEAK_ITERATIONS} iterations"
        )
        assert gpu_growth < _GPU_GROWTH_LIMIT_MB, (
            f"GPU mem grew by {gpu_growth} MB over {_LEAK_ITERATIONS} iterations"
        )
