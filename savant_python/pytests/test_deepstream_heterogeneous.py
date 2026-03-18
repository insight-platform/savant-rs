"""Tests for NonUniformBatch (zero-copy, nvstreammux2-style) — require CUDA/DeepStream runtime."""

from __future__ import annotations

import gc

import pytest

from conftest import HAS_DS_FEATURE, skip_no_ds_runtime
from deepstream_helpers import (
    GPU_GROWTH_LIMIT_MB,
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
        batch = ds.NonUniformBatch(gpu_id=0)
        assert batch is not None

    def test_properties(self):
        batch = ds.NonUniformBatch(gpu_id=0)
        assert batch.num_filled == 0
        assert batch.gpu_id == 0


@skip_no_ds_runtime
class TestHeterogeneousAdd:
    def test_add_single(self):
        gen = make_gen("RGBA", 640, 480)
        batch = ds.NonUniformBatch(gpu_id=0)
        buf = gen.acquire(id=1)
        view = ds.SurfaceView.from_buffer(buf)
        batch.add(view)
        assert batch.num_filled == 1

    def test_add_multiple_different_sizes(self):
        gen_small = make_gen("RGBA", 320, 240)
        gen_1080p = make_gen("RGBA", 1920, 1080)
        gen_720p = make_gen("RGBA", 1280, 720)
        batch = ds.NonUniformBatch(gpu_id=0)
        view1 = ds.SurfaceView.from_buffer(gen_small.acquire())
        view2 = ds.SurfaceView.from_buffer(gen_1080p.acquire())
        view3 = ds.SurfaceView.from_buffer(gen_720p.acquire())
        batch.add(view1)
        batch.add(view2)
        batch.add(view3)
        assert batch.num_filled == 3

    def test_add_different_formats(self):
        gen_rgba = make_gen("RGBA", 640, 480)
        gen_nv12 = make_gen("NV12", 640, 480)
        batch = ds.NonUniformBatch(gpu_id=0)
        view1 = ds.SurfaceView.from_buffer(gen_rgba.acquire())
        view2 = ds.SurfaceView.from_buffer(gen_nv12.acquire())
        batch.add(view1)
        batch.add(view2)
        assert batch.num_filled == 2


@skip_no_ds_runtime
class TestHeterogeneousFinalize:
    def test_finalize_returns_nonzero(self):
        gen = make_gen("RGBA", 640, 480)
        batch = ds.NonUniformBatch(gpu_id=0)
        buf = gen.acquire(id=1)
        view = ds.SurfaceView.from_buffer(buf)
        batch.add(view)
        shared = batch.finalize(ids=[(ds.SavantIdMetaKind.FRAME, 1)])
        assert bool(shared)

    def test_ids_propagated(self):
        gen = make_gen("RGBA", 640, 480)
        batch = ds.NonUniformBatch(gpu_id=0)
        views = []
        for frame_id in [10, 20, 30]:
            buf = gen.acquire(id=frame_id)
            view = ds.SurfaceView.from_buffer(buf)
            batch.add(view)
            views.append(view)
        shared = batch.finalize(ids=[(ds.SavantIdMetaKind.FRAME, fid) for fid in [10, 20, 30]])
        meta = ds.get_savant_id_meta(shared)
        meta_ids = [v for kind, v in meta if kind == ds.SavantIdMetaKind.FRAME]
        for frame_id in [10, 20, 30]:
            assert frame_id in meta_ids

    def test_finalize_empty_raises(self):
        """Finalizing an empty batch raises RuntimeError."""
        batch = ds.NonUniformBatch(gpu_id=0)
        with pytest.raises(RuntimeError, match="empty"):
            batch.finalize()

    def test_finalize_twice_raises(self):
        gen = make_gen("RGBA", 640, 480)
        batch = ds.NonUniformBatch(gpu_id=0)
        buf = gen.acquire()
        view = ds.SurfaceView.from_buffer(buf)
        batch.add(view)
        batch.finalize()
        with pytest.raises(RuntimeError, match="finalized"):
            batch.finalize()


# ── Memory-leak smoke tests ─────────────────────────────────────────────


@skip_no_ds_runtime
class TestHeterogeneousMemoryLeak:
    """Smoke tests: loop over heterogeneous batch provisioning / release."""

    def test_hetero_add_finalize_release_no_leak(self):
        """Full cycle: acquire srcs → add views to batch → finalize → RAII release.

        Uses a higher CPU RSS threshold (20 MB) because the SurfaceView-based
        API creates more transient Python/PyO3 objects per iteration than the
        old direct-buffer API, which amplifies pytest runner RSS noise.
        """
        gen = make_gen("RGBA", 1920, 1080)

        for _ in range(WARMUP_ITERATIONS):
            bufs = [gen.acquire(id=j) for j in range(4)]
            views = [ds.SurfaceView.from_buffer(b) for b in bufs]
            batch = ds.NonUniformBatch(gpu_id=0)
            for v in views:
                batch.add(v)
            shared = batch.finalize(ids=[(ds.SavantIdMetaKind.FRAME, j) for j in range(4)])
            del batch, shared, bufs, views, v

        gc.collect()
        cpu_before = cpu_rss_kb()
        gpu_before = gpu_mem_used_mb()

        for i in range(LEAK_ITERATIONS):
            bufs = [gen.acquire(id=j) for j in range(4)]
            views = [ds.SurfaceView.from_buffer(b) for b in bufs]
            batch = ds.NonUniformBatch(gpu_id=0)
            for v in views:
                batch.add(v)
            shared = batch.finalize(ids=[(ds.SavantIdMetaKind.FRAME, j) for j in range(4)])
            del batch, shared, bufs, views, v

        gc.collect()
        cpu_growth = cpu_rss_kb() - cpu_before
        gpu_growth = gpu_mem_used_mb() - gpu_before
        assert cpu_growth < 20_000, (
            f"hetero add/finalize/release: CPU RSS grew by {cpu_growth} KB "
            f"over {LEAK_ITERATIONS} iterations"
        )
        assert gpu_growth < GPU_GROWTH_LIMIT_MB, (
            f"hetero add/finalize/release: GPU mem grew by {gpu_growth} MB "
            f"over {LEAK_ITERATIONS} iterations"
        )

    def test_hetero_mixed_sizes_no_leak(self):
        """Different source sizes per iteration, RAII release."""
        gen_small = make_gen("RGBA", 320, 240)
        gen_large = make_gen("RGBA", 1920, 1080)

        for _ in range(WARMUP_ITERATIONS):
            buf1 = gen_small.acquire(id=1)
            buf2 = gen_large.acquire(id=2)
            view1 = ds.SurfaceView.from_buffer(buf1)
            view2 = ds.SurfaceView.from_buffer(buf2)
            batch = ds.NonUniformBatch(gpu_id=0)
            batch.add(view1)
            batch.add(view2)
            shared = batch.finalize(ids=[(ds.SavantIdMetaKind.FRAME, 1), (ds.SavantIdMetaKind.FRAME, 2)])
            del batch, shared, buf1, buf2, view1, view2

        gc.collect()
        cpu_before = cpu_rss_kb()
        gpu_before = gpu_mem_used_mb()

        for i in range(LEAK_ITERATIONS):
            buf1 = gen_small.acquire(id=i)
            buf2 = gen_large.acquire(id=i + 1000)
            view1 = ds.SurfaceView.from_buffer(buf1)
            view2 = ds.SurfaceView.from_buffer(buf2)
            batch = ds.NonUniformBatch(gpu_id=0)
            batch.add(view1)
            batch.add(view2)
            shared = batch.finalize(ids=[(ds.SavantIdMetaKind.FRAME, i), (ds.SavantIdMetaKind.FRAME, i + 1000)])
            del batch, shared, buf1, buf2, view1, view2

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
            buf = gen.acquire(id=1)
            view = ds.SurfaceView.from_buffer(buf)
            batch = ds.NonUniformBatch(gpu_id=0)
            batch.add(view)
            del batch, buf, view

        gc.collect()
        cpu_before = cpu_rss_kb()
        gpu_before = gpu_mem_used_mb()

        for i in range(LEAK_ITERATIONS):
            buf = gen.acquire(id=i)
            view = ds.SurfaceView.from_buffer(buf)
            batch = ds.NonUniformBatch(gpu_id=0)
            batch.add(view)
            del batch, buf, view

        gc.collect()
        assert_no_leak(
            "hetero drop without finalize",
            cpu_before, cpu_rss_kb(),
            gpu_before, gpu_mem_used_mb(),
        )
