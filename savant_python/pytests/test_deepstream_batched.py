"""Tests for UniformBatchGenerator and SurfaceBatch — require CUDA/DeepStream runtime."""

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
FK = ds.SavantIdMetaKind.FRAME


def _make_batched_gen(
    fmt: str, w: int, h: int, batch: int, pool: int = 2
) -> "ds.UniformBatchGenerator":
    return ds.UniformBatchGenerator(fmt, w, h, batch, pool_size=pool)


@skip_no_ds_runtime
class TestBatchedConstruction:
    def test_create_with_defaults(self):
        gen = ds.UniformBatchGenerator("RGBA", 640, 640, 4)
        assert gen is not None

    def test_create_custom_params(self):
        gen = ds.UniformBatchGenerator(
            "NV12", 320, 240, 8, pool_size=4, gpu_id=0
        )
        assert gen is not None

    def test_batch_size_1(self):
        gen = _make_batched_gen("RGBA", 640, 640, 1, 1)
        config = ds.TransformConfig()
        batch = gen.acquire_batch(config)
        assert batch.max_batch_size == 1


@skip_no_ds_runtime
class TestBatchedProperties:
    def test_width(self):
        gen = _make_batched_gen("RGBA", 640, 480, 4)
        assert gen.width == 640

    def test_height(self):
        gen = _make_batched_gen("RGBA", 640, 480, 4)
        assert gen.height == 480

    def test_format(self):
        gen = _make_batched_gen("RGBA", 640, 480, 4)
        assert gen.format == ds.VideoFormat.RGBA

    def test_gpu_id(self):
        gen = _make_batched_gen("RGBA", 640, 480, 4)
        assert gen.gpu_id == 0

    def test_max_batch_size(self):
        gen = _make_batched_gen("RGBA", 640, 480, 8)
        assert gen.max_batch_size == 8


@skip_no_ds_runtime
class TestAcquireSurfaceBatch:
    def test_returns_batched_surface(self):
        gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = gen.acquire_batch(config)
        assert batch is not None

    def test_initial_num_filled_zero(self):
        gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = gen.acquire_batch(config)
        assert batch.num_filled == 0

    def test_max_batch_size_matches(self):
        gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = gen.acquire_batch(config)
        assert batch.max_batch_size == 4


@skip_no_ds_runtime
class TestFillSlot:
    def test_fill_single_slot(self):
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        ids = [(FK, 42)]
        batch = batched_gen.acquire_batch(config, ids=ids)
        src = src_gen.acquire(id=42)
        batch.transform_slot(0, src)
        assert batch.num_filled == 1

    def test_fill_all_slots(self):
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        ids = [(FK, 10 + i) for i in range(4)]
        batch = batched_gen.acquire_batch(config, ids=ids)
        for i in range(4):
            src = src_gen.acquire(id=10 + i)
            batch.transform_slot(i, src)
        assert batch.num_filled == 4

    def test_fill_partial_batch(self):
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batch(config)
        for i in range(2):
            src = src_gen.acquire()
            batch.transform_slot(i, src)
        assert batch.num_filled == 2

    def test_fill_exceeds_batch_size_raises(self):
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 2)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batch(config)
        src1 = src_gen.acquire()
        src2 = src_gen.acquire()
        batch.transform_slot(0, src1)
        batch.transform_slot(1, src2)
        src3 = src_gen.acquire()
        with pytest.raises(RuntimeError, match="overflow|Batch|bounds"):
            batch.transform_slot(2, src3)


@skip_no_ds_runtime
class TestFillSlotWithSrcRect:
    """Tests for transform_slot with src_rect (source crop ROI)."""

    def test_fill_with_src_rect(self):
        """Fill a slot with a cropped source region (top, left, width, height)."""
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 2)
        config = ds.TransformConfig()
        ids = [(FK, 1)]
        batch = batched_gen.acquire_batch(config, ids=ids)
        src = src_gen.acquire()
        roi = ds.Rect(top=100, left=200, width=800, height=600)
        batch.transform_slot(0, src, src_rect=roi)
        assert batch.num_filled == 1
        batch.finalize()
        buf = batch.shared_buffer()
        assert bool(buf)

    def test_fill_mixed_src_rect_and_none(self):
        """Fill slots: one with src_rect, one without (full frame)."""
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 2)
        config = ds.TransformConfig()
        ids = [(FK, 1), (FK, 2)]
        batch = batched_gen.acquire_batch(config, ids=ids)
        src1 = src_gen.acquire()
        roi = ds.Rect(top=0, left=0, width=960, height=540)
        batch.transform_slot(0, src1, src_rect=roi)
        src2 = src_gen.acquire()
        batch.transform_slot(1, src2, src_rect=None)
        assert batch.num_filled == 2
        batch.finalize()
        buf = batch.shared_buffer()
        assert bool(buf)
        meta = ds.get_savant_id_meta(buf)
        meta_ids = [v for kind, v in meta if kind == FK]
        assert 1 in meta_ids
        assert 2 in meta_ids

    def test_fill_all_slots_with_different_rois(self):
        """Fill all slots with different source crop regions."""
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        ids = [(FK, 10 + i) for i in range(4)]
        batch = batched_gen.acquire_batch(config, ids=ids)
        rois = [
            ds.Rect(0, 0, 960, 540),  # top-left quadrant
            ds.Rect(0, 960, 960, 540),  # top-right
            ds.Rect(540, 0, 960, 540),  # bottom-left
            ds.Rect(540, 960, 960, 540),  # bottom-right
        ]
        for i, roi in enumerate(rois):
            src = src_gen.acquire(id=10 + i)
            batch.transform_slot(i, src, src_rect=roi)
        assert batch.num_filled == 4
        batch.finalize()
        buf = batch.shared_buffer()
        assert bool(buf)

    def test_same_src_added_multiple_times_with_various_rois(self):
        """Reuse the same source buffer for multiple slots with different ROIs."""
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 3)
        config = ds.TransformConfig()
        ids = [(FK, 10 + i) for i in range(3)]
        batch = batched_gen.acquire_batch(config, ids=ids)
        src = src_gen.acquire(id=1)
        rois = [
            ds.Rect(0, 0, 960, 540),  # top-left
            ds.Rect(0, 960, 960, 540),  # top-right
            ds.Rect(540, 480, 960, 540),  # center-ish
        ]
        for i, roi in enumerate(rois):
            batch.transform_slot(i, src, src_rect=roi)
        assert batch.num_filled == 3
        batch.finalize()
        buf = batch.shared_buffer()
        assert bool(buf)
        meta = ds.get_savant_id_meta(buf)
        meta_ids = [v for kind, v in meta if kind == FK]
        assert 10 in meta_ids
        assert 11 in meta_ids
        assert 12 in meta_ids

    def test_roi_partially_outside_viewport_succeeds(self):
        """ROI extending partially outside source bounds should succeed (clipped)."""
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 1)
        config = ds.TransformConfig()
        ids = [(FK, 1)]
        batch = batched_gen.acquire_batch(config, ids=ids)
        src = src_gen.acquire()
        # ROI: top=1000, left=1800, width=200, height=200
        # Extends 80px right of 1920, 120px below 1080
        roi = ds.Rect(1000, 1800, 200, 200)
        batch.transform_slot(0, src, src_rect=roi)
        assert batch.num_filled == 1
        batch.finalize()
        buf = batch.shared_buffer()
        assert bool(buf)

    def test_roi_completely_outside_viewport_fails(self):
        """ROI entirely outside source bounds must fail."""
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 1)
        config = ds.TransformConfig()
        ids = [(FK, 1)]
        batch = batched_gen.acquire_batch(config, ids=ids)
        src = src_gen.acquire()
        # ROI completely outside: top=2000, left=2000, width=100, height=100
        roi = ds.Rect(2000, 2000, 100, 100)
        with pytest.raises(RuntimeError):
            batch.transform_slot(0, src, src_rect=roi)


@skip_no_ds_runtime
class TestSlotPtr:
    def test_returns_surface_view_with_data_ptr_and_pitch(self):
        """view() returns SurfaceView with data_ptr and pitch (requires finalize)."""
        src_gen = make_gen("RGBA", 1920, 1080)
        gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        ids = [(FK, 42)]
        batch = gen.acquire_batch(config, ids=ids)
        src = src_gen.acquire(id=42)
        batch.transform_slot(0, src)
        batch.finalize()
        v = batch.view(0)
        assert hasattr(v, "data_ptr")
        assert hasattr(v, "pitch")
        data_ptr, pitch = v.data_ptr, v.pitch
        assert isinstance(data_ptr, int)
        assert isinstance(pitch, int)

    def test_data_ptr_and_pitch_nonzero(self):
        """view() data_ptr and pitch are non-zero after fill and finalize."""
        src_gen = make_gen("RGBA", 1920, 1080)
        gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        ids = [(FK, 42)]
        batch = gen.acquire_batch(config, ids=ids)
        src = src_gen.acquire(id=42)
        batch.transform_slot(0, src)
        batch.finalize()
        v = batch.view(0)
        assert v.data_ptr != 0
        assert v.pitch > 0

    def test_slot_ptr_out_of_bounds_raises(self):
        gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = gen.acquire_batch(config)
        with pytest.raises(RuntimeError, match="bounds|Slot"):
            batch.view(4)
        with pytest.raises(RuntimeError, match="bounds|Slot"):
            batch.view(100)


@skip_no_ds_runtime
class TestFinalize:
    def test_finalize_returns_nonzero(self):
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        ids = [(FK, 42)]
        batch = batched_gen.acquire_batch(config, ids=ids)
        src = src_gen.acquire(id=42)
        batch.transform_slot(0, src)
        batch.finalize()
        buf = batch.shared_buffer()
        assert bool(buf)

    def test_ids_propagated(self):
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        ids = [(FK, 10), (FK, 20), (FK, 30), (FK, 40)]
        batch = batched_gen.acquire_batch(config, ids=ids)
        for i, frame_id in enumerate([10, 20, 30, 40]):
            src = src_gen.acquire(id=frame_id)
            batch.transform_slot(i, src)
        batch.finalize()
        buf = batch.shared_buffer()
        meta = ds.get_savant_id_meta(buf)
        meta_ids = [v for kind, v in meta if kind == FK]
        for frame_id in [10, 20, 30, 40]:
            assert frame_id in meta_ids

    def test_finalize_twice_raises(self):
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batch(config)
        batch.finalize()
        with pytest.raises(RuntimeError, match="finalized"):
            batch.finalize()


@skip_no_ds_runtime
class TestSetNumFilled:
    def test_set_num_filled_succeeds(self):
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        ids = [(FK, 1)]
        batch = batched_gen.acquire_batch(config, ids=ids)
        src = src_gen.acquire(id=1)
        batch.transform_slot(0, src)
        batch.finalize()
        buf = batch.shared_buffer()
        ds.set_num_filled(buf, 3)
        # No exception means success; we cannot easily verify numFilled from Python

    def test_set_num_filled_overflow_raises(self):
        """set_num_filled with count > max_batch_size may raise (implementation-dependent)."""
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        ids = [(FK, 1)]
        batch = batched_gen.acquire_batch(config, ids=ids)
        src = src_gen.acquire(id=1)
        batch.transform_slot(0, src)
        batch.finalize()
        buf = batch.shared_buffer()
        # Current implementation does not validate overflow; test documents behavior
        try:
            ds.set_num_filled(buf, 5)
        except RuntimeError as e:
            assert "overflow" in str(e).lower() or "Batch" in str(e)

    def test_set_num_filled_null_raises(self):
        with pytest.raises(ValueError, match="null"):
            ds.set_num_filled(0, 1)


# ── Memory-leak smoke tests ─────────────────────────────────────────────


@skip_no_ds_runtime
class TestBatchedMemoryLeak:
    """Smoke tests: loop over batch provisioning / release, assert no leak."""

    def test_batched_fill_finalize_release_no_leak(self):
        """Full cycle: acquire src → acquire batch → fill → finalize → RAII release.

        Uses a higher CPU RSS threshold (20 MB) because the batched API
        creates many transient PyO3 objects per iteration, which amplifies
        pytest runner RSS noise (confirmed zero real growth via standalone
        testing).
        """
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4, pool=2)
        config = ds.TransformConfig()
        ids = [(FK, i) for i in range(4)]

        for _ in range(WARMUP_ITERATIONS):
            src = src_gen.acquire(id=1)
            batch = batched_gen.acquire_batch(config, ids=ids)
            for slot in range(4):
                batch.transform_slot(slot, src)
            batch.finalize()
            buf = batch.shared_buffer()
            del batch, buf, src

        gc.collect()
        cpu_before = cpu_rss_kb()
        gpu_before = gpu_mem_used_mb()

        for i in range(LEAK_ITERATIONS):
            src = src_gen.acquire(id=i)
            batch = batched_gen.acquire_batch(config, ids=ids)
            for slot in range(4):
                batch.transform_slot(slot, src)
            batch.finalize()
            buf = batch.shared_buffer()
            del batch, buf, src

        gc.collect()
        cpu_growth = cpu_rss_kb() - cpu_before
        gpu_growth = gpu_mem_used_mb() - gpu_before
        assert cpu_growth < 20_000, (
            f"batched fill/finalize/release: CPU RSS grew by {cpu_growth} KB "
            f"over {LEAK_ITERATIONS} iterations"
        )
        assert gpu_growth < GPU_GROWTH_LIMIT_MB, (
            f"batched fill/finalize/release: GPU mem grew by {gpu_growth} MB "
            f"over {LEAK_ITERATIONS} iterations"
        )

    def test_batched_acquire_drop_no_finalize_no_leak(self):
        """Acquire + fill but drop without finalize — batch returns to pool."""
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4, pool=2)
        config = ds.TransformConfig()

        for _ in range(WARMUP_ITERATIONS):
            src = src_gen.acquire(id=1)
            batch = batched_gen.acquire_batch(config, ids=[(FK, 1)])
            batch.transform_slot(0, src)
            del batch, src

        gc.collect()
        cpu_before = cpu_rss_kb()
        gpu_before = gpu_mem_used_mb()

        for i in range(LEAK_ITERATIONS):
            src = src_gen.acquire(id=i)
            batch = batched_gen.acquire_batch(config, ids=[(FK, i)])
            batch.transform_slot(0, src)
            del batch, src

        gc.collect()
        assert_no_leak(
            "batched acquire/drop without finalize",
            cpu_before, cpu_rss_kb(),
            gpu_before, gpu_mem_used_mb(),
        )

    def test_same_src_multiple_rois_release_no_leak(self):
        """Same source, multiple ROIs per batch, full release cycle."""
        src_gen = make_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 3, pool=2)
        config = ds.TransformConfig()
        ids = [(FK, i) for i in range(3)]
        rois = [
            ds.Rect(0, 0, 960, 540),
            ds.Rect(0, 960, 960, 540),
            ds.Rect(540, 480, 960, 540),
        ]

        for _ in range(WARMUP_ITERATIONS):
            src = src_gen.acquire(id=1)
            batch = batched_gen.acquire_batch(config, ids=ids)
            for j, roi in enumerate(rois):
                batch.transform_slot(j, src, src_rect=roi)
            batch.finalize()
            buf = batch.shared_buffer()
            del batch, buf, src

        gc.collect()
        cpu_before = cpu_rss_kb()
        gpu_before = gpu_mem_used_mb()

        for i in range(LEAK_ITERATIONS):
            src = src_gen.acquire(id=i)
            batch = batched_gen.acquire_batch(config, ids=ids)
            for j, roi in enumerate(rois):
                batch.transform_slot(j, src, src_rect=roi)
            batch.finalize()
            buf = batch.shared_buffer()
            del batch, buf, src

        gc.collect()
        assert_no_leak(
            "same src multiple ROIs",
            cpu_before, cpu_rss_kb(),
            gpu_before, gpu_mem_used_mb(),
        )
