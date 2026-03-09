"""Tests for DsNvUniformSurfaceBufferGenerator and DsNvUniformSurfaceBuffer — require CUDA/DeepStream runtime."""

from __future__ import annotations

import gc

import pytest

_ds = pytest.importorskip("savant_rs.deepstream")
if not hasattr(_ds, "DsNvUniformSurfaceBufferGenerator"):
    pytest.skip("savant_rs built without deepstream feature", allow_module_level=True)
ds = _ds


def _ds_runtime_available() -> bool:
    """Check if DeepStream + CUDA runtime is actually available."""
    try:
        ds.init_cuda(0)
        gen = ds.DsNvSurfaceBufferGenerator("RGBA", 64, 64, pool_size=1)
        _ = gen.acquire_surface()
        return True
    except Exception:
        return False


_has_runtime = _ds_runtime_available()
skip_no_runtime = pytest.mark.skipif(
    not _has_runtime, reason="CUDA/DeepStream not available"
)


def _make_src_gen(fmt: str, w: int, h: int) -> "ds.DsNvSurfaceBufferGenerator":
    return ds.DsNvSurfaceBufferGenerator(fmt, w, h, pool_size=4)


def _make_batched_gen(
    fmt: str, w: int, h: int, batch: int, pool: int = 2
) -> "ds.DsNvUniformSurfaceBufferGenerator":
    return ds.DsNvUniformSurfaceBufferGenerator(fmt, w, h, batch, pool_size=pool)


@skip_no_runtime
class TestBatchedConstruction:
    def test_create_with_defaults(self):
        gen = ds.DsNvUniformSurfaceBufferGenerator("RGBA", 640, 640, 4)
        assert gen is not None

    def test_create_custom_params(self):
        gen = ds.DsNvUniformSurfaceBufferGenerator(
            "NV12", 320, 240, 8, pool_size=4, gpu_id=0
        )
        assert gen is not None

    def test_batch_size_1(self):
        gen = _make_batched_gen("RGBA", 640, 640, 1, 1)
        config = ds.TransformConfig()
        batch = gen.acquire_batched_surface(config)
        assert batch.max_batch_size == 1


@skip_no_runtime
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


@skip_no_runtime
class TestAcquireDsNvUniformSurfaceBuffer:
    def test_returns_batched_surface(self):
        gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = gen.acquire_batched_surface(config)
        assert batch is not None

    def test_initial_num_filled_zero(self):
        gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = gen.acquire_batched_surface(config)
        assert batch.num_filled == 0

    def test_max_batch_size_matches(self):
        gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = gen.acquire_batched_surface(config)
        assert batch.max_batch_size == 4


@skip_no_runtime
class TestFillSlot:
    def test_fill_single_slot(self):
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        src = src_gen.acquire_surface()
        batch.fill_slot(src, id=42)
        assert batch.num_filled == 1

    def test_fill_all_slots(self):
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        for i in range(4):
            src = src_gen.acquire_surface(id=10 + i)
            batch.fill_slot(src, id=10 + i)
        assert batch.num_filled == 4

    def test_fill_partial_batch(self):
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        for _ in range(2):
            src = src_gen.acquire_surface()
            batch.fill_slot(src)
        assert batch.num_filled == 2

    def test_fill_exceeds_batch_size_raises(self):
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 2)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        src1 = src_gen.acquire_surface()
        src2 = src_gen.acquire_surface()
        batch.fill_slot(src1)
        batch.fill_slot(src2)
        src3 = src_gen.acquire_surface()
        with pytest.raises(RuntimeError, match="overflow|Batch"):
            batch.fill_slot(src3)


@skip_no_runtime
class TestFillSlotWithSrcRect:
    """Tests for fill_slot with src_rect (source crop ROI)."""

    def test_fill_with_src_rect(self):
        """Fill a slot with a cropped source region (top, left, width, height)."""
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 2)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        src = src_gen.acquire_surface()
        roi = ds.Rect(top=100, left=200, width=800, height=600)
        batch.fill_slot(src, src_rect=roi, id=1)
        assert batch.num_filled == 1
        batch.finalize()
        buf = batch.as_gst_buffer()
        assert buf.ptr != 0

    def test_fill_mixed_src_rect_and_none(self):
        """Fill slots: one with src_rect, one without (full frame)."""
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 2)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        src1 = src_gen.acquire_surface()
        roi = ds.Rect(top=0, left=0, width=960, height=540)
        batch.fill_slot(src1, src_rect=roi, id=1)
        src2 = src_gen.acquire_surface()
        batch.fill_slot(src2, src_rect=None, id=2)
        assert batch.num_filled == 2
        batch.finalize()
        buf = batch.as_gst_buffer()
        assert buf.ptr != 0
        meta = ds.get_savant_id_meta(buf)
        meta_ids = [v for _kind, v in meta if _kind == "frame"]
        assert 1 in meta_ids
        assert 2 in meta_ids

    def test_fill_all_slots_with_different_rois(self):
        """Fill all slots with different source crop regions."""
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        rois = [
            ds.Rect(0, 0, 960, 540),  # top-left quadrant
            ds.Rect(0, 960, 960, 540),  # top-right
            ds.Rect(540, 0, 960, 540),  # bottom-left
            ds.Rect(540, 960, 960, 540),  # bottom-right
        ]
        for i, roi in enumerate(rois):
            src = src_gen.acquire_surface(id=10 + i)
            batch.fill_slot(src, src_rect=roi, id=10 + i)
        assert batch.num_filled == 4
        batch.finalize()
        buf = batch.as_gst_buffer()
        assert buf.ptr != 0

    def test_same_src_added_multiple_times_with_various_rois(self):
        """Reuse the same source buffer for multiple slots with different ROIs."""
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 3)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        src = src_gen.acquire_surface(id=1)
        rois = [
            ds.Rect(0, 0, 960, 540),  # top-left
            ds.Rect(0, 960, 960, 540),  # top-right
            ds.Rect(540, 480, 960, 540),  # center-ish
        ]
        for i, roi in enumerate(rois):
            batch.fill_slot(src, src_rect=roi, id=10 + i)
        assert batch.num_filled == 3
        batch.finalize()
        buf = batch.as_gst_buffer()
        assert buf.ptr != 0
        meta = ds.get_savant_id_meta(buf)
        meta_ids = [v for _kind, v in meta if _kind == "frame"]
        assert 10 in meta_ids
        assert 11 in meta_ids
        assert 12 in meta_ids

    def test_roi_partially_outside_viewport_succeeds(self):
        """ROI extending partially outside source bounds should succeed (clipped)."""
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 1)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        src = src_gen.acquire_surface()
        # ROI: top=1000, left=1800, width=200, height=200
        # Extends 80px right of 1920, 120px below 1080
        roi = ds.Rect(1000, 1800, 200, 200)
        batch.fill_slot(src, src_rect=roi, id=1)
        assert batch.num_filled == 1
        batch.finalize()
        buf = batch.as_gst_buffer()
        assert buf.ptr != 0

    def test_roi_completely_outside_viewport_fails(self):
        """ROI entirely outside source bounds must fail."""
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 1)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        src = src_gen.acquire_surface()
        # ROI completely outside: top=2000, left=2000, width=100, height=100
        roi = ds.Rect(2000, 2000, 100, 100)
        with pytest.raises(RuntimeError):
            batch.fill_slot(src, src_rect=roi, id=1)


@skip_no_runtime
class TestSlotPtr:
    def test_returns_tuple(self):
        gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = gen.acquire_batched_surface(config)
        result = batch.slot_ptr(0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_data_ptr_and_pitch_nonzero(self):
        gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = gen.acquire_batched_surface(config)
        data_ptr, pitch = batch.slot_ptr(0)
        assert data_ptr != 0
        assert pitch > 0

    def test_slot_ptr_out_of_bounds_raises(self):
        gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = gen.acquire_batched_surface(config)
        with pytest.raises(RuntimeError, match="bounds|Slot"):
            batch.slot_ptr(4)
        with pytest.raises(RuntimeError, match="bounds|Slot"):
            batch.slot_ptr(100)


@skip_no_runtime
class TestFinalize:
    def test_finalize_returns_nonzero(self):
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        src = src_gen.acquire_surface(id=42)
        batch.fill_slot(src, id=42)
        batch.finalize()
        buf = batch.as_gst_buffer()
        assert buf.ptr != 0

    def test_ids_propagated(self):
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        ids = [10, 20, 30, 40]
        for frame_id in ids:
            src = src_gen.acquire_surface(id=frame_id)
            batch.fill_slot(src, id=frame_id)
        batch.finalize()
        buf = batch.as_gst_buffer()
        meta = ds.get_savant_id_meta(buf)
        meta_ids = [v for _kind, v in meta if _kind == "frame"]
        for frame_id in ids:
            assert frame_id in meta_ids

    def test_finalize_twice_raises(self):
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        batch.finalize()
        with pytest.raises(RuntimeError, match="finalized"):
            batch.finalize()


@skip_no_runtime
class TestSetNumFilled:
    def test_set_num_filled_succeeds(self):
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        src = src_gen.acquire_surface(id=1)
        batch.fill_slot(src, id=1)
        batch.finalize()
        buf = batch.as_gst_buffer()
        ds.set_num_filled(buf, 3)
        # No exception means success; we cannot easily verify numFilled from Python

    def test_set_num_filled_overflow_raises(self):
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4)
        config = ds.TransformConfig()
        batch = batched_gen.acquire_batched_surface(config)
        src = src_gen.acquire_surface(id=1)
        batch.fill_slot(src, id=1)
        batch.finalize()
        buf = batch.as_gst_buffer()
        with pytest.raises(RuntimeError, match="overflow|Batch"):
            ds.set_num_filled(buf, 5)

    def test_set_num_filled_null_raises(self):
        with pytest.raises(ValueError, match="null"):
            ds.set_num_filled(0, 1)


# ── Memory-leak smoke tests ─────────────────────────────────────────────

_LEAK_ITERATIONS = 200
_WARMUP_ITERATIONS = 10
_CPU_GROWTH_LIMIT_KB = 10_000  # 10 MB
_GPU_GROWTH_LIMIT_MB = 20


def _gpu_mem_used_mb() -> int:
    """Query current GPU memory usage (MiB)."""
    return ds.gpu_mem_used_mib()


def _cpu_rss_kb() -> int:
    """Current process RSS in KB (Linux), read from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


@skip_no_runtime
class TestBatchedMemoryLeak:
    """Smoke tests: loop over batch provisioning / release, assert no leak."""

    def test_batched_fill_finalize_release_no_leak(self):
        """Full cycle: acquire src → acquire batch → fill → finalize → RAII release."""
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4, pool=2)
        config = ds.TransformConfig()

        for _ in range(_WARMUP_ITERATIONS):
            src = src_gen.acquire_surface(id=1)
            batch = batched_gen.acquire_batched_surface(config)
            for slot in range(4):
                batch.fill_slot(src, id=slot)
            batch.finalize()
            buf = batch.as_gst_buffer()
            del batch, buf, src

        gc.collect()
        cpu_before = _cpu_rss_kb()
        gpu_before = _gpu_mem_used_mb()

        for i in range(_LEAK_ITERATIONS):
            src = src_gen.acquire_surface(id=i)
            batch = batched_gen.acquire_batched_surface(config)
            for slot in range(4):
                batch.fill_slot(src, id=slot)
            batch.finalize()
            buf = batch.as_gst_buffer()
            del batch, buf, src

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

    def test_batched_acquire_drop_no_finalize_no_leak(self):
        """Acquire + fill but drop without finalize — batch returns to pool."""
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 4, pool=2)
        config = ds.TransformConfig()

        for _ in range(_WARMUP_ITERATIONS):
            src = src_gen.acquire_surface(id=1)
            batch = batched_gen.acquire_batched_surface(config)
            batch.fill_slot(src, id=1)
            del batch, src

        gc.collect()
        cpu_before = _cpu_rss_kb()
        gpu_before = _gpu_mem_used_mb()

        for i in range(_LEAK_ITERATIONS):
            src = src_gen.acquire_surface(id=i)
            batch = batched_gen.acquire_batched_surface(config)
            batch.fill_slot(src, id=i)
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

    def test_same_src_multiple_rois_release_no_leak(self):
        """Same source, multiple ROIs per batch, full release cycle."""
        src_gen = _make_src_gen("RGBA", 1920, 1080)
        batched_gen = _make_batched_gen("RGBA", 640, 640, 3, pool=2)
        config = ds.TransformConfig()
        rois = [
            ds.Rect(0, 0, 960, 540),
            ds.Rect(0, 960, 960, 540),
            ds.Rect(540, 480, 960, 540),
        ]

        for _ in range(_WARMUP_ITERATIONS):
            src = src_gen.acquire_surface(id=1)
            batch = batched_gen.acquire_batched_surface(config)
            for j, roi in enumerate(rois):
                batch.fill_slot(src, src_rect=roi, id=j)
            batch.finalize()
            buf = batch.as_gst_buffer()
            del batch, buf, src

        gc.collect()
        cpu_before = _cpu_rss_kb()
        gpu_before = _gpu_mem_used_mb()

        for i in range(_LEAK_ITERATIONS):
            src = src_gen.acquire_surface(id=i)
            batch = batched_gen.acquire_batched_surface(config)
            for j, roi in enumerate(rois):
                batch.fill_slot(src, src_rect=roi, id=j)
            batch.finalize()
            buf = batch.as_gst_buffer()
            del batch, buf, src

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
