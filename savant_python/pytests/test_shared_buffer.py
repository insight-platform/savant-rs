"""Tests for SharedBuffer — require CUDA/DeepStream runtime."""

from __future__ import annotations

import gc
import pytest

# Check for SharedBuffer API before importing deepstream
try:
    from savant_rs import deepstream as _ds

    HAS_SHARED_BUFFER = hasattr(_ds, "BufferGenerator") and hasattr(_ds, "SharedBuffer")
except ImportError:
    HAS_SHARED_BUFFER = False
    _ds = None

if not HAS_SHARED_BUFFER:
    pytest.skip(
        "savant_rs.deepstream has no BufferGenerator/SharedBuffer",
        allow_module_level=True,
    )

import savant_rs.deepstream as ds


def _shared_buffer_runtime_available() -> bool:
    """Return True if BufferGenerator + acquire work (CUDA/DeepStream runtime)."""
    try:
        ds.init_cuda(0)
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=1)
        buf = gen.acquire()
        return buf is not None and hasattr(buf, "strong_count")
    except Exception:
        return False


HAS_SHARED_BUFFER_RUNTIME = _shared_buffer_runtime_available()

skip_no_shared_buffer_runtime = pytest.mark.skipif(
    not HAS_SHARED_BUFFER_RUNTIME,
    reason="CUDA/DeepStream runtime not available for SharedBuffer",
)

skip_no_consume_helper = pytest.mark.skipif(
    not hasattr(ds, "_test_consume_shared_buffer"),
    reason="_test_consume_shared_buffer only available in debug builds",
)


@skip_no_shared_buffer_runtime
class TestConstructionAndProperties:
    """Construction and basic properties."""

    def setup_class(self) -> None:
        ds.init_cuda(0)

    def test_acquire_returns_shared_buffer_with_strong_count_one(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        assert buf is not None
        assert isinstance(buf, ds.SharedBuffer)
        assert buf.strong_count == 1

    def test_pts_ns_get_set_roundtrip(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        buf.pts_ns = 123456789
        assert buf.pts_ns == 123456789

    def test_duration_ns_get_set_roundtrip(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        buf.duration_ns = 33333333
        assert buf.duration_ns == 33333333

    def test_savant_ids_get_set_roundtrip(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        buf.set_savant_ids([(ds.SavantIdMetaKind.FRAME, 42)])
        assert buf.savant_ids() == [(ds.SavantIdMetaKind.FRAME, 42)]

    def test_shared_buffer_cannot_be_instantiated_from_python(self) -> None:
        with pytest.raises(TypeError):
            ds.SharedBuffer()

    def test_shared_buffer_has_no_clone_attribute(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        assert not hasattr(buf, "clone")


@skip_no_shared_buffer_runtime
@skip_no_consume_helper
class TestConsumptionLifecycle:
    """Consumption lifecycle via mock consumer."""

    def setup_class(self) -> None:
        ds.init_cuda(0)

    def test_consume_succeeds_when_strong_count_one(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        assert buf.strong_count == 1
        ds._test_consume_shared_buffer(buf)
        assert buf.is_consumed is True
        assert bool(buf) is False

    def test_after_consumption_pts_ns_raises_runtime_error(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        ds._test_consume_shared_buffer(buf)
        with pytest.raises(RuntimeError):
            _ = buf.pts_ns

    def test_after_consumption_repr_shows_consumed(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        ds._test_consume_shared_buffer(buf)
        assert "consumed" in repr(buf).lower()

    def test_consume_on_already_consumed_raises_runtime_error(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        ds._test_consume_shared_buffer(buf)
        with pytest.raises(RuntimeError):
            ds._test_consume_shared_buffer(buf)


@skip_no_shared_buffer_runtime
class TestSurfaceViewIntegration:
    """SurfaceView integration."""

    def setup_class(self) -> None:
        ds.init_cuda(0)

    def test_surface_view_from_buffer_works_increases_strong_count(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        assert buf.strong_count == 1
        view = ds.SurfaceView.from_buffer(buf, 0)
        assert view is not None
        assert buf.strong_count >= 2

    def test_strong_count_returns_to_one_after_del_view(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        view = ds.SurfaceView.from_buffer(buf, 0)
        assert buf.strong_count >= 2
        del view
        gc.collect()
        assert buf.strong_count == 1

    @skip_no_consume_helper
    def test_consume_fails_when_surface_view_alive(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        view = ds.SurfaceView.from_buffer(buf, 0)
        with pytest.raises(RuntimeError, match="outstanding"):
            ds._test_consume_shared_buffer(buf)
        assert buf.is_consumed is False
        assert bool(buf) is True

    @skip_no_consume_helper
    def test_after_failed_consume_buf_restored(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        view = ds.SurfaceView.from_buffer(buf, 0)
        with pytest.raises(RuntimeError):
            ds._test_consume_shared_buffer(buf)
        del view
        gc.collect()
        assert buf.is_consumed is False
        assert bool(buf) is True
        assert buf.strong_count == 1


@skip_no_shared_buffer_runtime
class TestInteropWithFunctions:
    """Interop with existing functions."""

    def setup_class(self) -> None:
        ds.init_cuda(0)

    def test_set_num_filled_accepts_shared_buffer(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        ds.set_num_filled(buf, 1)
        # No exception means success

    def test_get_savant_id_meta_accepts_shared_buffer(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        meta = ds.get_savant_id_meta(buf)
        assert isinstance(meta, list)

    def test_get_nvbufsurface_info_returns_valid_tuple(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire()
        info = ds.get_nvbufsurface_info(buf)
        assert isinstance(info, tuple)
        assert len(info) == 4
        data_ptr, pitch, width, height = info
        assert data_ptr != 0
        assert pitch > 0
        assert width == 64
        assert height == 64


@skip_no_shared_buffer_runtime
class TestAcquireWithParams:
    """acquire_with_params."""

    def setup_class(self) -> None:
        ds.init_cuda(0)

    def test_acquire_with_params_returns_correct_pts_and_duration(self) -> None:
        gen = ds.BufferGenerator("RGBA", 64, 64, pool_size=2)
        buf = gen.acquire_with_params(pts_ns=42000, duration_ns=33333)
        assert buf is not None
        assert isinstance(buf, ds.SharedBuffer)
        assert buf.pts_ns == 42000
        assert buf.duration_ns == 33333
