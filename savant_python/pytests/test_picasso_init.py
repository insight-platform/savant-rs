"""Tests for Picasso engine initialization with various options.

Covers GeneralSpec, Callbacks, SourceSpec, CodecSpec, ConditionalSpec,
ObjectDrawSpec, and PicassoEngine construction plus set_source_spec,
remove_source_spec, and shutdown. Uses drop_frames/bypass codec specs
only so tests do not require GPU/CUDA.
"""

from __future__ import annotations

import pytest

ds = pytest.importorskip("savant_rs.picasso")
if not hasattr(ds, "PicassoEngine"):
    pytest.skip(
        "savant_rs.picasso not compiled (deepstream feature disabled)",
        allow_module_level=True,
    )

from savant_rs.picasso import (
    Callbacks,
    CodecSpec,
    ConditionalSpec,
    EvictionDecision,
    GeneralSpec,
    ObjectDrawSpec,
    PicassoEngine,
    SourceSpec,
)


# ─── GeneralSpec ───────────────────────────────────────────────────────────


class TestGeneralSpec:
    """GeneralSpec construction and defaults."""

    def test_default_idle_timeout(self) -> None:
        spec = GeneralSpec()
        assert spec.idle_timeout_secs == 30

    def test_custom_idle_timeout(self) -> None:
        spec = GeneralSpec(idle_timeout_secs=60)
        assert spec.idle_timeout_secs == 60

    def test_repr(self) -> None:
        spec = GeneralSpec(idle_timeout_secs=45)
        assert "45" in repr(spec)
        assert "GeneralSpec" in repr(spec)


# ─── Callbacks ──────────────────────────────────────────────────────────────


class TestCallbacks:
    """Callbacks construction with various options."""

    def test_empty_callbacks(self) -> None:
        cb = Callbacks()
        assert "Callbacks" in repr(cb)

    def test_callbacks_with_on_encoded_frame(self) -> None:
        def on_encoded(_output: object) -> None:
            pass

        cb = Callbacks(on_encoded_frame=on_encoded)
        assert cb.on_encoded_frame is not None

    def test_callbacks_with_on_eviction(self) -> None:
        def on_eviction(source_id: str) -> EvictionDecision:
            assert isinstance(source_id, str)
            return EvictionDecision.terminate()

        cb = Callbacks(on_eviction=on_eviction)
        assert cb.on_eviction is not None


# ─── EvictionDecision ───────────────────────────────────────────────────────


class TestEvictionDecision:
    """EvictionDecision factory methods."""

    def test_keep_for(self) -> None:
        d = EvictionDecision.keep_for(10)
        assert "keep_for" in repr(d) or "10" in repr(d)

    def test_terminate(self) -> None:
        d = EvictionDecision.terminate()
        assert "terminate" in repr(d).lower()

    def test_terminate_immediately(self) -> None:
        d = EvictionDecision.terminate_immediately()
        assert "terminate" in repr(d).lower() or "immediately" in repr(d).lower()


# ─── CodecSpec (drop / bypass only) ─────────────────────────────────────────


class TestCodecSpec:
    """CodecSpec drop_frames and bypass (no GPU)."""

    def test_drop_frames(self) -> None:
        spec = CodecSpec.drop_frames()
        assert spec.is_drop is True
        assert spec.is_bypass is False
        assert spec.is_encode is False

    def test_bypass(self) -> None:
        spec = CodecSpec.bypass()
        assert spec.is_drop is False
        assert spec.is_bypass is True
        assert spec.is_encode is False


# ─── ConditionalSpec ───────────────────────────────────────────────────────


class TestConditionalSpec:
    """ConditionalSpec construction."""

    def test_default_empty(self) -> None:
        spec = ConditionalSpec()
        assert spec.encode_attribute is None
        assert spec.render_attribute is None

    def test_with_encode_attribute(self) -> None:
        spec = ConditionalSpec(encode_attribute=("ns", "attr"))
        assert spec.encode_attribute == ("ns", "attr")

    def test_with_render_attribute(self) -> None:
        spec = ConditionalSpec(render_attribute=("ns", "render_attr"))
        assert spec.render_attribute == ("ns", "render_attr")


# ─── ObjectDrawSpec ────────────────────────────────────────────────────────


class TestObjectDrawSpec:
    """ObjectDrawSpec construction."""

    def test_empty_draw_spec(self) -> None:
        spec = ObjectDrawSpec()
        assert spec.is_empty() is True
        assert len(spec) == 0


# ─── SourceSpec ────────────────────────────────────────────────────────────


class TestSourceSpec:
    """SourceSpec construction with various options."""

    def test_defaults(self) -> None:
        spec = SourceSpec()
        assert spec.font_family == "sans-serif"
        assert spec.idle_timeout_secs is None
        assert spec.use_on_render is False
        assert spec.use_on_gpumat is False
        assert spec.codec.is_drop is True

    def test_with_drop_codec(self) -> None:
        spec = SourceSpec(codec=CodecSpec.drop_frames())
        assert spec.codec.is_drop is True

    def test_with_bypass_codec(self) -> None:
        spec = SourceSpec(codec=CodecSpec.bypass())
        assert spec.codec.is_bypass is True

    def test_with_conditional(self) -> None:
        cond = ConditionalSpec(encode_attribute=("n", "a"))
        spec = SourceSpec(conditional=cond)
        assert spec.conditional.encode_attribute == ("n", "a")

    def test_with_font_family(self) -> None:
        spec = SourceSpec(font_family="monospace")
        assert spec.font_family == "monospace"

    def test_with_idle_timeout(self) -> None:
        spec = SourceSpec(idle_timeout_secs=90)
        assert spec.idle_timeout_secs == 90

    def test_use_on_render(self) -> None:
        spec = SourceSpec(use_on_render=True)
        assert spec.use_on_render is True

    def test_use_on_gpumat(self) -> None:
        spec = SourceSpec(use_on_gpumat=True)
        assert spec.use_on_gpumat is True


# ─── PicassoEngine ────────────────────────────────────────────────────────


class TestPicassoEngineInit:
    """PicassoEngine creation with various GeneralSpec and Callbacks."""

    def test_engine_default_spec_no_callbacks(self) -> None:
        general = GeneralSpec()
        callbacks = Callbacks()
        engine = PicassoEngine(general, callbacks)
        assert "running" in repr(engine).lower() or "PicassoEngine" in repr(engine)

    def test_engine_custom_idle_timeout(self) -> None:
        general = GeneralSpec(idle_timeout_secs=120)
        callbacks = Callbacks()
        engine = PicassoEngine(general, callbacks)
        assert engine is not None

    def test_engine_with_on_encoded_callback(self) -> None:
        received: list[object] = []

        def on_encoded(output: object) -> None:
            received.append(output)

        general = GeneralSpec()
        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(general, callbacks)
        assert engine is not None
        engine.shutdown()

    def test_engine_with_on_eviction_callback(self) -> None:
        def on_eviction(source_id: str) -> EvictionDecision:
            return EvictionDecision.terminate()

        general = GeneralSpec()
        callbacks = Callbacks(on_eviction=on_eviction)
        engine = PicassoEngine(general, callbacks)
        assert engine is not None
        engine.shutdown()


class TestPicassoEngineSourceSpec:
    """set_source_spec and remove_source_spec."""

    def test_set_source_spec_drop(self) -> None:
        general = GeneralSpec()
        callbacks = Callbacks()
        engine = PicassoEngine(general, callbacks)
        spec = SourceSpec(codec=CodecSpec.drop_frames())
        engine.set_source_spec("source-1", spec)
        engine.remove_source_spec("source-1")
        engine.shutdown()

    def test_set_source_spec_bypass(self) -> None:
        general = GeneralSpec()
        callbacks = Callbacks()
        engine = PicassoEngine(general, callbacks)
        spec = SourceSpec(codec=CodecSpec.bypass())
        engine.set_source_spec("source-2", spec)
        engine.remove_source_spec("source-2")
        engine.shutdown()

    def test_set_source_spec_multiple_sources(self) -> None:
        general = GeneralSpec()
        callbacks = Callbacks()
        engine = PicassoEngine(general, callbacks)
        engine.set_source_spec("a", SourceSpec(codec=CodecSpec.drop_frames()))
        engine.set_source_spec("b", SourceSpec(codec=CodecSpec.bypass()))
        engine.remove_source_spec("a")
        engine.remove_source_spec("b")
        engine.shutdown()


class TestPicassoEngineShutdown:
    """Shutdown and post-shutdown behavior."""

    def test_shutdown_idempotent(self) -> None:
        engine = PicassoEngine(GeneralSpec(), Callbacks())
        engine.shutdown()
        engine.shutdown()

    def test_repr_after_shutdown(self) -> None:
        engine = PicassoEngine(GeneralSpec(), Callbacks())
        engine.shutdown()
        assert "shut_down" in repr(engine).lower() or "shutdown" in repr(engine).lower()

    def test_set_source_spec_after_shutdown_raises(self) -> None:
        engine = PicassoEngine(GeneralSpec(), Callbacks())
        engine.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            engine.set_source_spec("x", SourceSpec())

    def test_remove_source_spec_after_shutdown_raises(self) -> None:
        engine = PicassoEngine(GeneralSpec(), Callbacks())
        engine.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            engine.remove_source_spec("x")
