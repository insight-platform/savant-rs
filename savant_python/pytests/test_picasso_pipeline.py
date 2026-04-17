"""Integration tests for the Picasso GPU video pipeline.

Mirrors the Rust benchmark ``savant_deepstream/picasso/benches/skia_pipeline.rs``
but runs a small number of frames through the full encode path (Skia rendering +
H.264 encoding) and validates encoded output.

Requires the ``deepstream`` feature and a working DeepStream/CUDA runtime.
"""

from __future__ import annotations

import threading
import time

import pytest

from conftest import HAS_DS_FEATURE, HAS_DS_RUNTIME

if not HAS_DS_FEATURE:
    pytest.skip("savant_rs built without deepstream feature", allow_module_level=True)
if not HAS_DS_RUNTIME:
    pytest.skip("DeepStream/CUDA runtime not available", allow_module_level=True)


from savant_rs.deepstream import (
    MemType,
    BufferGenerator,
    SurfaceView,
    TransformConfig,
    VideoFormat,
    has_nvenc,
    init_cuda,
)
from savant_rs.draw_spec import (
    BoundingBoxDraw,
    ColorDraw,
    DotDraw,
    LabelDraw,
    LabelPosition,
    ObjectDraw,
    PaddingDraw,
)
from savant_rs.gstreamer import Codec
from savant_rs.picasso import (
    Av1DgpuProps,
    Callbacks,
    CodecSpec,
    DgpuPreset,
    EncoderConfig,
    EncoderProperties,
    EvictionDecision,
    GeneralSpec,
    H264DgpuProps,
    H264JetsonProps,
    HevcDgpuProps,
    HevcJetsonProps,
    JpegProps,
    ObjectDrawSpec,
    PicassoEngine,
    Platform,
    RateControl,
    SourceSpec,
    TuningPreset,
)
from savant_rs.primitives import (
    IdCollisionResolutionPolicy,
    VideoFrame,
    VideoFrameContent,
    VideoObject,
)
from savant_rs.match_query import MatchQuery
from savant_rs.primitives.geometry import RBBox

from picasso_helpers import (
    WIDTH,
    HEIGHT,
    FPS,
    FRAME_DURATION_NS,
    CLASSES,
    pseudo_rand,
    make_frame,
    make_nvmm_buffer,
    add_objects_to_frame,
    build_draw_spec,
    build_default_encoder_config as build_encoder_config,
    build_source_spec,
    poll_until,
    wait_for_eos,
)

NUM_FRAMES = 20
NUM_BOXES = 20


# ─── Encode pipeline tests ─────────────────────────────────────────────


class TestPicassoPipelineEncode:
    """End-to-end encode pipeline: frames -> Skia render -> H.264 output."""

    def test_encode_basic(self) -> None:
        """Feed frames through the full Skia + encode path and verify output."""
        init_cuda(0)

        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            with lock:
                results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", build_source_spec(use_render=False))

        gen = BufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(NUM_FRAMES):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        assert wait_for_eos(results, lock, timeout=15), "EOS not received in time"
        engine.shutdown()

        assert len(results) > 0, "expected at least one encoded frame"

        video_frames = [o for o in results if o.is_video_frame]
        assert len(video_frames) > 0, "expected at least one video frame output"

        vf = video_frames[0].as_video_frame()
        assert vf.source_id == "src-0"
        assert vf.width == WIDTH
        assert vf.height == HEIGHT
        assert vf.fps == (30, 1)

    def test_encode_objects_preserved(self) -> None:
        """Verify objects are carried through the encode pipeline."""
        init_cuda(0)

        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            with lock:
                results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", build_source_spec(use_render=False))

        gen = BufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(5):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        assert wait_for_eos(results, lock, timeout=15), "EOS not received in time"
        engine.shutdown()

        video_frames = [o for o in results if o.is_video_frame]
        assert len(video_frames) > 0
        for out in video_frames:
            vf = out.as_video_frame()
            objects = vf.get_all_objects()
            assert len(objects) == NUM_BOXES, (
                f"expected {NUM_BOXES} objects, got {len(objects)}"
            )

    def test_encode_multiple_sources(self) -> None:
        """Multiple sources encoding concurrently."""
        init_cuda(0)

        source_results: dict = {"src-0": [], "src-1": []}
        eos_sources: set[str] = set()
        lock = threading.Lock()

        def on_encoded(output) -> None:
            if output.is_video_frame:
                vf = output.as_video_frame()
                with lock:
                    if vf.source_id in source_results:
                        source_results[vf.source_id].append(vf)
            elif output.is_eos:
                eos = output.as_eos()
                with lock:
                    eos_sources.add(eos.source_id)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)

        for sid in source_results:
            engine.set_source_spec(sid, build_source_spec(use_render=False))

        gens = {
            sid: BufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)
            for sid in source_results
        }

        for i in range(10):
            for sid in source_results:
                frame = make_frame(sid)
                frame.pts = i * FRAME_DURATION_NS
                add_objects_to_frame(frame, i)
                buf_ptr = make_nvmm_buffer(gens[sid], i)
                engine.send_frame(sid, frame, buf_ptr)

        for sid in source_results:
            engine.send_eos(sid)

        expected_sources = set(source_results.keys())
        assert poll_until(
            lambda: expected_sources.issubset(eos_sources), timeout=15
        ), f"not all EOS received; got {eos_sources}"
        engine.shutdown()

        for sid, frames in source_results.items():
            assert len(frames) > 0, f"expected encoded frames from {sid}"
            for vf in frames:
                assert vf.source_id == sid
                assert vf.width == WIDTH
                assert vf.height == HEIGHT

    def test_encode_eos_delivered(self) -> None:
        """Verify EOS events are delivered via the encoded callback."""
        init_cuda(0)

        eos_received: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            if output.is_eos:
                with lock:
                    eos_received.append(output.as_eos())

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", build_source_spec(use_render=False))

        gen = BufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(3):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        assert poll_until(
            lambda: len(eos_received) >= 1, timeout=15
        ), "EOS not received in time"
        engine.shutdown()

        assert len(eos_received) >= 1, "expected EOS delivery"

    def test_encode_with_on_render_callback(self) -> None:
        """Verify on_render callback is invoked during encoding."""
        init_cuda(0)

        render_calls: list = []
        encoded_results: list = []
        lock = threading.Lock()

        def on_render(source_id, fbo_id, width, height, frame) -> None:
            with lock:
                render_calls.append((source_id, fbo_id, width, height))

        def on_encoded(output) -> None:
            with lock:
                encoded_results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded, on_render=on_render)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", build_source_spec(use_render=True))

        gen = BufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(10):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        assert wait_for_eos(
            encoded_results, lock, timeout=15
        ), "EOS not received in time"
        engine.shutdown()

        assert len(render_calls) > 0, "on_render was never called"
        for source_id, fbo_id, w, h in render_calls:
            assert source_id == "src-0"
            assert fbo_id > 0
            assert w == WIDTH
            assert h == HEIGHT

        video_frames = [o for o in encoded_results if o.is_video_frame]
        assert len(video_frames) > 0, "expected encoded output"


# ─── Bypass pipeline test ──────────────────────────────────────────────


class TestPicassoPipelineBypass:
    """Bypass mode: frames pass through without encoding."""

    def test_bypass_produces_output(self) -> None:
        init_cuda(0)

        bypass_results: list = []
        lock = threading.Lock()

        def on_bypass(output) -> None:
            with lock:
                bypass_results.append(output)

        callbacks = Callbacks(on_bypass_frame=on_bypass)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        spec = SourceSpec(codec=CodecSpec.bypass())
        engine.set_source_spec("src-0", spec)

        gen = BufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(5):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        assert wait_for_eos(
            bypass_results, lock, timeout=15
        ), "EOS not received in time"
        engine.shutdown()

        frames = [r for r in bypass_results if r.is_video_frame]
        assert len(frames) > 0, "expected bypass output"
        for out in frames:
            frame = out.as_video_frame()
            assert frame.source_id == "src-0"


# ─── Drop pipeline test ───────────────────────────────────────────────


class TestPicassoPipelineDrop:
    """Drop mode: frames are discarded entirely."""

    def test_drop_produces_no_encoded_output(self) -> None:
        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            with lock:
                results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        spec = SourceSpec(codec=CodecSpec.drop_frames())
        engine.set_source_spec("src-0", spec)

        gen = BufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(5):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        # Drop mode may or may not surface an EOS marker; either way
        # the pipeline should settle within a second. Poll to avoid a
        # fixed sleep dominating the test time.
        poll_until(lambda: any(o.is_eos for o in results), timeout=1.5)
        engine.shutdown()

        video_frames = [o for o in results if o.is_video_frame]
        assert len(video_frames) == 0, "drop mode should not produce video frames"


# ─── ObjectDrawSpec unit tests ─────────────────────────────────────────


class TestDrawSpecConstruction:
    """Verify ObjectDrawSpec can be built and populated like the benchmark."""

    def test_build_draw_spec(self) -> None:
        spec = build_draw_spec()
        assert not spec.is_empty()
        assert len(spec) == len(CLASSES)

    def test_lookup_existing_class(self) -> None:
        spec = build_draw_spec()
        for cls_name, _ in CLASSES:
            draw = spec.lookup("detector", cls_name)
            assert draw is not None, f"missing draw spec for {cls_name}"
            assert draw.bounding_box is not None
            assert draw.central_dot is not None
            assert draw.label is not None
            assert draw.blur is False

    def test_lookup_nonexistent(self) -> None:
        spec = build_draw_spec()
        assert spec.lookup("detector", "nonexistent") is None
        assert spec.lookup("other_ns", "person") is None

    def test_draw_spec_colors(self) -> None:
        """Verify per-class color correctness."""
        spec = build_draw_spec()
        for cls_name, (r, g, b) in CLASSES:
            draw = spec.lookup("detector", cls_name)
            assert draw is not None
            bc = draw.bounding_box.border_color
            assert (bc.red, bc.green, bc.blue, bc.alpha) == (r, g, b, 255)
            dc = draw.central_dot.color
            assert (dc.red, dc.green, dc.blue, dc.alpha) == (r, g, b, 255)

    def test_draw_spec_label_format(self) -> None:
        spec = build_draw_spec()
        draw = spec.lookup("detector", "person")
        assert draw is not None
        assert draw.label.format == ["{label} #{id}", "{confidence}"]
        assert draw.label.font_scale == pytest.approx(1.4)
        assert draw.label.thickness == 1


# ─── EncoderConfig unit tests ─────────────────────────────────────────


class TestEncoderConfig:
    """Verify encoder configuration matches the benchmark."""

    def test_h264_config_properties(self) -> None:
        cfg = build_encoder_config()
        r = repr(cfg)
        assert "Rgba" in r
        assert cfg.fps_num == FPS
        assert cfg.fps_den == 1
        assert cfg.encoder_params is not None

    def test_builder_methods(self) -> None:
        cfg = EncoderConfig(Codec.H264, 640, 480)
        cfg.format(VideoFormat.NV12)
        cfg.fps(60, 1)
        assert cfg.fps_num == 60
        assert cfg.fps_den == 1
        assert "Nv12" in repr(cfg)

    def test_encoder_properties_h264_dgpu(self) -> None:
        props = EncoderProperties.h264_dgpu(
            H264DgpuProps(
                bitrate=2_000_000,
                preset=DgpuPreset.P3,
                tuning_info=TuningPreset.HIGH_QUALITY,
                iframeinterval=15,
            )
        )
        assert props is not None

    def test_encoder_properties_hevc_dgpu(self) -> None:
        assert EncoderProperties.hevc_dgpu(HevcDgpuProps(bitrate=2_000_000)) is not None

    def test_encoder_properties_h264_jetson(self) -> None:
        assert EncoderProperties.h264_jetson(H264JetsonProps()) is not None

    def test_encoder_properties_hevc_jetson(self) -> None:
        assert EncoderProperties.hevc_jetson(HevcJetsonProps()) is not None

    def test_encoder_properties_jpeg(self) -> None:
        assert EncoderProperties.jpeg(JpegProps(quality=90)) is not None

    def test_encoder_properties_av1_dgpu(self) -> None:
        assert EncoderProperties.av1_dgpu(Av1DgpuProps(bitrate=3_000_000)) is not None


# ─── Platform enum tests ──────────────────────────────────────────────


class TestPlatformEnum:
    def test_dgpu(self) -> None:
        assert Platform.DGPU is not None

    def test_jetson(self) -> None:
        assert Platform.JETSON is not None

    def test_from_name_dgpu(self) -> None:
        assert Platform.from_name("dgpu") == Platform.DGPU

    def test_from_name_jetson(self) -> None:
        assert Platform.from_name("jetson") == Platform.JETSON

    def test_from_name_invalid(self) -> None:
        with pytest.raises(ValueError):
            Platform.from_name("unknown")


# ─── RateControl / preset enum tests ──────────────────────────────────


class TestEncoderEnums:
    def test_rate_control_variants(self) -> None:
        assert RateControl.VARIABLE_BITRATE is not None
        assert RateControl.CONSTANT_BITRATE is not None
        assert RateControl.CONSTANT_QP is not None

    def test_dgpu_preset_variants(self) -> None:
        for name in ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]:
            assert getattr(DgpuPreset, name) is not None

    def test_tuning_preset_variants(self) -> None:
        assert TuningPreset.HIGH_QUALITY is not None
        assert TuningPreset.LOW_LATENCY is not None
        assert TuningPreset.ULTRA_LOW_LATENCY is not None
        assert TuningPreset.LOSSLESS is not None


# ─── Frame + object generation tests ──────────────────────────────────


class TestFrameObjectGeneration:
    """Verify frame and object generation helpers produce valid data."""

    def test_make_frame(self) -> None:
        f = make_frame("test-src")
        assert f.source_id == "test-src"
        assert f.fps == (30, 1)
        assert f.width == WIDTH
        assert f.height == HEIGHT

    def test_add_objects(self) -> None:
        f = make_frame("test-src")
        add_objects_to_frame(f, 0)
        objects = f.get_all_objects()
        assert len(objects) == NUM_BOXES

    def test_objects_have_valid_labels(self) -> None:
        f = make_frame("test-src")
        add_objects_to_frame(f, 42)
        objects = f.access_objects(MatchQuery.idle())
        valid_labels = {c[0] for c in CLASSES}
        for obj in objects:
            assert obj.namespace == "detector"
            assert obj.label in valid_labels
            assert obj.detection_box is not None

    def test_objects_different_per_frame_idx(self) -> None:
        """Different frame_idx values produce different object positions."""
        f1 = make_frame("a")
        add_objects_to_frame(f1, 0)
        f2 = make_frame("b")
        add_objects_to_frame(f2, 100)

        objs1 = f1.get_all_objects()
        objs2 = f2.get_all_objects()
        boxes_differ = any(
            abs(o1.detection_box.xc - o2.detection_box.xc) > 0.001
            for o1, o2 in zip(objs1, objs2)
        )
        assert boxes_differ, "objects should move between frame indices"

    def test_pseudo_rand_deterministic(self) -> None:
        a = pseudo_rand(5, 100)
        b = pseudo_rand(5, 100)
        assert a == b

        c = pseudo_rand(5, 200)
        assert a != c

    def test_pseudo_rand_range(self) -> None:
        """Output is always in [0, 1)."""
        for s1 in range(50):
            for s2 in [100, 200, 300, 400, 500]:
                v = pseudo_rand(s1, s2)
                assert 0.0 <= v < 1.0, f"out of range: {v} for seeds ({s1}, {s2})"


# ─── SourceSpec / CodecSpec tests ─────────────────────────────────────


class TestSourceSpecVariants:
    def test_encode_spec(self) -> None:
        spec = build_source_spec(use_render=True)
        assert spec.codec.is_encode
        assert not spec.codec.is_drop
        assert not spec.codec.is_bypass
        assert spec.font_family == "monospace"
        assert spec.use_on_render is True

    def test_drop_spec(self) -> None:
        spec = SourceSpec(codec=CodecSpec.drop_frames())
        assert spec.codec.is_drop
        assert not spec.codec.is_encode
        assert not spec.codec.is_bypass

    def test_bypass_spec(self) -> None:
        spec = SourceSpec(codec=CodecSpec.bypass())
        assert spec.codec.is_bypass
        assert not spec.codec.is_encode
        assert not spec.codec.is_drop

    def test_source_spec_defaults(self) -> None:
        spec = SourceSpec()
        assert spec.codec.is_drop  # default is drop
        assert spec.font_family == "sans-serif"
        assert spec.use_on_render is False
        assert spec.use_on_gpumat is False
        assert spec.idle_timeout_secs is None


# ─── EvictionDecision tests ───────────────────────────────────────────


class TestEvictionDecision:
    def test_keep_for(self) -> None:
        d = EvictionDecision.keep_for(60)
        r = repr(d)
        assert "60" in r

    def test_terminate(self) -> None:
        d = EvictionDecision.terminate()
        r = repr(d)
        assert "terminate" in r.lower()

    def test_terminate_immediately(self) -> None:
        d = EvictionDecision.terminate_immediately()
        r = repr(d)
        assert "terminate" in r.lower()


# ─── Callbacks tests ──────────────────────────────────────────────────


class TestCallbacks:
    def test_empty_callbacks(self) -> None:
        cb = Callbacks()
        assert cb.on_encoded_frame is None
        assert cb.on_bypass_frame is None
        assert cb.on_render is None
        assert cb.on_object_draw_spec is None
        assert cb.on_gpumat is None
        assert cb.on_eviction is None
        assert "Callbacks" in repr(cb)

    def test_callbacks_with_functions(self) -> None:
        def noop(*_args):
            pass

        cb = Callbacks(
            on_encoded_frame=noop,
            on_bypass_frame=noop,
            on_render=noop,
            on_eviction=noop,
        )
        assert cb.on_encoded_frame is not None
        assert cb.on_bypass_frame is not None
        assert cb.on_render is not None
        assert cb.on_eviction is not None
        r = repr(cb)
        assert "on_encoded_frame" in r
        assert "on_bypass_frame" in r

    def test_callbacks_setter(self) -> None:
        cb = Callbacks()
        assert cb.on_encoded_frame is None

        def noop(*_args):
            pass

        cb.on_encoded_frame = noop
        assert cb.on_encoded_frame is not None

        cb.on_encoded_frame = None
        assert cb.on_encoded_frame is None


# ─── GeneralSpec tests ────────────────────────────────────────────────


class TestGeneralSpec:
    def test_default_idle_timeout(self) -> None:
        gs = GeneralSpec()
        assert gs.idle_timeout_secs == 30

    def test_custom_idle_timeout(self) -> None:
        gs = GeneralSpec(idle_timeout_secs=120)
        assert gs.idle_timeout_secs == 120

    def test_repr(self) -> None:
        gs = GeneralSpec(idle_timeout_secs=42)
        assert "42" in repr(gs)


# ─── Engine lifecycle tests ───────────────────────────────────────────


class TestPicassoEngineLifecycle:
    def test_engine_repr(self) -> None:
        engine = PicassoEngine(GeneralSpec(), Callbacks())
        assert "running" in repr(engine).lower()
        engine.shutdown()
        assert "shut_down" in repr(engine).lower()

    def test_shutdown_idempotent(self) -> None:
        engine = PicassoEngine(GeneralSpec(), Callbacks())
        engine.shutdown()
        engine.shutdown()  # should not raise

    def test_send_after_shutdown_raises(self) -> None:
        engine = PicassoEngine(GeneralSpec(), Callbacks())
        engine.shutdown()
        with pytest.raises(RuntimeError):
            engine.send_eos("src-0")

    def test_set_and_remove_source(self) -> None:
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), Callbacks())
        spec = SourceSpec(codec=CodecSpec.drop_frames())
        engine.set_source_spec("src-0", spec)
        engine.remove_source_spec("src-0")
        engine.shutdown()
