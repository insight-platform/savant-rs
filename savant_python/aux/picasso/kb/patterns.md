# Test Patterns & Recipes

## Two Test Categories

### NOGPU Tests (test_picasso_init.py style)
- No CUDA, no NvBufSurfaceGenerator, no send_frame
- Test: GeneralSpec, Callbacks, SourceSpec, CodecSpec (drop/bypass), ConditionalSpec, ObjectDrawSpec, EvictionDecision, engine init/shutdown
- Only need picasso import guard (no deepstream runtime check)

### GPU Tests (test_picasso_pipeline.py style)  
- Need CUDA + DeepStream runtime
- Test: full pipeline (send_frame, encode, bypass), callbacks invocation, encoded output
- Need additional runtime guard (see below)

---

## File Header Template (GPU tests)
```python
from __future__ import annotations
import threading
import time
import pytest

_mod = pytest.importorskip("savant_rs.picasso")
if not hasattr(_mod, "PicassoEngine"):
    pytest.skip("deepstream feature disabled", allow_module_level=True)

_ds = pytest.importorskip("savant_rs.deepstream")
if not hasattr(_ds, "NvBufSurfaceGenerator"):
    pytest.skip("deepstream feature disabled", allow_module_level=True)

def _ds_runtime_available() -> bool:
    try:
        from savant_rs.deepstream import init_cuda
        init_cuda(0)
        return True
    except Exception:
        return False

if not _ds_runtime_available():
    pytest.skip("DeepStream/CUDA runtime not available", allow_module_level=True)

# ... imports after guards ...
```

## File Header Template (NOGPU tests)
```python
from __future__ import annotations
import pytest

_mod = pytest.importorskip("savant_rs.picasso")
if not hasattr(_mod, "PicassoEngine"):
    pytest.skip("deepstream feature disabled", allow_module_level=True)

from savant_rs.picasso import (
    Callbacks, CodecSpec, ConditionalSpec, EvictionDecision,
    GeneralSpec, ObjectDrawSpec, PicassoEngine, SourceSpec,
)
```

---

## Engine Lifecycle Pattern
```python
# 1. Create engine
engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
# 2. Set source spec(s)
engine.set_source_spec("src-0", spec)
# 3. Send frames (GPU only)
engine.send_frame("src-0", frame, buf_ptr)
# 4. Send EOS when done
engine.send_eos("src-0")
# 5. Wait for async processing
time.sleep(2)  # adjust based on frame count
# 6. Shutdown
engine.shutdown()
```

⚠ Always call `engine.shutdown()` — even in test cleanup.
⚠ Use `time.sleep()` between EOS and shutdown to allow async workers to finish.
⚠ Callbacks fire on Rust threads — use `threading.Lock()` to protect shared state.

---

## Thread-Safe Callback Collection Pattern
```python
results: list = []
lock = threading.Lock()

def on_encoded(output) -> None:
    with lock:
        results.append(output)

callbacks = Callbacks(on_encoded_frame=on_encoded)
```

---

## Build Helpers (reusable)

### Make VideoFrame
```python
def make_frame(source_id: str, width=1280, height=720) -> VideoFrame:
    return VideoFrame(
        source_id=source_id,
        framerate="30/1",
        width=width, height=height,
        content=VideoFrameContent.none(),
        time_base=(1, 1_000_000_000),
        pts=0,
    )
```

### Make GPU Buffer
```python
FPS = 30
FRAME_DURATION_NS = 1_000_000_000 // FPS

gen = NvBufSurfaceGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)
buf_ptr = gen.acquire_surface(id=frame_idx)
# pts/duration are taken from the VideoFrame; set them before send_frame:
frame.pts = frame_idx * FRAME_DURATION_NS
frame.duration = FRAME_DURATION_NS
```

### Build H.264 Encoder Config
```python
def build_encoder_config(width=1280, height=720, fps=30) -> EncoderConfig:
    props = EncoderProperties.h264_dgpu(
        H264DgpuProps(bitrate=4_000_000, preset=DgpuPreset.P1,
                      tuning_info=TuningPreset.LOW_LATENCY, iframeinterval=30)
    )
    cfg = EncoderConfig(Codec.H264, width, height)
    cfg.format(VideoFormat.RGBA)
    cfg.fps(fps, 1)
    cfg.properties(props)
    return cfg
```

### Build Encode SourceSpec
```python
def build_encode_spec(width=1280, height=720) -> SourceSpec:
    return SourceSpec(
        codec=CodecSpec.encode(TransformConfig(), build_encoder_config(width, height)),
        draw=build_draw_spec(),  # optional
        font_family="monospace",
    )
```

### Build ObjectDrawSpec
```python
def build_draw_spec() -> ObjectDrawSpec:
    spec = ObjectDrawSpec()
    border = ColorDraw(255, 80, 80, 255)
    bg = ColorDraw(255, 80, 80, 50)
    bb = BoundingBoxDraw(border, bg, 2, PaddingDraw.default_padding())
    dot = DotDraw(ColorDraw(255, 80, 80, 255), 4)
    label = LabelDraw(
        font_color=ColorDraw(0, 0, 0, 255),
        background_color=ColorDraw(255, 80, 80, 200),
        border_color=ColorDraw(0, 0, 0, 0),
        font_scale=1.4, thickness=1,
        position=LabelPosition.default_position(),
        padding=PaddingDraw(4, 2, 4, 2),
        format=["{label} #{id}", "{confidence}"],
    )
    od = ObjectDraw(bounding_box=bb, central_dot=dot, label=label)
    spec.insert("detector", "person", od)
    return spec
```

### Add Objects to Frame
```python
def add_objects(frame: VideoFrame, n: int = 20) -> None:
    for i in range(n):
        obj = VideoObject(
            id=0, namespace="detector", label="person",
            detection_box=RBBox(640.0 + i * 10.0, 360.0, 100.0, 150.0),
            attributes=[], confidence=None, track_id=None, track_box=None,
        )
        frame.add_object(obj, IdCollisionResolutionPolicy.GenerateNewId)
```

---

## Full Encode Pipeline Test Template
```python
class TestEncode:
    def test_basic(self) -> None:
        init_cuda(0)
        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            with lock:
                results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", build_encode_spec())

        gen = NvBufSurfaceGenerator(VideoFormat.RGBA, 1280, 720, 30, 1, 0)

        for i in range(10):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            add_objects(frame)
            buf_ptr = gen.acquire_surface_with_params(
                pts_ns=i * FRAME_DURATION_NS,
                duration_ns=FRAME_DURATION_NS, id=i,
            )
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(3)
        engine.shutdown()

        assert len(results) > 0
        video_frames = [o for o in results if o.is_video_frame]
        assert len(video_frames) > 0
        vf = video_frames[0].as_video_frame()
        assert vf.source_id == "src-0"
```

## Bypass Pipeline Test Template
```python
class TestBypass:
    def test_basic(self) -> None:
        init_cuda(0)
        bypass_results: list = []
        lock = threading.Lock()

        def on_bypass(output) -> None:
            with lock:
                bypass_results.append(output)

        callbacks = Callbacks(on_bypass_frame=on_bypass)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", SourceSpec(codec=CodecSpec.bypass()))

        gen = NvBufSurfaceGenerator(VideoFormat.RGBA, 1280, 720, 30, 1, 0)

        for i in range(5):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            buf_ptr = gen.acquire_surface_with_params(
                pts_ns=i * FRAME_DURATION_NS,
                duration_ns=FRAME_DURATION_NS, id=i,
            )
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(3)
        engine.shutdown()

        assert len(bypass_results) > 0
        assert bypass_results[0].source_id == "src-0"
        assert bypass_results[0].frame is not None
```

## Drop Pipeline Test Template
```python
class TestDrop:
    def test_no_output(self) -> None:
        results: list = []
        callbacks = Callbacks(on_encoded_frame=lambda o: results.append(o))
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", SourceSpec(codec=CodecSpec.drop_frames()))

        gen = NvBufSurfaceGenerator(VideoFormat.RGBA, 1280, 720, 30, 1, 0)

        for i in range(5):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            buf_ptr = gen.acquire_surface_with_params(
                pts_ns=i * FRAME_DURATION_NS,
                duration_ns=FRAME_DURATION_NS, id=i,
            )
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(1)
        engine.shutdown()

        assert len([o for o in results if o.is_video_frame]) == 0
```

---

## Common Assertions

### Encoded output
```python
assert output.is_video_frame or output.is_eos
vf = output.as_video_frame()  # raises if EOS
assert vf.source_id == "src-0"
assert vf.width == 1280
assert vf.height == 720
assert vf.framerate == "30/1"
objects = vf.get_all_objects()
```

### Bypass output
```python
assert output.source_id == "src-0"
assert output.frame is not None
```

### Engine state
```python
assert "running" in repr(engine).lower()
engine.shutdown()
assert "shut_down" in repr(engine).lower()
```
