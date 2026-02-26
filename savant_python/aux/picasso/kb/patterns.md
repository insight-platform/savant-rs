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

### Build PNG Encoder Config (CPU-based, lossless)
```python
def build_png_encoder_config(width=1280, height=720, fps=30) -> EncoderConfig:
    props = EncoderProperties.png(PngProps(compression_level=6))
    cfg = EncoderConfig(Codec.PNG, width, height)
    cfg.format(VideoFormat.RGBA)  # PNG requires RGBA
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

## on_gpumat Callback Drawing Pattern (OpenCV CUDA)

The `on_gpumat` callback receives a raw CUDA device pointer that can be
wrapped as a `cv2.cuda.GpuMat` for zero-copy OpenCV CUDA drawing.

```python
import cv2

_RGBA_CV_TYPE = cv2.CV_8UC4

class GpuMatRenderer:
    """Callable on_gpumat callback for OpenCV CUDA rendering."""

    def __init__(self, width: int, height: int):
        self._frame_idx = 0

    def __call__(self, source_id, frame, data_ptr, pitch, width, height):
        gpumat = cv2.cuda.createGpuMatFromCudaMemory(
            int(height), int(width), _RGBA_CV_TYPE, int(data_ptr), int(pitch),
        )
        stream = cv2.cuda.Stream()
        gpumat.setTo((20, 20, 28, 255), stream=stream)  # draw something
        stream.waitForCompletion()
        self._frame_idx += 1

renderer = GpuMatRenderer(WIDTH, HEIGHT)
callbacks = Callbacks(on_gpumat=renderer)
spec = SourceSpec(
    codec=CodecSpec.encode(TransformConfig(), enc_cfg),
    use_on_gpumat=True,
)
```

⚠ A single source is processed sequentially on one worker thread; no
locking is needed for the animation state within a single source.

⚠ The `on_gpumat` callback receives raw `(data_ptr, pitch, width, height)`.
Use `nvbuf_as_gpu_mat(data_ptr, pitch, width, height)` inside the callback.
Use `nvgstbuf_as_gpu_mat(buf_ptr)` outside callbacks (e.g. pre-filling
backgrounds before `send_frame`) where you have a `GstBuffer*` pointer.

---

## on_render Callback Drawing Pattern (Skia → FBO)

The `on_render` callback provides an OpenGL FBO id managed by Picasso's
internal Skia renderer.  Create a Skia surface from the FBO to draw into it.

⚠ `on_render` fires **after** draw-spec bbox rendering.  Anything you draw
in `on_render` appears **on top** of the bboxes.  Use it for HUD overlays
(sidebar, footer, watermarks), **not** for backgrounds.  To place content
behind bboxes, pre-fill the input NvBufSurface before `send_frame`.

⚠ **Do NOT call `canvas.clear()`** in `on_render` if you use draw spec —
it erases the bboxes Picasso already rendered.

### Preferred: SkiaCanvas helper

`SkiaCanvas` from `savant_rs.deepstream` wraps the GL boilerplate
(`GrGLInterface.MakeEGL()`, `GrDirectContext`, `GrBackendRenderTarget`,
`Surface`) into a single object.  Cache it on first call:

```python
from savant_rs.deepstream import SkiaCanvas

class SkiaOverlayRenderer:
    """Callable on_render callback that draws overlays into Picasso's FBO."""

    def __init__(self):
        self._canvas: SkiaCanvas | None = None

    def __call__(self, source_id, fbo_id, width, height, frame):
        if self._canvas is None:
            self._canvas = SkiaCanvas.from_fbo(fbo_id, width, height)

        # ⚠ CRITICAL when combining draw spec + on_render:
        self._canvas.gr_context.resetContext()

        c = self._canvas.canvas()
        # ⚠ Do NOT clear — draw spec bboxes are already on the canvas
        # Draw sidebar, HUD, or watermark here...

        objects = frame.get_all_objects()
        for obj in objects:
            # obj.namespace, obj.label, obj.id, obj.confidence
            # obj.detection_box → RBBox (.xc, .yc, .width, .height)
            pass

        self._canvas.gr_context.flushAndSubmit()

renderer = SkiaOverlayRenderer()
callbacks = Callbacks(on_render=renderer)
spec = SourceSpec(
    codec=CodecSpec.encode(TransformConfig(), enc_cfg),
    use_on_render=True,
)
```

### Alternative: raw skia-python API

If you need finer control over the GL context lifecycle (e.g. texture
caching, multi-context), use the raw skia-python API:

⚠ **On headless EGL systems** (typical for DeepStream), `skia.GrDirectContext.MakeGL()`
with no arguments returns `None`.  You **must** use
`skia.GrGLInterface.MakeEGL()` to get the GL interface first:

```python
import skia

GL_RGBA8 = 0x8058

class SkiaOverlayRenderer:
    def __init__(self):
        self._gr_context = None

    def __call__(self, source_id, fbo_id, width, height, frame):
        if self._gr_context is None:
            interface = skia.GrGLInterface.MakeEGL()
            self._gr_context = skia.GrDirectContext.MakeGL(interface)

        self._gr_context.resetContext()  # ← CRITICAL with draw spec

        fb_info = skia.GrGLFramebufferInfo(fbo_id, GL_RGBA8)
        target = skia.GrBackendRenderTarget(width, height, 0, 8, fb_info)
        surface = skia.Surface.MakeFromBackendRenderTarget(
            self._gr_context, target,
            skia.kTopLeft_GrSurfaceOrigin,
            skia.kRGBA_8888_ColorType,
            None,
        )
        canvas = surface.getCanvas()
        # ... draw ...
        surface.flushAndSubmit()
```

⚠ GPU textures (`makeTextureImage`) must be uploaded using the same
`GrDirectContext` — defer asset loading to the first callback invocation
when the GL context is current.

---

## Mp4Muxer Pattern

`Mp4Muxer` is thread-safe (`Send`) and can be used directly from the
`on_encoded_frame` callback without a relay queue or dedicated thread:

```python
from savant_gstreamer import Mp4Muxer

muxer = Mp4Muxer(codec, "/tmp/out.mp4", fps_num=30)

def on_encoded(output) -> None:
    if output.is_video_frame:
        vf = output.as_video_frame()
        if vf.content.is_internal():
            data = vf.content.get_data()
            muxer.push(data, vf.pts, vf.dts or vf.pts, vf.duration)

# ... run pipeline ...
# after engine.shutdown():
muxer.finish()
```

---

## Draw Spec + Callbacks Composition Pattern

Combine `ObjectDrawSpec` (automatic bbox rendering) with callbacks for
a layered scene.  The composition order is:

1. **Input surface** (background) → pre-filled via `nvgstbuf_as_gpu_mat` before `send_frame`
2. **Draw spec** → Picasso renders bboxes/labels/dots for all `VideoObject`s
3. **`on_render`** → Skia overlays (sidebar, HUD) on top of bboxes
4. **`on_gpumat`** → final CUDA-level post-processing (optional)

```python
import cv2
from savant_rs.deepstream import nvgstbuf_as_gpu_mat
from savant_rs.draw_spec import (
    BoundingBoxDraw, ColorDraw, DotDraw, LabelDraw,
    LabelPosition, ObjectDraw, PaddingDraw,
)
from savant_rs.picasso import ObjectDrawSpec
from savant_rs.primitives import VideoObject, IdCollisionResolutionPolicy
from savant_rs.primitives.geometry import RBBox

# 1. Build draw spec (once)
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
    spec.insert("detector", "person", ObjectDraw(bounding_box=bb, central_dot=dot, label=label))
    return spec

# 2. Per-frame: pre-fill bg + add objects + send
buf_ptr = gen.acquire_surface(id=i)
with nvgstbuf_as_gpu_mat(buf_ptr) as (mat, stream):
    mat.setTo((18, 20, 28, 255), stream=stream)       # dark background

frame = VideoFrame(source_id="src-0", framerate="30/1",
                   width=1280, height=720,
                   content=VideoFrameContent.none(),
                   time_base=(1, 1_000_000_000), pts=pts_ns)
frame.duration = duration_ns

obj = VideoObject(id=0, namespace="detector", label="person",
                  detection_box=RBBox(640.0, 360.0, 100.0, 150.0),
                  attributes=[], confidence=0.95,
                  track_id=None, track_box=None)
frame.add_object(obj, IdCollisionResolutionPolicy.GenerateNewId)
engine.send_frame("src-0", frame, buf_ptr)

# 3. SourceSpec wiring
spec = SourceSpec(
    codec=CodecSpec.encode(TransformConfig(), enc_cfg),
    draw=build_draw_spec(),
    use_on_render=True,   # for sidebar overlay
)
```

⚠ `RBBox` uses **centre-x, centre-y, width, height** format — not top-left.
⚠ Objects attached to the `VideoFrame` are available in both the draw spec
renderer and the `on_render` callback (via `frame.get_all_objects()`).
⚠ `BorrowedVideoObject` properties: `.namespace`, `.label`, `.id`,
`.confidence` (Optional[float]), `.detection_box` → `RBBox` (`.xc`, `.yc`,
`.width`, `.height`).
⚠ **Skia `resetContext()` is mandatory when combining draw spec + on_render.**
Picasso's draw-spec renderer and the user's `on_render` Skia context share the
same GL context but cache state independently.  Call `gr_context.resetContext()`
at the **start** of every `on_render` invocation — without it, graphics are
corrupted.  Example:
```python
class MyRenderer:
    def __call__(self, source_id, fbo_id, width, height, frame):
        if self._gr_context is None:
            self._init_context()
        self._gr_context.resetContext()  # ← CRITICAL
        fb_info = skia.GrGLFramebufferInfo(fbo_id, 0x8058)
        target = skia.GrBackendRenderTarget(width, height, 0, 8, fb_info)
        surface = skia.Surface.MakeFromBackendRenderTarget(
            self._gr_context, target,
            skia.kTopLeft_GrSurfaceOrigin,
            skia.kRGBA_8888_ColorType, None)
        # ... draw on surface.getCanvas() ...
        surface.flushAndSubmit()
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
