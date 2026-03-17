# Picasso Rust Public API

Crate: `picasso`
Prelude: `use picasso::prelude::*;`

---

## prelude re-exports
```rust
pub use crate::callbacks::{
    Callbacks, OnBypassFrame, OnEncodedFrame, OnEviction, OnGpuMat, OnObjectDrawSpec, OnRender,
    OnStreamReset, StreamResetReason,
};
pub use crate::engine::PicassoEngine;
pub use crate::error::PicassoError;
pub use crate::message::OutputMessage;
pub use crate::spec::{
    CallbackInvocationOrder, CodecSpec, ConditionalSpec, EvictionDecision, GeneralSpec,
    ObjectDrawSpec, PtsResetPolicy, SourceSpec,
};
```

Also re-exported from `picasso` root (lib.rs):
```rust
pub use pipeline::encode::rewrite_frame_transformations;
pub use savant_core::primitives::eos::EndOfStream;
```

---

## PicassoEngine

```rust
pub struct PicassoEngine { /* private */ }
```

| Method | Signature | Notes |
|---|---|---|
| `new` | `(general: GeneralSpec, callbacks: Callbacks) → Self` | Spawns watchdog thread |
| `set_source_spec` | `(&self, source_id: &str, spec: SourceSpec) → Result<(), PicassoError>` | Creates worker on first call; sends UpdateSpec if worker exists |
| `remove_source_spec` | `(&self, source_id: &str)` | Sends Shutdown to worker, removes from map |
| `send_frame` | `(&self, source_id: &str, frame: VideoFrameProxy, view: SurfaceView, src_rect: Option<Rect>) → Result<(), PicassoError>` | Auto-creates worker with default Drop spec if source unknown. `src_rect` is optional per-frame crop. `SurfaceView` from `deepstream_nvbufsurface`. |
| `send_eos` | `(&self, source_id: &str) → Result<(), PicassoError>` | No-op if source not found |
| `shutdown` | `(&mut self)` | Drains all workers, joins watchdog. Idempotent via flag. |

- Implements `Drop` (calls `shutdown` if not already done).
- ⚠ `send_frame`/`send_eos`/`set_source_spec` return `Err(PicassoError::Shutdown)` after shutdown.

---

## GeneralSpec

```rust
#[derive(Debug, Clone)]
pub struct GeneralSpec {
    pub name: String,                // DEF: "" — used internally for logging and future extensibility
    pub idle_timeout_secs: u64,      // DEF: 30
    pub inflight_queue_size: usize,  // DEF: 8 — capacity of the per-worker crossbeam bounded channel
    pub pts_reset_policy: PtsResetPolicy, // DEF: PtsResetPolicy::EosOnDecreasingPts
}
```
Implements `Default`. `DEFAULT_INFLIGHT_QUEUE_SIZE` constant = 8.

---

## EvictionDecision

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictionDecision {
    KeepFor(u64),
    Terminate,
    TerminateImmediately,
}
```

---

## CodecSpec

```rust
#[derive(Debug, Clone)]
pub enum CodecSpec {
    Drop,
    Bypass,
    Encode {
        transform: TransformConfig,
        encoder: Box<EncoderConfig>,
    },
}
```

- `TransformConfig` from `deepstream_nvbufsurface`
- `EncoderConfig` from `deepstream_encoders`

---

## SourceSpec

```rust
#[derive(Debug, Clone)]
pub struct SourceSpec {
    pub codec: CodecSpec,              // DEF: CodecSpec::Drop
    pub conditional: ConditionalSpec,  // DEF: Default (no gates)
    pub draw: ObjectDrawSpec,          // DEF: Default (empty)
    pub font_family: String,           // DEF: "sans-serif"
    pub idle_timeout_secs: Option<u64>,// DEF: None (use engine's)
    pub use_on_render: bool,           // DEF: false
    pub use_on_gpumat: bool,           // DEF: false
    pub callback_order: CallbackInvocationOrder, // DEF: CallbackInvocationOrder::SkiaGpuMat
}
```
Implements `Default`.

---

## ConditionalSpec

```rust
#[derive(Debug, Clone, Default)]
pub struct ConditionalSpec {
    pub encode_attribute: Option<(String, String)>,  // (namespace, name)
    pub render_attribute: Option<(String, String)>,
}
```
- `encode_attribute`: frame must have this attribute to be processed at all
- `render_attribute`: frame must have this attribute for Skia rendering stage

---

## ObjectDrawSpec

```rust
#[derive(Debug, Clone, Default)]
pub struct ObjectDrawSpec { /* HashMap<(String,String), ObjectDraw> */ }
```

| Method | Signature |
|---|---|
| `new` | `() → Self` |
| `insert` | `(&mut self, namespace: &str, label: &str, draw: ObjectDraw)` |
| `lookup` | `(&self, namespace: &str, label: &str) → Option<&ObjectDraw>` |
| `iter` | `(&self) → impl Iterator<Item=(&(String,String), &ObjectDraw)>` |
| `is_empty` | `(&self) → bool` |
| `len` | `(&self) → usize` |

`ObjectDraw` from `savant_core::draw`.

---

## Callbacks

```rust
#[derive(Default)]
pub struct Callbacks {
    pub on_encoded_frame: Option<Arc<dyn OnEncodedFrame>>,
    pub on_bypass_frame:  Option<Arc<dyn OnBypassFrame>>,
    pub on_render:        Option<Arc<dyn OnRender>>,
    pub on_object_draw_spec: Option<Arc<dyn OnObjectDrawSpec>>,
    pub on_gpumat:        Option<Arc<dyn OnGpuMat>>,
    pub on_eviction:      Option<Arc<dyn OnEviction>>,
    pub on_stream_reset:  Option<Arc<dyn OnStreamReset>>,
}
```

### Callback Traits (all `Send + Sync + 'static`)

```rust
pub trait OnEncodedFrame {
    fn call(&self, output: OutputMessage);
}

pub trait OnBypassFrame {
    fn call(&self, output: OutputMessage);
}

pub trait OnRender {
    fn call(&self, source_id: &str, renderer: &mut SkiaRenderer, frame: &VideoFrameProxy);
}

pub trait OnObjectDrawSpec {
    fn call(&self, source_id: &str, object: &BorrowedVideoObject, current_spec: Option<&ObjectDraw>) → Option<ObjectDraw>;
}

pub trait OnGpuMat {
    fn call(&self, source_id: &str, frame: &VideoFrameProxy, view: &SurfaceView);
}

pub trait OnEviction {
    fn call(&self, source_id: &str) → EvictionDecision;
}

pub enum StreamResetReason {
    PtsDecreased { last_pts_ns: u64, new_pts_ns: u64 },
}

pub trait OnStreamReset {
    fn call(&self, source_id: &str, reason: StreamResetReason);
}
```

**Pointer validity (OnGpuMat):** The `data_ptr()` obtained from `&SurfaceView` is only valid for the duration of the callback. Storing the raw pointer for later use is undefined behaviour. On Jetson the pointer is tied to the EGL-CUDA registration; on dGPU it's the NvBufSurface `dataPtr`. The buffer may be recycled after the encode pipeline returns.

- ⚠ `OnObjectDrawSpec`: callback-returned `labelDraw.format` is resolved ephemerally and never written to the template cache.

---

## OutputMessage

```rust
pub enum OutputMessage {
    VideoFrame(VideoFrameProxy),
    EndOfStream(EndOfStream),
}
```

Bypass frames are now wrapped in `OutputMessage::VideoFrame`. EOS for bypass
sources is delivered via `OutputMessage::EndOfStream` through `on_bypass_frame`.

---

## WorkerMessage (pub, used in tests)

```rust
pub enum WorkerMessage {
    Frame(VideoFrameProxy, SurfaceView, Option<Rect>),
    Eos,
    UpdateSpec(Box<SourceSpec>),
    Shutdown,
}
```
Path: `picasso::message::WorkerMessage`

---

## SourceWorker (pub, used in low-level tests)

```rust
pub struct SourceWorker { /* private */ }
```

| Method | Signature |
|---|---|
| `spawn` | `(source_id: String, spec: SourceSpec, callbacks: Arc<Callbacks>, idle_timeout: Duration, queue_size: usize, pts_reset_policy: PtsResetPolicy) → Self` |
| `send` | `(&self, msg: WorkerMessage) → Result<(), SendError<WorkerMessage>>` |
| `is_alive` | `(&self) → bool` |

Path: `picasso::worker::SourceWorker`

---

## rewrite_frame_transformations (pub, used in geometry tests)

```rust
pub fn rewrite_frame_transformations(
    frame: &VideoFrameProxy,
    target_w: u32,
    target_h: u32,
    config: &TransformConfig,
    src_rect: Option<&Rect>,
) → Result<(), PicassoError>
```
Path: `picasso::rewrite_frame_transformations` (re-exported from lib.rs)

Appends GPU operations (crop + letterbox) to frame's transformation chain and calls `transform_forward`.
⚠ Frame must have exactly `[InitialSize(w, h)]` in its chain, matching frame's width/height.

---

## transform::compute_letterbox_params (pub)

```rust
pub fn compute_letterbox_params(
    src_w: u64, src_h: u64, dst_w: u64, dst_h: u64, padding: Padding,
    dst_padding: Option<DstPadding>,
) → anyhow::Result<(outer_w, outer_h, pad_left, pad_top, pad_right, pad_bottom)>
```
Path: `picasso::transform::compute_letterbox_params`

Returns `Err` if `dst_padding` reduces the effective width or height below `MIN_EFFECTIVE_DIM` (16 px).

---

## GPU Utilities (from deepstream_nvbufsurface)

### SharedMutableGstBuffer
Shared currency for the encode pipeline. Wraps a `gst::Buffer` in interior mutability so multiple `SurfaceView` instances can hold references while the encoder receives the same shared handle.

- `SharedMutableGstBuffer::from(buf: gst::Buffer) → Self` — wrap a buffer for shared use
- Passed directly to `NvEncoder::submit_frame`; no need to recover the buffer from the view

### SurfaceView
```rust
pub struct SurfaceView { /* private */ }
```
Zero-copy view of a single GPU surface with cached parameters. Wraps either a refcounted `gst::Buffer` or a `SharedMutableGstBuffer` containing an NvBufSurface descriptor.

| Constructor | Signature | Notes |
|---|---|---|
| `wrap` | `(buf: gst::Buffer) → Self` | Plain wrapper, surface params zeroed. For Drop/Bypass paths and NOGPU tests. |
| `from_buffer` | `(buf: gst::Buffer, slot_index: u32) → Result<Self, NvBufSurfaceError>` | Extract view from NvBufSurface-backed buffer (consumes buf by value). On Jetson, the pointer from `data_ptr()` is tied to a permanent EGL-CUDA registration that lives with the buffer. Recover buffer with `into_buffer()`. |
| `from_shared` | `(shared: SharedMutableGstBuffer, slot_index: u32) → Result<Self, NvBufSurfaceError>` | Create view from shared buffer. Use `view.buffer()` for mutable access. Drop view before passing `SharedMutableGstBuffer` to `submit_frame`. |
| `from_cuda_ptr` | `(data_ptr, pitch, width, height, gpu_id, channels, color_format, keepalive) → Result<Self, NvBufSurfaceError>` | Wrap arbitrary CUDA device memory with synthetic descriptor. |

| Accessor | Signature |
|---|---|
| `buffer` | `(&self) → MutexGuard<gst::Buffer>` | Replaces both old `buffer()` and `buffer_mut()`. Use for read and write access. |
| `into_buffer` | `(self) → Result<gst::Buffer, SurfaceView>` | Consumes the view and recovers the underlying buffer. Fails (returns `Err(self)`) if other refs exist (e.g. when using `from_shared`). For encode path, drop the view and pass `SharedMutableGstBuffer` to `submit_frame` instead. |
| `data_ptr` | `(&self) → *mut c_void` |
| `pitch` | `(&self) → u32` |
| `width` | `(&self) → u32` |
| `height` | `(&self) → u32` |
| `gpu_id` | `(&self) → u32` |
| `channels` | `(&self) → u32` |
| `color_format` | `(&self) → u32` |

Path: `deepstream_nvbufsurface::SurfaceView`

### buffer_gpu_id
```rust
pub fn buffer_gpu_id(buf: &gstreamer::BufferRef) → Result<u32, TransformError>
```
Path: `deepstream_nvbufsurface::buffer_gpu_id`

Extracts the `gpuId` from the NvBufSurface inside a GStreamer buffer. Used by `process_encode` for GPU affinity validation.

### DsNvSurfaceBufferGenerator.gpu_id()
```rust
pub fn gpu_id(&self) → u32
```
Returns the GPU device ID this generator allocates buffers on. Stored at construction time.

### NvEncoder::submit_frame (from deepstream_encoders)
```rust
pub fn submit_frame(&mut self, buf: SharedMutableGstBuffer, ...) → ...
```
Takes `SharedMutableGstBuffer` instead of `gst::Buffer`. The encode pipeline passes the shared handle directly after dropping the `SurfaceView`; no buffer recovery step.

---

## process_encode flow (pipeline/encode.rs)

```
dst_buf = generator.transform(...) → gst::Buffer
shared = SharedMutableGstBuffer::from(dst_buf)
view = SurfaceView::from_shared(shared.clone(), 0)
... rendering + callbacks using view.buffer() (MutexGuard) ...
drop(view)
encoder.submit_frame(shared, ...) → pass SharedMutableGstBuffer directly
```
