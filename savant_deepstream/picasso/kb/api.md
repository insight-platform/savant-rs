# Picasso Rust Public API

Crate: `picasso`
Prelude: `use picasso::prelude::*;`

---

## prelude re-exports
```rust
pub use crate::callbacks::{
    Callbacks, OnBypassFrame, OnEncodedFrame, OnEviction, OnGpuMat, OnObjectDrawSpec, OnRender,
};
pub use crate::engine::PicassoEngine;
pub use crate::error::PicassoError;
pub use crate::message::{BypassOutput, EncodedOutput};
pub use crate::spec::{
    CodecSpec, ConditionalSpec, EvictionDecision, GeneralSpec, ObjectDrawSpec, SourceSpec,
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
| `send_frame` | `(&self, source_id: &str, frame: VideoFrameProxy, buf: gstreamer::Buffer) → Result<(), PicassoError>` | Auto-creates worker with default Drop spec if source unknown |
| `send_eos` | `(&self, source_id: &str) → Result<(), PicassoError>` | No-op if source not found |
| `shutdown` | `(&mut self)` | Drains all workers, joins watchdog. Idempotent via flag. |

- Implements `Drop` (calls `shutdown` if not already done).
- ⚠ `send_frame`/`send_eos`/`set_source_spec` return `Err(PicassoError::Shutdown)` after shutdown.

---

## GeneralSpec

```rust
#[derive(Debug, Clone)]
pub struct GeneralSpec {
    pub idle_timeout_secs: u64,  // DEF: 30
}
```
Implements `Default`.

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
}
```

### Callback Traits (all `Send + Sync + 'static`)

```rust
pub trait OnEncodedFrame {
    fn call(&self, output: EncodedOutput);
}

pub trait OnBypassFrame {
    fn call(&self, output: BypassOutput);
}

pub trait OnRender {
    fn call(&self, source_id: &str, renderer: &mut SkiaRenderer, frame: &VideoFrameProxy);
}

pub trait OnObjectDrawSpec {
    fn call(&self, source_id: &str, object: &BorrowedVideoObject, current_spec: Option<&ObjectDraw>) → Option<ObjectDraw>;
}

pub trait OnGpuMat {
    fn call(&self, source_id: &str, frame: &VideoFrameProxy, data_ptr: usize, pitch: u32, width: u32, height: u32);
}

pub trait OnEviction {
    fn call(&self, source_id: &str) → EvictionDecision;
}
```

---

## EncodedOutput

```rust
pub enum EncodedOutput {
    VideoFrame(VideoFrameProxy),
    EndOfStream(EndOfStream),
}
```

---

## BypassOutput

```rust
pub struct BypassOutput {
    pub source_id: String,
    pub frame: VideoFrameProxy,
    pub buffer: gstreamer::Buffer,
}
```

---

## WorkerMessage (pub, used in tests)

```rust
pub enum WorkerMessage {
    Frame(VideoFrameProxy, gstreamer::Buffer),
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
| `spawn` | `(source_id: String, spec: SourceSpec, callbacks: Arc<Callbacks>, idle_timeout: Duration) → Self` |
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
) → (outer_w, outer_h, pad_left, pad_top, pad_right, pad_bottom)
```
Path: `picasso::transform::compute_letterbox_params`
