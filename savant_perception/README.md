# savant-perception-framework

Synchronous actor framework for real-time video-analytics pipelines
built on top of the `savant-rs` DeepStream stack, plus an end-to-end
reference example:

```
MP4 file -> NVDEC decode -> YOLOv11n inference -> NvDCF tracking -> Picasso bbox/label rendering -> H.264 encode -> MP4 mux
```

The pipeline is **fully streaming**: all stage boundaries are bounded
channels, so memory stays O(pipeline-depth) regardless of input
duration. Arbitrarily long files (hours) are supported without
running out of memory.

The crate ships in two layers:

* **Layer A — runtime primitives**: `Actor`, `Source`, `Handler`,
  `Envelope`, `Router`, `SharedStore`, `System`, …
* **Layer B — typed stages**: `Decoder`, `NvInfer`, `NvTracker`,
  `Mp4DemuxerSource`, `Mp4Muxer`, `Picasso`, `Function` — each with a
  grouped-builder API (`.inbox(...)`, `.results(...)`, `.common(...)`).

An executable reference pipeline lives under `examples/cars_demo/`
(standard Cargo [examples](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#examples)
layout; not part of the published library API).

## Examples

| Name | Description | Extra README |
|------|-------------|--------------|
| `cars-demo` | Detects, tracks and draws vehicles (car / bus / truck / motorbike) | examples/README.md |

## Prerequisites

- NVIDIA GPU (dGPU) or Jetson with DeepStream 7.1 runtime installed (the NvDCF tracker library at `/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so`).
- The `yolo11n.onnx` model and NvDCF YAML assets bundled with this workspace (under `savant_deepstream/nvinfer/assets/` and `savant_deepstream/nvtracker/assets/`).

## Architecture

See the crate documentation (`cargo doc --open -p savant-perception-framework`). Each example stage is its own module with a single `new()` constructor and one input method — no stage holds on to frames after the next stage has picked them up.
