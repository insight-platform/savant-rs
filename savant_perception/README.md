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
* **Layer B — typed templates**: `Decoder`, `NvInfer`, `NvTracker`,
  `Mp4DemuxerSource`, `Mp4Muxer`, `Picasso`, `Function` — each with a
  grouped-builder API (`.inbox(...)`, `.results(...)`, `.common(...)`).

An executable reference pipeline lives under `examples/cars_demo/`
(standard Cargo [examples](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#examples)
layout; not part of the published library API).

## Examples

| Name | Description |
|------|-------------|
| `cars-demo` | Detects, tracks and draws vehicles (car / bus / truck / motorbike) from an MP4 file into another MP4 file, printing FPS. |

## Prerequisites

- NVIDIA GPU (dGPU) or Jetson with DeepStream 7.1 runtime installed (the NvDCF tracker library at `/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so`).
- The `yolo11n.onnx` model and NvDCF YAML assets bundled with this workspace (under `savant_deepstream/nvinfer/assets/` and `savant_deepstream/nvtracker/assets/`).

## Input asset

Asset acquisition is **out of scope** of this crate. Place the sample clip on disk yourself:

```bash
curl -L -o savant_perception/assets/ny_city_center.mov \
  https://eu-central-1.linodeobjects.com/savant-data/demo/ny_city_center.mov
```

(Any other MP4 with H.264, HEVC, AV1, VP8, VP9 or JPEG video works too.)

## Running

```bash
cargo run --release -p savant-perception-framework --example cars-demo -- \
  --input assets/ny_city_center.mov \
  --output /tmp/ny_city_center.out.mp4 \
  --gpu 0
```

Full options:

```
--input  <PATH>         Input MP4 file (required; resolved relative to the crate manifest if not absolute).
--output <PATH>         Output MP4 file (required).
--gpu    <ID>           CUDA device ID [default: 0].
--conf   <THRESHOLD>    Detection confidence threshold [default: 0.25].
--iou    <THRESHOLD>    NMS IoU threshold [default: 0.45].
--channel-cap <N>       Inter-stage channel capacity [default: 4].
```

The example fails fast with a clear error message if `--input` does not exist.

## Architecture

See the crate documentation (`cargo doc --open -p savant-perception-framework`). Each example stage is its own module with a single `new()` constructor and one input method — no stage holds on to frames after the next stage has picked them up.
