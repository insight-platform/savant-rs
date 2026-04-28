# `savant_perception` — runnable examples

Two reference binaries live in `examples/cars_demo/`:

| Example          | Path                                          | Topology                                                              |
| ---------------- | --------------------------------------------- | --------------------------------------------------------------------- |
| `cars-demo`      | `examples/cars_demo/main.rs`                  | file/URI → decode → YOLO → NvDCF → Picasso → MP4                      |
| `cars-demo-zmq`  | `examples/cars_demo/main_zmq.rs`              | hybrid ZMQ binary with `producer` / `pipeline` / `consumer` subcommands |

Both share the heavy lifting: the pipeline stages, asset lookup, warmup,
and per-stage stats live in `examples/cars_demo/cars_demo/` and are
re-used between the two binaries via a small `PipelineHead` /
`PipelineTail` enum (see `cars_demo::pipeline::run_pipeline`).

The samples are standard [Cargo
examples](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#examples)
— they are **not** part of the published library API.

---

## Prerequisites

- An NVIDIA GPU (dGPU or Jetson) with **DeepStream 7.1** runtime
  installed; in particular the NvDCF tracker library at
  `/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so`.
- The bundled YOLO11n ONNX + NvDCF YAML assets — present in-tree under
  `savant_deepstream/nvinfer/assets/` and
  `savant_deepstream/nvtracker/assets/`.
- TensorRT plan files are built on first run and cached under
  `savant_perception/assets/cache/<model>/<platform_tag>/` (see
  `engine_cache_dir`).
- Pre-built `skia-binaries` (the workspace `.envrc` exports
  `SKIA_BINARIES_URL` so you do not recompile Skia from source — read
  `.cursor/rules/skia-prebuilt.mdc` once).

```bash
source /workspaces/savant-rs/.envrc
```

Build both examples in release mode — `cars-demo` does heavy GPU
work (decode → infer → track → encode) and is several times slower
under the default `dev` profile, so the README defaults to
`--release` end-to-end:

```bash
cargo build --release -p savant-perception-framework --example cars-demo --example cars-demo-zmq
```

All commands below show the `cargo run --release` form, which
builds the example on demand and forwards arguments after `--`:

```bash
cargo run --release -p savant-perception-framework --example cars-demo -- --help
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- --help
```

Equivalent direct invocations of
`target/release/examples/cars-demo` and
`target/release/examples/cars-demo-zmq` work just as well — pick
whichever you prefer.  Drop `--release` only when you explicitly
want the slower `dev` profile (e.g. for a debugger session).

---

## `cars-demo` — single-process pipeline

End-to-end pipeline in one process.  `--input` accepts either a
filesystem path (relative paths are resolved against the
`savant_perception` crate manifest) or a URI; the URI path
auto-selects the `UriDemuxerSource` stage so RTSP / HTTP(S) / HLS
all work out of the box.

### 1. Default — full pipeline with overlay

Detect, track, draw, encode → MP4.

```bash
cargo run --release -p savant-perception-framework --example cars-demo -- \
  --input  /path/to/cars.mp4 \
  --output /tmp/cars-demo.mp4
```

### 2. URI input (RTSP / HTTP(S) / HLS / RTMP / `file://`)

The dispatcher inside `cars_demo::pipeline::run` swaps in
`UriDemuxerSource` when `--input` resolves to a URI.

```bash
cargo run --release -p savant-perception-framework --example cars-demo -- \
  --input  rtsp://camera/stream \
  --output /tmp/cars-demo.mp4

cargo run --release -p savant-perception-framework --example cars-demo -- \
  --input  https://example.com/playlist.m3u8 \
  --output /tmp/cars-demo.mp4

cargo run --release -p savant-perception-framework --example cars-demo -- \
  --input  file:///abs/path/clip.mp4 \
  --output /tmp/cars-demo.mp4
```

### 3. `--no-draw` — transform + encode, no overlay

Picasso still runs the GPU transform + encoder; only the Skia draw
stage (bounding boxes, labels, frame-id overlay) is skipped.  Useful
to measure raw decode + infer + track + encode throughput, or to
produce a clean re-encode of the input.

```bash
cargo run --release -p savant-perception-framework --example cars-demo -- \
  --input  /path/to/cars.mp4 \
  --output /tmp/cars-demo-clean.mp4 \
  --no-draw
```

### 4. `--output null` — Picasso runs, file suppressed

Sentinel that keeps Picasso (transform + draw + encode) but discards
the encoded bitstream.  Lets you measure full pipeline cost without
disk I/O.

```bash
cargo run --release -p savant-perception-framework --example cars-demo -- \
  --input  /path/to/cars.mp4 \
  --output null
```

### 5. `--no-picasso` — decode + infer + track only

Picasso, the encoder, and the muxer are all bypassed; decoded frames
flow through YOLO + NvDCF and are dropped.  `--output` becomes
optional and is ignored if supplied.  Useful for measuring the cost
of the analytics tail in isolation.

```bash
cargo run --release -p savant-perception-framework --example cars-demo -- \
  --input  /path/to/cars.mp4 \
  --no-picasso
```

`--no-picasso --output null` is **rejected** — the resolver refuses
to silently pick which "no-output" path you meant.

### 6. Common knobs

| Flag                        | Default | Effect                                                                                  |
| --------------------------- | ------- | --------------------------------------------------------------------------------------- |
| `--gpu <id>`                | `0`     | CUDA device index used by NVDEC, nvinfer, nvtracker, transform and the encoder.         |
| `--conf <f32>`              | `0.25`  | Post-sigmoid detection-confidence threshold (YOLO).                                     |
| `--iou  <f32>`              | `0.45`  | NMS IoU threshold (YOLO).                                                               |
| `--channel-cap <n>`         | `4`     | Inter-stage channel capacity (in-flight messages per boundary).                         |
| `--fps-num` / `--fps-den`   | `25/1`  | Encoder rate-control + container framerate.  Pipeline PTS still drives playback timing. |
| `--stats-period <secs>`     | `30`    | Period between intermediate FPS / per-stage stats reports.  A final report is always emitted on shutdown regardless of the period.  Same flag is accepted by `cars-demo-zmq pipeline`. |
| `--debug`                   | off     | Raises only this binary's log level to `debug` (third-party crates stay at `info`).     |

`cars-demo --help` prints the full reference.

---

## `cars-demo-zmq` — split pipeline over ZeroMQ

Same logical pipeline as `cars-demo`, but the ingress, the analytics
middle, and the egress can each run as a separate process talking
over ZMQ.  Three subcommands are exposed by **one** binary:

| Subcommand  | Role                                       | Exits when                                         |
| ----------- | ------------------------------------------ | -------------------------------------------------- |
| `producer`  | `file/URI → decode-frame envelopes → ZMQ`  | upstream demuxer reaches end-of-input + sends EOS  |
| `pipeline`  | `ZMQ → decode → YOLO → NvDCF → Picasso → ZMQ` | only on `Ctrl+C` (it is a pure transit pipeline) |
| `consumer`  | `ZMQ → MP4`                                | first wire EOS arrives for the active source       |

The pipeline subcommand uses the **same** middle stages as `cars-demo`
(reused via `PipelineHead::Zmq` / `PipelineTail::Zmq`).  ZMQ sources
and sinks are always treated as multi-stream transports — drop
decisions belong upstream; the consumer overrides
`ZmqSource::on_source_eos` to return `Flow::Stop` so it terminates on
the first EOS.

### Wire contract — payload carrier

Encoded bitstream may travel either as an extra ZMQ multipart segment
(default, `PayloadCarrier::Multipart`) or embedded inside
`VideoFrameContent::Internal`.  `ZmqSource` accepts both transparently
(multipart wins; the frame content is cleared to avoid duplicating
the bytes).  `ZmqSink` is configurable; the demos use `Multipart`
end-to-end.

### Default socket wiring (zero-arg three-shell run)

Every subcommand ships compatible default values for `--zmq-in` /
`--zmq-out`, so the canonical IPC topology runs without any
transport flags at all.  The defaults are:

| Role / flag         | Default URL                                       | Side    |
| ------------------- | ------------------------------------------------- | ------- |
| `producer --zmq-out`| `dealer+connect:ipc:///tmp/savant_demo_in`        | connect |
| `pipeline --zmq-in` | `router+bind:ipc:///tmp/savant_demo_in`           | bind    |
| `pipeline --zmq-out`| `pub+bind:ipc:///tmp/savant_demo_out`             | bind    |
| `consumer --zmq-in` | `sub+connect:ipc:///tmp/savant_demo_out`          | connect |

`producer ↔ pipeline` is `dealer ↔ router` (lossless, request-style
multipart) and `pipeline ↔ consumer` is `pub ↔ sub` (broadcast).
The two endpoints (`/tmp/savant_demo_in` and
`/tmp/savant_demo_out`) line up across the chain.

### Topology — three shells

In three separate terminals on the same host (so the `ipc://` paths
resolve):

```bash
# Shell 1 — pipeline.  Binds both endpoints, so it must start first.
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- pipeline
```

```bash
# Shell 2 — consumer.  Subscribes to the pipeline's output.
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- \
  consumer --output /tmp/cars-demo-zmq.mp4
```

```bash
# Shell 3 — producer.  Pushes the source clip into the pipeline.
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- \
  producer --input /path/to/cars.mp4
```

Ordering rule: start the **bind** peers (`pipeline`, then `consumer`)
before the **connect** peer (`producer`).  The pipeline keeps
running across multiple producer sessions; press `Ctrl+C` to
terminate it.

### Subcommand reference

Each subcommand also accepts an explicit `--zmq-in` / `--zmq-out`
override; pass it whenever you need a non-default endpoint (TCP,
multiple producers, custom socket types, etc.).

#### `producer`

```bash
# Minimal: only --input is required.
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- \
  producer --input /path/to/cars.mp4

# Full set of overrides:
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- producer \
  --input       /path/to/cars.mp4 \
  --zmq-out     dealer+connect:ipc:///tmp/savant_demo_in \
  --source-id   cars-demo-zmq \
  --channel-cap 4

# Suppress the trailing wire EOS — useful when feeding a long-running
# pipeline that must keep the same `--source-id` across multiple
# back-to-back producer runs without inducing per-source teardown.
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- producer \
  --input  /path/to/cars.mp4 \
  --no-eos
```

`--input` accepts any URI that `cars-demo --input` accepts (file
path, `file://`, `rtsp://`, `http(s)://`, `rtmp://`, `hls://`, …).
Exits as soon as the upstream demuxer drains and broadcasts the
shutdown signal — i.e. when the source clip ends or the URI EOFs.

`--no-eos` suppresses the terminating
`EncodedMsg::SourceEos` envelope: the demuxer stops forwarding
frames, the supervisor broadcasts Shutdown as soon as the demuxer
source exits, and `ZmqSink` is torn down without ever issuing a
wire EOS.  Downstream pipeline / consumer therefore observe the
silence but never transition to a "drained" state.

#### `pipeline`

```bash
# Minimal: zero arguments — defaults handle the IPC wiring.
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- pipeline

# Full set of overrides:
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- pipeline \
  --zmq-in      router+bind:ipc:///tmp/savant_demo_in \
  --zmq-out     pub+bind:ipc:///tmp/savant_demo_out \
  --gpu         0 \
  --conf        0.25 --iou 0.45 \
  --channel-cap 4 \
  --fps-num     25  --fps-den 1 \
  --no-draw

# Drop Picasso *and* the trailing ZmqSink so the pipeline
# terminates at the tracker output (counts inference / tracker
# frames, no GPU transform / encoder / Skia overlay run at all).
# Stricter than `cars-demo --output null` (which keeps Picasso
# alive and only drops the bitstream) — useful for measuring raw
# decode → infer → track throughput, free of any encode / overlay
# cost, and for back-to-back producer load tests without the
# consumer-side backpressure.  Implies `--no-draw` (the Skia
# overlay is part of the Picasso stage that no longer exists).
# `--zmq-out` is ignored in this mode.
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- pipeline \
  --no-sink
```

Runs indefinitely; only `Ctrl+C` shuts the actor system down.
Per-stream EOS messages are forwarded on the wire but do **not**
terminate the pipeline.  `--no-sink` preserves the same multi-
stream semantics: per-source `SourceEos` is logged on the
function terminus but does not stop it, so back-to-back
`producer --no-eos` cycles keep flowing.

#### `consumer`

```bash
# Minimal: only --output is required.
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- \
  consumer --output /tmp/cars-demo-zmq.mp4

# Full set of overrides:
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- consumer \
  --zmq-in      sub+connect:ipc:///tmp/savant_demo_out \
  --output      /tmp/cars-demo-zmq.mp4 \
  --fps-num     25 --fps-den 1 \
  --channel-cap 4
```

Exits on the first EOS for any source — `ZmqSource::on_source_eos`
is overridden to return `Flow::Stop`, after which the `Mp4Muxer`
terminus drains and finalises the MP4 container.

### TCP transport (cross-host)

The default endpoints use `ipc://` because that is what works on a
single host with no extra setup.  For cross-host runs, override
each side with a matching `tcp://` URL:

```bash
# Pipeline on host A (binds both ends)
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- pipeline \
  --zmq-in  router+bind:tcp://0.0.0.0:6010 \
  --zmq-out pub+bind:tcp://0.0.0.0:6011

# Producer on host B
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- producer \
  --input   /path/to/cars.mp4 \
  --zmq-out dealer+connect:tcp://hostA:6010

# Consumer on host C
cargo run --release -p savant-perception-framework --example cars-demo-zmq -- consumer \
  --zmq-in  sub+connect:tcp://hostA:6011 \
  --output  /tmp/cars-demo-zmq.mp4
```

Run `cars-demo-zmq <subcommand> --help` (or `cargo run --release …
-- <sub> --help`) for the full per-subcommand flag reference.

---

## Logging

Every binary uses `env_logger` and reads `RUST_LOG`.  `--debug` raises
only this binary's modules to `debug` while keeping third-party
crates at `info` so the terminal does not get flooded by GStreamer /
TensorRT internals.  `RUST_LOG` overrides take precedence:

```bash
RUST_LOG=info,cars_demo=debug,savant_perception=debug \
  cargo run --release -p savant-perception-framework --example cars-demo -- \
    --input ... --output ...
```

---

## Where to look in the source

- `examples/cars_demo/cars_demo.rs` — module root for the shared
  pipeline, stats and warmup helpers.
- `examples/cars_demo/cars_demo/pipeline.rs` — `run` (cars-demo) and
  `run_pipeline` (cars-demo-zmq) plus `PipelineHead` / `PipelineTail` /
  `PipelineKnobs` / `ShutdownPolicy`.
- `examples/cars_demo/cli.rs` / `cli_zmq.rs` — clap definitions and
  resolver tests.
- `examples/cars_demo/main.rs` / `main_zmq.rs` — process entry points.
- `examples/cars_demo/assets.rs` — upstream asset + engine-cache
  layout helpers.
