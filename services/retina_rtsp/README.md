# savant-retina-rtsp

`savant-retina-rtsp` is an RTSP client and ingestion service for Savant built around Scott Lamb's `retina` stack, with optional GStreamer fallback and `cros-codecs`-based decode paths. It is designed for multi-camera ingress, RTCP sender-report synchronization, stream sync, and resilient reconnect behavior.

## What it does

The library exposes `Service`, `ServiceConfiguration`, `RtspSource`, `RtspSourceGroup`, `SyncConfiguration`, and backend selection via `RtspBackend::{Retina,Gstreamer}`. The binary loads one JSON config, opens a shared ZeroMQ sink, and spawns one task per RTSP source group; each group can carry multiple cameras, optional credentials, RTCP SR sync parameters, and backend-specific execution. `Service` also supports graceful shutdown, per-group stop requests, and automatic reconnect intervals, which makes the crate useful both as a standalone ingress service and as an embeddable component from Python through `savant_rs.retina_rtsp`.

## Install / run

```sh
cargo install savant-retina-rtsp
# or: docker pull ghcr.io/insight-platform/savant-retina-rtsp-x86:latest
retina_rtsp path/to/configuration.json
```

## Configuration

See `services/retina_rtsp/assets/configuration.json` for the shipped sample.

- `sink`: downstream Savant ZeroMQ writer configuration.
- `rtsp_sources`: named source groups, each with `sources`, optional `rtcp_sr_sync`, and a `backend`.
- `reconnect_interval`: delay before restarting failed groups.
- `eos_on_restart`: whether a restart emits EOS for the interrupted group.

## Documentation

- [Full service docs](https://insight-platform.github.io/savant-rs/services/retina_rtsp/index.html)
- [Source](https://github.com/insight-platform/savant-rs/tree/main/services/retina_rtsp)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).

