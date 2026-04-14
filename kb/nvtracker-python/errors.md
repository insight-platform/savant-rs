# NvTracker Python — Errors

PyO3 surfaces most failures as **`RuntimeError`** with the string from `NvTrackerError` (`Display`).

Typical cases:

| Situation | Python |
|-----------|--------|
| Invalid config (missing files, zero dims) | `RuntimeError` on `NvTracker(...)` |
| `SharedBuffer` still borrowed | `RuntimeError` on `submit` |
| Operation timeout (default 30 s) — pipeline failed | `RuntimeError` (`PipelineFailed`; tracker must be recreated) |
| Channel disconnect | `RuntimeError` (`ChannelDisconnected`) on `recv` / `submit` / `shutdown` |
| Shutting down | `RuntimeError` (`ShuttingDown`) on `submit` / `send_event` / `send_eos` / `reset_stream` after `graceful_shutdown` begins, or `RuntimeError` if engine already shut down |
| Framework / GStreamer pipeline error | `NvTrackerOutput.is_error` with message, or `RuntimeError` from `shutdown` |
| Unknown `element_properties` key / bad value | `RuntimeError` (`InvalidProperty`) |
| Double `shutdown()` | `RuntimeError: NvTracker is already shut down` |
| Use after `shutdown()` | `RuntimeError: NvTracker is shut down` |

Callback errors are logged (`NvTracker callback error: ...`) and not propagated into the pipeline caller.

For precise variant mapping, see `kb/nvtracker-rust/errors.md` (`NvTrackerError`).
