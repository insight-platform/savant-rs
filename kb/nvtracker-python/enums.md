# NvTracker Python — Enums

## `TrackingIdResetMode`

| Attribute | `int(...)` |
|-----------|------------|
| `NONE` | 0 |
| `ON_STREAM_RESET` | 1 |
| `ON_EOS` | 2 |
| `ON_STREAM_RESET_AND_EOS` | 3 |

Constructor kwarg `tracking_id_reset_mode` on `NvTrackerConfig` accepts these instances.

## `TrackState`

Aligned with DeepStream `TRACKER_STATE` / Rust `deepstream::TrackState`:

| Attribute | Value |
|-----------|-------|
| `EMPTY` | 0 |
| `ACTIVE` | 1 |
| `INACTIVE` | 2 |
| `TENTATIVE` | 3 |
| `PROJECTED` | 4 |

Exposed on `MiscTrackFrame.state` in tracker misc output.
