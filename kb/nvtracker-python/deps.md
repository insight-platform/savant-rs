# NvTracker Python — Dependencies & imports

## Required sibling modules

| Need | Import |
|------|--------|
| Batched NVMM buffer | `from savant_rs.deepstream import SharedBuffer, VideoFormat` |
| ROI geometry | `from savant_rs.nvinfer import Roi` |
| Bounding boxes | `from savant_rs.primitives.geometry import RBBox` |

## NvTracker package

```python
from savant_rs.nvtracker import (
    NvTracker,
    NvTrackerConfig,
    TrackingIdResetMode,
    TrackState,
    TrackerOutput,
    TrackedObject,
    MiscTrackData,
    MiscTrackFrame,
    TrackedFrame,
    NvTrackerBatchingOperatorConfig,
    TrackerBatchFormationResult,
    TrackerOperatorFrameOutput,
    SealedDeliveries,
    TrackerOperatorOutput,
    NvTrackerBatchingOperator,
)
```

## Type stubs

`savant_python/python/savant_rs/nvtracker/nvtracker.pyi` — keep in sync with PyO3 classes in `savant_core_py/src/nvtracker/`.

## Rust dependency chain (reference)

`savant_python` → `savant_core_py` (feature `deepstream`) → `nvtracker` crate → `deepstream`, `deepstream_buffers`, GStreamer.
