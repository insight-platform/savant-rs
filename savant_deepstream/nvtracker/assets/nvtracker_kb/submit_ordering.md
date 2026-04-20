# Submit ordering invariant (NvTracker crate)

This mirrors the nvinfer crate's `submit_ordering.md` — the tracker has
the same two-layer locking invariant, for the same reasons.

## The guarantee

Frames pushed into `NvTrackerBatchingOperator::add_frame` from a single
per-source caller thread must reach the result callback in the same
order.  Per-source frame-number accounting and the GStreamer feeder's
`StrictPts` policy both depend on it.

## Locks

### 1. `NvTracker::submit_mutex` (`pipeline.rs`)

Wraps `next_pts.fetch_add + prepare_batch + finalize_batch_buffer +
input_tx.send` in `NvTracker::submit`.  Same rationale as the nvinfer
version.

### 2. `SubmitContext::submit_lock`
    (`batching_operator/submit.rs`)

Wraps the **whole** `submit_batch_impl` window: state drain, batch
id assignment, per-source frame-counter increments, `pending_batches`
insertion, and `nvtracker.submit`.  Serialises the `add_frame` and
`timer_thread` paths against each other.

Without it, the same drain-vs-submit race documented in the nvinfer
KB can invert the order of submission to the tracker's GStreamer
pipeline, producing out-of-order tracker outputs and per-source frame
numbers that no longer match the video PTS order.

## Related

- `savant_deepstream/nvinfer/assets/nvinfer_kb/submit_ordering.md`
- `savant_samples/assets/savant_samples_kb/submit_serialisation.md`
  (cross-crate symptom walk-through and end-to-end verification).
