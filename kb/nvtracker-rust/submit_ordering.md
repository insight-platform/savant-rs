# Submit ordering invariant (NvTracker crate)

This mirrors the nvinfer crate's `submit_ordering.md` — the tracker has
the same two-layer locking invariant, for the same reasons.

## The guarantee

Frames pushed into `NvTrackerBatchingOperator::add_frame` from a single
per-source caller thread must reach the result callback in the same
order.  Per-source frame-number accounting and the GStreamer feeder's
`StrictPts` policy both depend on it.

## Gates

Both layers use `savant_gstreamer::submit_gate::SubmitGate`, which
owns the counter it protects.  No sibling `AtomicU64` or `Mutex<()>`
on either.

### 1. `NvTracker::submit_gate` (`pipeline.rs`)

Wraps `next_pts read + advance + prepare_batch + finalize_batch_buffer +
input_tx.send` in `NvTracker::submit` via `submit_gate.submit_with(…)`.
Same rationale as the nvinfer version.

### 2. `SubmitContext::submit_gate`
    (`batching_operator/submit.rs`)

Wraps the **whole** `submit_batch_impl` window: state drain,
`batch_id` assignment + counter advance, per-source frame-counter
increments, `pending_batches` insertion, and `nvtracker.submit`.
Serialises the `add_frame` and `timer_thread` paths against each
other.

Without it, the same drain-vs-submit race documented in the nvinfer
KB can invert the order of submission to the tracker's GStreamer
pipeline, producing out-of-order tracker outputs and per-source frame
numbers that no longer match the video PTS order.

## Why a `SubmitGate` instead of `AtomicU64 + Mutex<()>`

Previously this crate (like nvinfer) paired an `AtomicU64` counter
with a `Mutex<()>` purely to serialise access to that counter.  The
combination works but misleads readers: `fetch_add` reads as
"lock-free" while in fact the sibling mutex is always held.  A
`SubmitGate` owns the counter inside its mutex, so the compiler
rejects any code path that tries to advance the counter without
holding the gate.

## Related

- `kb/nvinfer-rust/submit_ordering.md`
- `kb/savant-samples/submit_serialisation.md`
  (cross-crate symptom walk-through and end-to-end verification).
- `kb/savant-gstreamer-rust/submit_gate.md`
  (primitive-level rationale and API shape).
