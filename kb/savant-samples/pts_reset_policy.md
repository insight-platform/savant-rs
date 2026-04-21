# Picasso PtsResetPolicy on dGPU

## Symptom on dGPU (absent on Jetson Orin)

With the default `PtsResetPolicy::EosOnDecreasingPts`, running `cars-demo`
on an x86 dGPU produced floods of:

```
WARN picasso::worker: PTS reset detected: source=cars-demo,
     prev_pts=19920000000, new_pts=19880000000,
     policy=EosOnDecreasingPts
WARN cars_tracking::pipeline: [encode-cb] muxer closed;
     dropping encoded frame
```

On Jetson Orin the same binary ran clean.  This looked like a Picasso
policy issue, but it was a **downstream-visible symptom** of a race
inside the NvInfer / NvTracker batching operators.

## Actual root cause

See `submit_serialisation.md` for the full analysis.  The short version:
`NvInferBatchingOperator::submit_batch_impl` and
`NvTrackerBatchingOperator::submit_batch_impl` can each be called
concurrently from `add_frame` (the caller thread) and the internal
`timer_thread`.  Before the fix, the "drain from `BatchState` → build
batch → call `NvInfer::submit` / `NvTracker::submit`" critical section
was **not** held under a single lock.  Two callers could drain frames
F₁ and F₂ in order, then race inside `nvinfer.submit` / `nvtracker.submit`,
and the loser of the race received the earlier internal PTS — inverting
the downstream order.

That inversion reached Picasso as a backward PTS step of exactly one
frame period (40 ms at 25 fps).  Picasso honoured
`EosOnDecreasingPts`, emitted a mid-stream EOS, recreated the encoder,
and our sample's `EncodedSink` happily forwarded that mid-stream EOS
to the MP4 muxer channel — which closed the muxer → floods of
`muxer closed; dropping encoded frame` → truncated output.

The dGPU-only appearance is explained by **thread-scheduling
characteristics**, not by a decoder-level difference: on Jetson Orin
the timer thread never wins the race, probably because of cache / core
locality on a smaller CPU.  The input video is all-I-frame H.264
(verified with `ffprobe`), so decoder reorder is definitively not the
cause.

## Fix

The fix lives in the batching operators, not in the sample:

- `savant_deepstream/nvinfer/src/batching_operator/submit.rs`
- `savant_deepstream/nvtracker/src/batching_operator/submit.rs`

A new `submit_lock: Mutex<()>` is held for the entire
`submit_batch_impl` window (drain → build → `pending_batches.insert`
→ `{nvinfer,nvtracker}.submit`).  This guarantees that the order in
which frames leave `BatchState::take` is also the order in which they
enter the inner pipeline — regardless of which thread is the caller.

After the fix the sample runs end-to-end with the default
`EosOnDecreasingPts` policy untouched, zero `PTS reset detected`
warnings, zero `muxer closed` warnings, and `decoded == encoded ==
1078` for the NY-city-center clip.

## When to revisit

- If Picasso later gains a "suppress encoder recreation on small
  backward jitter" policy, it becomes a second line of defence; but
  the current fix removes the jitter at its source, which is the
  correct layer.
- Any sample that deliberately wants to segment output on a PTS reset
  (split into multiple MP4s) must handle the `OutputMessage::EndOfStream`
  from Picasso explicitly — our current sample never needs that,
  because after the fix Picasso only emits the single terminal EOS at
  `picasso.send_eos(&source_id)` + `picasso.shutdown()` time.
