# NvInfer / NvTracker submit serialisation (two layers)

This note documents two related bugs that both manifest as PTS / ordering
problems when two threads reach the submit path concurrently.  Both were
fixed in `savant_deepstream`; samples did not have to change.

## Layer 1 — `NvInfer::submit` / `NvTracker::submit`

### Symptom

Running `cars-demo` on dGPU occasionally produced:

```
ERROR savant_gstreamer::pipeline::runner:
  nvinfer operator error: Framework pipeline error:
  Timestamp order violation (StrictPts): current 291 <= previous 292
```

The PTS values (tiny integers) pointed to the pipeline's internal
monotonic counter `next_pts: AtomicU64`, not the real video PTS.

### Root cause

The original `submit` looked like:

```rust
let pts = self.next_pts.fetch_add(1, Ordering::Relaxed); // 1
let buffer = self.prepare_buffer(batch, rois, pts)?;     // 2
self.input_tx.send(PipelineInput::Buffer(buffer))?;      // 3
```

Steps 1..3 are **not atomic** between threads.  Two concurrent callers
could interleave as:

```
T1: fetch_add -> 291
T2: fetch_add -> 292
T2: prepare_buffer + send           <-- 292 reaches pipeline first
T1: prepare_buffer + send           <-- 291 reaches pipeline second
```

The GStreamer feeder thread enforces `PtsPolicy::StrictPts`, so the
second buffer with `pts=291 <= previous_pts=292` is rejected.

### Fix

Added `submit_mutex: Mutex<()>` in both `NvInfer` and `NvTracker` and
wrapped the whole `fetch_add + prepare_buffer + send` sequence in it.

## Layer 2 — `submit_batch_impl` in the batching operators

Fixing Layer 1 removes the `StrictPts` crash but leaves a subtler
ordering bug that only shows up downstream as:

```
WARN picasso::worker: PTS reset detected: prev=19920000000,
     new=19880000000, policy=EosOnDecreasingPts
WARN cars_tracking::pipeline: [encode-cb] muxer closed;
     dropping encoded frame
```

(See also `pts_reset_policy.md`.)

### Root cause

`NvInferBatchingOperator::submit_batch_impl` and the nvtracker twin are
both invoked from two threads:

- the caller thread via `add_frame()` (when the batch fills up, or
  `max_batch_size = 1`, which always fills), and
- the operator's internal `timer_thread` (when
  `max_batch_wait` expires).

Before the fix the only lock taken inside `submit_batch_impl` covered
just the short `BatchState::take()` call.  After that the code did

```rust
let frames = state.lock().take();       // drain (short lock)
...                                      // build NonUniformBatch
let batch_id = next_batch_id.fetch_add(1, ...);
pending_batches.lock().insert(...);
nvinfer.submit(shared_buffer, rois)?;    // ← Layer-1 mutex here
```

with **no lock** held across drain → submit.  So two threads could:

```
T_timer:  state.take() -> [F_k]
T_add:    state.take() -> [F_{k+1}]
T_add:    nvinfer.submit(F_{k+1})   // wins layer-1 lock first
T_timer:  nvinfer.submit(F_k)
```

Even though `F_k` was drained *earlier* than `F_{k+1}`, the thread that
won the race inside `nvinfer.submit` got the smaller internal PTS.  The
GStreamer pipeline then processed `F_{k+1}` before `F_k` and the output
callback emitted them swapped.  At the next hop (Picasso) this looked
like a backward PTS jump of exactly one frame period.

This reproduced 100% of the time on an x86 dGPU box (RTX 4060) with
the provided all-I-frame NY-city-center clip at 25 fps; it did **not**
reproduce on Jetson Orin, almost certainly because of different
thread-scheduling behaviour.

### Fix

Added `submit_lock: Mutex<()>` to the `SubmitContext` of both batching
operators and hold it for the **entire** critical section:

```rust
let _submit_guard = self.submit_lock.lock();

let frames = { let mut st = self.state.lock(); st.take() };
// ... build batch, assign batch_id, insert into pending_batches ...
self.nvinfer.submit(shared_buffer, rois_arg.as_ref())?;
```

Because the inner state lock is nested inside `submit_lock`, ordering
is now "first thread to acquire `submit_lock` wins the whole drain +
submit window", and `BatchState::take()` always returns frames in push
order (`add_frame` pushes from a single caller thread per source).  The
two ordering guarantees compose:

    video-PTS order in → BatchState push order → take order → submit order

### Verification

After the fix, running `cars-demo` on dGPU:

- zero `Timestamp order violation (StrictPts)` errors,
- zero `PTS reset detected` warnings,
- zero `muxer closed; dropping encoded frame` warnings,
- `decoded == infer_frames == track_frames == encoded == 1078`,
- `ffprobe` reports 1078 packets in strict monotonic PTS order in
  `/tmp/ny_city_center.out.mp4`.

## Why both locks still exist

- `submit_mutex` in `NvInfer` / `NvTracker` covers **direct** callers
  of `submit()` (tests, benchmarks, any future user of the pipeline
  without going through a batching operator).  Keeping it is
  defence-in-depth at the right abstraction layer.
- `submit_lock` in the batching operators serialises the wider
  critical section that includes drain, batch construction and
  `pending_batches` accounting.  It is a strict superset of
  `submit_mutex` for the batching-operator path.

Removing either lock would reintroduce a real, reproducible ordering
bug.
