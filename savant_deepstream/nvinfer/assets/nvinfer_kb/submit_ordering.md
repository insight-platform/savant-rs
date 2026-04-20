# Submit ordering invariant (NvInfer crate)

## The guarantee

When frames are pushed in a fixed order through
`NvInferBatchingOperator::add_frame` from a single caller thread (the
typical per-source producer pattern), they must leave the internal
NvInfer pipeline — and therefore reach the result callback — in that
**same order**.

Downstream components (Picasso, `Mp4Muxer`, the GStreamer feeder
thread) assume monotonically non-decreasing frame PTS and respond to
violations with "PTS reset" / `EosOnDecreasingPts` / `StrictPts`
failures.  Those failures are not defensive noise — they are a
contract error.

## Where it is enforced

Two locks, at two different abstraction layers.  Both are required.

### 1. `NvInfer::submit_mutex` (`pipeline.rs`)

Wraps the `next_pts.fetch_add + prepare_buffer + input_tx.send`
sequence in `NvInfer::submit`.  Serialises **direct** callers of
`submit()` (tests, benchmarks, future embedders).  Without it, two
threads can win `fetch_add` but lose `send`, delivering buffers to
the GStreamer pipeline out of internal-PTS order; the feeder rejects
the later one with `StrictPts`.

### 2. `SubmitContext::submit_lock` (`batching_operator/submit.rs`)

Wraps the **whole** `submit_batch_impl` window (drain from
`BatchState`, build `NonUniformBatch`, assign `batch_id`, insert into
`pending_batches`, call `self.nvinfer.submit`).  Serialises the two
internal callers of that path:

- the caller-thread path through `NvInferBatchingOperator::add_frame`,
- the operator's internal `timer_thread`.

Without it, those two threads can each drain disjoint frames under
the short-held `state` lock, release it, and then race inside
`nvinfer.submit`.  The thread that wins the race assigns the earlier
internal PTS, even though it was the later frame.  Downstream sees
this as a single-frame backward PTS step per race, which
Picasso/`EosOnDecreasingPts` interprets as a stream boundary.

## Testing

- `cargo test -p savant-deepstream-nvinfer` covers the `engine_cache`
  and configuration units but does not exercise the concurrent
  `submit()` path (a true race test would need a fake `NvInfer`
  whose `submit()` sleeps and observes ordering).
- End-to-end reproduction path: `savant_samples/bin/cars-demo` on a
  dGPU with a single-source, single-FPS H.264 clip.  Before the fix
  this crashed with `StrictPts` and/or produced floods of
  `[encode-cb] muxer closed` warnings; after the fix it reports
  `decoded == encoded` and the output MP4 has strictly monotonic
  packet PTS.

## Gotchas when modifying the submit path

- Do **not** drop `submit_mutex` "because `submit_lock` covers it" —
  `submit_lock` only covers the batching-operator path, not direct
  `NvInfer::submit` callers.
- Do **not** take `submit_lock` *around* `state.lock()` ↔
  `condvar.wait` because the timer thread in `operator.rs` waits on
  the condvar holding the state lock; the two locks must **not**
  nest in the opposite order.  Current code acquires `submit_lock`
  first and the state lock strictly inside it in `submit_batch_impl`,
  and the timer's condvar wait only holds the state lock.  That is
  the safe nesting.
- When `max_batch_size > 1` the same invariant still applies: the
  order in which a single caller thread pushes frames into
  `BatchState` is the order in which `take()` returns them.
- A multi-source deployment where different sources are pushed from
  different caller threads will see interleaving between sources.
  That is expected — per-source monotonicity is what the downstream
  contract requires, and each source drains its own frames in order.
