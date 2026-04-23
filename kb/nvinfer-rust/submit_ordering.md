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

Two gates, at two different abstraction layers.  Both are required.
Each is a `savant_gstreamer::submit_gate::SubmitGate` that owns the
monotonic counter it protects — no sibling `AtomicU64`.

### 1. `NvInfer::submit_gate` (`pipeline.rs`)

Wraps the `next_pts read + advance + prepare_buffer + input_tx.send`
sequence in `NvInfer::submit` via `submit_gate.submit_with(…)`.
Serialises **direct** callers of `submit()` (tests, benchmarks,
future embedders).  Without it, two threads can read disjoint PTS
values but lose the `send` race, delivering buffers to the GStreamer
pipeline out of internal-PTS order; the feeder rejects the later one
with `StrictPts`.

Because the counter now lives inside the gate, it is a
compile-time error to advance the PTS without holding the
serialiser — the type system replaces the old convention that
guarded a sibling `AtomicU64` with a `Mutex<()>`.

### 2. `SubmitContext::submit_gate` (`batching_operator/submit.rs`)

Wraps the **whole** `submit_batch_impl` window (drain from
`BatchState`, assign `batch_id` + advance gate counter, build
`NonUniformBatch`, insert into `pending_batches`, call
`self.nvinfer.submit`).  Serialises the two internal callers of that
path:

- the caller-thread path through `NvInferBatchingOperator::add_frame`,
- the operator's internal `timer_thread`.

Without it, those two threads can each drain disjoint frames under
the short-held `state` lock, release it, and then race inside
`nvinfer.submit`.  The thread that wins the race assigns the earlier
internal PTS, even though it was the later frame.  Downstream sees
this as a single-frame backward PTS step per race, which
Picasso/`EosOnDecreasingPts` interprets as a stream boundary.

The counter itself (`batch_id`) lives inside the gate — there is no
longer an `Arc<AtomicU64> next_batch_id` sibling — so the same
compile-time guarantee applies here too.  `batch_id` is advanced
**after** the `state.is_empty()` early-return so empty flushes do
not burn ids.

## Testing

- `cargo test -p savant-deepstream-nvinfer` covers the `engine_cache`
  and configuration units but does not exercise the concurrent
  `submit()` path (a true race test would need a fake `NvInfer`
  whose `submit()` sleeps and observes ordering).
- `SubmitGate` itself carries a concurrent-monotonicity unit test in
  `savant-gstreamer` (`submit_gate::tests::concurrent_submitters_see_monotonic_counter`).
- End-to-end reproduction path: `savant_perception` `cars-demo` example on a
  dGPU with a single-source, single-FPS H.264 clip.  Before the fix
  this crashed with `StrictPts` and/or produced floods of
  `[encode-cb] muxer closed` warnings; after the fix it reports
  `decoded == encoded` and the output MP4 has strictly monotonic
  packet PTS.

## Gotchas when modifying the submit path

- Do **not** drop `NvInfer::submit_gate` "because
  `SubmitContext::submit_gate` covers it" —  the latter only covers
  the batching-operator path, not direct `NvInfer::submit` callers.
- Do **not** take `SubmitContext::submit_gate` *around* `state.lock()`
  ↔ `condvar.wait` because the timer thread in `operator.rs` waits on
  the condvar holding the state lock; the two locks must **not**
  nest in the opposite order.  Current code acquires the gate first
  and the state lock strictly inside it in `submit_batch_impl`,
  and the timer's condvar wait only holds the state lock.  That is
  the safe nesting.
- When `max_batch_size > 1` the same invariant still applies: the
  order in which a single caller thread pushes frames into
  `BatchState` is the order in which `take()` returns them.
- A multi-source deployment where different sources are pushed from
  different caller threads will see interleaving between sources.
  That is expected — per-source monotonicity is what the downstream
  contract requires, and each source drains its own frames in order.
- If you ever need the batch/PTS counter outside the critical
  section, **read it through `submit_with`**.  There is no
  free-standing accessor on purpose: the absence of one is what
  makes the invariant a compile-time guarantee.
