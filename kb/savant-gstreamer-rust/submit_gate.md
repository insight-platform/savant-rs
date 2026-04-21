# `SubmitGate`

`savant_gstreamer::submit_gate::SubmitGate` is a tiny synchronisation
primitive: a `parking_lot::Mutex<u64>` with a single public method,
`submit_with(|&mut u64| -> R) -> R`.  It exists to serialise access to
a monotonic counter that is paired with a non-trivial critical section
(e.g. build-a-buffer + push-on-channel).

## When to use it

Use `SubmitGate` whenever both conditions hold:

1. You have a monotonically-advancing `u64` counter (PTS, batch id,
   sequence number, …).
2. Advancing the counter has to appear atomic with some follow-up
   work that can itself fail or block — the follow-up is too long
   for `fetch_add` alone to be enough.

The `NvInfer` / `NvTracker` submit paths and both of their
batching-operator `submit_batch_impl` paths use `SubmitGate` for
exactly this reason; see
[`kb/nvinfer-rust/submit_ordering.md`](../nvinfer-rust/submit_ordering.md)
and [`kb/nvtracker-rust/submit_ordering.md`](../nvtracker-rust/submit_ordering.md).

## The anti-pattern it replaces

Before the gate existed, those call sites used:

```rust
next_counter: AtomicU64,
submit_mutex: Mutex<()>,
```

then, in the critical section:

```rust
let _guard = self.submit_mutex.lock();
let v = self.next_counter.fetch_add(1, Ordering::Relaxed);
// ... non-trivial build + send that must not observe a later v first ...
```

This works at runtime but is misleading:

- `fetch_add` reads as lock-free; it is not — the sibling mutex is
  always held, by convention.
- The compiler cannot catch a future contributor who "optimises"
  the mutex away.
- `Mutex<()>` forfeits Rust's core concurrency guarantee:
  `Mutex<T>` owns its `T` and is the only way to reach it.

`SubmitGate` takes the `u64` inside the mutex and exposes it only
through `submit_with`, so the invariant "you cannot read/advance
the counter without holding the serialiser" becomes a compile-time
property.

## API shape

```rust
pub struct SubmitGate { /* Mutex<u64> */ }

impl SubmitGate {
    pub fn new() -> Self;
    pub fn with_start(start: u64) -> Self;
    pub fn submit_with<R>(&self, f: impl FnOnce(&mut u64) -> R) -> R;
}
```

Notes:

- There is **no** free-standing accessor.  The absence of one is
  load-bearing: it is what turns the invariant into a compile-time
  guarantee.
- `submit_with` holds the lock for the entire duration of the
  closure.  Keep the closure body focused on the critical section —
  hoist anything that does not need to be serialised.
- Advance the counter at the policy boundary that matches the old
  `fetch_add` semantics: either unconditionally inside a non-empty
  batch (operator case) or as the first step after bail-out checks
  (pipeline case).  Both patterns are exercised in
  `savant_deepstream`.
- The gate is `Sync + Send` by virtue of `Mutex<u64>`; it is
  constructed once per owning struct and used by reference.

## Testing

`SubmitGate` itself carries a concurrent-monotonicity stress test
(`submit_gate::tests::concurrent_submitters_see_monotonic_counter`)
and a doc-test demonstrating the canonical call shape.  Call-site
crates rely on the cars-demo end-to-end run as the integration-level
proof: two back-to-back runs must produce `decoded == encoded` with
strictly monotonic PTS in the output MP4.
