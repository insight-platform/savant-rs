# Pipeline termination / EOS propagation

## Channel topology

```
decode_thread  ──►  rx_infer    ──►  infer_thread
infer_thread   ──►  rx_tracker  ──►  tracker_thread
tracker_thread ──►  rx_render   ──►  render_thread
render_thread  ──►  picasso workers (Arc<PicassoEngine>)
picasso cb     ──►  rx_encoded  ──►  mux_thread
```

Each inter-stage channel is a bounded `crossbeam::channel`, so every
stage naturally back-pressures its upstream.

## Termination order

1. `decode_thread` sees EOS from the demuxer, drops `tx_infer`.
2. `infer_thread` detects `rx_infer` closed, calls
   `operator.send_eos()` + `graceful_shutdown()`, drops `tx_tracker`.
3. `tracker_thread` mirrors step 2.
4. `render_thread`:
   - Waits for `rx_render` to close.
   - Calls `picasso.send_eos(&source_id)` → the per-source worker
     drains, finishes encoding, and delivers
     `OutputMessage::EndOfStream` on the callback.
   - Calls `picasso.shutdown()` to join all workers.
   - Drops the `Arc<PicassoEngine>` → last `tx_encoded` ref is dropped.
5. `mux_thread` exits either on `EncodedMsg::Eos` (the forwarded one) or
   on `RecvError` (channel closed) — both paths then call
   `muxer.finish()` to finalize the MP4 container.

## Important invariant

**Only the final** Picasso `OutputMessage::EndOfStream` must reach the
mux thread.  Any earlier EOS causes premature file finalisation and a
cascade of `muxer closed; dropping encoded frame` warnings — see
`pts_reset_policy.md`.

## Why we clone `tx_encoded` into Picasso and drop the original

```rust
let picasso = Arc::new(build_picasso_engine(
    tx_encoded.clone(),
    ...
));
drop(tx_encoded);
```

After this line the **only** `tx_encoded` references live inside
Picasso's `EncodedSink` callback.  Once Picasso drops (after
`shutdown()` joined all workers), the channel has no senders left and
the mux thread exits cleanly via `RecvError`, even if someone forgot to
forward EOS.

This is a defence-in-depth guarantee: EOS forwarding is the *fast*
termination path; senders-dropped is the *correct* termination path.
Both must converge on `muxer.finish()`.
