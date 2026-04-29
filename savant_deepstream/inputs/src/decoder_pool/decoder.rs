//! [`FlexibleDecoderPool`] — multi-stream wrapper over [`FlexibleDecoder`].
//!
//! Routes incoming frames to per-source [`FlexibleDecoder`] instances, creating
//! them on demand.  Idle streams are evicted by a background sweep thread;
//! an optional callback lets callers override the evict/keep decision.
//!
//! Submissions for a source whose eviction is in progress block (via
//! [`ReleaseSeal`]) until the eviction resolves, then either reuse the kept
//! decoder or transparently create a new one.

use crate::flexible_decoder::{
    FlexibleDecoder, FlexibleDecoderError, FlexibleDecoderOutput, PoolCacheRegistry,
};
use parking_lot::{Condvar, Mutex};
use savant_core::primitives::frame::VideoFrame;
use savant_core::utils::release_seal::ReleaseSeal;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::config::{EvictionDecision, FlexibleDecoderPoolConfig};

// ── Helpers ──────────────────────────────────────────────────────────────

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Minimum sweep interval. The condvar-based sleep ensures we never spin
/// even with a low floor — the thread wakes on signal or after the interval.
const MIN_SWEEP_INTERVAL: Duration = Duration::from_millis(100);

// ── StreamEntry ──────────────────────────────────────────────────────────

struct StreamEntry {
    decoder: Arc<FlexibleDecoder>,
    last_activity_ms: AtomicU64,
    eviction_seal: Option<Arc<ReleaseSeal>>,
}

// ── Type aliases ─────────────────────────────────────────────────────────

type EvictionCallback = Arc<dyn Fn(&str) -> EvictionDecision + Send + Sync>;

// ── FlexibleDecoderPool ──────────────────────────────────────────────────

/// Multi-stream pool of [`FlexibleDecoder`] instances.
///
/// Each unique `source_id` seen in [`submit`](Self::submit) gets its own
/// [`FlexibleDecoder`].  A background thread periodically checks for idle
/// streams (last activity older than `config.eviction_ttl`) and either
/// evicts them automatically or consults the optional eviction callback.
///
/// The API mirrors [`FlexibleDecoder`]: `submit`, `source_eos`,
/// `graceful_shutdown`, `shutdown`.
pub struct FlexibleDecoderPool {
    config: FlexibleDecoderPoolConfig,
    streams: Arc<Mutex<HashMap<String, StreamEntry>>>,
    on_output: Arc<dyn Fn(FlexibleDecoderOutput) + Send + Sync + 'static>,
    eviction_stop: Arc<(Mutex<bool>, Condvar)>,
    eviction_join: Option<JoinHandle<()>>,
    shut_down: AtomicBool,
    /// Per-source RGBA pool cache shared with every per-source
    /// [`FlexibleDecoder`] this pool creates.  Survives
    /// `source_eos`-driven decoder teardown so a fresh decoder for
    /// the same `source_id` can reuse the existing
    /// `BufferGenerator` (and its pooled `EglCudaMeta`
    /// registrations) instead of rebuilding from scratch.  Entries
    /// are cleared on eviction (`EvictionDecision::Evict`) and on
    /// pool [`Self::graceful_shutdown`] / [`Self::shutdown`] —
    /// each path runs the F2 sync before releasing the cached
    /// `Arc<Mutex<BufferGenerator>>`.
    pool_cache: Arc<PoolCacheRegistry>,
}

impl FlexibleDecoderPool {
    /// Create a pool **without** an eviction callback.
    ///
    /// Expired streams are evicted automatically (graceful drain).
    pub fn new<F>(config: FlexibleDecoderPoolConfig, on_output: F) -> Self
    where
        F: Fn(FlexibleDecoderOutput) + Send + Sync + 'static,
    {
        Self::build(config, Arc::new(on_output), None)
    }

    /// Create a pool **with** an eviction callback.
    ///
    /// When a stream's TTL expires the callback is invoked with its
    /// `source_id`.  Return [`EvictionDecision::Keep`] to reset the TTL or
    /// [`EvictionDecision::Evict`] to gracefully drain and remove the
    /// decoder.
    pub fn with_eviction_callback<F, E>(
        config: FlexibleDecoderPoolConfig,
        on_output: F,
        on_eviction: E,
    ) -> Self
    where
        F: Fn(FlexibleDecoderOutput) + Send + Sync + 'static,
        E: Fn(&str) -> EvictionDecision + Send + Sync + 'static,
    {
        Self::build(config, Arc::new(on_output), Some(Arc::new(on_eviction)))
    }

    fn build(
        config: FlexibleDecoderPoolConfig,
        on_output: Arc<dyn Fn(FlexibleDecoderOutput) + Send + Sync + 'static>,
        on_eviction: Option<EvictionCallback>,
    ) -> Self {
        let streams: Arc<Mutex<HashMap<String, StreamEntry>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let eviction_stop = Arc::new((Mutex::new(false), Condvar::new()));
        let pool_cache = Arc::new(PoolCacheRegistry::new());

        let sweep_interval = (config.eviction_ttl / 4).max(MIN_SWEEP_INTERVAL);

        let eviction_join = {
            let streams = Arc::clone(&streams);
            let stop = Arc::clone(&eviction_stop);
            let on_eviction = on_eviction.clone();
            let pool_cache_evict = Arc::clone(&pool_cache);
            let eviction_ttl_ms = config.eviction_ttl.as_millis() as u64;

            std::thread::Builder::new()
                .name("flex-pool-evict".into())
                .spawn(move || {
                    eviction_loop(
                        &streams,
                        &stop,
                        &on_eviction,
                        &pool_cache_evict,
                        eviction_ttl_ms,
                        sweep_interval,
                    );
                })
                .expect("failed to spawn eviction sweep thread")
        };

        Self {
            config,
            streams,
            on_output,
            eviction_stop,
            eviction_join: Some(eviction_join),
            shut_down: AtomicBool::new(false),
            pool_cache,
        }
    }

    /// Returns a clone of the [`PoolCacheRegistry`] `Arc` shared
    /// with every per-source [`FlexibleDecoder`] created by this
    /// pool.  Useful for tests that want to inspect cache state
    /// without holding a handle to a per-source decoder.
    pub fn pool_cache(&self) -> Arc<PoolCacheRegistry> {
        Arc::clone(&self.pool_cache)
    }

    // ── Public API ───────────────────────────────────────────────────

    /// Submit an encoded frame for decoding.
    ///
    /// The frame is routed to the [`FlexibleDecoder`] for
    /// `frame.get_source_id()`.  If none exists yet, one is created
    /// transparently.
    ///
    /// If an eviction is in progress for that source, this call **blocks**
    /// until the eviction completes (the seal is released), then retries.
    pub fn submit(
        &self,
        frame: &VideoFrame,
        data: Option<&[u8]>,
    ) -> Result<(), FlexibleDecoderError> {
        if self.shut_down.load(Ordering::Relaxed) {
            return Err(FlexibleDecoderError::ShutDown);
        }

        let source_id = frame.get_source_id();
        self.resolve_and_submit(&source_id, frame, data)
    }

    /// Inject a logical per-source EOS and tear down the per-source decoder.
    ///
    /// `source_eos` is the explicit "this contiguous PTS sequence is over"
    /// boundary for `source_id`.  The pool:
    ///
    ///  1. Removes the per-source [`FlexibleDecoder`] from the streams map
    ///     (so any concurrent [`submit`](Self::submit) for the same
    ///     `source_id` will create a fresh decoder).
    ///  2. Sends the in-band `SourceEos` custom downstream event so
    ///     remaining frames are delivered to the callback in order,
    ///     followed by the [`FlexibleDecoderOutput::SourceEos`] marker.
    ///  3. Drains the decoder via
    ///     [`FlexibleDecoder::graceful_shutdown`] (bounded by
    ///     `config.idle_timeout`) and tears down the underlying GstPipeline.
    ///
    /// Subsequent submits for the same `source_id` (typically a new
    /// producer run that legitimately rebases its PTS to 0) build a fresh
    /// per-source [`FlexibleDecoder`] with a fresh `nvv4l2decoder` —
    /// avoiding the stale internal monotonic-PTS state that would
    /// otherwise silently swallow the new sequence.
    ///
    /// **Pool-cache survival.**  The
    /// [`PoolCacheRegistry`] entry for `source_id` is **not** removed
    /// here: when a fresh decoder is created on the next
    /// [`submit`](Self::submit) it inherits the cached
    /// `Arc<Mutex<BufferGenerator>>` (and its pooled `EglCudaMeta`
    /// registrations) and starts streaming without rebuilding the
    /// pool — eliminating the rapid pool-churn pattern that drove
    /// the cars-demo-zmq CUDA-700 cascade.  Cache entries are
    /// released only on
    /// [`EvictionDecision::Evict`](super::config::EvictionDecision::Evict)
    /// or on full pool [`Self::graceful_shutdown`] /
    /// [`Self::shutdown`].
    ///
    /// If the source has no active decoder (it never had a frame
    /// submitted, or has already been evicted), the EOS marker is emitted
    /// directly via the output callback.
    pub fn source_eos(&self, source_id: &str) -> Result<(), FlexibleDecoderError> {
        if self.shut_down.load(Ordering::Relaxed) {
            return Err(FlexibleDecoderError::ShutDown);
        }

        // Atomically detach the entry from the streams map.  Any concurrent
        // `submit` racing past this point will see no entry and create a
        // fresh one for `source_id`.
        let entry = self.streams.lock().remove(source_id);

        let mut entry = match entry {
            Some(e) => e,
            None => {
                (self.on_output)(FlexibleDecoderOutput::SourceEos {
                    source_id: source_id.to_string(),
                });
                return Ok(());
            }
        };

        // Release any pending eviction seal so other waiters (if any)
        // unblock immediately.
        if let Some(seal) = entry.eviction_seal.take() {
            seal.release();
        }

        // 1. Inject the in-band per-source EOS marker so the worker emits
        //    `FlexibleDecoderOutput::SourceEos` ordered with the remaining
        //    decoded frames.  Errors here mean the underlying decoder is
        //    already dead — fall through to the drain anyway.
        let _ = entry.decoder.source_eos(source_id);

        // 2. Drain remaining frames and tear down the GstPipeline.  After
        //    this returns, the `FlexibleDecoder` is in `ShutDown` state and
        //    the only reference held by the pool was just dropped.
        entry.decoder.graceful_shutdown()
    }

    /// Force-flush pending rescue-eligible custom-downstream events on
    /// every active per-stream decoder in the pool.
    ///
    /// Iterates all streams and invokes [`FlexibleDecoder::flush_idle`]
    /// on each.  Returns the total number of events flushed across the
    /// pool (sum across streams).  Errors from individual streams are
    /// logged but do not short-circuit the iteration — the method
    /// returns the first `Err` encountered after attempting all streams.
    ///
    /// Typical use: call from the consumer's `recv_timeout(Timeout)`
    /// branch to let trailing per-source EOS markers escape their
    /// respective `nvv4l2decoder` elements without requiring a full
    /// [`graceful_shutdown`](Self::graceful_shutdown).
    pub fn flush_idle(&self) -> Result<usize, FlexibleDecoderError> {
        if self.shut_down.load(Ordering::Relaxed) {
            return Err(FlexibleDecoderError::ShutDown);
        }

        let decoders: Vec<Arc<FlexibleDecoder>> = {
            let streams = self.streams.lock();
            streams.values().map(|e| Arc::clone(&e.decoder)).collect()
        };

        let mut total = 0usize;
        let mut first_err: Option<FlexibleDecoderError> = None;
        for decoder in decoders {
            match decoder.flush_idle() {
                Ok(n) => total += n,
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        if let Some(e) = first_err {
            Err(e)
        } else {
            Ok(total)
        }
    }

    /// Drain every decoder in the pool and shut down.
    ///
    /// The eviction thread is stopped first.  Each per-stream decoder is
    /// gracefully drained (bounded by `config.idle_timeout`).  After all
    /// per-source decoders are drained the shared
    /// [`PoolCacheRegistry`] is cleared with a single device-sync
    /// fence — releasing every cached `BufferGenerator` only once
    /// the GPU has fully drained.
    ///
    /// Terminal — subsequent calls return `Err(ShutDown)`.
    pub fn graceful_shutdown(&mut self) -> Result<(), FlexibleDecoderError> {
        if self.shut_down.swap(true, Ordering::Relaxed) {
            return Err(FlexibleDecoderError::ShutDown);
        }
        self.stop_eviction_thread();

        let entries: Vec<(String, StreamEntry)> = {
            let mut streams = self.streams.lock();
            streams.drain().collect()
        };

        for (_source_id, mut entry) in entries {
            if let Some(seal) = entry.eviction_seal.take() {
                seal.release();
            }
            let _ = entry.decoder.graceful_shutdown();
        }
        // All per-source decoders are drained — clear the shared
        // pool cache (single device-sync fence) so the cached
        // `BufferGenerator`s are released cleanly.
        self.pool_cache.clear_all();
        Ok(())
    }

    /// Immediate teardown — frames in flight are lost.
    ///
    /// Terminal — subsequent calls return `Err(ShutDown)`.  After
    /// every per-source decoder has been torn down the shared
    /// [`PoolCacheRegistry`] is cleared with a single device-sync
    /// fence.
    pub fn shutdown(&mut self) {
        if self.shut_down.swap(true, Ordering::Relaxed) {
            return;
        }
        self.stop_eviction_thread();

        let entries: Vec<(String, StreamEntry)> = {
            let mut streams = self.streams.lock();
            streams.drain().collect()
        };

        for (_source_id, mut entry) in entries {
            if let Some(seal) = entry.eviction_seal.take() {
                seal.release();
            }
            entry.decoder.shutdown();
        }
        self.pool_cache.clear_all();
    }

    // ── Private helpers ──────────────────────────────────────────────

    fn stop_eviction_thread(&mut self) {
        let (lock, cvar) = &*self.eviction_stop;
        *lock.lock() = true;
        cvar.notify_one();
        if let Some(jh) = self.eviction_join.take() {
            let _ = jh.join();
        }
    }

    /// Resolve-or-create loop with seal blocking.
    fn resolve_and_submit(
        &self,
        source_id: &str,
        frame: &VideoFrame,
        data: Option<&[u8]>,
    ) -> Result<(), FlexibleDecoderError> {
        loop {
            let result = {
                let mut streams = self.streams.lock();

                if let Some(entry) = streams.get(source_id) {
                    if let Some(ref seal) = entry.eviction_seal {
                        // Eviction in progress — grab the seal and wait
                        // outside the lock.
                        Err(Arc::clone(seal))
                    } else {
                        entry.last_activity_ms.store(epoch_ms(), Ordering::Relaxed);
                        Ok(Arc::clone(&entry.decoder))
                    }
                } else {
                    // Create a new decoder for this source_id.
                    // Hand it the shared `pool_cache` registry so
                    // its first `submit` reuses any
                    // `BufferGenerator` left behind by a prior
                    // decoder (typical after `source_eos`-driven
                    // teardown of the same source) instead of
                    // rebuilding from scratch.
                    let flex_config = self.config.to_flexible_config(source_id);
                    let on_out = Arc::clone(&self.on_output);
                    let decoder = Arc::new(FlexibleDecoder::with_registry(
                        flex_config,
                        move |out| (on_out)(out),
                        Arc::clone(&self.pool_cache),
                    ));

                    let entry = StreamEntry {
                        decoder: Arc::clone(&decoder),
                        last_activity_ms: AtomicU64::new(epoch_ms()),
                        eviction_seal: None,
                    };
                    streams.insert(source_id.to_string(), entry);
                    Ok(decoder)
                }
            };

            match result {
                Ok(decoder) => return decoder.submit(frame, data),
                Err(seal) => seal.wait(),
            }
        }
    }
}

impl Drop for FlexibleDecoderPool {
    fn drop(&mut self) {
        if !self.shut_down.load(Ordering::Relaxed) {
            self.shutdown();
        }
    }
}

// ── Eviction sweep ──────────────────────────────────────────────────────

fn eviction_loop(
    streams: &Mutex<HashMap<String, StreamEntry>>,
    stop: &(Mutex<bool>, Condvar),
    on_eviction: &Option<EvictionCallback>,
    pool_cache: &PoolCacheRegistry,
    eviction_ttl_ms: u64,
    sweep_interval: Duration,
) {
    let (lock, cvar) = stop;
    loop {
        let mut stopped = lock.lock();
        if *stopped {
            break;
        }
        cvar.wait_for(&mut stopped, sweep_interval);
        if *stopped {
            break;
        }
        drop(stopped);
        eviction_sweep(streams, on_eviction, pool_cache, eviction_ttl_ms);
    }
}

fn eviction_sweep(
    streams: &Mutex<HashMap<String, StreamEntry>>,
    on_eviction: &Option<EvictionCallback>,
    pool_cache: &PoolCacheRegistry,
    eviction_ttl_ms: u64,
) {
    let now = epoch_ms();

    // Phase 1: identify expired streams and install seals.
    let expired: Vec<(String, Arc<ReleaseSeal>)> = {
        let mut map = streams.lock();
        let mut expired = Vec::new();

        for (source_id, entry) in map.iter_mut() {
            if entry.eviction_seal.is_some() {
                continue; // already being evicted
            }
            let last = entry.last_activity_ms.load(Ordering::Relaxed);
            if now.saturating_sub(last) > eviction_ttl_ms {
                let seal = Arc::new(ReleaseSeal::new());
                entry.eviction_seal = Some(Arc::clone(&seal));
                expired.push((source_id.clone(), seal));
            }
        }
        expired
    };
    // Lock released — blocked submit() calls can see the seal and wait.

    // Phase 2: consult callback and act.
    let mut to_drain: Vec<(String, Arc<FlexibleDecoder>)> = Vec::new();

    for (source_id, seal) in expired {
        let decision = on_eviction
            .as_ref()
            .map(|cb| cb(&source_id))
            .unwrap_or(EvictionDecision::Evict);

        match decision {
            EvictionDecision::Keep => {
                let mut map = streams.lock();
                if let Some(entry) = map.get_mut(&source_id) {
                    entry.last_activity_ms.store(epoch_ms(), Ordering::Relaxed);
                    entry.eviction_seal = None;
                }
                seal.release();
            }
            EvictionDecision::Evict => {
                let removed = {
                    let mut map = streams.lock();
                    map.remove(&source_id)
                };
                // Release seal first so blocked submitters can proceed
                // (they'll create a fresh decoder).
                seal.release();

                if let Some(entry) = removed {
                    to_drain.push((source_id, entry.decoder));
                }
            }
        }
    }

    // Phase 3: drain evicted decoders outside any lock.
    //
    // After the decoder drains we drop the cached pool entry for
    // that source — TTL eviction means the source has been
    // genuinely silent for `eviction_ttl_ms`, so retaining its
    // cached `BufferGenerator` would leak GPU memory.
    // `clear_source` runs a single device-sync fence before
    // releasing the cached `Arc<Mutex<BufferGenerator>>`.
    for (source_id, decoder) in to_drain {
        let _ = decoder.graceful_shutdown();
        pool_cache.clear_source(&source_id);
    }
}
