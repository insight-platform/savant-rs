//! Cross-decoder RGBA `BufferGenerator` cache.
//!
//! [`PoolCacheRegistry`] stores at most one [`CachedPool`] per
//! `source_id`.  Lifting the cache out of the per-stream
//! [`FlexibleDecoder`](super::FlexibleDecoder) and giving it to the
//! parent [`FlexibleDecoderPool`](crate::decoder_pool::FlexibleDecoderPool)
//! lets the pool survive across `source_eos`-driven decoder
//! teardown / recreation cycles (the cars-demo-zmq replay loop) so
//! we no longer rebuild a fresh `BufferGenerator` on every cycle —
//! eliminating the rapid pool-churn pattern that the F1+F2 fences
//! were guarding against.
//!
//! When the registry is owned exclusively by a single decoder
//! (the [`FlexibleDecoder::new`](super::FlexibleDecoder::new)
//! standalone path), dropping the decoder also drops the registry
//! and its [`Drop`] implementation drains all entries with a
//! [`cuda_device_synchronize`](deepstream_buffers::cuda_device_synchronize)
//! fence — preserving the F2 invariant that no GPU work is in
//! flight when the underlying NvBufSurfaces are freed.

use deepstream_buffers::{BufferGenerator, NvBufSurfaceMemType, VideoFormat};
use log::info;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

/// Cached RGBA output pool, kept alive across decoder restarts.
///
/// Whenever [`super::FlexibleDecoder::activate`] is invoked we
/// check the parent registry: if its descriptor matches the new
/// pool params, the existing [`BufferGenerator`] is reused (its
/// `GstBufferPool`, `NvBufSurface` backing memory, and Jetson
/// `EglCudaMeta` `GST_META_FLAG_POOLED` registrations all
/// survive). On a mismatch the previous pool is dropped and a
/// fresh one is built.
///
/// This is the central piece of the "deferred pool reprovisioning"
/// strategy described in `docs/cuda-race.md`: it eliminates pool
/// churn during rapid same-resolution restarts (the
/// cars-demo-zmq scenario), which removes one likely contributor
/// to the CUDA-700 cascade — without requiring any churn-time
/// fence on NVIDIA-internal streams.
pub(super) struct CachedPool {
    pub pool: Arc<Mutex<BufferGenerator>>,
    pub width: u32,
    pub height: u32,
    pub fps_num: i32,
    pub fps_den: i32,
    pub format: VideoFormat,
    pub gpu_id: u32,
    pub mem_type: NvBufSurfaceMemType,
    pub pool_size: u32,
}

impl CachedPool {
    /// Returns `true` when the cached pool can serve a decoder
    /// requested with the supplied parameters.
    ///
    /// All fields must match — width, height, pixel format, GPU
    /// id, memory type, pool size **and** framerate. The framerate
    /// enters the gst caps reported by
    /// [`BufferGenerator::nvmm_caps`] / [`BufferGenerator::raw_caps`];
    /// reusing a pool whose caps differ from the new stream would
    /// silently mis-advertise downstream.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn matches(
        &self,
        width: u32,
        height: u32,
        fps_num: i32,
        fps_den: i32,
        format: VideoFormat,
        gpu_id: u32,
        mem_type: NvBufSurfaceMemType,
        pool_size: u32,
    ) -> bool {
        self.width == width
            && self.height == height
            && self.fps_num == fps_num
            && self.fps_den == fps_den
            && self.format == format
            && self.gpu_id == gpu_id
            && self.mem_type == mem_type
            && self.pool_size == pool_size
    }
}

/// Process-shareable cache of [`CachedPool`] entries keyed by
/// `source_id`.
///
/// Lifetime decoupling:
///
/// * [`super::FlexibleDecoder::new`] constructs a fresh, **private**
///   `Arc<PoolCacheRegistry>`.  When the decoder is dropped this
///   Arc reaches zero strong refs, [`Drop for PoolCacheRegistry`](#impl-Drop)
///   runs the F2 sync, and every cached `BufferGenerator` is
///   released — preserving the contract of standalone tests.
///
/// * [`super::FlexibleDecoder::with_registry`] takes a clone of an
///   externally owned registry (typically held by
///   [`FlexibleDecoderPool`](crate::decoder_pool::FlexibleDecoderPool)).
///   The decoder's own `graceful_shutdown` / `shutdown` /
///   `Drop` then no longer affect the registry — its entries
///   survive `source_eos` teardown so the next decoder for the
///   same source reuses the existing pool.
///
/// Concurrency: every method takes a single short-held `Mutex`
/// guard and never invokes user callbacks while holding it.  The
/// underlying `Arc<Mutex<BufferGenerator>>` returned from
/// [`get_or_build`](Self::get_or_build) clones the cached `Arc`,
/// so the registry's lock is not retained while the caller
/// drives the pool.
pub struct PoolCacheRegistry {
    entries: Mutex<HashMap<String, CachedPool>>,
}

impl PoolCacheRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Reuse the cached RGBA output pool for `source_id` when its
    /// descriptor matches the requested params; otherwise build a
    /// fresh one and replace the cache entry.
    ///
    /// Logs `pool_cache hit` / `pool_cache replace` /
    /// `pool_cache build` exactly as the prior per-decoder
    /// implementation did, so existing log-mining stays valid.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn get_or_build(
        &self,
        source_id: &str,
        width: u32,
        height: u32,
        fps_num: i32,
        fps_den: i32,
        format: VideoFormat,
        gpu_id: u32,
        mem_type: NvBufSurfaceMemType,
        pool_size: u32,
    ) -> Result<Arc<Mutex<BufferGenerator>>, String> {
        let mut entries = self.entries.lock();
        if let Some(cached) = entries.get(source_id) {
            if cached.matches(
                width, height, fps_num, fps_den, format, gpu_id, mem_type, pool_size,
            ) {
                let pool = Arc::clone(&cached.pool);
                let strong = Arc::strong_count(&pool);
                info!(
                    "FlexibleDecoder pool_cache hit (source={}, {}x{}@{}/{}, format={:?}, \
                     mem_type={:?}, gpu_id={}, pool_size={}, strong_count={})",
                    source_id,
                    width,
                    height,
                    fps_num,
                    fps_den,
                    format,
                    mem_type,
                    gpu_id,
                    pool_size,
                    strong,
                );
                return Ok(pool);
            }
            info!(
                "FlexibleDecoder pool_cache replace (source={}, was {}x{}@{}/{} format={:?} \
                 mem_type={:?} pool_size={}, now {}x{}@{}/{} format={:?} mem_type={:?} \
                 pool_size={})",
                source_id,
                cached.width,
                cached.height,
                cached.fps_num,
                cached.fps_den,
                cached.format,
                cached.mem_type,
                cached.pool_size,
                width,
                height,
                fps_num,
                fps_den,
                format,
                mem_type,
                pool_size,
            );
        } else {
            info!(
                "FlexibleDecoder pool_cache build (source={}, {}x{}@{}/{}, format={:?}, \
                 mem_type={:?}, gpu_id={}, pool_size={})",
                source_id, width, height, fps_num, fps_den, format, mem_type, gpu_id, pool_size,
            );
        }

        let pool = BufferGenerator::builder(format, width, height)
            .fps(fps_num, fps_den)
            .gpu_id(gpu_id)
            .mem_type(mem_type)
            .min_buffers(pool_size)
            .max_buffers(pool_size)
            .build()
            .map_err(|e| format!("buffer pool creation failed: {e}"))?;
        let pool_arc = Arc::new(Mutex::new(pool));

        entries.insert(
            source_id.to_string(),
            CachedPool {
                pool: Arc::clone(&pool_arc),
                width,
                height,
                fps_num,
                fps_den,
                format,
                gpu_id,
                mem_type,
                pool_size,
            },
        );

        Ok(pool_arc)
    }

    /// Address of the cached `BufferGenerator` Arc for `source_id`,
    /// when present.
    ///
    /// Stable across decoder restarts when the resolution / format
    /// descriptor matches, so equality is a reliable cache-hit
    /// signal for tests.  Production code must treat the returned
    /// value as opaque — comparing addresses is only meaningful
    /// within a single process and only as long as the pool is
    /// alive.
    pub fn pool_addr(&self, source_id: &str) -> Option<usize> {
        self.entries
            .lock()
            .get(source_id)
            .map(|c| Arc::as_ptr(&c.pool) as usize)
    }

    /// Drop the cache entry for `source_id`, if any.
    ///
    /// Runs a single
    /// [`cuda_device_synchronize`](deepstream_buffers::cuda_device_synchronize)
    /// before releasing the cached `Arc<Mutex<BufferGenerator>>`,
    /// so any GPU work still referencing buffers from this
    /// source's pool (e.g. queued `NvBufSurfTransform`,
    /// `gst-nvinfer` preproc, or `nvtracker` `cuDCF` kernels) is
    /// forced to complete before the pool becomes droppable.
    /// Cost: at most one device sync per source-eviction or
    /// shutdown — never on hot path.
    pub fn clear_source(&self, source_id: &str) {
        let removed = self.entries.lock().remove(source_id);
        if removed.is_some() {
            info!("FlexibleDecoder pool_cache clear (source={source_id})");
            deepstream_buffers::cuda_device_synchronize(
                "PoolCacheRegistry::clear_source::pre_drop_sync",
                format_args!("source={source_id}"),
            );
        }
    }

    /// Drop every cache entry, with a single device-sync fence
    /// before all `Arc<Mutex<BufferGenerator>>` slots are
    /// released.
    ///
    /// Used by [`Drop`](#impl-Drop) and by the parent
    /// [`FlexibleDecoderPool`](crate::decoder_pool::FlexibleDecoderPool)
    /// on `graceful_shutdown` / `shutdown`.
    pub fn clear_all(&self) {
        let mut entries = self.entries.lock();
        if entries.is_empty() {
            return;
        }
        let count = entries.len();
        info!("FlexibleDecoder pool_cache clear_all (entries={count})");
        deepstream_buffers::cuda_device_synchronize(
            "PoolCacheRegistry::clear_all::pre_drop_sync",
            format_args!("entries={count}"),
        );
        entries.clear();
    }

    /// Number of currently cached entries.
    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    /// Returns `true` when the registry has no cached entries.
    pub fn is_empty(&self) -> bool {
        self.entries.lock().is_empty()
    }
}

impl Default for PoolCacheRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PoolCacheRegistry {
    fn drop(&mut self) {
        // `clear_all` runs a device sync before releasing the
        // cached `Arc<Mutex<BufferGenerator>>` slots — preserving
        // the F2 invariant that no GPU work is in flight when the
        // underlying NvBufSurfaces are freed.
        self.clear_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_starts_empty() {
        let r = PoolCacheRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.pool_addr("nonexistent").is_none());
    }

    #[test]
    fn clear_source_on_empty_is_noop() {
        let r = PoolCacheRegistry::new();
        r.clear_source("missing");
        assert!(r.is_empty());
    }

    #[test]
    fn clear_all_on_empty_is_noop() {
        let r = PoolCacheRegistry::new();
        r.clear_all();
        assert!(r.is_empty());
    }
}
