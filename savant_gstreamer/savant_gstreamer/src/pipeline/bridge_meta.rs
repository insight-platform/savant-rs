use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;

use gstreamer as gst;
use gstreamer::prelude::*;
use lru::LruCache;
use parking_lot::Mutex;

use crate::id_meta::{SavantIdMeta, SavantIdMetaKind};
use crate::pipeline::error::PipelineError;

/// Maximum number of in-flight PTS→meta entries before LRU eviction.
pub const MAX_BRIDGE_MAP_SIZE: usize = 1024;

/// Maximum number of meta entries stored per PTS key.
pub const MAX_ENTRIES_PER_PTS: usize = 32;

/// Internal bridge map: PTS → FIFO of captured meta entries, with LRU
/// eviction across PTS keys.
pub(crate) type BridgeMap = LruCache<u64, VecDeque<Vec<SavantIdMetaKind>>>;

/// Apply the capture step for one buffer: insert `ids` under `pts_ns`,
/// emit the capacity / collision / per-PTS-overflow log records, and
/// enforce [`MAX_ENTRIES_PER_PTS`] via FIFO eviction.
///
/// Extracted from the capture-pad probe so that the overflow / eviction
/// behaviour can be unit-tested deterministically without GStreamer
/// threading.  See the `bridge_map_per_pts_eviction` test.
pub(crate) fn capture_into_map(map: &mut BridgeMap, pts_ns: u64, ids: Vec<SavantIdMetaKind>) {
    if map.len() == map.cap().get() {
        log::error!(
            "bridge_savant_id_meta_across: PTS map reached capacity {}; \
             LRU eviction will occur",
            map.cap()
        );
    }
    if let Some(existing) = map.get(&pts_ns) {
        log::warn!(
            "bridge_savant_id_meta_across: PTS collision at {} ns; \
             existing entries={}, new meta={:?}",
            pts_ns,
            existing.len(),
            ids
        );
    }
    let entries = map.get_or_insert_mut(pts_ns, VecDeque::new);
    entries.push_back(ids);
    if entries.len() > MAX_ENTRIES_PER_PTS {
        let evicted = entries.pop_front();
        log::error!(
            "bridge_savant_id_meta_across: per-PTS limit ({}) exceeded at {} ns; \
             evicted={:?}, added={:?}",
            MAX_ENTRIES_PER_PTS,
            pts_ns,
            evicted,
            entries.back()
        );
    }
}

/// Bridge [`SavantIdMeta`] between two pads using PTS-keyed side storage.
///
/// This is intended for wrapping an internal pipeline segment:
/// - `capture_pad`: where input buffers enter the segment
/// - `restore_pad`: where output buffers leave the segment
pub fn bridge_savant_id_meta_across(
    capture_pad: &gst::Pad,
    restore_pad: &gst::Pad,
) -> Result<(), PipelineError> {
    let map: Arc<Mutex<BridgeMap>> = Arc::new(Mutex::new(LruCache::new(
        NonZeroUsize::new(MAX_BRIDGE_MAP_SIZE).expect("non-zero"),
    )));

    let capture_map = map.clone();
    capture_pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        if let Some(buffer) = info.buffer() {
            if let (Some(meta), Some(pts)) = (buffer.meta::<SavantIdMeta>(), buffer.pts()) {
                let ids = meta.ids().to_vec();
                let pts_ns = pts.nseconds();
                let mut map = capture_map.lock();
                capture_into_map(&mut map, pts_ns, ids);
            }
        }
        gst::PadProbeReturn::Ok
    });

    let restore_map = map;
    restore_pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        if let Some(buffer) = info.buffer_mut() {
            if let Some(pts) = buffer.pts() {
                let pts_ns = pts.nseconds();
                let mut map = restore_map.lock();
                let mut should_remove = false;
                if let Some(entries) = map.get_mut(&pts_ns) {
                    if let Some(ids) = entries.pop_front() {
                        SavantIdMeta::replace(buffer.make_mut(), ids);
                    }
                    should_remove = entries.is_empty();
                }
                if should_remove {
                    map.pop(&pts_ns);
                }
            }
        }
        gst::PadProbeReturn::Ok
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_map() -> BridgeMap {
        LruCache::new(NonZeroUsize::new(MAX_BRIDGE_MAP_SIZE).expect("non-zero"))
    }

    /// Pushing exactly `MAX_ENTRIES_PER_PTS` entries under a single PTS
    /// keeps them all; pushing one more evicts the oldest entry and
    /// preserves FIFO order for the rest.
    ///
    /// This covers the per-PTS overflow branch
    /// (`bridge_savant_id_meta_across: per-PTS limit (N) exceeded …`) that
    /// is otherwise not reachable deterministically from an e2e pipeline
    /// test because `appsrc::push_buffer` is asynchronous and upstream
    /// pushes can interleave with downstream restores.
    #[test]
    fn bridge_map_per_pts_eviction() {
        let pts_ns: u64 = 7_000;
        let mut map = new_map();

        for i in 0..MAX_ENTRIES_PER_PTS as u128 {
            capture_into_map(&mut map, pts_ns, vec![SavantIdMetaKind::Frame(i)]);
        }
        let entries = map.get(&pts_ns).expect("entry present");
        assert_eq!(entries.len(), MAX_ENTRIES_PER_PTS);
        assert_eq!(entries.front(), Some(&vec![SavantIdMetaKind::Frame(0)]));
        assert_eq!(
            entries.back(),
            Some(&vec![SavantIdMetaKind::Frame(
                MAX_ENTRIES_PER_PTS as u128 - 1
            )])
        );

        capture_into_map(
            &mut map,
            pts_ns,
            vec![SavantIdMetaKind::Frame(MAX_ENTRIES_PER_PTS as u128)],
        );

        let entries = map.get(&pts_ns).expect("entry present after overflow");
        assert_eq!(
            entries.len(),
            MAX_ENTRIES_PER_PTS,
            "len must stay clamped at MAX_ENTRIES_PER_PTS after overflow"
        );
        assert_eq!(
            entries.front(),
            Some(&vec![SavantIdMetaKind::Frame(1)]),
            "oldest (Frame 0) must have been evicted from the front"
        );
        assert_eq!(
            entries.back(),
            Some(&vec![SavantIdMetaKind::Frame(MAX_ENTRIES_PER_PTS as u128)]),
            "newest must be at the back"
        );
    }

    /// Two distinct PTS keys each get their own independent FIFO; overflow
    /// on one key does not affect the other.
    #[test]
    fn bridge_map_distinct_pts_independent_overflow() {
        let pts_a: u64 = 100;
        let pts_b: u64 = 200;
        let mut map = new_map();

        capture_into_map(&mut map, pts_b, vec![SavantIdMetaKind::Frame(42)]);

        for i in 0..=MAX_ENTRIES_PER_PTS as u128 {
            capture_into_map(&mut map, pts_a, vec![SavantIdMetaKind::Frame(i)]);
        }

        let entries_a = map.get(&pts_a).expect("pts_a present");
        assert_eq!(entries_a.len(), MAX_ENTRIES_PER_PTS);
        assert_eq!(entries_a.front(), Some(&vec![SavantIdMetaKind::Frame(1)]));

        let entries_b = map.get(&pts_b).expect("pts_b present");
        assert_eq!(entries_b.len(), 1);
        assert_eq!(entries_b.front(), Some(&vec![SavantIdMetaKind::Frame(42)]));
    }
}
