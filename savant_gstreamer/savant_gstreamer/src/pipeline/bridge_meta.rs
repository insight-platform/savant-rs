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

/// Bridge [`SavantIdMeta`] between two pads using PTS-keyed side storage.
///
/// This is intended for wrapping an internal pipeline segment:
/// - `capture_pad`: where input buffers enter the segment
/// - `restore_pad`: where output buffers leave the segment
pub fn bridge_savant_id_meta_across(
    capture_pad: &gst::Pad,
    restore_pad: &gst::Pad,
) -> Result<(), PipelineError> {
    let map: Arc<Mutex<LruCache<u64, VecDeque<Vec<SavantIdMetaKind>>>>> = Arc::new(Mutex::new(
        LruCache::new(NonZeroUsize::new(MAX_BRIDGE_MAP_SIZE).expect("non-zero")),
    ));

    let capture_map = map.clone();
    capture_pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        if let Some(buffer) = info.buffer() {
            if let (Some(meta), Some(pts)) = (buffer.meta::<SavantIdMeta>(), buffer.pts()) {
                let ids = meta.ids().to_vec();
                let pts_ns = pts.nseconds();
                let mut map = capture_map.lock();
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
