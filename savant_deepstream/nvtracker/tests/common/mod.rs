//! Shared test setup for nvtracker integration tests.

use deepstream_buffers::{cuda_init, SavantIdMetaKind};
use deepstream_nvtracker::{
    NvTracker, NvTrackerError, NvTrackerOutput, Result, TrackedFrame, TrackerOutput,
};
use gstreamer as gst;
use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        let _ = env_logger::try_init();
        gst::init().expect("gst init");
        cuda_init(0).expect("cuda_init — GPU required for nvtracker tests");
    });
}

/// Submit a batch and block until [`TrackerOutput`] is available (skips stray events).
///
/// Not every integration test binary uses this helper (see `test_detection_meta_count`);
/// those crates still compile `common` for shared `init`, so suppress `dead_code`.
#[allow(dead_code)]
pub fn track_sync(
    tracker: &NvTracker,
    frames: &[TrackedFrame],
    ids: Vec<SavantIdMetaKind>,
) -> Result<TrackerOutput> {
    tracker.submit(frames, ids)?;
    loop {
        match tracker.recv()? {
            NvTrackerOutput::Tracking(t) => return Ok(t),
            NvTrackerOutput::Event(_) => continue,
            NvTrackerOutput::Eos { source_id } => {
                return Err(NvTrackerError::PipelineError(format!(
                    "unexpected EOS during track_sync: {source_id}"
                )));
            }
            NvTrackerOutput::Error(e) => return Err(e),
        }
    }
}
