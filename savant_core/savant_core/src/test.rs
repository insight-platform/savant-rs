use crate::primitives::any_object::AnyObject;
use crate::primitives::attribute_value::AttributeValue;
use crate::primitives::frame::{
    VideoFrameBuilder, VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use crate::primitives::object::{
    IdCollisionResolutionPolicy, ObjectOperations, VideoObject, VideoObjectBuilder,
};
use crate::primitives::{RBBox, WithAttributes};
use crate::utils::uuid_v7::incremental_uuid_v7;
use std::sync::Arc;

pub fn gen_empty_frame() -> VideoFrameProxy {
    VideoFrameProxy::from_inner(
        VideoFrameBuilder::default()
            .source_id("test".to_string())
            .pts(0)
            .fps((30, 1))
            .width(0)
            .uuid(incremental_uuid_v7().as_u128())
            .height(0)
            .content(Arc::new(VideoFrameContent::None))
            .transcoding_method(VideoFrameTranscodingMethod::Copy)
            .codec(None)
            .keyframe(None)
            .build()
            .unwrap(),
    )
}

pub fn gen_frame() -> VideoFrameProxy {
    let mut f = VideoFrameProxy::from_inner(
        VideoFrameBuilder::default()
            .source_id("test".to_string())
            .pts(1000000)
            .fps((30, 1))
            .width(1280)
            .uuid(incremental_uuid_v7().as_u128())
            .height(720)
            .content(Arc::new(VideoFrameContent::None))
            .transcoding_method(VideoFrameTranscodingMethod::Copy)
            .codec(None)
            .keyframe(None)
            .build()
            .unwrap(),
    );

    let parent_object = VideoObjectBuilder::default()
        .id(0)
        .detection_box(RBBox::new(0.0, 0.0, 0.0, 0.0, None))
        .attributes(Vec::default())
        .confidence(None)
        .namespace("test".to_string())
        .label("test2".to_string())
        .build()
        .unwrap();

    let c1 = VideoObjectBuilder::default()
        .id(1)
        .detection_box(RBBox::new(0.0, 0.0, 0.0, 0.0, None))
        .parent_id(Some(parent_object.get_id()))
        .attributes(Vec::default())
        .confidence(None)
        .namespace("test2".to_string())
        .label("test".to_string())
        .build()
        .unwrap();

    let c2 = VideoObjectBuilder::default()
        .id(2)
        .detection_box(RBBox::new(0.0, 0.0, 0.0, 0.0, None))
        .parent_id(Some(parent_object.get_id()))
        .attributes(Vec::default())
        .confidence(None)
        .namespace("test2".to_string())
        .label("test2".to_string())
        .build()
        .unwrap();

    f.add_object(parent_object, IdCollisionResolutionPolicy::Error)
        .unwrap();
    f.add_object(c1, IdCollisionResolutionPolicy::Error)
        .unwrap();
    f.add_object(c2, IdCollisionResolutionPolicy::Error)
        .unwrap();

    f.set_persistent_attribute(
        "system",
        "test",
        &Some("test"),
        false,
        vec![AttributeValue::string("1", None)],
    );

    f.set_persistent_attribute(
        "system2",
        "test2",
        &None,
        false,
        vec![AttributeValue::string("2", None)],
    );

    f.set_persistent_attribute(
        "system",
        "test2",
        &Some("test"),
        false,
        vec![AttributeValue::string("3", None)],
    );

    f.set_persistent_attribute(
        "test",
        "test",
        &Some("hint"),
        false,
        vec![
            AttributeValue::bytes(&[8, 3, 8, 8], &[0; 192], None),
            AttributeValue::integer_vector([0, 1, 2, 3, 4, 5].into(), None),
            AttributeValue::string("incoming", Some(0.56)),
            AttributeValue::temporary_value(AnyObject::new(Box::new(1.0)), None),
        ],
    );
    f
}

pub fn gen_object(id: i64) -> VideoObject {
    let mut o = VideoObject {
        id,
        namespace: s("peoplenet"),
        label: s("face"),
        confidence: Some(0.5),
        detection_box: RBBox::new(1.0, 2.0, 10.0, 20.0, None),
        track_id: Some(id),
        track_box: Some(RBBox::new(100.0, 200.0, 10.0, 20.0, None)),
        ..Default::default()
    };

    o.set_persistent_attribute("some", "attribute", &Some("hint"), false, vec![]);
    o
}

#[inline(always)]
pub fn s(a: &str) -> String {
    a.to_string()
}

/// Process-wide log capture utility for tests.
///
/// Tests sometimes need to assert that a specific `log::error!` / `warn!` /
/// `info!` line was (or was *not*) emitted — e.g. a code path documented as
/// "fires a warning under backpressure" should actually fire it.
///
/// The standard `env_logger` is fire-and-forget: it writes to stderr and
/// forgets; tests have no way to inspect what was emitted.  This module
/// installs a process-wide [`log::Log`] implementation that captures every
/// record (level + rendered message) into a shared `Vec`, *and* echoes each
/// record to stderr so `cargo test -- --nocapture` still shows output.
///
/// # Usage
///
/// ```rust,no_run
/// use savant_core::test::log_capture::log_records;
/// use log::Level;
///
/// # fn main() {
/// let records = log_records();          // installs the logger on first call
/// let baseline = records.lock().unwrap().len();
/// log::warn!("overflow at pts=42");
/// let new_records: Vec<_> = {
///     let guard = records.lock().unwrap();
///     guard[baseline..].to_vec()
/// };
/// assert!(new_records
///     .iter()
///     .any(|(lvl, msg)| *lvl == Level::Warn && msg.contains("overflow")));
/// # }
/// ```
///
/// # Caveats
///
/// * Installation is best-effort: `log::set_boxed_logger` rejects any
///   subsequent install, so if another test / library installed its own
///   logger first (including `env_logger`), this capture becomes an
///   empty-buffer no-op.  Tests relying on the capture should document this.
/// * The captured `Vec` grows unboundedly for the life of the process.
///   Tests typically snapshot `records.lock().len()` at start and slice
///   from there, so this is fine in practice.
/// * Shared across threads via `Arc<Mutex<_>>` — safe under contention.
pub mod log_capture {
    use log::{Level, LevelFilter, Log, Metadata, Record};
    use std::sync::{Arc, Mutex, OnceLock};

    /// Shared buffer of captured `(level, rendered-message)` pairs.
    pub type LogRecords = Arc<Mutex<Vec<(Level, String)>>>;

    struct CapturingLogger {
        records: LogRecords,
    }

    impl Log for CapturingLogger {
        fn enabled(&self, _metadata: &Metadata) -> bool {
            true
        }

        fn log(&self, record: &Record) {
            let msg = format!("{}", record.args());
            if let Ok(mut v) = self.records.lock() {
                v.push((record.level(), msg.clone()));
            }
            eprintln!("[{} {}] {}", record.level(), record.target(), msg);
        }

        fn flush(&self) {}
    }

    /// Install the capturing logger (idempotent) and return a handle to the
    /// shared record buffer.  See module docs for semantics and caveats.
    pub fn log_records() -> LogRecords {
        static RECORDS: OnceLock<LogRecords> = OnceLock::new();
        RECORDS
            .get_or_init(|| {
                let records: LogRecords = Arc::new(Mutex::new(Vec::new()));
                let logger = Box::new(CapturingLogger {
                    records: records.clone(),
                });
                let _ = log::set_boxed_logger(logger);
                log::set_max_level(LevelFilter::Debug);
                records
            })
            .clone()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn captures_warn_and_error() {
            let records = log_records();
            let baseline = records.lock().unwrap().len();
            log::warn!("log_capture_test marker WARN alpha");
            log::error!("log_capture_test marker ERROR beta");
            let new: Vec<_> = {
                let guard = records.lock().unwrap();
                guard[baseline..].to_vec()
            };
            assert!(new
                .iter()
                .any(|(lvl, msg)| *lvl == Level::Warn && msg.contains("marker WARN alpha")));
            assert!(new
                .iter()
                .any(|(lvl, msg)| *lvl == Level::Error && msg.contains("marker ERROR beta")));
        }
    }
}
