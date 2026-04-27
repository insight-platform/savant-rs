//! NvTracker configuration.

use crate::error::{NvTrackerError, Result};
use deepstream_buffers::{MetaClearPolicy, VideoFormat};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

/// Default width passed to the tracker element (`tracker-width`).
pub const DEFAULT_TRACKER_WIDTH: u32 = 640;
/// Default height passed to the tracker element (`tracker-height`).
pub const DEFAULT_TRACKER_HEIGHT: u32 = 384;
/// Default maximum batch size for the tracker element (`batch-size`).
pub const DEFAULT_MAX_BATCH_SIZE: u32 = 16;

/// How tracking IDs behave on stream reset / EOS (matches DeepStream `tracking-id-reset-mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum TrackingIdResetMode {
    #[default]
    None = 0,
    OnStreamReset = 1,
    OnEos = 2,
    OnStreamResetAndEos = 3,
}

impl TrackingIdResetMode {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Configuration for [`crate::pipeline::NvTracker`].
#[derive(Debug, Clone)]
pub struct NvTrackerConfig {
    pub name: String,
    pub tracker_width: u32,
    pub tracker_height: u32,
    pub max_batch_size: u32,
    /// Path to `libnvds_nvmultiobjecttracker.so`.
    pub ll_lib_file: String,
    /// Path to low-level tracker YAML (e.g. IOU config).
    pub ll_config_file: String,
    pub gpu_id: u32,
    pub input_format: VideoFormat,
    pub element_properties: HashMap<String, String>,
    pub tracking_id_reset_mode: TrackingIdResetMode,
    /// Controls clearing of `NvDsObjectMeta` before submit and on output drop.
    /// Default: `Before`.
    pub meta_clear_policy: MetaClearPolicy,
    /// Maximum time to wait for a submitted buffer to produce a result.
    /// Passed to the GStreamer pipeline framework as the in-flight watchdog
    /// deadline. When exceeded, the pipeline enters a terminal failed state.
    /// Default: 30 s.
    pub operation_timeout: Duration,
    /// Bounded channel capacity for [`crate::pipeline::NvTracker::submit`].
    pub input_channel_capacity: usize,
    /// Bounded channel capacity for decoded [`crate::pipeline::NvTrackerOutput`].
    pub output_channel_capacity: usize,
    /// How often the framework drain thread polls `appsink` when idle.
    pub drain_poll_interval: Duration,
    /// When `Some`, the inner [`savant_gstreamer::pipeline::GstPipeline`]
    /// auto-invokes
    /// [`flush_idle`](savant_gstreamer::pipeline::GstPipeline::flush_idle)
    /// every interval in a background thread.  Ensures per-source EOS
    /// markers escape `nvtracker` without an explicit caller flush or
    /// full graceful_shutdown.  Default: 10 ms.
    pub idle_flush_interval: Option<Duration>,
    /// When `Some(d)`, a background stale-source evictor thread runs and
    /// invokes
    /// [`reset_stream`](crate::pipeline::NvTracker::reset_stream)
    /// for any source whose last-seen PTS is older than `d`.  Default:
    /// `Some(5s)`.  Set to `None` to disable.
    ///
    /// Together with the per-source release sequence in
    /// [`reset_stream`](crate::pipeline::NvTracker::reset_stream) this
    /// keeps long-running pipelines from accumulating one pinned
    /// prev-frame buffer per ever-seen source.
    pub stale_source_after: Option<Duration>,
}

impl NvTrackerConfig {
    /// Create a config with required tracker library paths; other fields use defaults.
    pub fn new(ll_lib_file: impl Into<String>, ll_config_file: impl Into<String>) -> Self {
        Self {
            name: String::new(),
            tracker_width: DEFAULT_TRACKER_WIDTH,
            tracker_height: DEFAULT_TRACKER_HEIGHT,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            ll_lib_file: ll_lib_file.into(),
            ll_config_file: ll_config_file.into(),
            gpu_id: 0,
            input_format: VideoFormat::RGBA,
            element_properties: HashMap::new(),
            tracking_id_reset_mode: TrackingIdResetMode::None,
            meta_clear_policy: MetaClearPolicy::default(),
            operation_timeout: Duration::from_secs(30),
            input_channel_capacity: 16,
            output_channel_capacity: 16,
            drain_poll_interval: Duration::from_millis(100),
            idle_flush_interval: Some(Duration::from_millis(10)),
            stale_source_after: Some(Duration::from_secs(5)),
        }
    }

    /// Configure the stale-source evictor.  See
    /// [`Self::stale_source_after`].  Pass `None` to disable.
    pub fn stale_source_after(mut self, after: Option<Duration>) -> Self {
        self.stale_source_after = after;
        self
    }

    /// Set the bounded input channel capacity (framework backpressure).
    pub fn input_channel_capacity(mut self, capacity: usize) -> Self {
        self.input_channel_capacity = capacity;
        self
    }

    /// Set the bounded output channel capacity.
    pub fn output_channel_capacity(mut self, capacity: usize) -> Self {
        self.output_channel_capacity = capacity;
        self
    }

    /// Set the framework drain thread poll interval when no sample is ready.
    pub fn drain_poll_interval(mut self, interval: Duration) -> Self {
        self.drain_poll_interval = interval;
        self
    }

    /// Enable or disable the auto-flush thread for pending custom
    /// downstream events.  See [`NvTrackerConfig::idle_flush_interval`].
    pub fn idle_flush_interval(mut self, interval: Option<Duration>) -> Self {
        self.idle_flush_interval = interval;
        self
    }

    /// Set the `NvDsObjectMeta` clearing policy.
    pub fn meta_clear_policy(mut self, policy: MetaClearPolicy) -> Self {
        self.meta_clear_policy = policy;
        self
    }

    /// Validate paths and dimensions.
    pub fn validate(&self) -> Result<()> {
        if !Path::new(&self.ll_lib_file).is_file() {
            return Err(NvTrackerError::ConfigError(format!(
                "ll_lib_file does not exist: {}",
                self.ll_lib_file
            )));
        }
        if !Path::new(&self.ll_config_file).is_file() {
            return Err(NvTrackerError::ConfigError(format!(
                "ll_config_file does not exist: {}",
                self.ll_config_file
            )));
        }
        if self.tracker_width == 0 || self.tracker_height == 0 {
            return Err(NvTrackerError::ConfigError(
                "tracker_width and tracker_height must be non-zero".into(),
            ));
        }
        if self.max_batch_size == 0 {
            return Err(NvTrackerError::ConfigError(
                "max_batch_size must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_missing_files() {
        let c = NvTrackerConfig::new("/no/such/lib.so", "/no/such/config.yml");
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_dimensions() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let lib = dir.join(format!("nvtracker_ut_lib_{pid}.so"));
        let yml = dir.join(format!("nvtracker_ut_cfg_{pid}.yml"));
        std::fs::write(&lib, b"x").unwrap();
        std::fs::write(&yml, b"y").unwrap();

        let mut c = NvTrackerConfig::new(
            lib.to_string_lossy().into_owned(),
            yml.to_string_lossy().into_owned(),
        );
        c.tracker_width = 0;
        assert!(c.validate().is_err());
        c.tracker_width = 640;
        c.tracker_height = 0;
        assert!(c.validate().is_err());

        let _ = std::fs::remove_file(&lib);
        let _ = std::fs::remove_file(&yml);
    }

    #[test]
    fn new_defaults_meta_clear_policy_before() {
        let c = NvTrackerConfig::new("/tmp/lib.so", "/tmp/cfg.yml");
        assert_eq!(c.meta_clear_policy, MetaClearPolicy::Before);
    }

    #[test]
    fn fluent_meta_clear_policy_setter_updates_value() {
        let c = NvTrackerConfig::new("/tmp/lib.so", "/tmp/cfg.yml")
            .meta_clear_policy(MetaClearPolicy::Both);
        assert_eq!(c.meta_clear_policy, MetaClearPolicy::Both);
    }
}
