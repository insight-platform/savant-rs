/// Policy applied when a frame's PTS is not strictly greater than the
/// previous frame's PTS (non-monotonic / backward jump).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PtsResetPolicy {
    /// Emit a synthetic EOS (drain + flush encoder, fire EOS sentinel),
    /// then recreate the encoder and process the offending frame normally.
    /// Downstream sees a clean EOS boundary between the old and new streams.
    #[default]
    EosOnDecreasingPts,
    /// Silently destroy and recreate the encoder without emitting EOS.
    /// The frame that triggered the reset is processed on the new encoder.
    RecreateOnDecreasingPts,
}

/// Global defaults for the Picasso engine.
#[derive(Debug, Clone)]
pub struct GeneralSpec {
    /// Optional name for this engine instance, used internally for logging and
    /// future extensibility.
    pub name: String,
    /// Default idle timeout in seconds before a source is considered for
    /// eviction. Individual sources can override this via
    /// [`super::source::SourceSpec::idle_timeout_secs`].
    pub idle_timeout_secs: u64,
    /// Capacity of the per-worker inflight message queue.
    ///
    /// This controls how many frames can be buffered between the engine's
    /// `send_frame` call and the worker thread consuming them.  A larger
    /// value absorbs bursts but increases memory usage and latency.
    pub inflight_queue_size: usize,
    /// Policy for handling non-monotonic (decreasing) PTS values.
    pub pts_reset_policy: PtsResetPolicy,
}

/// Default inflight queue capacity for per-source worker channels.
pub const DEFAULT_INFLIGHT_QUEUE_SIZE: usize = 8;

impl Default for GeneralSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            idle_timeout_secs: 30,
            inflight_queue_size: DEFAULT_INFLIGHT_QUEUE_SIZE,
            pts_reset_policy: PtsResetPolicy::default(),
        }
    }
}

impl GeneralSpec {
    /// Create a new builder starting from default values.
    pub fn builder() -> GeneralSpecBuilder {
        GeneralSpecBuilder(GeneralSpec::default())
    }
}

/// Builder for [`GeneralSpec`] — uses `Default` values for unset fields.
///
/// # Example
///
/// ```rust,ignore
/// let spec = GeneralSpec::builder()
///     .name("my-engine")
///     .idle_timeout_secs(60)
///     .build();
/// ```
pub struct GeneralSpecBuilder(GeneralSpec);

impl GeneralSpecBuilder {
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.0.name = name.into();
        self
    }

    pub fn idle_timeout_secs(mut self, secs: u64) -> Self {
        self.0.idle_timeout_secs = secs;
        self
    }

    pub fn inflight_queue_size(mut self, size: usize) -> Self {
        self.0.inflight_queue_size = size;
        self
    }

    pub fn pts_reset_policy(mut self, policy: PtsResetPolicy) -> Self {
        self.0.pts_reset_policy = policy;
        self
    }

    /// Finish building and return the [`GeneralSpec`].
    pub fn build(self) -> GeneralSpec {
        self.0
    }
}

/// Decision returned by the `OnEviction` callback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictionDecision {
    /// Keep the source alive for at least the given number of seconds.
    KeepFor(u64),
    /// Drain the encoder (send EOS) then terminate the worker.
    Terminate,
    /// Terminate the worker immediately without draining.
    TerminateImmediately,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_general_spec() {
        let s = GeneralSpec::default();
        assert!(s.name.is_empty());
        assert_eq!(s.idle_timeout_secs, 30);
        assert_eq!(s.inflight_queue_size, DEFAULT_INFLIGHT_QUEUE_SIZE);
        assert_eq!(s.pts_reset_policy, PtsResetPolicy::EosOnDecreasingPts);
    }

    #[test]
    fn pts_reset_policy_variants() {
        assert_eq!(
            PtsResetPolicy::default(),
            PtsResetPolicy::EosOnDecreasingPts
        );
        assert_ne!(
            PtsResetPolicy::EosOnDecreasingPts,
            PtsResetPolicy::RecreateOnDecreasingPts
        );
    }

    #[test]
    fn eviction_decision_eq() {
        assert_eq!(EvictionDecision::Terminate, EvictionDecision::Terminate);
        assert_eq!(EvictionDecision::KeepFor(10), EvictionDecision::KeepFor(10));
        assert_ne!(EvictionDecision::KeepFor(10), EvictionDecision::KeepFor(20));
    }

    #[test]
    fn builder_default_matches_struct_default() {
        let built = GeneralSpec::builder().build();
        let direct = GeneralSpec::default();
        assert_eq!(built.idle_timeout_secs, direct.idle_timeout_secs);
        assert_eq!(built.inflight_queue_size, direct.inflight_queue_size);
    }

    #[test]
    fn builder_overrides_fields() {
        let spec = GeneralSpec::builder()
            .name("test-engine")
            .idle_timeout_secs(60)
            .inflight_queue_size(16)
            .build();
        assert_eq!(spec.name, "test-engine");
        assert_eq!(spec.idle_timeout_secs, 60);
        assert_eq!(spec.inflight_queue_size, 16);
    }
}
