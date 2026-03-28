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
}
