/// Global defaults for the Picasso engine.
#[derive(Debug, Clone)]
pub struct GeneralSpec {
    /// Default idle timeout in seconds before a source is considered for
    /// eviction. Individual sources can override this via
    /// [`super::source::SourceSpec::idle_timeout_secs`].
    pub idle_timeout_secs: u64,
}

impl Default for GeneralSpec {
    fn default() -> Self {
        Self {
            idle_timeout_secs: 30,
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
        assert_eq!(s.idle_timeout_secs, 30);
    }

    #[test]
    fn eviction_decision_eq() {
        assert_eq!(EvictionDecision::Terminate, EvictionDecision::Terminate);
        assert_eq!(EvictionDecision::KeepFor(10), EvictionDecision::KeepFor(10));
        assert_ne!(EvictionDecision::KeepFor(10), EvictionDecision::KeepFor(20));
    }
}
