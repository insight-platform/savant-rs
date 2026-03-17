use picasso::prelude::{EvictionDecision, GeneralSpec, PtsResetPolicy};
use pyo3::prelude::*;

/// Policy for handling non-monotonic (decreasing) PTS values.
///
/// Construct via the factory static methods
/// [`eos_on_decreasing_pts`] or [`recreate_on_decreasing_pts`].
#[pyclass(from_py_object, name = "PtsResetPolicy", module = "savant_rs.picasso")]
#[derive(Debug, Clone)]
pub struct PyPtsResetPolicy {
    inner: PtsResetPolicy,
}

#[pymethods]
impl PyPtsResetPolicy {
    /// Emit a synthetic EOS before recreating the encoder (default).
    ///
    /// Downstream sees a clean EOS boundary between old and new streams.
    #[staticmethod]
    fn eos_on_decreasing_pts() -> Self {
        Self {
            inner: PtsResetPolicy::EosOnDecreasingPts,
        }
    }

    /// Silently recreate the encoder without emitting EOS.
    #[staticmethod]
    fn recreate_on_decreasing_pts() -> Self {
        Self {
            inner: PtsResetPolicy::RecreateOnDecreasingPts,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            PtsResetPolicy::EosOnDecreasingPts => {
                "PtsResetPolicy.eos_on_decreasing_pts()".to_string()
            }
            PtsResetPolicy::RecreateOnDecreasingPts => {
                "PtsResetPolicy.recreate_on_decreasing_pts()".to_string()
            }
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __ne__(&self, other: &Self) -> bool {
        self.inner != other.inner
    }

    fn __hash__(&self) -> u64 {
        self.inner as u64
    }
}

impl PyPtsResetPolicy {
    pub(crate) fn to_rust(&self) -> PtsResetPolicy {
        self.inner
    }
}

/// Global defaults for the Picasso engine.
#[pyclass(from_py_object, name = "GeneralSpec", module = "savant_rs.picasso")]
#[derive(Debug, Clone)]
pub struct PyGeneralSpec {
    /// Optional name for this engine instance, used internally for logging and
    /// future extensibility.
    #[pyo3(get, set)]
    pub name: String,
    /// Default idle timeout in seconds before a source is considered for
    /// eviction.
    #[pyo3(get, set)]
    pub idle_timeout_secs: u64,
    /// Capacity of the per-worker inflight message queue.
    #[pyo3(get, set)]
    pub inflight_queue_size: usize,
    /// Policy for handling non-monotonic (decreasing) PTS values.
    #[pyo3(get, set)]
    pub pts_reset_policy: PyPtsResetPolicy,
}

#[pymethods]
impl PyGeneralSpec {
    #[new]
    #[pyo3(signature = (name = "picasso", idle_timeout_secs = 30, inflight_queue_size = 8, pts_reset_policy = None))]
    fn new(
        name: &str,
        idle_timeout_secs: u64,
        inflight_queue_size: usize,
        pts_reset_policy: Option<PyPtsResetPolicy>,
    ) -> Self {
        Self {
            name: name.to_string(),
            idle_timeout_secs,
            inflight_queue_size,
            pts_reset_policy: pts_reset_policy
                .unwrap_or_else(|| PyPtsResetPolicy::eos_on_decreasing_pts()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GeneralSpec(name={:?}, idle_timeout_secs={}, inflight_queue_size={}, pts_reset_policy={})",
            self.name, self.idle_timeout_secs, self.inflight_queue_size, self.pts_reset_policy.__repr__()
        )
    }
}

impl PyGeneralSpec {
    pub(crate) fn to_rust(&self) -> GeneralSpec {
        GeneralSpec {
            name: self.name.clone(),
            idle_timeout_secs: self.idle_timeout_secs,
            inflight_queue_size: self.inflight_queue_size,
            pts_reset_policy: self.pts_reset_policy.to_rust(),
        }
    }
}

/// Decision returned by the `OnEviction` callback.
///
/// Construct via the factory static methods [`keep_for`], [`terminate`], or
/// [`terminate_immediately`].
#[pyclass(
    from_py_object,
    name = "EvictionDecision",
    module = "savant_rs.picasso"
)]
#[derive(Debug, Clone)]
pub struct PyEvictionDecision {
    inner: EvictionDecision,
}

#[pymethods]
impl PyEvictionDecision {
    /// Keep the source alive for at least `secs` more seconds.
    #[staticmethod]
    fn keep_for(secs: u64) -> Self {
        Self {
            inner: EvictionDecision::KeepFor(secs),
        }
    }

    /// Drain the encoder (send EOS) then terminate the worker.
    #[staticmethod]
    fn terminate() -> Self {
        Self {
            inner: EvictionDecision::Terminate,
        }
    }

    /// Terminate the worker immediately without draining.
    #[staticmethod]
    fn terminate_immediately() -> Self {
        Self {
            inner: EvictionDecision::TerminateImmediately,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            EvictionDecision::KeepFor(s) => format!("EvictionDecision.keep_for({s})"),
            EvictionDecision::Terminate => "EvictionDecision.terminate()".to_string(),
            EvictionDecision::TerminateImmediately => {
                "EvictionDecision.terminate_immediately()".to_string()
            }
        }
    }
}

impl PyEvictionDecision {
    pub(crate) fn to_rust(&self) -> EvictionDecision {
        self.inner.clone()
    }

    #[allow(dead_code)]
    pub(crate) fn from_rust(d: EvictionDecision) -> Self {
        Self { inner: d }
    }
}
