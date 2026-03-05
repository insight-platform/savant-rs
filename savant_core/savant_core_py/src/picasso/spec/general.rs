use picasso::prelude::{EvictionDecision, GeneralSpec};
use pyo3::prelude::*;

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
}

#[pymethods]
impl PyGeneralSpec {
    #[new]
    #[pyo3(signature = (name = "picasso", idle_timeout_secs = 30))]
    fn new(name: &str, idle_timeout_secs: u64) -> Self {
        Self {
            name: name.to_string(),
            idle_timeout_secs,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GeneralSpec(name={:?}, idle_timeout_secs={})",
            self.name, self.idle_timeout_secs
        )
    }
}

impl PyGeneralSpec {
    pub(crate) fn to_rust(&self) -> GeneralSpec {
        GeneralSpec {
            name: self.name.clone(),
            idle_timeout_secs: self.idle_timeout_secs,
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
