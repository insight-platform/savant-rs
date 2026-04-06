use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use retina_rtsp::configuration;

// ── RtspBackend ──────────────────────────────────────────────────────

#[pyclass(name = "RtspBackend", module = "savant_rs.retina_rtsp", from_py_object, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PyRtspBackend {
    Retina = 0,
    Gstreamer = 1,
}

impl From<PyRtspBackend> for configuration::RtspBackend {
    fn from(val: PyRtspBackend) -> Self {
        match val {
            PyRtspBackend::Retina => configuration::RtspBackend::Retina,
            PyRtspBackend::Gstreamer => configuration::RtspBackend::Gstreamer,
        }
    }
}

#[pymethods]
impl PyRtspBackend {
    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

// ── RtspSourceOptions ────────────────────────────────────────────────

#[pyclass(name = "RtspSourceOptions", module = "savant_rs.retina_rtsp", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyRtspSourceOptions {
    inner: configuration::RtspSourceOptions,
}

#[pymethods]
impl PyRtspSourceOptions {
    #[new]
    fn new(username: String, password: String) -> Self {
        Self {
            inner: configuration::RtspSourceOptions { username, password },
        }
    }

    #[getter]
    fn username(&self) -> &str {
        &self.inner.username
    }

    #[getter]
    fn password(&self) -> &str {
        &self.inner.password
    }

    pub fn __repr__(&self) -> String {
        format!(
            "RtspSourceOptions(username={:?}, password=***)",
            self.inner.username
        )
    }
}

// ── RtspSource ───────────────────────────────────────────────────────

#[pyclass(name = "RtspSource", module = "savant_rs.retina_rtsp", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyRtspSource {
    inner: configuration::RtspSource,
}

#[pymethods]
impl PyRtspSource {
    #[new]
    #[pyo3(signature = (source_id, url, stream_position=None, options=None))]
    fn new(
        source_id: String,
        url: String,
        stream_position: Option<usize>,
        options: Option<PyRtspSourceOptions>,
    ) -> Self {
        Self {
            inner: configuration::RtspSource {
                source_id,
                url,
                stream_position,
                options: options.map(|o| o.inner),
            },
        }
    }

    #[getter]
    fn source_id(&self) -> &str {
        &self.inner.source_id
    }

    #[getter]
    fn url(&self) -> &str {
        &self.inner.url
    }

    #[getter]
    fn stream_position(&self) -> Option<usize> {
        self.inner.stream_position
    }

    #[getter]
    fn options(&self) -> Option<PyRtspSourceOptions> {
        self.inner
            .options
            .as_ref()
            .map(|o| PyRtspSourceOptions { inner: o.clone() })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "RtspSource(source_id={:?}, url={:?})",
            self.inner.source_id, self.inner.url
        )
    }
}

// ── SyncConfiguration ────────────────────────────────────────────────

#[pyclass(name = "SyncConfiguration", module = "savant_rs.retina_rtsp", from_py_object)]
#[derive(Debug, Clone)]
pub struct PySyncConfiguration {
    inner: configuration::SyncConfiguration,
}

#[pymethods]
impl PySyncConfiguration {
    #[new]
    #[pyo3(signature = (group_window_duration_ms, batch_duration_ms, network_skew_correction=false, rtcp_once=false))]
    fn new(
        group_window_duration_ms: u64,
        batch_duration_ms: u64,
        network_skew_correction: bool,
        rtcp_once: bool,
    ) -> Self {
        Self {
            inner: configuration::SyncConfiguration {
                group_window_duration: Duration::from_millis(group_window_duration_ms),
                batch_duration: Duration::from_millis(batch_duration_ms),
                network_skew_correction: Some(network_skew_correction),
                rtcp_once: Some(rtcp_once),
            },
        }
    }

    #[getter]
    fn group_window_duration_ms(&self) -> u64 {
        self.inner.group_window_duration.as_millis() as u64
    }

    #[getter]
    fn batch_duration_ms(&self) -> u64 {
        self.inner.batch_duration.as_millis() as u64
    }

    #[getter]
    fn network_skew_correction(&self) -> bool {
        self.inner.network_skew_correction.unwrap_or(false)
    }

    #[getter]
    fn rtcp_once(&self) -> bool {
        self.inner.rtcp_once.unwrap_or(false)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SyncConfiguration(group_window_duration_ms={}, batch_duration_ms={})",
            self.group_window_duration_ms(),
            self.batch_duration_ms()
        )
    }
}

// ── RtspSourceGroup ──────────────────────────────────────────────────

#[pyclass(name = "RtspSourceGroup", module = "savant_rs.retina_rtsp", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyRtspSourceGroup {
    inner: configuration::RtspSourceGroup,
}

impl PyRtspSourceGroup {
    pub fn to_rust(&self) -> configuration::RtspSourceGroup {
        self.inner.clone()
    }
}

#[pymethods]
impl PyRtspSourceGroup {
    #[new]
    #[pyo3(signature = (sources, backend=PyRtspBackend::Retina, rtcp_sr_sync=None))]
    fn new(
        sources: Vec<PyRtspSource>,
        backend: PyRtspBackend,
        rtcp_sr_sync: Option<PySyncConfiguration>,
    ) -> Self {
        Self {
            inner: configuration::RtspSourceGroup {
                sources: sources.into_iter().map(|s| s.inner).collect(),
                backend: backend.into(),
                rtcp_sr_sync: rtcp_sr_sync.map(|s| s.inner),
            },
        }
    }

    #[getter]
    fn sources(&self) -> Vec<PyRtspSource> {
        self.inner
            .sources
            .iter()
            .map(|s| PyRtspSource { inner: s.clone() })
            .collect()
    }

    #[getter]
    fn backend(&self) -> PyRtspBackend {
        match self.inner.backend {
            configuration::RtspBackend::Retina => PyRtspBackend::Retina,
            configuration::RtspBackend::Gstreamer => PyRtspBackend::Gstreamer,
        }
    }

    #[getter]
    fn rtcp_sr_sync(&self) -> Option<PySyncConfiguration> {
        self.inner
            .rtcp_sr_sync
            .as_ref()
            .map(|s| PySyncConfiguration { inner: s.clone() })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "RtspSourceGroup(sources={}, backend={:?})",
            self.inner.sources.len(),
            self.inner.backend
        )
    }
}

// ── RetinarRtspService ───────────────────────────────────────────────

#[pyclass(name = "RetinarRtspService", module = "savant_rs.retina_rtsp")]
pub struct PyRetinarRtspService {
    service: Arc<retina_rtsp::Service>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyRetinarRtspService {
    #[new]
    fn new(config_path: String) -> PyResult<Self> {
        let conf = configuration::ServiceConfiguration::new(&config_path)
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
        let service = retina_rtsp::Service::new(&conf)
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
        Ok(Self {
            service: Arc::new(service),
            runtime: Arc::new(runtime),
        })
    }

    /// Run a group.  Blocks (GIL released) until stopped or fatal error.
    fn run_group(
        &self,
        py: Python<'_>,
        group: &PyRtspSourceGroup,
        name: String,
    ) -> PyResult<()> {
        let svc = self.service.clone();
        let rt = self.runtime.clone();
        let group = group.to_rust();
        py.detach(move || {
            rt.block_on(svc.run_group(&group, name))
                .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))
        })
    }

    /// Stop a running group.  Blocks (GIL released) until the group exits.
    fn stop_group(&self, py: Python<'_>, name: String) -> PyResult<()> {
        let svc = self.service.clone();
        let rt = self.runtime.clone();
        py.detach(move || {
            rt.block_on(svc.stop_group(&name));
            Ok(())
        })
    }

    /// Stop all running groups.  Blocks (GIL released).
    fn shutdown(&self, py: Python<'_>) -> PyResult<()> {
        let svc = self.service.clone();
        let rt = self.runtime.clone();
        py.detach(move || {
            rt.block_on(svc.shutdown());
            Ok(())
        })
    }

    #[getter]
    fn running_groups(&self) -> Vec<String> {
        self.service.running_groups()
    }

    pub fn __repr__(&self) -> String {
        let groups = self.service.running_groups();
        format!("RetinarRtspService(running_groups={:?})", groups)
    }
}

// ── Module registration ──────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRtspBackend>()?;
    m.add_class::<PyRtspSourceOptions>()?;
    m.add_class::<PyRtspSource>()?;
    m.add_class::<PySyncConfiguration>()?;
    m.add_class::<PyRtspSourceGroup>()?;
    m.add_class::<PyRetinarRtspService>()?;
    Ok(())
}
