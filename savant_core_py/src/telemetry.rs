use pyo3::{pyclass, pyfunction, pymethods};
use savant_core::telemetry;
use std::time::Duration;

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum ContextPropagationFormat {
    Jaeger,
    W3C,
}

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum Protocol {
    Grpc,
    HttpBinary,
    HttpJson,
}

#[pyclass]
#[derive(Clone)]
pub struct Identity(telemetry::Identity);

#[pymethods]
impl Identity {
    #[new]
    pub fn new(key: String, certificate: String) -> Self {
        Self(telemetry::Identity { key, certificate })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ClientTlsConfig(telemetry::ClientTlsConfig);

#[pymethods]
impl ClientTlsConfig {
    #[new]
    #[pyo3(signature = (ca=None, identity=None))]
    pub fn new(ca: Option<String>, identity: Option<Identity>) -> Self {
        Self(telemetry::ClientTlsConfig {
            ca,
            identity: identity.map(|e| e.0),
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct TracerConfiguration(telemetry::TracerConfiguration);

#[pymethods]
impl TracerConfiguration {
    #[new]
    #[pyo3(signature = (service_name, protocol, endpoint, tls=None, timeout=None))]
    pub fn new(
        service_name: String,
        protocol: Protocol,
        endpoint: String,
        tls: Option<ClientTlsConfig>,
        timeout: Option<u64>,
    ) -> Self {
        Self(telemetry::TracerConfiguration {
            service_name,
            protocol: protocol.into(),
            endpoint,
            tls: tls.map(|e| e.0),
            timeout: timeout.map(Duration::from_millis),
        })
    }
}

#[pyclass]
pub struct TelemetryConfiguration(telemetry::TelemetryConfiguration);

#[pymethods]
impl TelemetryConfiguration {
    #[new]
    #[pyo3(signature = (context_propagation_format=None, tracer=None))]
    pub fn new(
        context_propagation_format: Option<ContextPropagationFormat>,
        tracer: Option<TracerConfiguration>,
    ) -> Self {
        Self(telemetry::TelemetryConfiguration {
            context_propagation_format: context_propagation_format.map(|e| e.into()),
            tracer: tracer.map(|e| e.0),
        })
    }

    #[staticmethod]
    pub fn no_op() -> Self {
        Self(telemetry::TelemetryConfiguration::no_op())
    }
}

impl From<ContextPropagationFormat> for telemetry::ContextPropagationFormat {
    fn from(value: ContextPropagationFormat) -> Self {
        match value {
            ContextPropagationFormat::Jaeger => telemetry::ContextPropagationFormat::Jaeger,
            ContextPropagationFormat::W3C => telemetry::ContextPropagationFormat::W3C,
        }
    }
}

impl From<Protocol> for telemetry::Protocol {
    fn from(value: Protocol) -> Self {
        match value {
            Protocol::Grpc => telemetry::Protocol::Grpc,
            Protocol::HttpBinary => telemetry::Protocol::HttpBinary,
            Protocol::HttpJson => telemetry::Protocol::HttpJson,
        }
    }
}

/// Initializes OpenTelemetry.
///
/// Params
/// ------
/// config: :py:class:`TelemetryConfiguration`
///   The configuration for OpenTelemetry
///
#[pyfunction]
pub fn init(config: &TelemetryConfiguration) {
    telemetry::init(&config.0);
}

/// Initializes OpenTelemetry from a file.
///
/// Configuration file sample:
///  
/// {
///     "tracer": {
///         "service_name": "savant-core",
///         "protocol": "grpc",
///         "endpoint": "http://localhost:4318",
///         "timeout": {
///             "secs": 10,
///             "nanos": 0
///         },
///         "tls": {
///             "ca": "path/to/ca.pem",
///             "identity": {
///                 "key": "path/to/key.pem",
///                 "certificate": "path/to/certificate.pem"
///             }
///         }
///     }
/// }
///
/// Explore more details on the configuration file format at https://docs.savant-ai.io/develop/advanced_topics/9_open_telemetry.html
///
/// Params
/// ------
/// path: :py:class:`str`
///   The path to the file containing the configuration for OpenTelemetry
///
#[pyfunction]
pub fn init_from_file(path: &str) {
    telemetry::init_from_file(path);
}

/// Shuts down OpenTelemetry.
#[pyfunction]
pub fn shutdown() {
    telemetry::shutdown()
}
