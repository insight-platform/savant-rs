use log::error;
use opentelemetry::global;
use opentelemetry_jaeger_propagator::Propagator;
use opentelemetry_otlp::{SpanExporterBuilder, WithExportConfig};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{Config, TracerProvider};
use opentelemetry_sdk::{runtime, Resource};
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_NAMESPACE};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::cell::OnceCell;
use std::fs;
use std::time::Duration;
use twelf::{config, Layer};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ContextPropagationFormat {
    #[serde(rename = "jaeger")]
    Jaeger,
    #[serde(rename = "w3c")]
    W3C,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Protocol {
    #[serde(rename = "grpc")]
    Grpc,
    #[serde(rename = "http-binary")]
    HttpBinary,
    #[serde(rename = "http-json")]
    HttpJson,
}

impl From<Protocol> for opentelemetry_otlp::Protocol {
    fn from(value: Protocol) -> Self {
        match value {
            Protocol::Grpc => opentelemetry_otlp::Protocol::Grpc,
            Protocol::HttpBinary => opentelemetry_otlp::Protocol::HttpBinary,
            Protocol::HttpJson => opentelemetry_otlp::Protocol::HttpJson,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub key: String,
    pub certificate: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientTlsConfig {
    pub ca: Option<String>,
    pub identity: Option<Identity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerConfiguration {
    pub service_name: String,
    pub protocol: Protocol,
    pub endpoint: String,
    pub tls: Option<ClientTlsConfig>,
    pub timeout: Option<Duration>,
}

#[config]
#[derive(Debug, Serialize, Clone)]
pub struct TelemetryConfiguration {
    pub context_propagation_format: Option<ContextPropagationFormat>,
    pub tracer: Option<TracerConfiguration>,
}

impl TelemetryConfiguration {
    pub fn no_op() -> Self {
        Self {
            context_propagation_format: Some(ContextPropagationFormat::W3C),
            tracer: None,
        }
    }

    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let config = TelemetryConfiguration::with_layers(&[Layer::Json(path.into())])?;
        Ok(config)
    }
}

pub struct Configurator;

impl Configurator {
    /// Initializes global tracer provider, propagator and error handler.
    pub fn new(service_namespace: &str, config: &TelemetryConfiguration) -> Self {
        let tracer_provider = match config.tracer.as_ref() {
            Some(tracer_config) => {
                let exporter: SpanExporterBuilder = match tracer_config.protocol {
                    Protocol::Grpc => {
                        let mut builder = opentelemetry_otlp::new_exporter()
                            .tonic()
                            .with_endpoint(tracer_config.endpoint.clone());

                        builder = if let Some(timeout) = tracer_config.timeout {
                            builder.with_timeout(timeout)
                        } else {
                            builder
                        };

                        let mut tonic_tls_config =
                            tonic::transport::ClientTlsConfig::new().with_enabled_roots();

                        tonic_tls_config = if let Some(tls_config) = tracer_config.tls.as_ref() {
                            tonic_tls_config =
                                if let Some(root_certificate) = tls_config.ca.as_ref() {
                                    let buf = fs::read(root_certificate)
                                        .expect("Failed to load root certificate");
                                    let cert = tonic::transport::Certificate::from_pem(buf);

                                    tonic_tls_config.ca_certificate(cert)
                                } else {
                                    tonic_tls_config
                                };

                            tonic_tls_config =
                                if let Some(identity_conf) = tls_config.identity.as_ref() {
                                    let cert = fs::read(&identity_conf.certificate)
                                        .expect("Failed to load identity certificate");
                                    let key = fs::read(&identity_conf.key)
                                        .expect("Failed to load identity key");
                                    let identity = tonic::transport::Identity::from_pem(cert, key);

                                    tonic_tls_config.identity(identity)
                                } else {
                                    tonic_tls_config
                                };

                            tonic_tls_config
                        } else {
                            tonic_tls_config
                        };
                        builder = builder.with_tls_config(tonic_tls_config);

                        builder.into()
                    }
                    Protocol::HttpBinary | Protocol::HttpJson => {
                        let mut builder = opentelemetry_otlp::new_exporter()
                            .http()
                            .with_endpoint(tracer_config.endpoint.clone())
                            .with_protocol(tracer_config.protocol.clone().into());

                        builder = if let Some(timeout) = tracer_config.timeout {
                            builder.with_timeout(timeout)
                        } else {
                            builder
                        };

                        let mut client_builder = reqwest::Client::builder()
                            .tls_built_in_root_certs(true)
                            .use_rustls_tls();
                        client_builder = if let Some(tls_config) = tracer_config.tls.as_ref() {
                            client_builder = if let Some(certificate) = tls_config.ca.as_ref() {
                                let buf =
                                    fs::read(certificate).expect("Failed to read root certificate");
                                let cert = reqwest::Certificate::from_pem(&buf)
                                    .expect("Failed to load root certificate");

                                client_builder.add_root_certificate(cert)
                            } else {
                                client_builder
                            };

                            client_builder = if let Some(identity) = tls_config.identity.as_ref() {
                                let mut buf = Vec::new();
                                buf.append(
                                    &mut fs::read(&identity.key)
                                        .expect("Failed to read identity key"),
                                );
                                buf.append(
                                    &mut fs::read(&identity.certificate)
                                        .expect("Failed to read identity certificate"),
                                );

                                let identity = reqwest::Identity::from_pem(&buf)
                                    .expect("Failed to load identity");

                                client_builder.identity(identity)
                            } else {
                                client_builder
                            };
                            client_builder
                        } else {
                            client_builder
                        };

                        let client = client_builder.build().expect("Failed to create a client");
                        builder = builder.with_http_client(client);
                        builder.into()
                    }
                };

                opentelemetry_otlp::new_pipeline()
                    .tracing()
                    .with_exporter(exporter)
                    .with_trace_config(Config::default().with_resource(Resource::new(vec![
                        opentelemetry::KeyValue::new(
                            SERVICE_NAME,
                            tracer_config.service_name.clone(),
                        ),
                        opentelemetry::KeyValue::new(
                            SERVICE_NAMESPACE,
                            service_namespace.to_string(),
                        ),
                    ])))
                    .install_batch(runtime::Tokio)
                    .expect("Failed to install OpenTelemetry tracer globally")
            }
            None => {
                let exporter = opentelemetry_stdout::SpanExporter::builder()
                    .with_writer(std::io::sink())
                    .build();
                TracerProvider::builder()
                    .with_simple_exporter(exporter)
                    .build()
            }
        };
        global::set_tracer_provider(tracer_provider);

        match config.context_propagation_format {
            None | Some(ContextPropagationFormat::Jaeger) => {
                global::set_text_map_propagator(Propagator::new())
            }
            Some(ContextPropagationFormat::W3C) => {
                global::set_text_map_propagator(TraceContextPropagator::new())
            }
        }

        global::set_error_handler(|e| {
            error!(target: "opentelemetry", "{}", e);
        })
        .expect("Failed to set OpenTelemetry error handler");

        Self
    }

    pub fn shutdown(&mut self) {
        global::shutdown_tracer_provider();
    }
}

static CONFIGURATOR: Mutex<OnceCell<Configurator>> = Mutex::new(OnceCell::new());

pub fn init(config: &TelemetryConfiguration) {
    let configurator = CONFIGURATOR.lock();
    match configurator.get() {
        Some(_) => panic!("Open Telemetry has been configured"),
        None => {
            let c = Configurator::new("savant", config);
            let result = configurator.set(c);
            if result.is_err() {
                // should not happen
                panic!("Failed to configure OpenTelemetry");
            }
        }
    }
}

pub fn init_from_file(path: &str) {
    let config = TelemetryConfiguration::from_file(path).unwrap_or_else(|e| {
        panic!(
            "Failed to load telemetry configuration from {}, error: {}",
            path, e
        )
    });
    init(&config);
}

pub fn shutdown() {
    let mut configurator = CONFIGURATOR.lock();
    if let Some(mut c) = configurator.take() {
        c.shutdown()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn test_init_from_file() -> anyhow::Result<()> {
        let conf_json = r#"
        {
            "tracer": {
                "service_name": "savant-core",
                "protocol": "grpc",
                "endpoint": "http://localhost:4318",
                "timeout": {
                    "secs": 10,
                    "nanos": 0
                },
                "tls": {
                    "ca": "path/to/ca.pem",
                    "identity": {
                        "key": "path/to/key.pem",
                        "certificate": "path/to/certificate.pem"
                    }
                }
            }
        }
        "#;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(conf_json.as_bytes()).unwrap();
        let _ = TelemetryConfiguration::from_file(file.path().to_str().unwrap())?;
        Ok(())
    }
}
