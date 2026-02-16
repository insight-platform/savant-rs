use std::time::Duration;

use anyhow::{Context, Result};
use savant_services_common::job_writer::SinkConfiguration;
use savant_services_common::source::SourceConfiguration;
use serde::{Deserialize, Serialize};
use twelf::{config, Layer};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IngressConfiguration {
    pub socket: SourceConfiguration,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EgressConfiguration {
    pub socket: SinkConfiguration,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub enum InvocationContext {
    #[default]
    AfterReceive,
    BeforeSend,
}

fn default_invocation_context() -> InvocationContext {
    InvocationContext::AfterReceive
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HandlerInitConfiguration {
    pub python_root: String,
    pub module_name: String,
    pub function_name: String,
    pub args: Option<serde_json::Value>,
    #[serde(default = "default_invocation_context")]
    pub invocation_context: InvocationContext,
}

fn default_telemetry_port() -> u16 {
    8080
}

fn default_stats_log_interval() -> Duration {
    Duration::from_secs(60)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TelemetryConfiguration {
    #[serde(default = "default_telemetry_port")]
    pub port: u16,
    #[serde(default = "default_stats_log_interval")]
    pub stats_log_interval: Duration,
    pub metrics_extra_labels: Option<serde_json::Value>,
}

fn default_max_length() -> usize {
    1_000_000
}

fn default_full_threshold_percentage() -> usize {
    90
}

fn default_reset_on_start() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BufferConfiguration {
    pub path: String,
    #[serde(default = "default_max_length")]
    pub max_length: usize,
    #[serde(default = "default_full_threshold_percentage")]
    pub full_threshold_percentage: usize,
    #[serde(default = "default_reset_on_start")]
    pub reset_on_start: bool,
}

fn default_idle_sleep() -> Duration {
    Duration::from_millis(1)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CommonConfiguration {
    pub message_handler_init: Option<HandlerInitConfiguration>,
    pub telemetry: TelemetryConfiguration,
    pub buffer: BufferConfiguration,
    #[serde(default = "default_idle_sleep")]
    pub idle_sleep: Duration,
}

#[config]
#[derive(Debug, Serialize, Clone)]
pub struct ServiceConfiguration {
    pub ingress: IngressConfiguration,
    pub egress: EgressConfiguration,
    pub common: CommonConfiguration,
}

impl ServiceConfiguration {
    pub(crate) fn validate(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn new(path: &str) -> Result<Self> {
        let mut conf = Self::with_layers(&[Layer::Json(path.into())]).with_context(|| {
            let cwd = std::env::current_dir().unwrap();
            format!(
                "Failed to load configuration from {}, current working directory: {}",
                path,
                cwd.display()
            )
        })?;
        conf.validate()?;
        Ok(conf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate() -> anyhow::Result<()> {
        let _ = ServiceConfiguration::new("assets/configuration_full.json")?;
        Ok(())
    }
}
