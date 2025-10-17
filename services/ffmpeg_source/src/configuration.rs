use std::time::Duration;

use anyhow::{Context, Result};
use savant_services_common::job_writer::SinkConfiguration;
use serde::{Deserialize, Serialize};
use twelf::{config, Layer};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EgressConfiguration {
    pub socket: SinkConfiguration,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EventHandlerConfiguration {
    pub python_root: String,
    pub module_name: String,
    pub function_name: String,
    pub args: Option<serde_json::Value>,
    pub callbacks: CallbackConfiguration,
}

fn default_frame_event() -> bool {
    true
}

fn default_probe_event() -> bool {
    true
}

fn default_stream_termination_event() -> bool {
    true
}

fn default_create_streams_request() -> Duration {
    Duration::from_secs(1)
}

fn default_stop_streams_request() -> Duration {
    Duration::from_secs(1)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CallbackConfiguration {
    #[serde(default = "default_frame_event")]
    pub frame_event: bool,
    #[serde(default = "default_probe_event")]
    pub probe_event: bool,
    #[serde(default = "default_stream_termination_event")]
    pub stream_termination_event: bool,
    #[serde(default = "default_create_streams_request")]
    pub create_streams_request: Duration,
    #[serde(default = "default_stop_streams_request")]
    pub stop_streams_request: Duration,
}

#[config]
#[derive(Debug, Serialize, Clone)]
pub struct ServiceConfiguration {
    pub egress: EgressConfiguration,
    pub event_handlers: EventHandlerConfiguration,
}

impl ServiceConfiguration {
    pub fn new(path: &str) -> Result<Self> {
        let conf = Self::with_layers(&[Layer::Json(path.into())]).with_context(|| {
            let cwd = std::env::current_dir().unwrap();
            format!(
                "Failed to load configuration from {}, current working directory: {}",
                path,
                cwd.display()
            )
        })?;
        Ok(conf)
    }
}

pub const STOP_STREAMS_REQUEST_LABEL: &str = "stop_streams_request";
pub const CREATE_STREAMS_REQUEST_LABEL: &str = "create_streams_request";
pub const STREAM_TERMINATION_EVENT_LABEL: &str = "stream_termination_event";
pub const PROBE_EVENT_LABEL: &str = "probe_event";
pub const FRAME_EVENT_LABEL: &str = "frame_event";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate() -> anyhow::Result<()> {
        let _ = ServiceConfiguration::new("assets/configuration.json")?;
        Ok(())
    }
}
