use std::time::Duration;

use anyhow::Result;
use savant_services_common::job_writer::SinkConfiguration;
use savant_services_common::source::SourceConfiguration;
use serde::{Deserialize, Serialize};
use twelf::{config, Layer};

const DEFAULT_MAX_DURATION: Duration = Duration::from_secs(5);
pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_millis(1);

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum EosPolicy {
    Allow,
    Deny,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IngressConfiguration {
    pub name: String,
    pub socket: SourceConfiguration,
    pub handler: Option<String>,
    pub eos_policy: Option<EosPolicy>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EgressConfiguration {
    pub socket: SinkConfiguration,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NameCacheConfiguration {
    pub ttl: Duration,
    pub size: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HandlerInitConfiguration {
    pub python_root: String,
    pub module_name: String,
    pub function_name: String,
    pub args: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CallbacksConfiguration {
    pub on_merge: String,
    pub on_head_expire: String,
    pub on_head_ready: String,
    pub on_late_arrival: String,
    pub on_unsupported_message: Option<String>,
    pub on_send: Option<String>,
}

fn default_max_duration() -> Duration {
    DEFAULT_MAX_DURATION
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QueueConfiguration {
    #[serde(default = "default_max_duration")]
    pub max_duration: Duration,
}

fn default_queue_configuration() -> QueueConfiguration {
    QueueConfiguration {
        max_duration: DEFAULT_MAX_DURATION,
    }
}

fn default_idle_sleep() -> Duration {
    DEFAULT_IDLE_TIMEOUT
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CommonConfiguration {
    pub init: Option<HandlerInitConfiguration>,
    pub callbacks: CallbacksConfiguration,
    #[serde(default = "default_idle_sleep")]
    pub idle_sleep: Duration,
    #[serde(default = "default_queue_configuration")]
    pub queue: QueueConfiguration,
}

#[config]
#[derive(Debug, Serialize, Clone)]
pub struct ServiceConfiguration {
    pub ingress: Vec<IngressConfiguration>,
    pub egress: EgressConfiguration,
    pub common: CommonConfiguration,
}

impl ServiceConfiguration {
    pub(crate) fn validate(&mut self) -> Result<()> {
        let python_root = self.common.init.as_ref().unwrap().python_root.clone();
        let metadata = std::fs::metadata(&python_root)?;
        if !metadata.is_dir() {
            return Err(anyhow::anyhow!("{} is not a directory", python_root));
        }

        // check that allow policy is set no more than once
        let allow_policy_count = self
            .ingress
            .iter()
            .filter(|ingress| matches!(ingress.eos_policy, Some(EosPolicy::Allow)))
            .count();
        if allow_policy_count > 1 {
            anyhow::bail!("The eos_policy == 'allow' is set more than once which can lead to multiple EOS delivery to downstream services.")
        }

        Ok(())
    }

    pub fn new(path: &str) -> Result<Self> {
        let mut conf = Self::with_layers(&[Layer::Json(path.into())])?;
        conf.validate()?;
        Ok(conf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate() -> anyhow::Result<()> {
        std::env::set_var("PYTHON_MODULE_ROOT", "assets/python");
        let _ = ServiceConfiguration::new("assets/configuration.json")?;
        Ok(())
    }
}
