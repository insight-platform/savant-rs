use std::time::Duration;

use anyhow::Result;
use savant_services_common::job_writer::SinkConfiguration;
use savant_services_common::source::SourceConfiguration;
use serde::{Deserialize, Serialize};
use twelf::{config, Layer};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IngressConfiguration {
    pub name: String,
    pub socket: SourceConfiguration,
    pub handler: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EgressConfiguration {
    pub name: String,
    pub socket: Option<SinkConfiguration>,
    pub matcher: Option<String>,
    pub rename_handler: Option<String>,
}

pub const DEFAULT_NAME_CACHE_TTL: Duration = Duration::from_secs(10);
pub const DEFAULT_NAME_CACHE_SIZE: usize = 1000;

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

pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_millis(2);

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CommonConfiguration {
    pub name_cache: Option<NameCacheConfiguration>,
    pub init: Option<HandlerInitConfiguration>,
    pub idle_sleep: Option<Duration>,
}

impl Default for NameCacheConfiguration {
    fn default() -> Self {
        Self {
            ttl: DEFAULT_NAME_CACHE_TTL,
            size: DEFAULT_NAME_CACHE_SIZE,
        }
    }
}

#[config]
#[derive(Debug, Serialize, Clone)]
pub struct ServiceConfiguration {
    #[serde(rename = "ingres")]
    pub ingress: Vec<IngressConfiguration>,
    pub egress: Vec<EgressConfiguration>,
    pub common: CommonConfiguration,
}

impl ServiceConfiguration {
    pub(crate) fn validate(&mut self) -> Result<()> {
        if self.common.name_cache.is_none() {
            self.common.name_cache = Some(NameCacheConfiguration::default());
        }

        if self.common.idle_sleep.is_none() {
            self.common.idle_sleep = Some(DEFAULT_IDLE_TIMEOUT);
        }

        let python_root = self.common.init.as_ref().unwrap().python_root.clone();
        let metadata = std::fs::metadata(&python_root)?;
        if !metadata.is_dir() {
            return Err(anyhow::anyhow!("{} is not a directory", python_root));
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
        let _ = ServiceConfiguration::new("assets/configuration.json")?;
        Ok(())
    }
}
