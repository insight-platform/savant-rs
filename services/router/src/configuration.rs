use anyhow::Result;
use savant_services_common::job_writer::SinkConfiguration;
use savant_services_common::source::SourceConfiguration;
use serde::{Deserialize, Serialize};
use twelf::{config, Layer};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HandlerConfiguration {
    pub module: String,
    pub class_name: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IngressConfiguration {
    pub name: Option<String>,
    pub socket: SourceConfiguration,
    pub handler: Option<HandlerConfiguration>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EgressConfiguration {
    pub name: Option<String>,
    pub socket: SinkConfiguration,
    pub matcher: Option<String>,
    pub handler: Option<HandlerConfiguration>,
}

#[config]
#[derive(Debug, Serialize, Clone)]
pub struct ServiceConfiguration {
    #[serde(rename = "ingres")]
    pub ingress: Vec<IngressConfiguration>,
    pub egress: Vec<EgressConfiguration>,
}

impl ServiceConfiguration {
    pub(crate) fn validate(&self) -> Result<()> {
        Ok(())
    }

    pub fn new(path: &str) -> Result<Self> {
        let conf = Self::with_layers(&[Layer::Json(path.into())])?;
        conf.validate()?;
        Ok(conf)
    }
}
