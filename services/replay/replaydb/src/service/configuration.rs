use crate::job_writer::{SinkConfiguration, SinkOptions};
use anyhow::{bail, Result};
use savant_core::{
    telemetry::init_from_file,
    transport::zeromq::{NonBlockingReader, ReaderConfigBuilder},
};
use serde::{Deserialize, Serialize};
use std::result;
use std::time::Duration;
use twelf::{config, Layer};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum TopicPrefixSpec {
    #[serde(rename = "source_id")]
    SourceId(String),
    #[serde(rename = "prefix")]
    Prefix(String),
    #[serde(rename = "none")]
    None,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SourceOptions {
    pub(crate) receive_timeout: Duration,
    pub(crate) receive_hwm: usize,
    pub(crate) topic_prefix_spec: TopicPrefixSpec,
    pub(crate) source_cache_size: usize,
    pub(crate) fix_ipc_permissions: Option<u32>,
    pub(crate) inflight_ops: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SourceConfiguration {
    pub(crate) url: String,
    pub(crate) options: Option<SourceOptions>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Storage {
    #[serde(rename = "rocksdb")]
    RocksDB {
        path: String,
        data_expiration_ttl: Duration,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CommonConfiguration {
    pub management_port: u16,
    pub stats_period: Duration,
    pub pass_metadata_only: bool,
    pub job_writer_cache_max_capacity: u64,
    pub job_writer_cache_ttl: Duration,
    pub job_eviction_ttl: Duration,
    pub default_job_sink_options: Option<SinkOptions>,
    pub telemetry_config_file: Option<String>,
    pub stats_frame_period: Option<i64>,
    pub stats_timestamp_period: Option<Duration>,
}

#[config]
#[derive(Debug, Serialize, Clone)]
pub struct ServiceConfiguration {
    pub common: CommonConfiguration,
    pub in_stream: SourceConfiguration,
    pub out_stream: Option<SinkConfiguration>,
    pub storage: Storage,
}

impl ServiceConfiguration {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.common.management_port <= 1024 {
            bail!("Management port must be set to a value greater than 1024!");
        }
        if let Some(telemetry_config_file) = &self.common.telemetry_config_file {
            init_from_file(telemetry_config_file.as_str());
        }
        Ok(())
    }

    pub fn new(path: &str) -> Result<Self> {
        let conf = Self::with_layers(&[Layer::Json(path.into())])?;
        conf.validate()?;
        Ok(conf)
    }
}

impl From<&TopicPrefixSpec> for savant_core::transport::zeromq::TopicPrefixSpec {
    fn from(value: &TopicPrefixSpec) -> Self {
        match value {
            TopicPrefixSpec::SourceId(value) => Self::SourceId(value.clone()),
            TopicPrefixSpec::Prefix(value) => Self::Prefix(value.clone()),
            TopicPrefixSpec::None => Self::None,
        }
    }
}

impl TryFrom<&SourceConfiguration> for NonBlockingReader {
    type Error = anyhow::Error;

    fn try_from(
        source_conf: &SourceConfiguration,
    ) -> result::Result<NonBlockingReader, Self::Error> {
        let conf = ReaderConfigBuilder::default().url(&source_conf.url)?;
        let conf = if let Some(options) = &source_conf.options {
            let conf = if let Some(fix_ipc_permissions) = options.fix_ipc_permissions {
                conf.with_fix_ipc_permissions(Some(fix_ipc_permissions))?
            } else {
                conf
            };
            conf.with_receive_timeout(options.receive_timeout.as_millis() as i32)?
                .with_receive_hwm(options.receive_hwm as i32)?
                .with_topic_prefix_spec((&options.topic_prefix_spec).into())?
                .with_routing_cache_size(options.source_cache_size)?
        } else {
            conf
        };
        let conf = conf.build()?;

        let inflight_ops = source_conf
            .options
            .as_ref()
            .map_or(100, |opts| opts.inflight_ops);
        let mut reader = NonBlockingReader::new(&conf, inflight_ops)?;
        reader.start()?;
        Ok(reader)
    }
}
