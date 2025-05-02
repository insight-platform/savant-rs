use savant_core::transport::zeromq::{NonBlockingReader, ReaderConfigBuilder};
use serde::{Deserialize, Serialize};
use std::result;
use std::time::Duration;

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
    pub receive_timeout: Duration,
    pub receive_hwm: usize,
    pub topic_prefix_spec: TopicPrefixSpec,
    pub source_cache_size: usize,
    pub fix_ipc_permissions: Option<u32>,
    pub inflight_ops: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SourceConfiguration {
    pub url: String,
    pub options: Option<SourceOptions>,
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
