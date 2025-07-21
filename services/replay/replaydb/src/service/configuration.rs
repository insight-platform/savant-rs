use anyhow::{bail, Result};
use log::info;
use savant_services_common::{
    job_writer::{SinkConfiguration, SinkOptions},
    source::SourceConfiguration,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use twelf::{config, Layer};

const DEFAULT_COMPACTION_PERIOD: Duration = Duration::from_secs(60); // 1 minute

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Storage {
    #[serde(rename = "rocksdb")]
    RocksDB {
        path: String,
        data_expiration_ttl: Duration,
        compaction_period: Option<Duration>,
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
    pub(crate) fn validate(&mut self) -> Result<()> {
        match &mut self.storage {
            Storage::RocksDB {
                path: _,
                data_expiration_ttl: _,
                compaction_period,
            } => {
                if compaction_period.is_none() {
                    *compaction_period = Some(DEFAULT_COMPACTION_PERIOD);
                }
                info!("Compaction period is set to: {:?}", compaction_period);
            }
        }
        if self.common.management_port <= 1024 {
            bail!("Management port must be set to a value greater than 1024!");
        }
        Ok(())
    }

    pub fn new(path: &str) -> Result<Self> {
        let mut conf = Self::with_layers(&[Layer::Json(path.into())])?;
        conf.validate()?;
        Ok(conf)
    }
}
