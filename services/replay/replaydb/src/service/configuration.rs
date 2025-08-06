use anyhow::{bail, Result};
use log::info;
use savant_services_common::{
    job_writer::{SinkConfiguration, SinkOptions},
    source::SourceConfiguration,
};
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, time::Duration};
use twelf::{config, Layer};

#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum CompactionStyle {
    #[serde(rename = "level")]
    Level,
    #[serde(rename = "universal")]
    #[default]
    Universal,
    #[serde(rename = "fifo")]
    Fifo,
}

impl From<CompactionStyle> for rocksdb::DBCompactionStyle {
    fn from(style: CompactionStyle) -> Self {
        match style {
            CompactionStyle::Level => rocksdb::DBCompactionStyle::Level,
            CompactionStyle::Universal => rocksdb::DBCompactionStyle::Universal,
            CompactionStyle::Fifo => rocksdb::DBCompactionStyle::Fifo,
        }
    }
}

const MAX_TOTAL_WAL_SIZE: u64 = 1024 * 1024 * 1024;

fn default_max_total_wal_size() -> u64 {
    MAX_TOTAL_WAL_SIZE
}

fn default_keep_log_file_num() -> usize {
    1000
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Storage {
    #[serde(rename = "rocksdb")]
    RocksDB {
        path: PathBuf,
        data_expiration_ttl: Duration,
        #[serde(default)]
        disable_wal: bool,
        #[serde(default = "default_max_total_wal_size")]
        max_total_wal_size: u64,
        #[serde(default)]
        compaction_style: CompactionStyle,
        #[serde(default)]
        max_log_file_size: usize,
        #[serde(default = "default_keep_log_file_num")]
        keep_log_file_num: usize,
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
                disable_wal,
                max_total_wal_size,
                compaction_style,
                max_log_file_size,
                keep_log_file_num,
            } => {
                info!("Disable WAL is set to: {:?}", disable_wal);
                info!("Max total WAL size is set to: {:?}", max_total_wal_size);
                info!("Compaction style is set to: {:?}", compaction_style);
                info!("Max log file size is set to: {:?}", max_log_file_size);
                info!("Keep log file num is set to: {:?}", keep_log_file_num);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_config_default_values() {
        let conf = serde_json::from_str::<Storage>(r#"
        {
            "rocksdb": {
                "path": "test_path",
                "data_expiration_ttl": {
                    "secs": 60,
                    "nanos": 0
                }
            }
        }"#).unwrap();

        let Storage::RocksDB {
            path,
            data_expiration_ttl,
            disable_wal,
            max_total_wal_size,
            compaction_style,
            max_log_file_size,
            keep_log_file_num,
        } = conf;
        assert_eq!(path, PathBuf::from("test_path"));
        assert_eq!(data_expiration_ttl, Duration::from_secs(60));
        assert_eq!(disable_wal, false);
        assert_eq!(max_total_wal_size, MAX_TOTAL_WAL_SIZE);
        assert_eq!(compaction_style, CompactionStyle::Universal);
        assert_eq!(max_log_file_size, 0);
        assert_eq!(keep_log_file_num, 1000);
    }
}