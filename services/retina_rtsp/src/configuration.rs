use anyhow::Result;
use hashbrown::{HashMap, HashSet};
use replaydb::job_writer::SinkConfiguration;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use twelf::{config, Layer};

const DEFAULT_RECONNECT_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RtspSourceOptions {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RtspSource {
    pub source_id: String,
    pub url: String,
    pub stream_position: Option<usize>,
    pub options: Option<RtspSourceOptions>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SyncConfiguration {
    pub group_window_duration: Duration,
    pub batch_duration: Duration,
    pub network_skew_correction: Option<bool>,
    pub rtcp_once: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RtspSourceGroup {
    pub sources: Vec<RtspSource>,
    pub rtcp_sr_sync: Option<SyncConfiguration>,
}

#[config]
#[derive(Debug, Serialize, Clone)]
pub struct ServiceConfiguration {
    pub sink: SinkConfiguration,
    pub rtsp_sources: HashMap<String, RtspSourceGroup>,
    pub reconnect_interval: Option<Duration>,
    pub eos_on_restart: Option<bool>,
}

impl ServiceConfiguration {
    pub fn new(path: &str) -> Result<Self> {
        let conf = {
            let mut conf = Self::with_layers(&[Layer::Json(path.into())])?;
            if conf.reconnect_interval.is_none() {
                conf.reconnect_interval = Some(DEFAULT_RECONNECT_INTERVAL);
            }

            // if eos_on_restart is not set, set it to true, default value
            if conf.eos_on_restart.is_none() {
                conf.eos_on_restart = Some(true);
            }
            conf
        };

        {
            // check uniqueness of group_name
            let group_names = conf.rtsp_sources.keys().collect::<HashSet<_>>();
            if group_names.len() != conf.rtsp_sources.len() {
                return Err(anyhow::anyhow!(
                    "group names must be unique among the groups."
                ));
            }

            // all sources must have unique source_id
            let mut all_source_ids: Vec<String> = Vec::new();
            for group in conf.rtsp_sources.values() {
                let source_ids = group
                    .sources
                    .iter()
                    .map(|s| s.source_id.clone())
                    .collect::<Vec<_>>();
                all_source_ids.extend_from_slice(&source_ids);
            }
            let unique_source_ids = all_source_ids.iter().collect::<HashSet<_>>();
            if unique_source_ids.len() != all_source_ids.len() {
                return Err(anyhow::anyhow!(
                    "The source_id field must be unique among the sources."
                ));
            }
        }

        Ok(conf)
    }
}
