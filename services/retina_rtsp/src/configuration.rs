use anyhow::Result;
use hashbrown::HashMap;
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
pub struct SyncWindowDurationConfiguration {
    pub group_window_duration: Duration,
    pub batch_duration: Duration,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RtspSourceGroup {
    pub sources: Vec<RtspSource>,
    pub rtcp_sync: Option<SyncWindowDurationConfiguration>,
}

#[config]
#[derive(Debug, Serialize, Clone)]
pub struct ServiceConfiguration {
    pub sink: SinkConfiguration,
    pub rtsp_sources: HashMap<String, RtspSourceGroup>,
    pub reconnect_interval: Option<Duration>,
}

impl ServiceConfiguration {
    pub fn new(path: &str) -> Result<Self> {
        let mut conf = Self::with_layers(&[Layer::Json(path.into())])?;
        if conf.reconnect_interval.is_none() {
            conf.reconnect_interval = Some(DEFAULT_RECONNECT_INTERVAL);
        }

        Ok(conf)
    }
}
