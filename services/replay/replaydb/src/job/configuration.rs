use crate::job::{RoutingLabelsUpdateStrategy, STD_FPS};
use anyhow::bail;
use derive_builder::Builder;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize, Clone, Builder)]
#[builder(default)]
pub struct JobConfiguration {
    pub(crate) ts_sync: bool,
    pub(crate) skip_intermediary_eos: bool,
    pub(crate) send_eos: bool,
    pub(crate) stop_on_incorrect_ts: bool,
    pub(crate) ts_discrepancy_fix_duration: Duration,
    pub(crate) min_duration: Duration,
    pub(crate) max_duration: Duration,
    pub(crate) stored_stream_id: String,
    pub(crate) resulting_stream_id: String,
    pub(crate) routing_labels: RoutingLabelsUpdateStrategy,
    pub(crate) max_idle_duration: Duration,
    pub(crate) max_delivery_duration: Duration,
    pub(crate) send_metadata_only: bool,
    pub(crate) labels: Option<HashMap<String, String>>,
}

impl Default for JobConfiguration {
    fn default() -> Self {
        Self {
            ts_sync: false,
            skip_intermediary_eos: false,
            send_eos: false,
            stop_on_incorrect_ts: false,
            ts_discrepancy_fix_duration: Duration::from_secs_f64(1_f64 / STD_FPS),
            min_duration: Duration::from_secs_f64(1_f64 / STD_FPS),
            max_duration: Duration::from_secs_f64(1_f64 / STD_FPS),
            stored_stream_id: String::new(),
            resulting_stream_id: String::new(),
            routing_labels: RoutingLabelsUpdateStrategy::Bypass,
            max_idle_duration: Duration::from_secs(10),
            max_delivery_duration: Duration::from_secs(10),
            send_metadata_only: false,
            labels: None,
        }
    }
}

impl JobConfigurationBuilder {
    pub fn build_and_validate(&mut self) -> anyhow::Result<JobConfiguration> {
        let c = self.build()?;
        if c.min_duration > c.max_duration {
            bail!("Min PTS delta is greater than max PTS delta!");
        }
        if c.stored_stream_id.is_empty() || c.resulting_stream_id.is_empty() {
            bail!("Stored source id or resulting source id is empty!");
        }
        Ok(c)
    }
}
