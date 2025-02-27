use crate::job::configuration::JobConfiguration;
use crate::job::stop_condition::JobStopCondition;
use crate::job_writer::SinkConfiguration;
use crate::store::JobOffset;
use anyhow::Result;
use savant_core::primitives::Attribute;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct JobQuery {
    pub sink: SinkConfiguration,
    pub configuration: JobConfiguration,
    pub stop_condition: JobStopCondition,
    pub anchor_keyframe: String,
    pub anchor_wait_duration: Option<Duration>,
    pub offset: JobOffset,
    pub attributes: Vec<Attribute>,
}

impl JobQuery {
    pub fn new(
        sink: SinkConfiguration,
        configuration: JobConfiguration,
        stop_condition: JobStopCondition,
        anchor_keyframe: Uuid,
        anchor_wait_duration: Option<Duration>,
        offset: JobOffset,
        attributes: Vec<Attribute>,
    ) -> Self {
        let anchor_keyframe = anchor_keyframe.to_string();
        Self {
            sink,
            configuration,
            stop_condition,
            offset,
            attributes,
            anchor_keyframe,
            anchor_wait_duration,
        }
    }

    pub fn json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn json_pretty(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::job::configuration::JobConfigurationBuilder;
    use crate::job::query::JobQuery;
    use crate::job::stop_condition::JobStopCondition;
    use crate::job_writer::SinkConfiguration;
    use crate::store::JobOffset;
    use savant_core::primitives::attribute_value::AttributeValue;
    use savant_core::primitives::Attribute;
    use savant_core::utils::uuid_v7::incremental_uuid_v7;
    use std::time::Duration;

    #[test]
    fn test_job_query() {
        let configuration = JobConfigurationBuilder::default()
            .min_duration(Duration::from_millis(700))
            .max_duration(Duration::from_secs_f64(1_f64 / 30_f64))
            .stored_stream_id("stored_source_id".to_string())
            .resulting_stream_id("resulting_source_id".to_string())
            .labels(Some(
                vec![("key".to_string(), "value".to_string())]
                    .into_iter()
                    .collect(),
            ))
            .build()
            .unwrap();
        let stop_condition = JobStopCondition::frame_count(1);
        let offset = JobOffset::Blocks(0);
        let job_query = JobQuery::new(
            SinkConfiguration::default(),
            configuration,
            stop_condition,
            incremental_uuid_v7(),
            Some(Duration::from_secs(1)),
            offset,
            vec![Attribute::persistent(
                "key",
                "value",
                vec![
                    AttributeValue::integer(1, Some(0.5)),
                    AttributeValue::float_vector(vec![1.0, 2.0, 3.0], None),
                ],
                &None,
                false,
            )],
        );
        let json = job_query.json_pretty().unwrap();
        println!("{}", json);
        let _ = JobQuery::from_json(&json).unwrap();
    }
}
