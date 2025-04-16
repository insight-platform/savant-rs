use crate::metrics::{get_or_create_counter_family, get_or_create_gauge_family};
use crate::pipeline::get_registered_pipelines;
use crate::rust::FrameProcessingStatRecordType;
use log::debug;

#[derive(Debug)]
pub(crate) struct PipelineMetricBuilder;

impl PipelineMetricBuilder {
    pub(crate) async fn build() -> anyhow::Result<()> {
        fn adjust_labels(src: &[&str], extra: &[&str]) -> Vec<String> {
            src.iter()
                .map(|s| s.to_string())
                .chain(extra.iter().map(|s| s.to_string()))
                .collect()
        }

        fn record_type_to_str(rt: &FrameProcessingStatRecordType) -> &'static str {
            match rt {
                FrameProcessingStatRecordType::Initial => "initial",
                FrameProcessingStatRecordType::Frame => "frame",
                FrameProcessingStatRecordType::Timestamp => "timestamp",
            }
        }

        debug!("Building pipeline metrics");
        let label_names = ["record_type"].as_slice();
        let stage_performance_label_names = ["record_type", "stage_name"].as_slice();
        let stage_latency_label_names =
            ["record_type", "destination_stage_name", "source_stage_name"].as_slice();

        let registered_pipelines = get_registered_pipelines();
        debug!(
            "Found {} registered pipeline(s)",
            registered_pipelines.len()
        );
        for p in registered_pipelines.values() {
            let stats = p.get_stat_records(1);
            if stats.is_empty() {
                debug!("No stats for pipeline {:?}", p.get_name());
                continue;
            }
            let last_record = &stats[0];
            let pipeline_name = p.get_name();
            debug!("Building metrics for pipeline {:?}", &pipeline_name);
            let (additional_label_names, additional_label_values) =
                (["pipeline_name"].as_slice(), vec![pipeline_name.clone()]);

            let additional_label_value_refs = additional_label_values
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<&str>>();

            let adjusted_label_names = adjust_labels(label_names, additional_label_names);
            let aln_refs: Vec<&str> = adjusted_label_names.iter().map(|s| s.as_str()).collect();
            let adjusted_stage_performance_label_names =
                adjust_labels(stage_performance_label_names, additional_label_names);
            let aspln_refs: Vec<&str> = adjusted_stage_performance_label_names
                .iter()
                .map(|s| s.as_str())
                .collect();
            let adjusted_stage_latency_label_names =
                adjust_labels(stage_latency_label_names, additional_label_names);
            let aslln_refs: Vec<&str> = adjusted_stage_latency_label_names
                .iter()
                .map(|s| s.as_str())
                .collect();

            let frame_counter = get_or_create_counter_family(
                "frame_counter",
                Some("Number of frames passed through the module"),
                &aln_refs,
                None,
            );
            let object_counter = get_or_create_counter_family(
                "object_counter",
                Some("Number of objects passed through the module"),
                &aln_refs,
                None,
            );
            let stage_queue_length = get_or_create_gauge_family(
                "stage_queue_length",
                Some("Number of frames or batches in the stage queue"),
                &aspln_refs,
                None,
            );
            let stage_frame_counter = get_or_create_counter_family(
                "stage_frame_counter",
                Some("Number of frames passed through the stage"),
                &aspln_refs,
                None,
            );
            let stage_object_counter = get_or_create_counter_family(
                "stage_object_counter",
                Some("Number of objects passed through the stage"),
                &aspln_refs,
                None,
            );
            let stage_batch_counter = get_or_create_counter_family(
                "stage_batch_counter",
                Some("Number of batches passed through the stage"),
                &aspln_refs,
                None,
            );
            let stage_min_latency = get_or_create_gauge_family(
                "stage_min_latency",
                Some("Minimum latency of the stage"),
                &aslln_refs,
                None,
            );
            let stage_max_latency = get_or_create_gauge_family(
                "stage_max_latency",
                Some("Maximum latency of the stage"),
                &aslln_refs,
                None,
            );
            let stage_avg_latency = get_or_create_gauge_family(
                "stage_avg_latency",
                Some("Average latency of the stage"),
                &aslln_refs,
                None,
            );
            let stage_latency_samples = get_or_create_gauge_family(
                "stage_latency_samples",
                Some("Number of samples used to calculate the latency"),
                &aslln_refs,
                None,
            );
            let rt = record_type_to_str(&last_record.record_type);
            let labels = adjust_labels(&[rt], &additional_label_value_refs);
            let label_refs = labels.iter().map(|s| s.as_str()).collect::<Vec<&str>>();

            frame_counter
                .lock()
                .set(last_record.frame_no as u64, &label_refs)?;

            object_counter
                .lock()
                .set(last_record.object_counter as u64, &label_refs)?;

            debug!("Building metrics for stages");
            for (sps, sls) in &last_record.stage_stats {
                debug!("Building metrics for stage {:?}", &sps.stage_name);
                // stage_performance_labels = record_type_str, sps.stage_name
                let stage_performance_labels =
                    adjust_labels(&[rt, &sps.stage_name], &additional_label_value_refs);
                let stage_performance_label_refs = stage_performance_labels
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<&str>>();

                stage_queue_length
                    .lock()
                    .set(sps.queue_length as f64, &stage_performance_label_refs)?;

                stage_frame_counter
                    .lock()
                    .set(sps.frame_counter as u64, &stage_performance_label_refs)?;

                stage_object_counter
                    .lock()
                    .set(sps.object_counter as u64, &stage_performance_label_refs)?;
                stage_batch_counter
                    .lock()
                    .set(sps.batch_counter as u64, &stage_performance_label_refs)?;
                debug!(
                    "Building metrics for stage latencies: {}",
                    sls.latencies.len()
                );
                for (_, measurement) in &sls.latencies {
                    let source_stage_name = if let Some(ssn) = &measurement.source_stage_name {
                        ssn.as_str()
                    } else {
                        "unknown"
                    };

                    debug!(
                        "Building metrics for stage transition {:?} -> {:?}",
                        source_stage_name, &sls.stage_name
                    );
                    let stage_latency_labels = adjust_labels(
                        &[rt, &sls.stage_name, source_stage_name],
                        &additional_label_value_refs,
                    );
                    let stage_latency_label_refs = stage_latency_labels
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<&str>>();

                    stage_min_latency.lock().set(
                        measurement.min_latency.as_micros() as f64,
                        &stage_latency_label_refs,
                    )?;

                    stage_max_latency.lock().set(
                        measurement.max_latency.as_micros() as f64,
                        &stage_latency_label_refs,
                    )?;

                    stage_avg_latency.lock().set(
                        measurement.accumulated_latency.as_micros() as f64
                            / measurement.count as f64,
                        &stage_latency_label_refs,
                    )?;

                    stage_latency_samples
                        .lock()
                        .set(measurement.count as f64, &stage_latency_label_refs)?;
                }
            }
        }
        Ok(())
    }
}
