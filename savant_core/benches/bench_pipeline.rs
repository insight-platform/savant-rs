#![feature(test)]

extern crate test;

use std::sync::Once;
use test::Bencher;

use anyhow::Result;
use opentelemetry::trace::TraceContextExt;

use savant_core::pipeline::Pipeline;
use savant_core::pipeline::PipelineStagePayloadType;
use savant_core::rust::PipelineConfigurationBuilder;
use savant_core::telemetry::{
    ContextPropagationFormat, init, Protocol, shutdown, TelemetryConfiguration, TracerConfiguration,
};
use savant_core::test::gen_frame;

static INIT: Once = Once::new();

fn init_telemetry() {
    INIT.call_once(|| init(&TelemetryConfiguration::no_op()))
}

fn get_pipeline(
    append_frame_meta_to_otlp_span: bool,
) -> Result<(Pipeline, Vec<(String, PipelineStagePayloadType)>)> {
    let mut stages = vec![
        (String::from("add"), PipelineStagePayloadType::Frame),
        // intermediate
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        // batches
        (format!("{}", line!()), PipelineStagePayloadType::Batch),
        (format!("{}", line!()), PipelineStagePayloadType::Batch),
        (format!("{}", line!()), PipelineStagePayloadType::Batch),
        (format!("{}", line!()), PipelineStagePayloadType::Batch),
        (format!("{}", line!()), PipelineStagePayloadType::Batch),
        (format!("{}", line!()), PipelineStagePayloadType::Batch),
        (format!("{}", line!()), PipelineStagePayloadType::Batch),
        (format!("{}", line!()), PipelineStagePayloadType::Batch),
        (format!("{}", line!()), PipelineStagePayloadType::Batch),
        (format!("{}", line!()), PipelineStagePayloadType::Batch),
        // frames
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        (format!("{}", line!()), PipelineStagePayloadType::Frame),
        // inter
        (String::from("drop"), PipelineStagePayloadType::Frame),
    ];
    let conf = PipelineConfigurationBuilder::default()
        .append_frame_meta_to_otlp_span(append_frame_meta_to_otlp_span)
        .collection_history(100)
        .frame_period(Some(100))
        .build()?;

    let pipeline_stages = stages
        .iter()
        .map(|(name, payload)| (name.clone(), payload.clone(), None, None))
        .collect();

    let pipeline = Pipeline::new(pipeline_stages, conf)?;
    stages.pop();
    pipeline.set_root_span_name("bench".to_owned())?;
    Ok((pipeline, stages))
}

fn benchmark(
    b: &mut Bencher,
    pipeline: &mut Pipeline,
    stages: Vec<(String, PipelineStagePayloadType)>,
) {
    b.iter(|| {
        let f = gen_frame();
        let mut current_id = pipeline.add_frame("add", f).expect("Cannot add frame");
        let mut current_payload_type = PipelineStagePayloadType::Frame;
        for (next_stage, next_payload_type) in &stages {
            match (&current_payload_type, next_payload_type) {
                (PipelineStagePayloadType::Frame, PipelineStagePayloadType::Frame) => {
                    pipeline
                        .move_as_is(next_stage, vec![current_id])
                        .expect("Cannot move");
                }
                (PipelineStagePayloadType::Frame, PipelineStagePayloadType::Batch) => {
                    current_id = pipeline
                        .move_and_pack_frames(next_stage, vec![current_id])
                        .expect("Cannot move");
                }
                (PipelineStagePayloadType::Batch, PipelineStagePayloadType::Batch) => {
                    pipeline
                        .move_as_is(next_stage, vec![current_id])
                        .expect("Cannot move");
                }
                (PipelineStagePayloadType::Batch, PipelineStagePayloadType::Frame) => {
                    let ids = pipeline
                        .move_and_unpack_batch(next_stage, current_id)
                        .expect("Cannot move");
                    current_id = ids[0];
                }
            }
            current_payload_type = next_payload_type.clone();
        }
        let results = pipeline.delete(current_id).expect("Cannot delete");
        for (_, ctx) in results {
            ctx.span().end();
        }
        assert_eq!(pipeline.get_id_locations_len(), 0);
    });
}

#[bench]
fn bench_pipeline2_sampling_none(b: &mut Bencher) -> Result<()> {
    init_telemetry();

    let (mut pipeline, stages) = get_pipeline(false)?;
    pipeline.set_sampling_period(0)?;
    benchmark(b, &mut pipeline, stages);

    pipeline.log_final_fps();
    let records = pipeline.get_stat_records(1);
    dbg!(&records);

    Ok(())
}

#[bench]
fn bench_pipeline2_sampling_none_with_json(b: &mut Bencher) -> Result<()> {
    init_telemetry();

    let (mut pipeline, stages) = get_pipeline(true)?;
    pipeline.set_sampling_period(0)?;
    benchmark(b, &mut pipeline, stages);
    Ok(())
}

#[bench]
fn bench_pipeline2_sampling_every(b: &mut Bencher) -> Result<()> {
    init_telemetry();

    let (mut pipeline, stages) = get_pipeline(false)?;
    pipeline.set_sampling_period(1)?;
    benchmark(b, &mut pipeline, stages);

    Ok(())
}

#[bench]
#[ignore]
fn bench_pipeline2_with_jaeger_no_sampling(b: &mut Bencher) -> Result<()> {
    let tracer_config = TracerConfiguration {
        service_name: "bench-pipeline".to_string(),
        protocol: Protocol::Grpc,
        endpoint: "http://localhost:4317".to_string(),
        tls: None,
        timeout: None,
    };
    let config = TelemetryConfiguration {
        context_propagation_format: Some(ContextPropagationFormat::Jaeger),
        tracer: Some(tracer_config),
    };
    init(&config);
    let (mut pipeline, stages) = get_pipeline(false)?;
    pipeline.set_sampling_period(1)?;
    benchmark(b, &mut pipeline, stages);

    shutdown();

    Ok(())
}
