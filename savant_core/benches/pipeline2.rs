#![feature(test)]

extern crate test;

use anyhow::Result;
use opentelemetry::global;
use opentelemetry::sdk::export::trace::stdout;
use opentelemetry::sdk::propagation::TraceContextPropagator;
use opentelemetry::trace::TraceContextExt;
use savant_core::pipeline::PipelineStagePayloadType;
use savant_core::pipeline2::Pipeline;
use savant_core::telemetry::init_jaeger_tracer;
use savant_core::test::gen_frame;
use std::io::sink;
use test::Bencher;

fn get_pipeline() -> Result<(Pipeline, Vec<(String, PipelineStagePayloadType)>)> {
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
    let pipeline = Pipeline::new(stages.clone())?;
    stages.pop();
    pipeline.set_root_span_name("bench_batch_snapshot".to_owned())?;
    Ok((pipeline, stages))
}

fn benchmark(b: &mut Bencher, pipeline: Pipeline, stages: Vec<(String, PipelineStagePayloadType)>) {
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
fn bench_pipeline2_sampling_1of100(b: &mut Bencher) -> Result<()> {
    stdout::new_pipeline().with_writer(sink()).install_simple();
    global::set_text_map_propagator(TraceContextPropagator::new());

    let (pipeline, stages) = get_pipeline()?;
    pipeline.set_sampling_period(100)?;
    benchmark(b, pipeline, stages);
    Ok(())
}

#[bench]
fn bench_pipeline2_sampling_1of1(b: &mut Bencher) -> Result<()> {
    stdout::new_pipeline().with_writer(sink()).install_simple();
    global::set_text_map_propagator(TraceContextPropagator::new());

    let (pipeline, stages) = get_pipeline()?;
    pipeline.set_sampling_period(1)?;
    benchmark(b, pipeline, stages);
    Ok(())
}

#[bench]
#[ignore]
fn bench_pipeline2_with_jaeger_no_sampling(b: &mut Bencher) -> Result<()> {
    init_jaeger_tracer("bench-pipeline", "localhost:6831");
    let (pipeline, stages) = get_pipeline()?;
    pipeline.set_sampling_period(1)?;
    benchmark(b, pipeline, stages);
    Ok(())
}
