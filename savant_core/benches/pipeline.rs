#![feature(test)]

extern crate test;

use opentelemetry::trace::TraceContextExt;
use savant_core::pipeline::Pipeline;
use savant_core::pipeline::PipelineStagePayloadType;
use savant_core::telemetry::{init_jaeger_tracer, init_noop_tracer};
use savant_core::test::gen_frame;
use test::Bencher;

fn get_pipeline() -> (Pipeline, Vec<(String, PipelineStagePayloadType)>) {
    let pipeline = Pipeline::default();
    pipeline.set_root_span_name("bench_batch_snapshot".to_owned());
    let stages = vec![
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

    pipeline
        .add_stage("add", PipelineStagePayloadType::Frame)
        .expect("Cannot add stage");
    for (name, payload_type) in &stages {
        pipeline
            .add_stage(name, payload_type.clone())
            .expect("Cannot add stage");
    }
    (pipeline, stages)
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
                    current_id = ids.values().next().unwrap().clone();
                }
            }
            current_payload_type = next_payload_type.clone();
        }
        let results = pipeline.delete(current_id).expect("Cannot delete");
        for (_, ctx) in results {
            ctx.span().end();
        }
        assert!(pipeline.get_id_locations().is_empty())
    });
}

#[bench]
fn bench_pipeline_sampling_none(b: &mut Bencher) {
    init_noop_tracer();

    let (pipeline, stages) = get_pipeline();
    pipeline.set_sampling_period(0);
    benchmark(b, pipeline, stages);
}

#[bench]
fn bench_pipeline_sampling_every(b: &mut Bencher) {
    init_noop_tracer();

    let (pipeline, stages) = get_pipeline();
    pipeline.set_sampling_period(1);
    benchmark(b, pipeline, stages);
}

#[bench]
#[ignore]
fn bench_pipeline_with_jaeger_no_sampling(b: &mut Bencher) {
    init_jaeger_tracer("bench-pipeline", "localhost:6831");
    let (pipeline, stages) = get_pipeline();
    pipeline.set_sampling_period(1);
    benchmark(b, pipeline, stages);
}
