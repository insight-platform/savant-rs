#![feature(test)]

extern crate test;

use opentelemetry::global;
use opentelemetry::sdk::export::trace::stdout;
use opentelemetry::sdk::propagation::TraceContextPropagator;
use opentelemetry::trace::TraceContextExt;
use savant_core::pipeline::PipelineStagePayloadType;
use savant_core::rust::Pipeline;
use savant_core::telemetry::init_jaeger_tracer;
use savant_core::test::gen_frame;
use std::io::sink;
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
fn bench_pipeline_sampling_1of100(b: &mut Bencher) {
    stdout::new_pipeline().with_writer(sink()).install_simple();
    global::set_text_map_propagator(TraceContextPropagator::new());

    let (pipeline, stages) = get_pipeline();
    pipeline.set_sampling_period(100);
    benchmark(b, pipeline, stages);
}

#[bench]
fn bench_pipeline_no_sampling(b: &mut Bencher) {
    stdout::new_pipeline().with_writer(sink()).install_simple();
    global::set_text_map_propagator(TraceContextPropagator::new());

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
