use criterion::{black_box, criterion_group, criterion_main, Criterion};
use savant_core::primitives::rust::VideoFrameProxy;
use savant_core::protobuf::{from_pb, ToProtobuf};
use savant_core::test::gen_frame;

fn frame_pb_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_protobuf");

    group.bench_function("save_load_video_frame_pb", |b| {
        let frame = gen_frame();
        b.iter(|| {
            let res = black_box(frame.to_pb().unwrap());
            black_box(from_pb::<savant_core::protobuf::VideoFrame, VideoFrameProxy>(&res).unwrap());
        })
    });

    group.finish();
}

criterion_group!(benches, frame_pb_benchmarks);
criterion_main!(benches);
