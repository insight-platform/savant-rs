use criterion::{black_box, criterion_group, criterion_main, Criterion};
use savant_core::match_query::MatchQuery;
use savant_core::message::{load_message, save_message, Message};
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::frame_batch::VideoFrameBatch;
use savant_core::primitives::frame_update::VideoFrameUpdate;
use savant_core::primitives::object::ObjectOperations;
use savant_core::primitives::WithAttributes;
use savant_core::test::gen_frame;

fn message_save_load_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_save_load");
    
    group.bench_function("video_frame", |b| {
        let message = Message::video_frame(&gen_frame());
        b.iter(|| {
            let res = black_box(save_message(&message).unwrap());
            let m = black_box(load_message(&res));
            assert!(m.is_video_frame());
        })
    });
    
    group.bench_function("eos", |b| {
        let eos = EndOfStream::new("test".to_string());
        let message = Message::end_of_stream(eos);
        b.iter(|| {
            let res = black_box(save_message(&message).unwrap());
            let m = black_box(load_message(&res));
            assert!(m.is_end_of_stream());
        })
    });
    
    group.bench_function("batch", |b| {
        let mut batch = VideoFrameBatch::new();
        batch.add(1, gen_frame());
        batch.add(2, gen_frame());
        batch.add(3, gen_frame());
        batch.add(4, gen_frame());
        let message = Message::video_frame_batch(&batch);
        b.iter(|| {
            let res = black_box(save_message(&message).unwrap());
            let m = black_box(load_message(&res));
            assert!(m.is_video_frame_batch());
        })
    });
    
    group.bench_function("frame_update", |b| {
        let f = gen_frame();
        let mut update = VideoFrameUpdate::default();
        for o in f.access_objects(&MatchQuery::Idle) {
            update.add_object(o.detached_copy(), None);
        }
        let attrs = f.get_attributes();
        for (namespace, label) in attrs {
            update.add_frame_attribute(f.get_attribute(&namespace, &label).unwrap());
        }
        let message = Message::video_frame_update(update);
        
        b.iter(|| {
            let res = black_box(save_message(&message).unwrap());
            let m = black_box(load_message(&res));
            assert!(m.is_video_frame_update());
        })
    });
    
    group.finish();
}

criterion_group!(benches, message_save_load_benchmarks);
criterion_main!(benches);
