use std::hint::black_box;
use std::time::{Duration, Instant};

use buffer_ng::rocksdb::PersistentQueueWithCapacity;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rocksdb::Options;

const QUEUE_CAPACITY: usize = 2_000_000;
const PAYLOAD_SIZES: [usize; 3] = [128, 128 * 1024, 1024 * 1024];

fn bench_rocksdb(c: &mut Criterion) {
    let mut group = c.benchmark_group("rocksdb");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(10));

    for &payload_size in &PAYLOAD_SIZES {
        group.bench_with_input(
            BenchmarkId::new("push_latency", payload_size),
            &payload_size,
            |b, &payload_size| {
                let path = &format!("/tmp/test/rocksdb/push_latency/{payload_size}");
                std::fs::remove_dir_all(path).unwrap_or_default();

                let mut db =
                    PersistentQueueWithCapacity::new(path, QUEUE_CAPACITY, 90, Options::default())
                        .unwrap();

                let payload = vec![0u8; payload_size];

                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();
                        db.push(black_box(&[&payload])).unwrap();
                        total += start.elapsed();
                    }
                    db.pop(iters as usize).unwrap();
                    total
                });
            },
        );
    }

    for &payload_size in &PAYLOAD_SIZES {
        group.bench_with_input(
            BenchmarkId::new("pop_latency", payload_size),
            &payload_size,
            |b, &payload_size| {
                let path = &format!("/tmp/test/rocksdb/pop_latency/{payload_size}");
                std::fs::remove_dir_all(path).unwrap_or_default();

                let mut db =
                    PersistentQueueWithCapacity::new(path, QUEUE_CAPACITY, 90, Options::default())
                        .unwrap();

                let payload = vec![0u8; payload_size];

                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        db.push(&[&payload]).unwrap();
                        let start = Instant::now();
                        db.pop(1).unwrap();
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.bench_function("pop_latency_empty_queue", |b| {
        let path = "/tmp/test/rocksdb/pop_latency_empty_queue";
        std::fs::remove_dir_all(path).unwrap_or_default();

        let mut db =
            PersistentQueueWithCapacity::new(path, QUEUE_CAPACITY, 90, Options::default()).unwrap();

        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();
                db.pop(1).unwrap();
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

criterion_group!(benches, bench_rocksdb);
criterion_main!(benches);
