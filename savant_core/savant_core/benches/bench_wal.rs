use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use savant_core::wal::PersistentQueueWithCapacity;

const QUEUE_CAPACITY: usize = 2_000_000;
const SEGMENT_MAX_BYTES: usize = 64 * 1024 * 1024; // 64 MiB
const PAYLOAD_SIZES: [usize; 3] = [128, 128 * 1024, 1024 * 1024];

fn bench_wal(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(10));

    for &payload_size in &PAYLOAD_SIZES {
        group.bench_with_input(
            BenchmarkId::new("push_latency", payload_size),
            &payload_size,
            |b, &payload_size| {
                let path = &format!("/tmp/test/wal/push_latency/{payload_size}");
                std::fs::remove_dir_all(path).unwrap_or_default();

                let mut q =
                    PersistentQueueWithCapacity::new(path, QUEUE_CAPACITY, 90, SEGMENT_MAX_BYTES)
                        .unwrap();

                let payload = vec![0u8; payload_size];

                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();
                        q.push(black_box(&[&payload])).unwrap();
                        total += start.elapsed();
                    }
                    q.pop(iters as usize).unwrap();
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
                let path = &format!("/tmp/test/wal/pop_latency/{payload_size}");
                std::fs::remove_dir_all(path).unwrap_or_default();

                let mut q =
                    PersistentQueueWithCapacity::new(path, QUEUE_CAPACITY, 90, SEGMENT_MAX_BYTES)
                        .unwrap();

                let payload = vec![0u8; payload_size];

                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        q.push(&[&payload]).unwrap();
                        let start = Instant::now();
                        q.pop(1).unwrap();
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.bench_function("pop_latency_empty_queue", |b| {
        let path = "/tmp/test/wal/pop_latency_empty_queue";
        std::fs::remove_dir_all(path).unwrap_or_default();

        let mut q =
            PersistentQueueWithCapacity::new(path, QUEUE_CAPACITY, 90, SEGMENT_MAX_BYTES).unwrap();

        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();
                q.pop(1).unwrap();
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

criterion_group!(benches, bench_wal);
criterion_main!(benches);
