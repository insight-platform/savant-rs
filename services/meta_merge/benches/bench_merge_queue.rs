use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pyo3::{
    types::{PyBytes, PyList},
    Python,
};
use savant_core_py::test::utils::gen_frame;
use std::{hint::black_box, time::Duration};

// Import the merge_queue module
use meta_merge::egress::merge_queue::MergeQueue;

const DATA_SIZES: &[(usize, &str)] = &[
    (10 * 1024, "10KB"),
    (100 * 1024, "100KB"),
    (1024 * 1024, "1MB"),
];

const PACKETS_PER_ITERATION: usize = 1000;

fn create_test_data(size: usize) -> Vec<u8> {
    vec![0u8; size]
}

fn bench_push_frame(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_queue_push");

    for &(data_size, size_label) in DATA_SIZES {
        // Total throughput = data_size * number of packets processed per iteration
        group.throughput(Throughput::Bytes(
            (data_size * PACKETS_PER_ITERATION) as u64,
        ));

        group.bench_function(BenchmarkId::new("push_frame", size_label), |b| {
            b.iter(|| {
                Python::attach(|py| {
                    let mut queue = MergeQueue::new(Duration::from_secs(1));

                    // Push frames until we get an error (late frame)
                    let mut count = 0;
                    loop {
                        let data = create_test_data(data_size);
                        let bytes = PyBytes::new(py, &data);
                        let py_data = PyList::new(py, &[bytes]).unwrap();
                        let labels = PyList::empty(py);
                        let frame = gen_frame();
                        let result = queue.push_frame(
                            black_box(frame),
                            black_box(py_data.unbind()),
                            black_box(labels.unbind()),
                        );

                        if result.is_err() {
                            break;
                        }
                        count += 1;

                        // Limit iterations to prevent excessive memory usage
                        if count >= PACKETS_PER_ITERATION {
                            break;
                        }
                    }
                    count
                })
            });
        });
    }

    group.finish();
}

fn bench_push_and_fetch(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_queue_combined");

    for &(data_size, size_label) in DATA_SIZES {
        // Total throughput = data_size * number of packets processed per iteration
        group.throughput(Throughput::Bytes(
            (data_size * PACKETS_PER_ITERATION) as u64,
        ));

        group.bench_function(BenchmarkId::new("push_and_fetch", size_label), |b| {
            b.iter(|| {
                Python::attach(|py| {
                    let mut queue = MergeQueue::new(Duration::from_millis(0));

                    // Push and fetch frames
                    let mut uuids = Vec::new();
                    let mut count = 0;
                    loop {
                        let data = create_test_data(data_size);
                        let bytes = PyBytes::new(py, &data);
                        let py_data = PyList::new(py, &[bytes]).unwrap();
                        let labels = PyList::empty(py);
                        let frame = gen_frame();
                        let uuid = frame.0.get_uuid();
                        if queue
                            .push_frame(
                                black_box(frame),
                                black_box(py_data.unbind()),
                                black_box(labels.unbind()),
                            )
                            .is_ok()
                        {
                            uuids.push(uuid);
                        }
                        if let Ok((_, _eos, _reason)) = black_box(queue.fetch_head()) {
                            count += 1;
                        }
                        if count >= PACKETS_PER_ITERATION {
                            break;
                        }
                    }
                })
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_push_frame,
    // bench_fetch_head,
    bench_push_and_fetch
);
criterion_main!(benches);
