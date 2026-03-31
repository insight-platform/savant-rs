//! Criterion benchmarks for non-uniform (heterogeneous) batch assembly.
//!
//! Measures the time to build a [`NonUniformBatch`] from
//! pre-existing GstBuffers at batch sizes 1, 4, 8 and 16.  Encoder
//! creation and surface acquisition are done in the setup closure and
//! excluded from timing.
//!
//! Run with:
//! ```sh
//! cargo bench -p deepstream_buffers --bench non_uniform_batch
//! ```

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use deepstream_buffers::{
    cuda_init, BufferGenerator, NonUniformBatch, NvBufSurfaceMemType, SavantIdMetaKind,
    SharedBuffer, SurfaceView, VideoFormat,
};
use std::sync::Once;

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(|| {
        gstreamer::init().expect("GStreamer init failed");
        cuda_init(0).expect("CUDA init failed");
    });
}

fn make_gen(format: VideoFormat, w: u32, h: u32, pool: u32) -> BufferGenerator {
    BufferGenerator::builder(format, w, h)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(pool)
        .max_buffers(pool)
        .build()
        .expect("failed to build generator")
}

/// Pre-allocate `n` SharedBuffers from a generator, along with their SurfaceViews.
fn acquires(gen: &BufferGenerator, n: usize) -> Vec<(SharedBuffer, SurfaceView)> {
    (0..n)
        .map(|i| {
            let shared = gen.acquire(Some(i as u128)).expect("acquire");
            let view = SurfaceView::from_buffer(&shared, 0).expect("surface view");
            (shared, view)
        })
        .collect()
}

/// Assemble a non-uniform batch from pre-existing views.
/// This is the operation being benchmarked.
fn assemble_batch(items: &[(SharedBuffer, SurfaceView)]) -> SharedBuffer {
    let mut batch = NonUniformBatch::new(0);
    let mut ids = Vec::new();
    for (i, (_shared, view)) in items.iter().enumerate() {
        batch.add(view).expect("batch add");
        ids.push(SavantIdMetaKind::Frame(i as u128));
    }
    batch.finalize(ids).expect("batch finalize")
}

fn bench_non_uniform_batch_assembly(c: &mut Criterion) {
    ensure_init();

    let mut group = c.benchmark_group("non_uniform_batch_assembly");
    group.sample_size(50);

    for &batch_size in &[1, 4, 8, 16] {
        group.throughput(criterion::Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &n| {
                let gen = make_gen(VideoFormat::RGBA, 1920, 1080, n as u32);
                b.iter_batched(
                    || acquires(&gen, n),
                    |items| {
                        let _shared = assemble_batch(&items);
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_non_uniform_batch_assembly);
criterion_main!(benches);
