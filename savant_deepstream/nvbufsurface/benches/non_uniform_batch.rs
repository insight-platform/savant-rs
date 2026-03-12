//! Criterion benchmarks for non-uniform (heterogeneous) batch assembly.
//!
//! Measures the time to build a [`DsNvNonUniformSurfaceBuffer`] from
//! pre-existing GstBuffers at batch sizes 1, 4, 8 and 16.  Encoder
//! creation and surface acquisition are done in the setup closure and
//! excluded from timing.
//!
//! Run with:
//! ```sh
//! cargo bench -p deepstream_nvbufsurface --bench non_uniform_batch
//! ```

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use deepstream_nvbufsurface::{
    cuda_init, DsNvNonUniformSurfaceBuffer, DsNvSurfaceBufferGenerator, NvBufSurfaceMemType,
    VideoFormat,
};
use std::sync::Once;

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(|| {
        gstreamer::init().expect("GStreamer init failed");
        cuda_init(0).expect("CUDA init failed");
    });
}

fn make_gen(format: VideoFormat, w: u32, h: u32, pool: u32) -> DsNvSurfaceBufferGenerator {
    DsNvSurfaceBufferGenerator::builder(format, w, h)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(pool)
        .max_buffers(pool)
        .build()
        .expect("failed to build generator")
}

/// Pre-allocate `n` GstBuffers from a generator.
fn acquire_buffers(gen: &DsNvSurfaceBufferGenerator, n: usize) -> Vec<gstreamer::Buffer> {
    (0..n)
        .map(|i| {
            gen.acquire_surface(Some(i as i64))
                .expect("acquire_surface")
        })
        .collect()
}

/// Assemble a non-uniform batch from pre-existing buffers.
/// This is the operation being benchmarked.
fn assemble_batch(buffers: &[gstreamer::Buffer]) -> DsNvNonUniformSurfaceBuffer {
    let n = buffers.len() as u32;
    let mut batch = DsNvNonUniformSurfaceBuffer::new(n, 0).expect("batch creation");
    for (i, buf) in buffers.iter().enumerate() {
        batch.add(buf, Some(i as i64)).expect("batch add");
    }
    batch.finalize().expect("batch finalize");
    batch
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
                    || acquire_buffers(&gen, n),
                    |buffers| {
                        let batch = assemble_batch(&buffers);
                        let _ = batch.as_gst_buffer().expect("as_gst_buffer");
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
