use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cudarc::runtime::sys as cuda;
use deepstream_nvinfer::{
    BatchInput, Context, InferContextInitParams, InferFormat, InferNetworkMode, InferTensorOrder,
};
use std::{ffi::c_void, ptr};

const WIDTH: usize = 112;
const HEIGHT: usize = 112;
const CHANNELS: usize = 3;

fn setup_cuda_memory() -> anyhow::Result<(*mut c_void, usize)> {
    // Select device (creates the primary context for the runtime API)
    unsafe {
        cuda::cudaSetDevice(0).result()?;
    }

    let mut dptr: *mut c_void = ptr::null_mut();
    let mut row_width: usize = 0;

    let row_bytes = WIDTH * CHANNELS * std::mem::size_of::<u8>();

    // Allocate pitched 2D memory: height rows, each at least row_bytes
    unsafe {
        cuda::cudaMallocPitch(&mut dptr, &mut row_width as *mut usize, row_bytes, HEIGHT)
            .result()?;
    }

    Ok((dptr, row_width))
}

fn setup_infer_context() -> anyhow::Result<Context> {
    let mut init_params = InferContextInitParams::new();
    init_params
        .set_gpu_id(0)
        .set_max_batch_size(16)
        .set_unique_id(1)
        .set_network_mode(InferNetworkMode::FP16)
        .set_onnx_file_path("assets/adaface_ir50_webface4m.onnx")?
        .set_engine_file_path("assets/adaface_ir50_webface4m.onnx_b16_gpu0_fp16.engine")?
        .set_network_scale_factor(0.007843137254902f32)
        .set_offsets(&[127.5, 127.5, 127.5])?
        .set_net_input_order(InferTensorOrder::NCHW)
        .set_net_input_format(InferFormat::RGB)
        .set_infer_input_dims(CHANNELS, WIDTH, HEIGHT);

    Ok(Context::new_with_default_logging(init_params)?)
}

fn benchmark_inference_batch_sizes(c: &mut Criterion) {
    let mut infer_context = setup_infer_context().expect("Failed to setup infer context");

    let (dptr, row_width) = setup_cuda_memory().expect("Failed to setup CUDA memory");
    let mut group = c.benchmark_group("inference_batch_sizes");

    for batch_size in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let mut batch_input = BatchInput::new();
                    let frame_ptrs: Vec<*mut std::ffi::c_void> =
                        (0..batch_size).map(|_| dptr).collect();
                    batch_input.set_frames(frame_ptrs, InferFormat::RGB, row_width);
                    infer_context
                        .queue_input_batch(&batch_input)
                        .expect("Failed to queue input");
                    let batch_output = infer_context
                        .dequeue_output_batch()
                        .expect("Failed to dequeue output");
                    let frame_outputs = batch_output.frame_outputs();
                    std::hint::black_box(frame_outputs);
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_inference_batch_sizes);
criterion_main!(benches);
