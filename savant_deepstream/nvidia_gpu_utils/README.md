# savant-nvidia-gpu-utils

`savant-nvidia-gpu-utils` is the NVIDIA device-discovery and memory-inspection crate used by Savant, the real-time video-analytics framework. It combines NVML on dGPU hosts with CUDA attributes and Linux `/proc` inspection on Jetson to answer practical questions such as "which GPU is this?", "does it have NVENC?", "how much memory is in use?", and "which platform tag should I use for TensorRT engine caches?".

## What's inside

- `JetsonModel` is the crate's platform model enum for Jetson boards. It distinguishes devices such as `AgxOrin64GB`, `OrinNx16GB`, `OrinNano8GB`, `XavierNx`, and `Nano`, and provides `as_str()`, `is_orin_nano()`, and `platform_tag()`.
- `gpu_mem_used_mib` reports GPU memory usage. On `x86_64` it uses `nvml-wrapper` and NVML data similar to `nvidia-smi`; on Jetson it falls back to `/proc/meminfo` and documents the limitations of unified-memory accounting for `NvBufSurface` allocations.
- `process_rss_mib` reports the current process RSS from `/proc/self/status`, which is useful alongside GPU memory numbers in tests, benchmarks, and long-running services.
- `jetson_model` and `is_jetson_kernel` detect whether the current system is a Jetson device and, if so, classify it using CUDA SM count, compute capability, and total memory.
- `has_nvenc` reports hardware encoder availability. It is aware of Jetson special cases such as Orin Nano and of dGPU architectures where NVENC is not present.
- `gpu_architecture` returns the dGPU architecture family from NVML, and `gpu_platform_tag` turns either the Jetson model or dGPU architecture into a stable lowercase tag for cache directories and deployment logic.
- `mem_total_mib` exposes total system memory as a simple helper for scripts and runtime heuristics.
- `GpuUtilsError` centralizes NVML, CUDA, I/O, and parse failures.
- The bundled `nvidia_gpu_info` CLI prints shell-safe `KEY='value'` pairs so scripts can `eval` GPU facts directly.

## Usage

```rust
use nvidia_gpu_utils::{gpu_platform_tag, has_nvenc, jetson_model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tag = gpu_platform_tag(0)?;
    let nvenc = has_nvenc(0)?;
    println!("platform={tag} nvenc={nvenc}");

    if let Some(model) = jetson_model(0)? {
        println!("jetson={} orin_nano={}", model, model.is_orin_nano());
    }

    Ok(())
}
```

## Install

```toml
[dependencies]
savant-nvidia-gpu-utils = "2"
```

This crate does not expose Cargo features today.

## System requirements

Requires Linux with the NVIDIA driver stack at run time. On `x86_64`, `savant-nvidia-gpu-utils` uses NVML through `nvml-wrapper`; on Jetson, it uses CUDA device attributes plus `/proc/version` and `/proc/meminfo`.

The `docs.rs` build is limited to `x86_64-unknown-linux-gnu`.

## Documentation

- [docs.rs](https://docs.rs/savant-nvidia-gpu-utils)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
