//! Platform-aware GPU and process memory monitoring utilities.
//!
//! - **dGPU (x86_64)**: Uses NVML to query GPU device memory.
//! - **Jetson (aarch64)**: Reads `/proc/meminfo` for unified memory (RAM used).
//! - **Process RSS**: Reads `/proc/self/status` `VmRSS` (works on any Linux).

use std::io;
use thiserror::Error;

/// Error type for GPU utils operations.
#[derive(Debug, Error)]
pub enum GpuUtilsError {
    /// NVML initialization or query failed (dGPU only).
    #[cfg(target_arch = "x86_64")]
    #[error("NVML error: {0}")]
    Nvml(#[from] nvml_wrapper::error::NvmlError),

    /// I/O error reading a `/proc` file.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Failed to parse a `/proc` file.
    #[error("Parse error: {0}")]
    Parse(String),

    /// Platform not supported.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[error("GPU memory monitoring not supported on this platform")]
    UnsupportedPlatform,
}

/// Returns GPU memory currently used, in MiB.
///
/// - **dGPU (x86_64)**: Uses NVML to query device `gpu_id` (same data as `nvidia-smi --query-gpu=memory.used`).
/// - **Jetson (aarch64)**: Reads `/proc/meminfo` and returns `(MemTotal - MemAvailable) / 1024` (unified memory).
///
/// # Errors
///
/// Returns an error if NVML is unavailable (dGPU), `/proc/meminfo` cannot be read (Jetson),
/// or the platform is not supported.
///
/// # Example
///
/// ```rust,no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let used_mib = nvidia_gpu_utils::gpu_mem_used_mib(0)?;
/// println!("GPU memory used: {} MiB", used_mib);
/// # Ok(())
/// # }
/// ```
pub fn gpu_mem_used_mib(gpu_id: u32) -> Result<u64, GpuUtilsError> {
    #[cfg(target_arch = "x86_64")]
    {
        gpu_mem_used_mib_nvml(gpu_id)
    }

    #[cfg(target_arch = "aarch64")]
    {
        let _ = gpu_id;
        gpu_mem_used_mib_meminfo()
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = gpu_id;
        Err(GpuUtilsError::UnsupportedPlatform)
    }
}

/// Returns the current process RSS (Resident Set Size) in MiB.
///
/// Reads `VmRSS` from `/proc/self/status`.  Works on any Linux system
/// (dGPU, Jetson, CI) and measures **CPU** memory used by the calling
/// process.
///
/// # Errors
///
/// Returns an error if `/proc/self/status` cannot be read or parsed.
///
/// # Example
///
/// ```rust,no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rss = nvidia_gpu_utils::process_rss_mib()?;
/// println!("Process RSS: {} MiB", rss);
/// # Ok(())
/// # }
/// ```
pub fn process_rss_mib() -> Result<u64, GpuUtilsError> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| GpuUtilsError::Parse("VmRSS value not parseable".into()))?;
            return Ok(kb / 1024);
        }
    }
    Err(GpuUtilsError::Parse(
        "VmRSS not found in /proc/self/status".into(),
    ))
}

#[cfg(target_arch = "x86_64")]
fn gpu_mem_used_mib_nvml(gpu_id: u32) -> Result<u64, GpuUtilsError> {
    let nvml = nvml_wrapper::Nvml::init()?;
    let device = nvml.device_by_index(gpu_id)?;
    let info = device.memory_info()?;
    Ok(info.used / (1024 * 1024))
}

#[cfg(target_arch = "aarch64")]
fn gpu_mem_used_mib_meminfo() -> Result<u64, GpuUtilsError> {
    let content = std::fs::read_to_string("/proc/meminfo")?;
    let mut mem_total_kb: Option<u64> = None;
    let mut mem_available_kb: Option<u64> = None;

    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            mem_total_kb = parse_meminfo_line(line);
        } else if line.starts_with("MemAvailable:") {
            mem_available_kb = parse_meminfo_line(line);
        }
        if mem_total_kb.is_some() && mem_available_kb.is_some() {
            break;
        }
    }

    let total = mem_total_kb.ok_or_else(|| GpuUtilsError::Parse("MemTotal not found".into()))?;
    let available =
        mem_available_kb.ok_or_else(|| GpuUtilsError::Parse("MemAvailable not found".into()))?;
    let used_kb = total.saturating_sub(available);
    Ok(used_kb / 1024)
}

#[cfg(target_arch = "aarch64")]
fn parse_meminfo_line(line: &str) -> Option<u64> {
    line.split_whitespace().nth(1)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_mem_used_mib_returns_positive_on_gpu_host() {
        match gpu_mem_used_mib(0) {
            Ok(mib) => assert!(
                mib > 0,
                "Expected positive GPU memory on GPU host, got {}",
                mib
            ),
            Err(e) => {
                eprintln!("gpu_mem_used_mib(0) failed (expected in CI): {}", e);
            }
        }
    }

    #[test]
    fn test_process_rss_mib_returns_positive() {
        let rss = process_rss_mib().expect("process_rss_mib should work on Linux");
        assert!(rss > 0, "Expected positive RSS, got {}", rss);
    }
}
