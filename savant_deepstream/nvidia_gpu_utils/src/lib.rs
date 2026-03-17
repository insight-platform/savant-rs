//! Platform-aware GPU and process memory monitoring utilities.
//!
//! - **dGPU (x86_64)**: Uses NVML to query GPU device memory and NVENC capability.
//! - **Jetson (aarch64)**: Reads `/proc/meminfo` for unified memory (RAM used).
//! - **Process RSS**: Reads `/proc/self/status` `VmRSS` (works on any Linux).
//! - **Jetson detection**: Uses CUDA SM count + `/proc/meminfo` MemTotal to identify
//!   Jetson Orin/Xavier variants when running in a container (no device-tree access).
//! - **NVENC detection**: [`has_nvenc`] checks hardware encoder availability — returns
//!   `false` for Orin Nano and datacenter GPUs without NVENC (H100, A100, A30, etc.).

use std::io;
use thiserror::Error;

/// NVIDIA Jetson model identified by SM count and total memory.
///
/// Mapping is based on (SM count, MemTotal) since device-tree is often not
/// available inside containers. Orin family uses compute capability 8.7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JetsonModel {
    /// AGX Orin 64GB (16 SMs, ~64 GB unified).
    AgxOrin64GB,
    /// AGX Orin 32GB (12 SMs, ~32 GB unified).
    AgxOrin32GB,
    /// Orin NX 16GB (8 SMs, ~16 GB unified).
    OrinNx16GB,
    /// Orin NX 8GB (6 SMs, ~8 GB unified).
    OrinNx8GB,
    /// Orin Nano 8GB (8 SMs, ~8 GB unified).
    OrinNano8GB,
    /// Orin Nano 4GB (6 SMs, ~4 GB unified).
    OrinNano4GB,
    /// Xavier NX (6 SMs, Volta, ~8 GB).
    XavierNx,
    /// AGX Xavier (8 SMs, Volta, ~32 GB).
    AgxXavier,
    /// Jetson TX2 (2 SMs, Pascal).
    Tx2,
    /// Jetson Nano (1 SM, Maxwell).
    Nano,
    /// Jetson detected but model could not be determined.
    Unknown,
}

impl JetsonModel {
    /// Human-readable model name.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AgxOrin64GB => "AGX Orin 64GB",
            Self::AgxOrin32GB => "AGX Orin 32GB",
            Self::OrinNx16GB => "Orin NX 16GB",
            Self::OrinNx8GB => "Orin NX 8GB",
            Self::OrinNano8GB => "Orin Nano 8GB",
            Self::OrinNano4GB => "Orin Nano 4GB",
            Self::XavierNx => "Xavier NX",
            Self::AgxXavier => "AGX Xavier",
            Self::Tx2 => "TX2",
            Self::Nano => "Nano",
            Self::Unknown => "Unknown Jetson",
        }
    }

    /// Returns `true` if this is an Orin Nano (8GB or 4GB).
    #[must_use]
    pub fn is_orin_nano(&self) -> bool {
        matches!(self, Self::OrinNano8GB | Self::OrinNano4GB)
    }
}

impl std::fmt::Display for JetsonModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Error type for GPU utils operations.
#[derive(Debug, Error)]
pub enum GpuUtilsError {
    /// NVML initialization or query failed (dGPU only).
    #[cfg(target_arch = "x86_64")]
    #[error("NVML error: {0}")]
    Nvml(#[from] nvml_wrapper::error::NvmlError),

    /// CUDA driver API error (e.g. cuInit, cuDeviceGetAttribute).
    #[error("CUDA error: {0}")]
    Cuda(String),

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

/// Returns the Jetson model if running on a Jetson device, or `None` if not.
///
/// Uses CUDA SM count and `/proc/meminfo` MemTotal to identify the model.
/// Works inside containers where `/proc/device-tree` is typically not mounted.
/// Requires `uname -r` to contain "tegra" (Jetson kernel) and a working CUDA.
///
/// # Errors
///
/// Returns an error if CUDA initialization fails or `/proc/meminfo` cannot be read.
///
/// # Example
///
/// ```rust,no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// if let Ok(Some(model)) = nvidia_gpu_utils::jetson_model(0) {
///     println!("Jetson: {}", model);
///     if model.is_orin_nano() {
///         println!("Detected Orin Nano");
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub fn jetson_model(gpu_id: u32) -> Result<Option<JetsonModel>, GpuUtilsError> {
    if !is_jetson_kernel() {
        return Ok(None);
    }
    let (sm_count, cc_major, cc_minor) = cuda_device_attrs(gpu_id)?;
    let mem_total_mib = mem_total_mib()?;
    Ok(Some(map_to_jetson(
        sm_count,
        cc_major,
        cc_minor,
        mem_total_mib,
    )))
}

/// Returns `true` if the kernel is a Jetson (Tegra) kernel.
///
/// Reads `/proc/version` and checks for "tegra". The result is cached
/// in a [`OnceLock`](std::sync::OnceLock) so the file is read at most once.
#[must_use]
pub fn is_jetson_kernel() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::fs::read_to_string("/proc/version")
            .map(|s| s.contains("tegra"))
            .unwrap_or(false)
    })
}

/// Returns (SM count, compute capability major, compute capability minor).
fn cuda_device_attrs(gpu_id: u32) -> Result<(i32, i32, i32), GpuUtilsError> {
    // CUDA driver API attributes
    const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

    extern "C" {
        fn cuInit(flags: u32) -> u32;
        fn cuDeviceGet(device: *mut i32, ordinal: i32) -> u32;
        fn cuDeviceGetAttribute(pi: *mut i32, attrib: i32, dev: i32) -> u32;
    }

    let mut device: i32 = 0;
    // SAFETY: cuInit(0) is the standard CUDA driver API initialisation call.
    // It is safe to call multiple times (idempotent) and requires no
    // pre-conditions beyond a working CUDA installation.
    let err = unsafe { cuInit(0) };
    if err != 0 {
        return Err(GpuUtilsError::Cuda(format!("cuInit failed: {}", err)));
    }
    // SAFETY: cuInit succeeded, so the driver is initialised. `device` is a
    // valid mutable pointer to a stack-local i32, and `gpu_id` is the
    // caller-supplied ordinal (bounds-checked by the driver itself).
    let err = unsafe { cuDeviceGet(&mut device, gpu_id as i32) };
    if err != 0 {
        return Err(GpuUtilsError::Cuda(format!(
            "cuDeviceGet({}) failed: {}",
            gpu_id, err
        )));
    }
    let mut sm_count: i32 = 0;
    // SAFETY: `device` is a valid CUdevice handle returned by cuDeviceGet.
    // `sm_count` is a valid mutable pointer to a stack-local i32.
    let err = unsafe {
        cuDeviceGetAttribute(
            &mut sm_count,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            device,
        )
    };
    if err != 0 {
        return Err(GpuUtilsError::Cuda(format!(
            "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed: {}",
            err
        )));
    }
    let mut major: i32 = 0;
    // SAFETY: same as above — valid device handle and output pointer.
    let err = unsafe {
        cuDeviceGetAttribute(
            &mut major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        )
    };
    if err != 0 {
        return Err(GpuUtilsError::Cuda(format!(
            "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR) failed: {}",
            err
        )));
    }
    let mut minor: i32 = 0;
    // SAFETY: same as above — valid device handle and output pointer.
    let err = unsafe {
        cuDeviceGetAttribute(
            &mut minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        )
    };
    if err != 0 {
        return Err(GpuUtilsError::Cuda(format!(
            "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR) failed: {}",
            err
        )));
    }
    Ok((sm_count, major, minor))
}

/// Returns total system memory in MiB from `/proc/meminfo` MemTotal.
///
/// Useful for scripts that need to know total RAM (e.g. on Jetson for unified memory).
pub fn mem_total_mib() -> Result<u64, GpuUtilsError> {
    let content = std::fs::read_to_string("/proc/meminfo")?;
    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            let kb = parse_meminfo_line(line)
                .ok_or_else(|| GpuUtilsError::Parse("MemTotal not parseable".into()))?;
            return Ok(kb / 1024);
        }
    }
    Err(GpuUtilsError::Parse("MemTotal not found".into()))
}

fn parse_meminfo_line(line: &str) -> Option<u64> {
    line.split_whitespace().nth(1)?.parse().ok()
}

/// Maps (SM count, compute cap, MemTotal MiB) to Jetson model.
///
/// Orin: CC 8.7, SMs 16/12/8/6; Xavier: CC 7.2, SMs 8/6; TX2: CC 6.2, 2 SMs; Nano: CC 5.3, 1 SM.
pub(crate) fn map_to_jetson(
    sm_count: i32,
    cc_major: i32,
    cc_minor: i32,
    mem_total_mib: u64,
) -> JetsonModel {
    let is_orin = cc_major == 8 && cc_minor == 7;
    let is_xavier = cc_major == 7 && cc_minor == 2;
    let is_pascal = cc_major == 6;
    let is_maxwell = cc_major == 5;

    if is_orin {
        match sm_count {
            16 if mem_total_mib >= 60_000 => JetsonModel::AgxOrin64GB,
            12 if (30_000..60_000).contains(&mem_total_mib) => JetsonModel::AgxOrin32GB,
            8 if (14_000..20_000).contains(&mem_total_mib) => JetsonModel::OrinNx16GB,
            8 if (7_000..10_000).contains(&mem_total_mib) => JetsonModel::OrinNano8GB,
            6 if (7_000..10_000).contains(&mem_total_mib) => JetsonModel::OrinNx8GB,
            6 if (3_500..5_500).contains(&mem_total_mib) => JetsonModel::OrinNano4GB,
            _ => JetsonModel::Unknown,
        }
    } else if is_xavier {
        match sm_count {
            8 if (28_000..36_000).contains(&mem_total_mib) => JetsonModel::AgxXavier,
            6 if (7_000..10_000).contains(&mem_total_mib) => JetsonModel::XavierNx,
            _ => JetsonModel::Unknown,
        }
    } else if is_pascal && sm_count == 2 {
        JetsonModel::Tx2
    } else if is_maxwell && sm_count == 1 {
        JetsonModel::Nano
    } else {
        JetsonModel::Unknown
    }
}

/// Returns `true` if the GPU has NVENC hardware encoding support.
///
/// - **Jetson**: Orin Nano is the only Jetson without NVENC; all others have it.
///   `Unknown` models conservatively return `false`.
/// - **dGPU (x86_64)**: Uses NVML `encoder_capacity(H264)` — returns `false` for
///   datacenter GPUs without NVENC (H100, A100, A30, B200, B300, GB200, etc.).
///
/// # Errors
///
/// Returns an error if CUDA/NVML initialization fails.
///
/// # Example
///
/// ```rust,no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// if !nvidia_gpu_utils::has_nvenc(0)? {
///     eprintln!("This GPU does not support NVENC hardware encoding");
/// }
/// # Ok(())
/// # }
/// ```
pub fn has_nvenc(gpu_id: u32) -> Result<bool, GpuUtilsError> {
    // Jetson path: Orin Nano has no NVENC; Unknown is treated as no-NVENC.
    if let Some(model) = jetson_model(gpu_id)? {
        return Ok(!matches!(
            model,
            JetsonModel::OrinNano8GB | JetsonModel::OrinNano4GB | JetsonModel::Unknown
        ));
    }

    // dGPU path: query NVML encoder capacity.
    #[cfg(target_arch = "x86_64")]
    {
        return has_nvenc_nvml(gpu_id);
    }

    // Fallback: if not Jetson and not x86_64, assume available.
    #[cfg(not(target_arch = "x86_64"))]
    Ok(true)
}

#[cfg(target_arch = "x86_64")]
fn cached_nvml() -> Result<&'static nvml_wrapper::Nvml, GpuUtilsError> {
    use std::sync::OnceLock;
    static NVML: OnceLock<nvml_wrapper::Nvml> = OnceLock::new();
    if let Some(nvml) = NVML.get() {
        return Ok(nvml);
    }
    let nvml = nvml_wrapper::Nvml::init()?;
    let _ = NVML.set(nvml);
    Ok(NVML
        .get()
        .expect("OnceLock was just set or set by another thread"))
}

#[cfg(target_arch = "x86_64")]
fn has_nvenc_nvml(gpu_id: u32) -> Result<bool, GpuUtilsError> {
    use nvml_wrapper::enum_wrappers::device::EncoderType;

    let nvml = cached_nvml()?;
    let device = nvml.device_by_index(gpu_id)?;
    match device.encoder_capacity(EncoderType::H264) {
        Ok(cap) => Ok(cap > 0),
        Err(nvml_wrapper::error::NvmlError::NotSupported) => Ok(false),
        Err(e) => Err(GpuUtilsError::Nvml(e)),
    }
}

#[cfg(target_arch = "x86_64")]
fn gpu_mem_used_mib_nvml(gpu_id: u32) -> Result<u64, GpuUtilsError> {
    let nvml = cached_nvml()?;
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

    #[test]
    fn test_is_jetson_kernel_does_not_panic() {
        let _ = is_jetson_kernel();
    }

    #[test]
    fn test_jetson_model_does_not_panic() {
        let _ = jetson_model(0);
    }

    #[test]
    fn test_jetson_model_display() {
        assert_eq!(JetsonModel::OrinNano8GB.as_str(), "Orin Nano 8GB");
        assert!(JetsonModel::OrinNano8GB.is_orin_nano());
        assert!(JetsonModel::OrinNano4GB.is_orin_nano());
        assert!(!JetsonModel::OrinNx8GB.is_orin_nano());
    }

    #[test]
    fn test_has_nvenc_does_not_panic() {
        let _ = has_nvenc(0);
    }

    // ---- map_to_jetson exhaustive tests ----

    #[test]
    fn test_agx_orin_64gb() {
        assert_eq!(map_to_jetson(16, 8, 7, 62_000), JetsonModel::AgxOrin64GB);
        assert_eq!(map_to_jetson(16, 8, 7, 60_000), JetsonModel::AgxOrin64GB);
        assert_eq!(map_to_jetson(16, 8, 7, 65_000), JetsonModel::AgxOrin64GB);
    }

    #[test]
    fn test_agx_orin_32gb() {
        assert_eq!(map_to_jetson(12, 8, 7, 31_000), JetsonModel::AgxOrin32GB);
        assert_eq!(map_to_jetson(12, 8, 7, 30_000), JetsonModel::AgxOrin32GB);
        assert_eq!(map_to_jetson(12, 8, 7, 59_999), JetsonModel::AgxOrin32GB);
    }

    #[test]
    fn test_orin_nx_16gb() {
        assert_eq!(map_to_jetson(8, 8, 7, 15_000), JetsonModel::OrinNx16GB);
        assert_eq!(map_to_jetson(8, 8, 7, 14_000), JetsonModel::OrinNx16GB);
        assert_eq!(map_to_jetson(8, 8, 7, 19_999), JetsonModel::OrinNx16GB);
    }

    #[test]
    fn test_orin_nano_8gb() {
        assert_eq!(map_to_jetson(8, 8, 7, 8_000), JetsonModel::OrinNano8GB);
        assert_eq!(map_to_jetson(8, 8, 7, 7_000), JetsonModel::OrinNano8GB);
        assert_eq!(map_to_jetson(8, 8, 7, 9_999), JetsonModel::OrinNano8GB);
    }

    #[test]
    fn test_orin_nx_8gb() {
        assert_eq!(map_to_jetson(6, 8, 7, 8_000), JetsonModel::OrinNx8GB);
        assert_eq!(map_to_jetson(6, 8, 7, 7_000), JetsonModel::OrinNx8GB);
        assert_eq!(map_to_jetson(6, 8, 7, 9_999), JetsonModel::OrinNx8GB);
    }

    #[test]
    fn test_orin_nano_4gb() {
        assert_eq!(map_to_jetson(6, 8, 7, 4_000), JetsonModel::OrinNano4GB);
        assert_eq!(map_to_jetson(6, 8, 7, 3_500), JetsonModel::OrinNano4GB);
        assert_eq!(map_to_jetson(6, 8, 7, 5_499), JetsonModel::OrinNano4GB);
    }

    #[test]
    fn test_agx_xavier() {
        assert_eq!(map_to_jetson(8, 7, 2, 32_000), JetsonModel::AgxXavier);
        assert_eq!(map_to_jetson(8, 7, 2, 28_000), JetsonModel::AgxXavier);
        assert_eq!(map_to_jetson(8, 7, 2, 35_999), JetsonModel::AgxXavier);
    }

    #[test]
    fn test_xavier_nx() {
        assert_eq!(map_to_jetson(6, 7, 2, 8_000), JetsonModel::XavierNx);
        assert_eq!(map_to_jetson(6, 7, 2, 7_000), JetsonModel::XavierNx);
        assert_eq!(map_to_jetson(6, 7, 2, 9_999), JetsonModel::XavierNx);
    }

    #[test]
    fn test_tx2() {
        assert_eq!(map_to_jetson(2, 6, 2, 8_000), JetsonModel::Tx2);
        assert_eq!(map_to_jetson(2, 6, 1, 4_000), JetsonModel::Tx2);
    }

    #[test]
    fn test_nano() {
        assert_eq!(map_to_jetson(1, 5, 3, 4_000), JetsonModel::Nano);
        assert_eq!(map_to_jetson(1, 5, 0, 2_000), JetsonModel::Nano);
    }

    #[test]
    fn test_orin_memory_gap_between_ranges() {
        // 8 SMs, Orin CC, memory between NX 16GB and Nano 8GB ranges (10_000..14_000)
        assert_eq!(map_to_jetson(8, 8, 7, 12_000), JetsonModel::Unknown);
        // 6 SMs, Orin CC, memory between Nano 4GB and NX 8GB ranges (5_500..7_000)
        assert_eq!(map_to_jetson(6, 8, 7, 6_000), JetsonModel::Unknown);
    }

    #[test]
    fn test_orin_memory_below_lowest_range() {
        assert_eq!(map_to_jetson(6, 8, 7, 2_000), JetsonModel::Unknown);
        assert_eq!(map_to_jetson(8, 8, 7, 5_000), JetsonModel::Unknown);
    }

    #[test]
    fn test_xavier_memory_out_of_range() {
        assert_eq!(map_to_jetson(8, 7, 2, 50_000), JetsonModel::Unknown);
        assert_eq!(map_to_jetson(6, 7, 2, 3_000), JetsonModel::Unknown);
    }

    #[test]
    fn test_orin_sm_count_not_recognized() {
        assert_eq!(map_to_jetson(4, 8, 7, 16_000), JetsonModel::Unknown);
        assert_eq!(map_to_jetson(10, 8, 7, 32_000), JetsonModel::Unknown);
    }

    #[test]
    fn test_cc_mismatch_right_sm_wrong_cc() {
        // 16 SMs but not Orin CC — should fall through to Unknown
        assert_eq!(map_to_jetson(16, 7, 0, 62_000), JetsonModel::Unknown);
        assert_eq!(map_to_jetson(16, 9, 0, 62_000), JetsonModel::Unknown);
        // 8 SMs with Xavier CC but memory in Orin NX range
        assert_eq!(map_to_jetson(8, 7, 2, 15_000), JetsonModel::Unknown);
    }

    #[test]
    fn test_zero_sm_count() {
        assert_eq!(map_to_jetson(0, 8, 7, 32_000), JetsonModel::Unknown);
        assert_eq!(map_to_jetson(0, 7, 2, 32_000), JetsonModel::Unknown);
        assert_eq!(map_to_jetson(0, 0, 0, 0), JetsonModel::Unknown);
    }

    #[test]
    fn test_agx_orin_64gb_boundary_below_threshold() {
        // Just below the 60_000 threshold
        assert_eq!(map_to_jetson(16, 8, 7, 59_999), JetsonModel::Unknown);
    }

    #[test]
    fn test_agx_orin_32gb_boundary_at_edges() {
        assert_eq!(map_to_jetson(12, 8, 7, 29_999), JetsonModel::Unknown);
        assert_eq!(map_to_jetson(12, 8, 7, 60_000), JetsonModel::Unknown);
    }
}
