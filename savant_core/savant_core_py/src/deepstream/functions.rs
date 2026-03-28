//! Standalone `#[pyfunction]` items for the `savant_rs.deepstream` module.

#[cfg(debug_assertions)]
use super::buffer::PySharedBuffer;
use super::buffer::{extract_buf_ptr, with_mut_buffer_ref};
use super::enums::{from_rust_id_kind, PySavantIdMetaKind};
use deepstream_buffers::transform;
use gstreamer as gst;
use pyo3::prelude::*;
use savant_gstreamer::id_meta::SavantIdMeta;

/// Set numFilled on a batched NvBufSurface GstBuffer.
///
/// Args:
///     buf (SharedBuffer | int): Buffer containing a batched NvBufSurface.
///     count (int): Number of filled slots.
#[pyfunction]
#[pyo3(name = "set_num_filled")]
pub fn py_set_num_filled(buf: &Bound<'_, PyAny>, count: u32) -> PyResult<()> {
    with_mut_buffer_ref(buf, |buf_ref| {
        let surf_ptr = unsafe {
            transform::extract_nvbufsurface(buf_ref)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        };
        unsafe { (*surf_ptr).numFilled = count };
        Ok(())
    })
}

/// Initialize CUDA context for the given GPU device.
///
/// Args:
///     gpu_id (int): GPU device ID (default 0).
#[pyfunction]
#[pyo3(name = "init_cuda", signature = (gpu_id=0))]
pub fn py_init_cuda(gpu_id: u32) -> PyResult<()> {
    deepstream_buffers::cuda_init(gpu_id)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Returns GPU memory currently used, in MiB.
///
/// - dGPU (x86_64): Uses NVML to query device ``gpu_id``.
/// - Jetson (aarch64): Reads /proc/meminfo (unified memory).
///
/// Args:
///     gpu_id (int): GPU device ID (default 0).
///
/// Returns:
///     int: GPU memory used in MiB.
///
/// Raises:
///     RuntimeError: If NVML or /proc/meminfo is unavailable.
#[pyfunction]
#[pyo3(name = "gpu_mem_used_mib", signature = (gpu_id=0))]
pub fn py_gpu_mem_used_mib(gpu_id: u32) -> PyResult<u64> {
    nvidia_gpu_utils::gpu_mem_used_mib(gpu_id)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Returns the Jetson model name if running on a Jetson device, or ``None`` if not.
///
/// Uses CUDA SM count and /proc/meminfo MemTotal to identify the model.
/// Works inside containers where /proc/device-tree is typically not mounted.
/// Requires ``uname -r`` to contain "tegra" and a working CUDA.
///
/// Args:
///     gpu_id (int): GPU device ID (default 0).
///
/// Returns:
///     str | None: Model name (e.g. "Orin Nano 8GB") or None if not Jetson.
///
/// Raises:
///     RuntimeError: If CUDA or /proc/meminfo is unavailable.
#[pyfunction]
#[pyo3(name = "jetson_model", signature = (gpu_id=0))]
pub fn py_jetson_model(gpu_id: u32) -> PyResult<Option<String>> {
    nvidia_gpu_utils::jetson_model(gpu_id)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        .map(|opt| opt.map(|m| m.to_string()))
}

/// Returns ``True`` if the kernel is a Jetson (Tegra) kernel.
///
/// Checks ``uname -r`` for the "tegra" suffix.
#[pyfunction]
#[pyo3(name = "is_jetson_kernel")]
pub fn py_is_jetson_kernel() -> bool {
    nvidia_gpu_utils::is_jetson_kernel()
}

/// Returns the GPU architecture family name (x86_64 dGPU only, via NVML).
///
/// Returns a lowercase architecture name such as ``"ampere"``, ``"ada"``,
/// ``"hopper"``, ``"turing"``, etc.  Returns ``None`` on Jetson/aarch64.
///
/// Args:
///     gpu_id (int): GPU device ID (default 0).
///
/// Returns:
///     str | None: Architecture name or None if not on x86_64.
///
/// Raises:
///     RuntimeError: If NVML initialization fails.
#[pyfunction]
#[pyo3(name = "gpu_architecture", signature = (gpu_id=0))]
pub fn py_gpu_architecture(gpu_id: u32) -> PyResult<Option<String>> {
    nvidia_gpu_utils::gpu_architecture(gpu_id)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Returns a directory-safe platform tag for TensorRT engine caching.
///
/// - **Jetson**: Jetson model name (e.g. ``"agx_orin_64gb"``, ``"orin_nano_8gb"``).
/// - **dGPU (x86_64)**: GPU architecture family (e.g. ``"ampere"``, ``"ada"``).
/// - **Unknown**: ``"unknown"`` if the platform cannot be determined.
///
/// Args:
///     gpu_id (int): GPU device ID (default 0).
///
/// Returns:
///     str: Platform tag string.
///
/// Raises:
///     RuntimeError: If CUDA/NVML initialization fails.
#[pyfunction]
#[pyo3(name = "gpu_platform_tag", signature = (gpu_id=0))]
pub fn py_gpu_platform_tag(gpu_id: u32) -> PyResult<String> {
    nvidia_gpu_utils::gpu_platform_tag(gpu_id)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Returns ``True`` if the GPU has NVENC hardware encoding support.
///
/// - **Jetson**: Orin Nano is the only Jetson without NVENC; all others have it.
///   ``Unknown`` models conservatively return ``False``.
/// - **dGPU (x86_64)**: Uses NVML ``encoder_capacity(H264)`` — returns ``False``
///   for datacenter GPUs without NVENC (H100, A100, A30, etc.).
///
/// Args:
///     gpu_id (int): GPU device ID (default 0).
///
/// Returns:
///     bool: ``True`` if NVENC is available.
///
/// Raises:
///     RuntimeError: If CUDA/NVML initialization fails.
#[pyfunction]
#[pyo3(name = "has_nvenc", signature = (gpu_id=0))]
pub fn py_has_nvenc(gpu_id: u32) -> PyResult<bool> {
    nvidia_gpu_utils::has_nvenc(gpu_id)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Read ``SavantIdMeta`` from a GStreamer buffer.
///
/// Returns:
///     list[tuple[str, int]]: Meta entries, e.g. ``[("frame", 42)]``.
#[pyfunction]
#[pyo3(name = "get_savant_id_meta")]
pub fn py_get_savant_id_meta(buf: &Bound<'_, PyAny>) -> PyResult<Vec<(PySavantIdMetaKind, i64)>> {
    let buf_ptr = extract_buf_ptr(buf)?;
    unsafe {
        let buf_ref = gst::BufferRef::from_ptr(buf_ptr as *const gst::ffi::GstBuffer);
        match buf_ref.meta::<SavantIdMeta>() {
            Some(meta) => Ok(meta.ids().iter().map(from_rust_id_kind).collect()),
            None => Ok(vec![]),
        }
    }
}

/// Extract NvBufSurface descriptor fields from an existing GstBuffer.
///
/// Returns:
///     tuple[int, int, int, int]: ``(data_ptr, pitch, width, height)``
#[pyfunction]
#[pyo3(name = "get_nvbufsurface_info")]
pub fn py_get_nvbufsurface_info(buf: &Bound<'_, PyAny>) -> PyResult<(usize, u32, u32, u32)> {
    let buf_ptr = extract_buf_ptr(buf)?;
    unsafe {
        let buf_ref = gst::BufferRef::from_ptr(buf_ptr as *const gst::ffi::GstBuffer);
        let surf_ptr = transform::extract_nvbufsurface(buf_ref)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let surf = &*surf_ptr;
        let params = &*surf.surfaceList;
        Ok((
            params.dataPtr as usize,
            params.pitch as u32,
            params.width,
            params.height,
        ))
    }
}

/// Debug-only mock consumer that simulates Rust-side buffer deconstruction.
///
/// Calls ``take_inner()`` followed by ``into_buffer()`` on the
/// ``SharedBuffer``, verifying the full consumption lifecycle.
/// Fails if the buffer has outstanding `Arc` references (e.g. a live
/// ``SurfaceView``).
///
/// Available only in debug builds.
#[cfg(debug_assertions)]
#[pyfunction]
#[pyo3(name = "_test_consume_shared_buffer")]
pub fn py_test_consume_shared_buffer(buf: &Bound<'_, PyAny>) -> PyResult<()> {
    let mut sb = buf.extract::<PyRefMut<'_, PySharedBuffer>>()?;
    let shared = sb.take_inner()?;
    shared.into_buffer().map_err(|returned| {
        sb.restore(returned);
        pyo3::exceptions::PyRuntimeError::new_err(
            "SharedBuffer has outstanding references; drop all SurfaceViews before consuming",
        )
    })?;
    Ok(())
}
