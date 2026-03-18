//! PyO3 bindings for the `deepstream_buffers` crate.
//!
//! These types are registered in the `savant_rs.deepstream` Python submodule
//! by `savant_python` when the `deepstream` feature is enabled.

pub mod buffer;
pub mod config;
pub mod enums;
pub mod functions;
pub mod generators;
pub mod skia;
pub mod surface_view;

// Re-export types used by other crate modules (picasso, nvinfer).
pub use buffer::PySharedBuffer;
pub(crate) use buffer::{extract_gst_buffer, extract_shared_buffer};
pub use config::{PyDstPadding, PyRect, PyTransformConfig};
pub use enums::{PyMemType, PySavantIdMetaKind, PyVideoFormat};
pub use surface_view::PySurfaceView;

use gstreamer as gst;
use pyo3::prelude::*;

/// Register the DeepStream Python classes on the given module.
///
/// Initialises GStreamer once at import time so that individual functions
/// do not have to repeat the call.
pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    gst::init().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    m.add_class::<enums::PyPadding>()?;
    m.add_class::<enums::PyInterpolation>()?;
    m.add_class::<enums::PyComputeMode>()?;
    m.add_class::<enums::PyVideoFormat>()?;
    m.add_class::<enums::PyMemType>()?;
    m.add_class::<enums::PySavantIdMetaKind>()?;
    m.add_class::<config::PyRect>()?;
    m.add_class::<config::PyDstPadding>()?;
    m.add_class::<config::PyTransformConfig>()?;
    m.add_class::<buffer::PySharedBuffer>()?;
    m.add_class::<surface_view::PySurfaceView>()?;
    m.add_class::<generators::PyBufferGenerator>()?;
    m.add_class::<generators::PyUniformBatchGenerator>()?;
    m.add_class::<generators::PySurfaceBatch>()?;
    m.add_class::<generators::PyNonUniformBatch>()?;
    m.add_function(pyo3::wrap_pyfunction!(functions::py_set_num_filled, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(functions::py_init_cuda, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(functions::py_gpu_mem_used_mib, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(functions::py_jetson_model, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(functions::py_is_jetson_kernel, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(functions::py_has_nvenc, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(functions::py_get_savant_id_meta, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        functions::py_get_nvbufsurface_info,
        m
    )?)?;
    #[cfg(debug_assertions)]
    m.add_function(pyo3::wrap_pyfunction!(
        functions::py_test_consume_shared_buffer,
        m
    )?)?;
    m.add_class::<skia::PySkiaContext>()?;
    Ok(())
}
