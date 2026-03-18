//! `SkiaContext` — GPU-accelerated Skia rendering context backed by CUDA-GL interop.

use super::buffer::{extract_shared_buffer, with_mut_buffer_ref};
use super::config::PyTransformConfig;
use super::surface_view::PySurfaceView;
use pyo3::prelude::*;

/// GPU-accelerated Skia rendering context backed by CUDA-GL interop.
#[pyclass(name = "SkiaContext", module = "savant_rs.deepstream", unsendable)]
pub struct PySkiaContext {
    inner: deepstream_buffers::SkiaRenderer,
}

#[pymethods]
impl PySkiaContext {
    #[new]
    #[pyo3(signature = (width, height, gpu_id=0))]
    fn new(width: u32, height: u32, gpu_id: u32) -> PyResult<Self> {
        let inner = deepstream_buffers::SkiaRenderer::new(width, height, gpu_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create from an existing NvBufSurface buffer or ``SurfaceView``.
    ///
    /// Accepts a ``SurfaceView`` (preferred) or a ``SharedBuffer``
    /// / raw ``int`` pointer.
    #[staticmethod]
    #[pyo3(signature = (buf, gpu_id=0))]
    fn from_nvbuf(buf: &Bound<'_, PyAny>, gpu_id: u32) -> PyResult<Self> {
        if let Ok(sv) = buf.extract::<PyRef<'_, PySurfaceView>>() {
            let view = sv.inner_ref()?;
            let inner = unsafe {
                deepstream_buffers::SkiaRenderer::from_nvbuf(
                    view.width(),
                    view.height(),
                    gpu_id,
                    view.data_ptr() as *const std::ffi::c_void,
                    view.pitch() as usize,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            };
            return Ok(Self { inner });
        }
        let shared = extract_shared_buffer(buf)?;
        let view = deepstream_buffers::SurfaceView::from_buffer(&shared, 0)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let inner = unsafe {
            deepstream_buffers::SkiaRenderer::from_nvbuf(
                view.width(),
                view.height(),
                gpu_id,
                view.data_ptr() as *const std::ffi::c_void,
                view.pitch() as usize,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        };
        Ok(Self { inner })
    }

    #[getter]
    fn fbo_id(&self) -> u32 {
        self.inner.fbo_id()
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    #[pyo3(signature = (buf, config=None))]
    fn render_to_nvbuf(
        &mut self,
        buf: &Bound<'_, PyAny>,
        config: Option<&PyTransformConfig>,
    ) -> PyResult<()> {
        let rust_config = config.map(|c| c.to_rust());
        with_mut_buffer_ref(buf, |buf_ref| {
            self.inner
                .render_to_nvbuf(buf_ref, rust_config.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}
