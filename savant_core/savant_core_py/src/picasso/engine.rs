use super::callbacks::PyCallbacks;
use super::error::to_py_err;
use super::spec::general::PyGeneralSpec;
use super::spec::source::PySourceSpec;
use crate::deepstream::{PyGstBuffer, PyRect, PySurfaceView};
use glib::translate::from_glib_none;
use picasso::prelude::PicassoEngine;
use pyo3::prelude::*;

/// Extracts the GPU device ID from a CuPy array or PyTorch CUDA tensor.
///
/// CuPy exposes `obj.device.id`; PyTorch exposes `obj.device.index`.
/// Returns 0 if the device cannot be determined.
fn extract_cuda_gpu_id(obj: &Bound<'_, PyAny>) -> u32 {
    if let Ok(device) = obj.getattr("device") {
        // CuPy: device.id
        if let Ok(id) = device.getattr("id").and_then(|v| v.extract::<u32>()) {
            return id;
        }
        // PyTorch: device.index
        if let Ok(idx) = device.getattr("index") {
            if let Ok(id) = idx.extract::<u32>() {
                return id;
            }
        }
    }
    0
}

/// The main entry point for the Picasso frame-processing pipeline.
///
/// Manages per-source worker threads, a watchdog for idle-source eviction,
/// and dispatches frames to the appropriate worker.
#[pyclass(name = "PicassoEngine", module = "savant_rs.picasso")]
pub struct PyPicassoEngine {
    inner: Option<PicassoEngine>,
}

#[pymethods]
impl PyPicassoEngine {
    /// Create a new engine with the given global defaults and callbacks.
    ///
    /// Spawns the watchdog thread immediately.
    #[new]
    fn new(py: Python<'_>, general: &PyGeneralSpec, callbacks: &PyCallbacks) -> Self {
        let engine = PicassoEngine::new(general.to_rust(), callbacks.to_rust(py));
        Self {
            inner: Some(engine),
        }
    }

    /// Set or replace the processing spec for a specific source.
    fn set_source_spec(
        &self,
        py: Python<'_>,
        source_id: &str,
        spec: &PySourceSpec,
    ) -> PyResult<()> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine is shut down"))?;
        let rust_spec = spec.to_rust();
        py.detach(|| {
            engine
                .set_source_spec(source_id, rust_spec)
                .map_err(to_py_err)
        })
    }

    /// Remove the spec for a source.  The worker will be shut down.
    fn remove_source_spec(&self, py: Python<'_>, source_id: &str) -> PyResult<()> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine is shut down"))?;
        py.detach(|| engine.remove_source_spec(source_id));
        Ok(())
    }

    /// Submit a video frame for processing.
    ///
    /// Accepts one of:
    ///
    /// - ``SurfaceView`` — the preferred input type.
    /// - Any object with ``__cuda_array_interface__`` (CuPy array,
    ///   PyTorch CUDA tensor) — automatically wrapped in a ``SurfaceView``.
    /// - ``GstBuffer`` or raw ``int`` pointer (legacy API) — the buffer is
    ///   wrapped with ``SurfaceView.wrap`` (no surface parameter extraction).
    ///
    /// Args:
    ///     source_id (str): Source identifier.
    ///     frame (VideoFrame): Frame metadata.
    ///     buf: Surface data (``SurfaceView``, ``__cuda_array_interface__``
    ///         object, ``GstBuffer``, or ``int``).
    ///     src_rect (Rect | None): Optional per-frame source crop.
    ///
    /// Raises:
    ///     RuntimeError: If the engine is shut down.
    ///     TypeError: If ``buf`` is not a supported type.
    ///     ValueError: If ``buf`` is null.
    #[pyo3(signature = (source_id, frame, buf, src_rect=None))]
    fn send_frame(
        &self,
        py: Python<'_>,
        source_id: &str,
        frame: &crate::primitives::frame::VideoFrame,
        buf: &Bound<'_, PyAny>,
        src_rect: Option<&PyRect>,
    ) -> PyResult<()> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine is shut down"))?;

        let frame_proxy = frame.0.clone();
        let src_rect_rust = src_rect.map(|r| r.into_rust());

        // Dispatch: PySurfaceView → __cuda_array_interface__ → GstBuffer/int
        if let Ok(mut sv) = buf.extract::<PyRefMut<'_, PySurfaceView>>() {
            let view = sv.take()?;
            py.detach(|| {
                engine
                    .send_frame(source_id, frame_proxy, view, src_rect_rust)
                    .map_err(to_py_err)
            })
        } else if buf.hasattr("__cuda_array_interface__")? {
            let gpu_id = extract_cuda_gpu_id(buf);
            let mut py_sv = PySurfaceView::from_cuda_iface(py, buf.clone(), gpu_id)?;
            let view = py_sv.take()?;
            py.detach(|| {
                engine
                    .send_frame(source_id, frame_proxy, view, src_rect_rust)
                    .map_err(to_py_err)
            })
        } else {
            let is_guard = buf.extract::<PyRef<'_, PyGstBuffer>>().is_ok();
            let buf_ptr = crate::deepstream::extract_buf_ptr(buf)?;
            py.detach(|| {
                let _ = gstreamer::init();
                let gst_buf = unsafe {
                    if is_guard {
                        from_glib_none(buf_ptr as *const gstreamer::ffi::GstBuffer)
                    } else {
                        gstreamer::Buffer::from_glib_full(buf_ptr as *mut gstreamer::ffi::GstBuffer)
                    }
                };
                let view = deepstream_nvbufsurface::SurfaceView::wrap(gst_buf);
                engine
                    .send_frame(source_id, frame_proxy, view, src_rect_rust)
                    .map_err(to_py_err)
            })
        }
    }

    /// Send an end-of-stream signal to a specific source.
    fn send_eos(&self, py: Python<'_>, source_id: &str) -> PyResult<()> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("engine is shut down"))?;
        py.detach(|| engine.send_eos(source_id).map_err(to_py_err))
    }

    /// Gracefully shut down all workers and the watchdog.
    ///
    /// Releases the GIL while joining worker threads so that any pending
    /// Python callbacks (on_encoded_frame, on_render, etc.) can acquire
    /// the GIL and complete without deadlocking.
    fn shutdown(&mut self, py: Python<'_>) {
        if let Some(mut engine) = self.inner.take() {
            py.detach(|| engine.shutdown());
        }
    }

    fn __repr__(&self) -> String {
        if self.inner.is_some() {
            "PicassoEngine(running)".to_string()
        } else {
            "PicassoEngine(shut_down)".to_string()
        }
    }
}
