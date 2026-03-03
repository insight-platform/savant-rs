use super::callbacks::PyCallbacks;
use super::error::to_py_err;
use super::spec::general::PyGeneralSpec;
use super::spec::source::PySourceSpec;
use crate::deepstream::{PyGstBuffer, PyRect};
use glib::translate::from_glib_none;
use picasso::prelude::PicassoEngine;
use pyo3::prelude::*;

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
    /// Args:
    ///     source_id (str): Source identifier.
    ///     frame (VideoFrame): Frame metadata.
    ///     buf (GstBuffer | int): Buffer — either a ``GstBuffer`` RAII guard
    ///         or a raw ``GstBuffer*`` pointer as ``int``.
    ///     src_rect (Rect | None): Optional per-frame source crop.
    ///
    /// Raises:
    ///     RuntimeError: If the engine is shut down.
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

        let is_guard = buf.extract::<PyRef<'_, PyGstBuffer>>().is_ok();
        let buf_ptr = crate::deepstream::extract_buf_ptr(buf)?;

        let frame_proxy = frame.0.clone();
        let src_rect_rust = src_rect.map(|r| r.into_rust());
        py.detach(|| {
            let _ = gstreamer::init();
            let gst_buf = unsafe {
                if is_guard {
                    // GstBuffer guard keeps its own ref; borrow by adding a ref.
                    from_glib_none(buf_ptr as *const gstreamer::ffi::GstBuffer)
                } else {
                    // Raw int: caller transfers ownership (legacy API).
                    gstreamer::Buffer::from_glib_full(buf_ptr as *mut gstreamer::ffi::GstBuffer)
                }
            };
            engine
                .send_frame(source_id, frame_proxy, gst_buf, src_rect_rust)
                .map_err(to_py_err)
        })
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
