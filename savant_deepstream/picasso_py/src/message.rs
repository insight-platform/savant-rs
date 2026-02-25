use picasso::prelude::{BypassOutput, EncodedOutput};
use pyo3::prelude::*;

/// Output produced by the encoding pipeline.
///
/// Use the ``is_video_frame`` / ``is_eos`` properties to discriminate,
/// then ``as_video_frame()`` or ``as_eos()`` to extract the payload.
#[pyclass(name = "EncodedOutput", module = "picasso._native")]
pub struct PyEncodedOutput {
    inner: EncodedOutput,
}

#[pymethods]
impl PyEncodedOutput {
    /// `True` when this output carries an encoded video frame.
    #[getter]
    fn is_video_frame(&self) -> bool {
        matches!(self.inner, EncodedOutput::VideoFrame(_))
    }

    /// `True` when this output is an end-of-stream signal.
    #[getter]
    fn is_eos(&self) -> bool {
        matches!(self.inner, EncodedOutput::EndOfStream(_))
    }

    /// Extract the encoded ``VideoFrame``.
    ///
    /// Raises:
    ///     RuntimeError: If this is an EOS output, not a video frame.
    fn as_video_frame(&self) -> PyResult<savant_core_py::primitives::frame::VideoFrame> {
        match &self.inner {
            EncodedOutput::VideoFrame(f) => {
                Ok(savant_core_py::primitives::frame::VideoFrame(f.clone()))
            }
            EncodedOutput::EndOfStream(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "EncodedOutput is EndOfStream, not VideoFrame",
            )),
        }
    }

    /// Extract the ``EndOfStream`` signal.
    ///
    /// Raises:
    ///     RuntimeError: If this is a video-frame output, not EOS.
    fn as_eos(&self) -> PyResult<savant_core_py::primitives::eos::EndOfStream> {
        match &self.inner {
            EncodedOutput::EndOfStream(e) => Ok(savant_core_py::primitives::eos::EndOfStream::new(
                e.source_id.clone(),
            )),
            EncodedOutput::VideoFrame(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "EncodedOutput is VideoFrame, not EndOfStream",
            )),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            EncodedOutput::VideoFrame(_) => "EncodedOutput.VideoFrame(...)".to_string(),
            EncodedOutput::EndOfStream(e) => {
                format!("EncodedOutput.EndOfStream(source_id={:?})", e.source_id)
            }
        }
    }
}

impl PyEncodedOutput {
    pub(crate) fn from_rust(output: EncodedOutput) -> Self {
        Self { inner: output }
    }
}

/// Output for bypass mode — frame with bboxes transformed back to initial
/// coordinates.
#[pyclass(name = "BypassOutput", module = "picasso._native")]
pub struct PyBypassOutput {
    source_id: String,
    frame: savant_core::primitives::frame::VideoFrameProxy,
    buffer_ptr: usize,
}

#[pymethods]
impl PyBypassOutput {
    /// Source identifier.
    #[getter]
    fn source_id(&self) -> &str {
        &self.source_id
    }

    /// The ``VideoFrame`` with bboxes in the initial coordinate space.
    #[getter]
    fn frame(&self) -> savant_core_py::primitives::frame::VideoFrame {
        savant_core_py::primitives::frame::VideoFrame(self.frame.clone())
    }

    /// Raw pointer to the ``GstBuffer`` (for PyGObject interop).
    #[getter]
    fn buffer_ptr(&self) -> usize {
        self.buffer_ptr
    }

    fn __repr__(&self) -> String {
        format!(
            "BypassOutput(source_id={:?}, buffer_ptr=0x{:x})",
            self.source_id, self.buffer_ptr
        )
    }
}

impl PyBypassOutput {
    pub(crate) fn from_rust(output: BypassOutput) -> Self {
        use glib::translate::IntoGlibPtr;
        // SAFETY: we transfer ownership of the GstBuffer to Python.
        // The caller (via PyGObject) is expected to manage the refcount.
        let buffer_ptr = unsafe { output.buffer.into_glib_ptr() } as usize;
        Self {
            source_id: output.source_id,
            frame: output.frame,
            buffer_ptr,
        }
    }
}
