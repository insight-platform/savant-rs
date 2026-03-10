use picasso::prelude::EncodedOutput;
use pyo3::prelude::*;

/// Output produced by the encoding pipeline or bypass mode.
///
/// Use the ``is_video_frame`` / ``is_eos`` properties to discriminate,
/// then ``as_video_frame()`` or ``as_eos()`` to extract the payload.
#[pyclass(name = "EncodedOutput", module = "savant_rs.picasso")]
pub struct PyEncodedOutput {
    inner: EncodedOutput,
}

#[pymethods]
impl PyEncodedOutput {
    /// `True` when this output carries a video frame.
    #[getter]
    fn is_video_frame(&self) -> bool {
        matches!(self.inner, EncodedOutput::VideoFrame(_))
    }

    /// `True` when this output is an end-of-stream signal.
    #[getter]
    fn is_eos(&self) -> bool {
        matches!(self.inner, EncodedOutput::EndOfStream(_))
    }

    /// Extract the ``VideoFrame``.
    ///
    /// Raises:
    ///     RuntimeError: If this is an EOS output, not a video frame.
    fn as_video_frame(&self) -> PyResult<crate::primitives::frame::VideoFrame> {
        match &self.inner {
            EncodedOutput::VideoFrame(f) => Ok(crate::primitives::frame::VideoFrame(f.clone())),
            EncodedOutput::EndOfStream(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "EncodedOutput is EndOfStream, not VideoFrame",
            )),
        }
    }

    /// Extract the ``EndOfStream`` signal.
    ///
    /// Raises:
    ///     RuntimeError: If this is a video-frame output, not EOS.
    fn as_eos(&self) -> PyResult<crate::primitives::eos::EndOfStream> {
        match &self.inner {
            EncodedOutput::EndOfStream(e) => Ok(crate::primitives::eos::EndOfStream::new(
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
