use picasso::prelude::OutputMessage;
use pyo3::prelude::*;

/// Output produced by the encoding pipeline or bypass mode.
///
/// Use the ``is_video_frame`` / ``is_eos`` properties to discriminate,
/// then ``as_video_frame()`` or ``as_eos()`` to extract the payload.
#[pyclass(name = "OutputMessage", module = "savant_rs.picasso")]
pub struct PyOutputMessage {
    inner: OutputMessage,
}

#[pymethods]
impl PyOutputMessage {
    /// `True` when this output carries a video frame.
    #[getter]
    fn is_video_frame(&self) -> bool {
        matches!(self.inner, OutputMessage::VideoFrame(_))
    }

    /// `True` when this output is an end-of-stream signal.
    #[getter]
    fn is_eos(&self) -> bool {
        matches!(self.inner, OutputMessage::EndOfStream(_))
    }

    /// Extract the ``VideoFrame``.
    ///
    /// Raises:
    ///     RuntimeError: If this is an EOS output, not a video frame.
    fn as_video_frame(&self) -> PyResult<crate::primitives::frame::VideoFrame> {
        match &self.inner {
            OutputMessage::VideoFrame(f) => Ok(crate::primitives::frame::VideoFrame(f.clone())),
            OutputMessage::EndOfStream(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "OutputMessage is EndOfStream, not VideoFrame",
            )),
        }
    }

    /// Extract the ``EndOfStream`` signal.
    ///
    /// Raises:
    ///     RuntimeError: If this is a video-frame output, not EOS.
    fn as_eos(&self) -> PyResult<crate::primitives::eos::EndOfStream> {
        match &self.inner {
            OutputMessage::EndOfStream(e) => Ok(crate::primitives::eos::EndOfStream::new(
                e.source_id.clone(),
            )),
            OutputMessage::VideoFrame(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "OutputMessage is VideoFrame, not EndOfStream",
            )),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            OutputMessage::VideoFrame(_) => "OutputMessage.VideoFrame(...)".to_string(),
            OutputMessage::EndOfStream(e) => {
                format!("OutputMessage.EndOfStream(source_id={:?})", e.source_id)
            }
        }
    }
}

impl PyOutputMessage {
    pub(crate) fn from_rust(output: OutputMessage) -> Self {
        Self { inner: output }
    }
}
