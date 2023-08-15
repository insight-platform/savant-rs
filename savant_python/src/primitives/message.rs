pub mod eos;
pub mod loader;
pub mod saver;
pub mod unspecified;
pub mod video;

use crate::primitives::message::unspecified::UnspecifiedData;
use crate::primitives::message::video::frame_update::VideoFrameUpdate;
use crate::primitives::VideoFrame;
use crate::primitives::{EndOfStream, VideoFrameBatch};
use crate::utils::otlp::PropagatedContext;
use pyo3::{pyclass, pymethods, Py, PyAny};
use savant_core::primitives::rust as rust_primitives;

#[pyclass]
pub struct Message(pub(crate) rust_primitives::Message);

#[pymethods]
impl Message {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Create a new undefined message
    ///
    /// Parameters
    /// ----------
    /// s : str
    ///   The message text
    ///
    /// Returns
    /// -------
    /// :class:`savant_rs.utils.serialization.Message`
    ///   The message of Unknown type
    ///
    #[staticmethod]
    pub fn unknown(s: String) -> Self {
        Self(rust_primitives::Message::unknown(s))
    }

    /// Create a new unspecified message
    ///
    /// Parameters
    /// ----------
    /// t : savant_rs.primitives.UnspecifiedData
    ///   The unspecified message
    ///
    /// Returns
    /// -------
    /// :class:`savant_rs.utils.serialization.Message`
    ///   The message of EndOfStream type
    ///
    #[staticmethod]
    pub fn unspecified(data: UnspecifiedData) -> Self {
        Self(rust_primitives::Message::unspecified(data.0))
    }

    /// Create a new end of stream message
    ///
    /// Parameters
    /// ----------
    /// eos : savant_rs.primitives.EndOfStream
    ///   The end of stream message
    ///
    /// Returns
    /// -------
    /// :class:`savant_rs.utils.serialization.Message`
    ///   The message of EndOfStream type
    ///
    #[staticmethod]
    pub fn end_of_stream(eos: EndOfStream) -> Self {
        Self(rust_primitives::Message::end_of_stream(eos.0))
    }
    /// Create a new video frame message
    ///
    /// Parameters
    /// ----------
    /// frame : savant_rs.primitives.VideoFrame
    ///   The video frame
    ///
    /// Returns
    /// -------
    /// :class:`savant_rs.utils.serialization.Message`
    ///   The message of VideoFrame type
    ///
    #[staticmethod]
    pub fn video_frame(frame: &VideoFrame) -> Self {
        Self(rust_primitives::Message::video_frame(&frame.0))
    }

    /// Create a new video frame batch message
    ///
    /// Parameters
    /// ----------
    /// batch : savant_rs.primitives.VideoFrameBatch
    ///   The video frame batch
    ///
    /// Returns
    /// -------
    /// :class:`savant_rs.utils.serialization.Message`
    ///   The message of VideoFrameBatch type
    ///
    #[staticmethod]
    pub fn video_frame_batch(batch: &VideoFrameBatch) -> Self {
        Self(rust_primitives::Message::video_frame_batch(&batch.0))
    }

    /// Create a new video frame update message
    ///
    /// Parameters
    /// ----------
    /// update : savant_rs.primitives.VideoFrameUpdate
    ///   The update struct
    ///
    /// Returns
    /// -------
    /// :class:`savant_rs.utils.serialization.Message`
    ///   The message of VideoFrameUpdate type
    ///
    #[staticmethod]
    pub fn video_frame_update(update: VideoFrameUpdate) -> Self {
        Self(rust_primitives::Message::video_frame_update(update.0))
    }

    #[getter]
    fn get_labels(&self) -> Vec<String> {
        self.0.meta().routing_labels.clone()
    }

    #[setter]
    fn set_labels(&mut self, labels: Vec<String>) {
        self.0.meta_mut().routing_labels = labels;
    }

    #[setter]
    fn set_span_context(&mut self, context: PropagatedContext) {
        self.0.meta_mut().span_context = context.0;
    }

    #[getter]
    fn get_span_context(&self) -> PropagatedContext {
        PropagatedContext(self.0.meta().span_context.clone())
    }

    /// Checks if the message is of Unknown type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of Unknown type, False otherwise
    ///
    pub fn is_unknown(&self) -> bool {
        self.0.is_unknown()
    }

    /// Checks if the message is of EndOfStream type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of EndOfStream type, False otherwise
    ///
    pub fn is_end_of_stream(&self) -> bool {
        self.0.is_end_of_stream()
    }

    /// Checks if the message is of Telemetry type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of Telemetry type, False otherwise
    ///
    pub fn is_unspecified(&self) -> bool {
        self.0.is_unspecified()
    }

    /// Checks if the message is of VideoFrame type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of VideoFrame type, False otherwise
    ///
    pub fn is_video_frame(&self) -> bool {
        self.0.is_video_frame()
    }

    /// Checks if the message is of VideoFrameUpdate type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of VideoFrameUpdate type, False otherwise
    ///
    pub fn is_video_frame_update(&self) -> bool {
        self.0.is_video_frame_update()
    }

    /// Checks if the message is of VideoFrameBatch type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of VideoFrameBatch type, False otherwise
    ///
    pub fn is_video_frame_batch(&self) -> bool {
        self.0.is_video_frame_batch()
    }

    /// Returns the message as Unknown type
    ///
    /// Returns
    /// -------
    /// str
    ///   The message as Unknown type
    /// None
    ///   If the message is not of Unknown type
    ///
    pub fn as_unknown(&self) -> Option<String> {
        self.0.as_unknown()
    }

    /// Returns the message as EndOfStream type
    ///
    /// Returns
    /// -------
    /// :class:`savant_rs.primitives.EndOfStream`
    ///   The message as EndOfStream type
    /// None
    ///   If the message is not of EndOfStream type
    ///
    pub fn as_end_of_stream(&self) -> Option<EndOfStream> {
        let eos = self.0.as_end_of_stream()?;
        Some(EndOfStream(eos.clone()))
    }

    /// Returns the message as Telemetry type
    ///
    /// Returns
    /// -------
    /// :class:`savant_rs.primitives.Telemetry`
    ///   The message as Telemetry type
    /// None
    ///   If the message is not of Telemetry type
    ///
    pub fn as_unspecified(&self) -> Option<UnspecifiedData> {
        let data = self.0.as_unspecified()?;
        Some(UnspecifiedData(data.clone()))
    }

    /// Returns the message as VideoFrame type
    ///
    /// Returns
    /// -------
    /// :class:`savant_rs.primitives.VideoFrame`
    ///   The message as VideoFrame type
    /// None
    ///   If the message is not of VideoFrame type
    ///
    pub fn as_video_frame(&self) -> Option<VideoFrame> {
        let frame = self.0.as_video_frame()?;
        Some(VideoFrame(frame.clone()))
    }

    /// Returns the message as VideoFrameUpdate type
    ///
    /// Returns
    /// -------
    /// :class:`savant_rs.primitives.VideoFrameUpdate`
    ///   The message as VideoFrameUpdate type
    /// None
    ///   If the message is not of VideoFrameUpdate type
    ///
    pub fn as_video_frame_update(&self) -> Option<VideoFrameUpdate> {
        let update = self.0.as_video_frame_update()?;
        Some(VideoFrameUpdate(update.clone()))
    }

    /// Returns the message as VideoFrameBatch type
    ///
    /// Returns
    /// -------
    /// :class:`savant_rs.primitives.VideoFrameBatch`
    ///   The message as VideoFrameBatch type
    /// None
    ///   If the message is not of VideoFrameBatch type
    ///
    pub fn as_video_frame_batch(&self) -> Option<VideoFrameBatch> {
        let batch = self.0.as_video_frame_batch()?;
        Some(VideoFrameBatch(batch.clone()))
    }
}
