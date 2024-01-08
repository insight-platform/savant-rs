pub mod loader;
pub mod saver;

use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::user_data::UserData;
use crate::primitives::VideoFrame;
use crate::primitives::{EndOfStream, Shutdown, VideoFrameBatch};
use crate::utils::otlp::PropagatedContext;
use pyo3::{pyclass, pyfunction, pymethods, Py, PyAny};
use savant_core::primitives::rust as rust_primitives;

#[pyclass]
#[derive(Clone, Debug)]
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
    /// Params
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

    /// Create a new shutdown message
    ///
    /// Params
    /// ------
    /// shutdown : :py:class:`savant_rs.primitives.Shutdown`
    ///   The shutdown message
    ///
    /// Returns
    /// -------
    /// py:class:`savant_rs.utils.serialization.Message`
    ///   The message of Shutdown type
    ///
    #[staticmethod]
    pub fn shutdown(shutdown: Shutdown) -> Self {
        Self(rust_primitives::Message::shutdown(shutdown.0))
    }

    /// Create a new unspecified message
    ///
    /// Params
    /// ------
    /// t : :py:class:`savant_rs.primitives.UnspecifiedData`
    ///   The unspecified message
    ///
    /// Returns
    /// -------
    /// py:class:`savant_rs.utils.serialization.Message`
    ///   The message of EndOfStream type
    ///
    #[staticmethod]
    pub fn user_data(data: UserData) -> Self {
        Self(rust_primitives::Message::user_data(data.0))
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

    /// Checks if the message is of Shutdown type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of Shutdown type, False otherwise
    ///
    pub fn is_shutdown(&self) -> bool {
        self.0.is_shutdown()
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
    pub fn is_user_data(&self) -> bool {
        self.0.is_user_data()
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

    /// Returns the message as Shutdown type
    ///
    /// Returns
    /// -------
    /// py:class:`savant_rs.primitives.Shutdown`
    ///   The message as Shutdown type
    ///
    pub fn as_shutdown(&self) -> Option<Shutdown> {
        let shutdown = self.0.as_shutdown()?;
        Some(Shutdown(shutdown.clone()))
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
    pub fn as_user_data(&self) -> Option<UserData> {
        let data = self.0.as_user_data()?;
        Some(UserData(data.clone()))
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

    /// Allows validating the sequence id of the message
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the sequence id is valid, False otherwise
    ///
    pub fn validate_seq_id(&self) -> bool {
        savant_core::message::validate_seq_id(&self.0)
    }
}

#[pyfunction]
pub(crate) fn clear_source_seq_id(source: &str) {
    savant_core::message::clear_source_seq_id(source);
}
