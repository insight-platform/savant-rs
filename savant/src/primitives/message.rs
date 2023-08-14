pub mod eos;
pub mod loader;
pub mod saver;
pub mod unspecified;
pub mod video;

use crate::primitives::attribute::AttributeMethods;
use crate::primitives::message::unspecified::UnspecifiedData;
use crate::primitives::message::video::frame::frame_update::VideoFrameUpdate;
use crate::primitives::message::video::frame::VideoFrame;
use crate::primitives::message::video::query::MatchQuery;
use crate::primitives::VideoFrameProxy;
use crate::primitives::{EndOfStream, VideoFrameBatch};
use crate::release_gil;
use crate::utils::otlp::PropagatedContext;
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};
use savant_core::primitives::rust as rust_primitives;
use savant_core::{rust, version_to_bytes_le};
use std::mem;

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub enum MessageEnvelope {
    EndOfStream(rust_primitives::EndOfStream),
    VideoFrame(Box<VideoFrame>),
    VideoFrameBatch(VideoFrameBatch),
    VideoFrameUpdate(VideoFrameUpdate),
    Unspecified(rust_primitives::UnspecifiedData),
    Unknown(String),
}

pub const VERSION_LEN: usize = 4;

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct MessageMeta {
    pub(crate) lib_version: [u8; VERSION_LEN],
    pub(crate) routing_labels: Vec<String>,
    pub(crate) span_context: rust::PropagatedContext,
}

impl Default for MessageMeta {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageMeta {
    pub fn new() -> Self {
        Self {
            lib_version: version_to_bytes_le(),
            routing_labels: Vec::default(),
            span_context: rust::PropagatedContext::default(),
        }
    }
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct Message {
    meta: MessageMeta,
    payload: MessageEnvelope,
}

impl Message {
    pub fn unknown(s: String) -> Self {
        Self {
            meta: MessageMeta::new(),
            payload: MessageEnvelope::Unknown(s),
        }
    }

    pub fn unspecified(mut t: UnspecifiedData) -> Self {
        t.exclude_temporary_attributes();

        Self {
            meta: MessageMeta::new(),
            payload: MessageEnvelope::Unspecified(unsafe {
                mem::transmute::<UnspecifiedData, rust_primitives::UnspecifiedData>(t)
            }),
        }
    }

    pub fn end_of_stream(eos: EndOfStream) -> Self {
        Self {
            meta: MessageMeta::new(),
            payload: MessageEnvelope::EndOfStream(unsafe {
                mem::transmute::<EndOfStream, rust_primitives::EndOfStream>(eos)
            }),
        }
    }
    pub fn video_frame(frame: &VideoFrameProxy) -> Self {
        let frame_copy = frame.deep_copy();

        frame_copy.exclude_temporary_attributes();
        frame_copy
            .access_objects(&MatchQuery::Idle)
            .iter()
            .for_each(|o| {
                o.exclude_temporary_attributes();
            });
        frame_copy.make_snapshot();

        let inner = frame_copy.inner.read().clone();

        Self {
            meta: MessageMeta::new(),
            payload: MessageEnvelope::VideoFrame(inner),
        }
    }

    pub fn video_frame_batch(batch: &VideoFrameBatch) -> Self {
        let mut batch_copy = batch.deep_copy();
        batch_copy.prepare_before_save();
        Self {
            meta: MessageMeta::new(),
            payload: MessageEnvelope::VideoFrameBatch(batch_copy),
        }
    }

    pub fn video_frame_update(update: VideoFrameUpdate) -> Self {
        Self {
            meta: MessageMeta::new(),
            payload: MessageEnvelope::VideoFrameUpdate(update),
        }
    }

    pub fn meta(&self) -> &MessageMeta {
        &self.meta
    }

    pub fn meta_mut(&mut self) -> &mut MessageMeta {
        &mut self.meta
    }
}

#[pymethods]
impl Message {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
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
    #[pyo3(name = "unknown")]
    pub fn unknown_py(s: String) -> Self {
        Message::unknown(s)
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
    #[pyo3(name = "video_frame")]
    #[pyo3(signature = (frame, no_gil=true))]
    fn video_frame_gil(frame: &VideoFrameProxy, no_gil: bool) -> Self {
        release_gil!(no_gil, || Message::video_frame(frame))
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
    #[pyo3(name = "video_frame_batch")]
    #[pyo3(signature = (batch, no_gil=true))]
    fn video_frame_batch_gil(batch: &VideoFrameBatch, no_gil: bool) -> Self {
        release_gil!(no_gil, || Message::video_frame_batch(batch))
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
    #[pyo3(name = "end_of_stream")]
    fn end_of_stream_py(eos: EndOfStream) -> Self {
        Self::end_of_stream(eos)
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
    #[pyo3(name = "unspecified")]
    #[pyo3(signature = (t, no_gil=true))]
    fn unspecified_gil(t: &UnspecifiedData, no_gil: bool) -> Self {
        release_gil!(no_gil, || Message::unspecified(t.clone()))
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
    #[pyo3(name = "video_frame_update")]
    #[pyo3(signature = (update, no_gil=true))]
    fn video_frame_update_gil(update: &VideoFrameUpdate, no_gil: bool) -> Self {
        release_gil!(no_gil, || Message::video_frame_update(update.clone()))
    }

    #[getter]
    fn get_labels(&self) -> Vec<String> {
        self.meta.routing_labels.clone()
    }

    #[setter]
    fn set_labels(&mut self, labels: Vec<String>) {
        self.meta.routing_labels = labels;
    }

    #[setter]
    fn set_span_context(&mut self, context: PropagatedContext) {
        self.meta.span_context =
            unsafe { mem::transmute::<PropagatedContext, rust::PropagatedContext>(context) };
    }

    #[getter]
    fn get_span_context(&self) -> PropagatedContext {
        unsafe {
            mem::transmute::<rust::PropagatedContext, PropagatedContext>(
                self.meta.span_context.clone(),
            )
        }
    }

    /// Checks if the message is of Unknown type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of Unknown type, False otherwise
    ///
    pub fn is_unknown(&self) -> bool {
        matches!(self.payload, MessageEnvelope::Unknown(_))
    }

    /// Checks if the message is of EndOfStream type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of EndOfStream type, False otherwise
    ///
    pub fn is_end_of_stream(&self) -> bool {
        matches!(self.payload, MessageEnvelope::EndOfStream(_))
    }

    /// Checks if the message is of Telemetry type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of Telemetry type, False otherwise
    ///
    pub fn is_unspecified(&self) -> bool {
        matches!(self.payload, MessageEnvelope::Unspecified(_))
    }

    /// Checks if the message is of VideoFrame type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of VideoFrame type, False otherwise
    ///
    pub fn is_video_frame(&self) -> bool {
        matches!(self.payload, MessageEnvelope::VideoFrame(_))
    }

    /// Checks if the message is of VideoFrameUpdate type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of VideoFrameUpdate type, False otherwise
    ///
    pub fn is_video_frame_update(&self) -> bool {
        matches!(self.payload, MessageEnvelope::VideoFrameUpdate(_))
    }

    /// Checks if the message is of VideoFrameBatch type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of VideoFrameBatch type, False otherwise
    ///
    pub fn is_video_frame_batch(&self) -> bool {
        matches!(self.payload, MessageEnvelope::VideoFrameBatch(_))
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
        match &self.payload {
            MessageEnvelope::Unknown(s) => Some(s.clone()),
            _ => None,
        }
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
        match &self.payload {
            MessageEnvelope::EndOfStream(eos) => Some(unsafe {
                mem::transmute::<rust_primitives::EndOfStream, EndOfStream>(eos.clone())
            }),
            _ => None,
        }
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
        match &self.payload {
            MessageEnvelope::Unspecified(t) => Some(unsafe {
                mem::transmute::<rust_primitives::UnspecifiedData, UnspecifiedData>(t.clone())
            }),
            _ => None,
        }
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
    pub fn as_video_frame(&self) -> Option<VideoFrameProxy> {
        match &self.payload {
            MessageEnvelope::VideoFrame(frame) => Some(VideoFrameProxy::from_inner(*frame.clone())),
            _ => None,
        }
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
        match &self.payload {
            MessageEnvelope::VideoFrameUpdate(update) => Some(update.clone()),
            _ => None,
        }
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
        match &self.payload {
            MessageEnvelope::VideoFrameBatch(batch) => Some(batch.clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::message::loader::load_message;
    use crate::primitives::{
        save_message, Attribute, EndOfStream, Message, UnspecifiedData, VideoFrameBatch,
    };
    use crate::test::utils::gen_frame;

    #[test]
    fn test_save_load_eos() {
        pyo3::prepare_freethreaded_python();
        let eos = EndOfStream::new("test".to_string());
        let m = Message::end_of_stream(eos);
        let res = save_message(&m);
        let m = load_message(&res);
        assert!(m.is_end_of_stream());
    }

    #[test]
    fn test_save_load_unspecified() {
        pyo3::prepare_freethreaded_python();
        let t = UnspecifiedData::new("test".to_string());
        let m = Message::unspecified(t);
        let res = save_message(&m);
        let m = load_message(&res);
        assert!(m.is_unspecified());
    }

    #[test]
    fn test_save_load_video_frame() {
        pyo3::prepare_freethreaded_python();
        let m = Message::video_frame(&gen_frame());
        let res = save_message(&m);
        let m = load_message(&res);
        assert!(m.is_video_frame());
    }

    #[test]
    fn test_save_load_unknown() {
        pyo3::prepare_freethreaded_python();
        let m = Message::unknown("x".to_string());
        let res = save_message(&m);
        let m = load_message(&res);
        assert!(m.is_unknown());
    }

    #[test]
    fn test_save_load_batch() {
        pyo3::prepare_freethreaded_python();
        let mut batch = VideoFrameBatch::new();
        batch.add(1, gen_frame());
        batch.add(2, gen_frame());
        batch.add(3, gen_frame());
        let m = Message::video_frame_batch(&batch);
        let res = save_message(&m);
        let m = load_message(&res);
        assert!(m.is_video_frame_batch());

        let b = m.as_video_frame_batch().unwrap();
        assert!(b.get(1).is_some());
        assert!(b.get(2).is_some());
        assert!(b.get(3).is_some());
        let f = b.get(1).unwrap();
        let mut attrs = f.get_attributes();
        attrs.sort();

        assert_eq!(
            attrs,
            vec![
                ("system".into(), "test".into()),
                ("system".into(), "test2".into()),
                ("system2".into(), "test2".into()),
                ("test".into(), "test".into()),
            ]
        );

        let _ = f.access_objects_by_id(&vec![0]).pop().unwrap();
    }

    #[test]
    fn test_save_load_frame_with_temp_attributes() {
        pyo3::prepare_freethreaded_python();

        let f = gen_frame();
        let tmp_attr =
            Attribute::temporary("chronos".to_string(), "temp".to_string(), vec![], None);
        let attrs = f.get_attributes();
        assert_eq!(attrs.len(), 4);
        f.set_attribute(tmp_attr);
        let attrs = f.get_attributes();
        assert_eq!(attrs.len(), 5);
        let m = Message::video_frame(&f);
        let res = save_message(&m);
        let m = load_message(&res);
        assert!(m.is_video_frame());
        let f = m.as_video_frame().unwrap();
        let attrs = f.get_attributes();
        assert_eq!(attrs.len(), 4);
    }
}
