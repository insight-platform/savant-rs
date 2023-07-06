pub mod eos;
pub mod loader;
pub mod saver;
pub mod telemetry;
pub mod video;

use crate::primitives::message::telemetry::Telemetry;
use crate::primitives::message::video::frame::frame_update::VideoFrameUpdate;
use crate::primitives::VideoFrameProxy;
use crate::primitives::{EndOfStream, VideoFrameBatch};
use crate::utils::python::release_gil;
use pyo3::{pyclass, pymethods, Py, PyAny};

#[derive(Debug, Clone)]
pub enum NativeMessage {
    EndOfStream(EndOfStream),
    VideoFrame(VideoFrameProxy),
    VideoFrameBatch(VideoFrameBatch),
    VideoFrameUpdate(VideoFrameUpdate),
    Telemetry(Telemetry),
    Unknown(String),
}

#[repr(u32)]
#[derive(Debug)]
enum NativeMessageTypeConsts {
    EndOfStream,
    VideoFrame,
    VideoFrameBatch,
    VideFrameUpdate,
    Telemetry,
    Unknown,
}

pub const NATIVE_MESSAGE_MARKER_LEN: usize = 4;
pub const VERSION_LEN: usize = 4;
pub type NativeMessageMarkerType = [u8; NATIVE_MESSAGE_MARKER_LEN];

impl From<NativeMessageTypeConsts> for NativeMessageMarkerType {
    fn from(value: NativeMessageTypeConsts) -> Self {
        match value {
            NativeMessageTypeConsts::EndOfStream => [0, 0, 0, 0],
            NativeMessageTypeConsts::VideoFrame => [1, 0, 0, 0],
            NativeMessageTypeConsts::VideoFrameBatch => [2, 0, 0, 0],
            NativeMessageTypeConsts::VideFrameUpdate => [3, 0, 0, 0],
            NativeMessageTypeConsts::Telemetry => [4, 0, 0, 0],
            NativeMessageTypeConsts::Unknown => [255, 255, 255, 255],
        }
    }
}

impl From<&NativeMessageMarkerType> for NativeMessageTypeConsts {
    fn from(value: &NativeMessageMarkerType) -> Self {
        match value {
            [0, 0, 0, 0] => NativeMessageTypeConsts::EndOfStream,
            [1, 0, 0, 0] => NativeMessageTypeConsts::VideoFrame,
            [2, 0, 0, 0] => NativeMessageTypeConsts::VideoFrameBatch,
            [3, 0, 0, 0] => NativeMessageTypeConsts::VideFrameUpdate,
            [4, 0, 0, 0] => NativeMessageTypeConsts::Telemetry,
            _ => NativeMessageTypeConsts::Unknown,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Message {
    payload: NativeMessage,
}

impl Message {
    pub fn telemetry(t: Telemetry) -> Self {
        Self {
            payload: NativeMessage::Telemetry(t),
        }
    }

    pub fn end_of_stream(eos: EndOfStream) -> Self {
        Self {
            payload: NativeMessage::EndOfStream(eos),
        }
    }
    pub fn video_frame(frame: VideoFrameProxy) -> Self {
        Self {
            payload: NativeMessage::VideoFrame(frame),
        }
    }

    pub fn video_frame_batch(batch: VideoFrameBatch) -> Self {
        Self {
            payload: NativeMessage::VideoFrameBatch(batch),
        }
    }

    pub fn video_frame_update(update: VideoFrameUpdate) -> Self {
        release_gil(|| Self {
            payload: NativeMessage::VideoFrameUpdate(update),
        })
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
    pub fn unknown(s: String) -> Self {
        Self {
            payload: NativeMessage::Unknown(s),
        }
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
    fn video_frame_gil(frame: &VideoFrameProxy) -> Self {
        release_gil(|| Self {
            payload: NativeMessage::VideoFrame(frame.clone()),
        })
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
    fn video_frame_batch_gil(batch: &VideoFrameBatch) -> Self {
        release_gil(|| Self {
            payload: NativeMessage::VideoFrameBatch(batch.clone()),
        })
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
    fn end_of_stream_gil(eos: &EndOfStream) -> Self {
        release_gil(|| Self {
            payload: NativeMessage::EndOfStream(eos.clone()),
        })
    }

    /// Create a new telemetry message
    ///
    /// Parameters
    /// ----------
    /// t : savant_rs.primitives.Telemetry
    ///   The telemetry message
    ///
    /// Returns
    /// -------
    /// :class:`savant_rs.utils.serialization.Message`
    ///   The message of EndOfStream type
    ///
    #[staticmethod]
    #[pyo3(name = "telemetry")]
    fn telemetry_gil(t: &Telemetry) -> Self {
        release_gil(|| Self {
            payload: NativeMessage::Telemetry(t.clone()),
        })
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
    fn video_frame_update_gil(update: &VideoFrameUpdate) -> Self {
        Self {
            payload: NativeMessage::VideoFrameUpdate(update.clone()),
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
        matches!(self.payload, NativeMessage::Unknown(_))
    }

    /// Checks if the message is of EndOfStream type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of EndOfStream type, False otherwise
    ///
    pub fn is_end_of_stream(&self) -> bool {
        matches!(self.payload, NativeMessage::EndOfStream(_))
    }

    /// Checks if the message is of Telemetry type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of Telemetry type, False otherwise
    ///
    pub fn is_telemetry(&self) -> bool {
        matches!(self.payload, NativeMessage::Telemetry(_))
    }

    /// Checks if the message is of VideoFrame type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of VideoFrame type, False otherwise
    ///
    pub fn is_video_frame(&self) -> bool {
        matches!(self.payload, NativeMessage::VideoFrame(_))
    }

    /// Checks if the message is of VideoFrameUpdate type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of VideoFrameUpdate type, False otherwise
    ///
    pub fn is_video_frame_update(&self) -> bool {
        matches!(self.payload, NativeMessage::VideoFrameUpdate(_))
    }

    /// Checks if the message is of VideoFrameBatch type
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the message is of VideoFrameBatch type, False otherwise
    ///
    pub fn is_video_frame_batch(&self) -> bool {
        matches!(self.payload, NativeMessage::VideoFrameBatch(_))
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
            NativeMessage::Unknown(s) => Some(s.clone()),
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
            NativeMessage::EndOfStream(eos) => Some(eos.clone()),
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
    pub fn as_telemetry(&self) -> Option<Telemetry> {
        match &self.payload {
            NativeMessage::Telemetry(t) => Some(t.clone()),
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
            NativeMessage::VideoFrame(frame) => Some(frame.clone()),
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
            NativeMessage::VideoFrameUpdate(update) => Some(update.clone()),
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
            NativeMessage::VideoFrameBatch(batch) => Some(batch.clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::message::loader::load_message;
    use crate::primitives::message::{
        NativeMessageMarkerType, NativeMessageTypeConsts, NATIVE_MESSAGE_MARKER_LEN, VERSION_LEN,
    };
    use crate::primitives::{
        save_message, Attribute, EndOfStream, Message, Telemetry, VideoFrameBatch,
    };
    use crate::test::utils::gen_frame;
    use std::ops::AddAssign;

    #[test]
    fn test_save_load_eos() {
        pyo3::prepare_freethreaded_python();
        let eos = EndOfStream::new("test".to_string());
        let m = Message::end_of_stream(eos);
        let res = save_message(m);
        let type_start = res.len() - NATIVE_MESSAGE_MARKER_LEN - VERSION_LEN;
        let type_end = type_start + VERSION_LEN;

        assert_eq!(
            res[type_start..type_end].as_ref(),
            NativeMessageMarkerType::from(NativeMessageTypeConsts::EndOfStream).as_ref()
        );
        let m = load_message(&res);
        assert!(m.is_end_of_stream());
    }

    #[test]
    fn test_save_load_telemetry() {
        pyo3::prepare_freethreaded_python();
        let t = Telemetry::new("test".to_string());
        let m = Message::telemetry(t);
        let res = save_message(m);
        let type_start = res.len() - NATIVE_MESSAGE_MARKER_LEN - VERSION_LEN;
        let type_end = type_start + VERSION_LEN;

        assert_eq!(
            res[type_start..type_end].as_ref(),
            NativeMessageMarkerType::from(NativeMessageTypeConsts::Telemetry).as_ref()
        );
        let m = load_message(&res);
        assert!(m.is_telemetry());
    }

    #[test]
    fn test_save_load_video_frame() {
        pyo3::prepare_freethreaded_python();
        let m = Message::video_frame(gen_frame());
        let res = save_message(m);
        let type_start = res.len() - NATIVE_MESSAGE_MARKER_LEN - VERSION_LEN;
        let type_end = type_start + VERSION_LEN;
        assert_eq!(
            res[type_start..type_end].as_ref(),
            NativeMessageMarkerType::from(NativeMessageTypeConsts::VideoFrame).as_ref()
        );
        let m = load_message(&res);
        assert!(m.is_video_frame());
    }

    #[test]
    fn test_save_load_unknown() {
        pyo3::prepare_freethreaded_python();
        let m = Message::unknown("x".to_string());
        let res = save_message(m);
        let type_start = res.len() - NATIVE_MESSAGE_MARKER_LEN - VERSION_LEN;
        let type_end = type_start + VERSION_LEN;
        assert_eq!(
            res[type_start..type_end].as_ref(),
            NativeMessageMarkerType::from(NativeMessageTypeConsts::Unknown).as_ref()
        );
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
        let m = Message::video_frame_batch(batch);
        let res = save_message(m);
        let type_start = res.len() - NATIVE_MESSAGE_MARKER_LEN - VERSION_LEN;
        let type_end = type_start + VERSION_LEN;

        assert_eq!(
            res[type_start..type_end].as_ref(),
            NativeMessageMarkerType::from(NativeMessageTypeConsts::VideoFrameBatch).as_ref()
        );
        let m = load_message(&res);
        assert!(m.is_video_frame_batch());

        let b = m.as_video_frame_batch().unwrap();
        assert!(b.get(1).is_some());
        assert!(b.get(2).is_some());
        assert!(b.get(3).is_some());
        let f = b.get(1).unwrap();
        let mut attrs = f.attributes_gil();
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
        let m = Message::video_frame(f);
        let res = save_message(m);
        let m = load_message(&res);
        assert!(m.is_video_frame());
        let f = m.as_video_frame().unwrap();
        let attrs = f.get_attributes();
        assert_eq!(attrs.len(), 4);
    }

    #[test]
    fn test_wrong_version() {
        pyo3::prepare_freethreaded_python();
        let eos = EndOfStream::new("test".to_string());
        let m = Message::end_of_stream(eos);
        let mut res = save_message(m);
        let version_start = res.len() - VERSION_LEN;
        let version_end = version_start + VERSION_LEN;
        res[version_end - 1].add_assign(1);
        let m = load_message(&res);
        assert!(m.is_unknown());
        let m = m.as_unknown().unwrap();
        assert!(m.contains("CRC32"));
    }
}
