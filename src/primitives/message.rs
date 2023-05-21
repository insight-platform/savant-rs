pub mod eos;
pub mod loader;
pub mod saver;
pub mod video;

use crate::primitives::VideoFrame;
use crate::primitives::{EndOfStream, VideoFrameBatch};
use pyo3::{pyclass, pymethods, Py, PyAny};

#[derive(Debug, Clone)]
pub enum NativeMessage {
    EndOfStream(EndOfStream),
    VideoFrame(VideoFrame),
    VideoFrameBatch(VideoFrameBatch),
    Unknown(String),
}

#[repr(u32)]
#[derive(Debug)]
enum NativeMessageTypeConsts {
    EndOfStream,
    VideoFrame,
    VideoFrameBatch,
    Unknown,
}

pub const NATIVE_MESSAGE_MARKER_LEN: usize = 4;
pub type NativeMessageMarkerType = [u8; NATIVE_MESSAGE_MARKER_LEN];

impl From<NativeMessageTypeConsts> for NativeMessageMarkerType {
    fn from(value: NativeMessageTypeConsts) -> Self {
        match value {
            NativeMessageTypeConsts::EndOfStream => [0, 0, 0, 0],
            NativeMessageTypeConsts::VideoFrame => [1, 0, 0, 0],
            NativeMessageTypeConsts::VideoFrameBatch => [2, 0, 0, 0],
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
            _ => NativeMessageTypeConsts::Unknown,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Message {
    payload: NativeMessage,
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

    #[staticmethod]
    pub fn unknown(s: String) -> Self {
        Self {
            payload: NativeMessage::Unknown(s),
        }
    }

    #[staticmethod]
    pub fn video_frame(frame: VideoFrame) -> Self {
        Self {
            payload: NativeMessage::VideoFrame(frame),
        }
    }

    #[staticmethod]
    pub fn video_frame_batch(batch: VideoFrameBatch) -> Self {
        Self {
            payload: NativeMessage::VideoFrameBatch(batch),
        }
    }

    #[staticmethod]
    pub fn end_of_stream(eos: EndOfStream) -> Self {
        Self {
            payload: NativeMessage::EndOfStream(eos),
        }
    }

    #[getter]
    pub fn is_unknown(&self) -> bool {
        matches!(self.payload, NativeMessage::Unknown(_))
    }

    #[getter]
    pub fn is_end_of_stream(&self) -> bool {
        matches!(self.payload, NativeMessage::EndOfStream(_))
    }

    #[getter]
    pub fn is_video_frame(&self) -> bool {
        matches!(self.payload, NativeMessage::VideoFrame(_))
    }

    #[getter]
    pub fn is_video_frame_batch(&self) -> bool {
        matches!(self.payload, NativeMessage::VideoFrameBatch(_))
    }

    #[getter]
    pub fn as_unknown(&self) -> Option<String> {
        match &self.payload {
            NativeMessage::Unknown(s) => Some(s.clone()),
            _ => None,
        }
    }

    #[getter]
    pub fn as_end_of_stream(&self) -> Option<EndOfStream> {
        match &self.payload {
            NativeMessage::EndOfStream(eos) => Some(eos.clone()),
            _ => None,
        }
    }

    #[getter]
    pub fn as_video_frame(&self) -> Option<VideoFrame> {
        match &self.payload {
            NativeMessage::VideoFrame(frame) => Some(frame.clone()),
            _ => None,
        }
    }

    #[getter]
    pub fn as_video_frame_batch(&self) -> Option<VideoFrameBatch> {
        match &self.payload {
            NativeMessage::VideoFrameBatch(batch) => Some(batch.clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::message::loader::load_message;
    use crate::primitives::message::saver::save_message_py;
    use crate::primitives::message::{
        NativeMessageMarkerType, NativeMessageTypeConsts, NATIVE_MESSAGE_MARKER_LEN,
    };
    use crate::primitives::{EndOfStream, Message, VideoFrameBatch};
    use crate::test::utils::gen_frame;

    #[test]
    fn test_save_load_eos() {
        pyo3::prepare_freethreaded_python();
        let eos = EndOfStream::new("test".to_string());
        let m = Message::end_of_stream(eos);
        let res = save_message_py(m);
        assert_eq!(
            res[(res.len() - NATIVE_MESSAGE_MARKER_LEN)..].as_ref(),
            NativeMessageMarkerType::from(NativeMessageTypeConsts::EndOfStream).as_ref()
        );
        let m = load_message(res);
        assert!(m.is_end_of_stream());
    }

    #[test]
    fn test_save_load_video_frame() {
        pyo3::prepare_freethreaded_python();
        let m = Message::video_frame(gen_frame());
        let res = save_message_py(m);
        assert_eq!(
            res[(res.len() - NATIVE_MESSAGE_MARKER_LEN)..].as_ref(),
            NativeMessageMarkerType::from(NativeMessageTypeConsts::VideoFrame).as_ref()
        );
        let m = load_message(res);
        assert!(m.is_video_frame());
    }

    #[test]
    fn test_save_load_unknown() {
        pyo3::prepare_freethreaded_python();
        let m = Message::unknown("x".to_string());
        let res = save_message_py(m);
        assert_eq!(
            res[(res.len() - NATIVE_MESSAGE_MARKER_LEN)..].as_ref(),
            NativeMessageMarkerType::from(NativeMessageTypeConsts::Unknown).as_ref()
        );
        let m = load_message(res);
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
        let res = save_message_py(m);
        assert_eq!(
            res[(res.len() - NATIVE_MESSAGE_MARKER_LEN)..].as_ref(),
            NativeMessageMarkerType::from(NativeMessageTypeConsts::VideoFrameBatch).as_ref()
        );
        let m = load_message(res);
        assert!(m.is_video_frame_batch());

        let b = m.as_video_frame_batch().unwrap();
        assert!(b.get(1).is_some());
        assert!(b.get(2).is_some());
        assert!(b.get(3).is_some());
        let f = b.get(1).unwrap();
        let mut attrs = f.attributes();
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

        let _ = f.access_objects_by_id_py(vec![0]).pop().unwrap();
    }
}
