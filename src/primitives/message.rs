pub mod eos;
pub mod loader;
pub mod saver;
pub mod video;

use crate::primitives::EndOfStream;
use crate::primitives::ProxyVideoFrame;
use pyo3::{pyclass, pymethods, Py, PyAny};

#[derive(Debug, Clone)]
pub enum NativeMessage {
    EndOfStream(EndOfStream),
    VideoFrame(ProxyVideoFrame),
    Unknown,
}

#[repr(u32)]
enum NativeMessageTypeConsts {
    EndOfStream,
    VideoFrame,
    Unknown,
}

pub const NATIVE_MESSAGE_MARKER_LEN: usize = 4;
pub type NativeMessageMarkerType = [u8; NATIVE_MESSAGE_MARKER_LEN];

impl From<NativeMessageTypeConsts> for NativeMessageMarkerType {
    fn from(value: NativeMessageTypeConsts) -> Self {
        match value {
            NativeMessageTypeConsts::EndOfStream => [0, 0, 0, 0],
            NativeMessageTypeConsts::VideoFrame => [1, 0, 0, 0],
            NativeMessageTypeConsts::Unknown => [255, 255, 255, 255],
        }
    }
}

impl From<&NativeMessageMarkerType> for NativeMessageTypeConsts {
    fn from(value: &NativeMessageMarkerType) -> Self {
        match value {
            [0, 0, 0, 0] => NativeMessageTypeConsts::EndOfStream,
            [1, 0, 0, 0] => NativeMessageTypeConsts::VideoFrame,
            _ => NativeMessageTypeConsts::Unknown,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Message {
    frame: NativeMessage,
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
    pub fn unknown() -> Self {
        Self {
            frame: NativeMessage::Unknown,
        }
    }

    #[staticmethod]
    pub fn video_frame(frame: ProxyVideoFrame) -> Self {
        Self {
            frame: NativeMessage::VideoFrame(frame),
        }
    }

    #[staticmethod]
    pub fn end_of_stream(eos: EndOfStream) -> Self {
        Self {
            frame: NativeMessage::EndOfStream(eos),
        }
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self.frame, NativeMessage::Unknown)
    }

    pub fn is_end_of_stream(&self) -> bool {
        matches!(self.frame, NativeMessage::EndOfStream(_))
    }

    pub fn is_video_frame(&self) -> bool {
        matches!(self.frame, NativeMessage::VideoFrame(_))
    }

    pub fn as_end_of_stream(&self) -> Option<EndOfStream> {
        match &self.frame {
            NativeMessage::EndOfStream(eos) => Some(eos.clone()),
            _ => None,
        }
    }

    pub fn as_video_frame(&self) -> Option<ProxyVideoFrame> {
        match &self.frame {
            NativeMessage::VideoFrame(frame) => Some(frame.clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::message::loader::load_message;
    use crate::primitives::message::saver::save_message;
    use crate::primitives::message::video::frame::proxy::ProxyVideoFrame;
    use crate::primitives::message::{
        NativeMessageMarkerType, NativeMessageTypeConsts, NATIVE_MESSAGE_MARKER_LEN,
    };
    use crate::primitives::{EndOfStream, Message};
    use crate::test::utils::gen_frame;

    #[test]
    fn test_save_load_eos() {
        pyo3::prepare_freethreaded_python();
        let eos = EndOfStream::new("test".to_string());
        let frame = Message::end_of_stream(eos);
        let res = save_message(frame);
        assert_eq!(
            res[(res.len() - NATIVE_MESSAGE_MARKER_LEN)..].as_ref(),
            NativeMessageMarkerType::from(NativeMessageTypeConsts::EndOfStream).as_ref()
        );
        let frame = load_message(res);
        assert!(frame.is_end_of_stream());
    }

    #[test]
    fn test_save_video_frame() {
        pyo3::prepare_freethreaded_python();
        let frame = Message::video_frame(ProxyVideoFrame::new(gen_frame()));
        let res = save_message(frame);
        assert_eq!(
            res[(res.len() - NATIVE_MESSAGE_MARKER_LEN)..].as_ref(),
            NativeMessageMarkerType::from(NativeMessageTypeConsts::VideoFrame).as_ref()
        );
        let frame = load_message(res);
        assert!(frame.is_video_frame());
    }

    #[test]
    fn test_save_unknown() {
        pyo3::prepare_freethreaded_python();
        let frame = Message::unknown();
        let res = save_message(frame);
        assert_eq!(
            res[(res.len() - NATIVE_MESSAGE_MARKER_LEN)..].as_ref(),
            NativeMessageMarkerType::from(NativeMessageTypeConsts::Unknown).as_ref()
        );
        let frame = load_message(res);
        assert!(frame.is_unknown());
    }
}
