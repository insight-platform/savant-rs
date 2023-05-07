pub mod loader;
pub mod saver;

use crate::primitives::{EndOfStream, VideoFrame};
use pyo3::{pyclass, pymethods, Py, PyAny};

#[derive(Debug, Clone)]
pub enum NativeFrame {
    EndOfStream(EndOfStream),
    VideoFrame(Box<VideoFrame>),
    Unknown,
}

#[repr(u32)]
enum NativeFrameTypeConsts {
    EndOfStream,
    VideoFrame,
    Unknown,
}

pub const NATIVE_FRAME_MARKER_LEN: usize = 4;
pub type NativeFrameMarkerType = [u8; NATIVE_FRAME_MARKER_LEN];

impl From<NativeFrameTypeConsts> for NativeFrameMarkerType {
    fn from(value: NativeFrameTypeConsts) -> Self {
        match value {
            NativeFrameTypeConsts::EndOfStream => [0, 0, 0, 0],
            NativeFrameTypeConsts::VideoFrame => [1, 0, 0, 0],
            NativeFrameTypeConsts::Unknown => [255, 255, 255, 255],
        }
    }
}

impl From<&NativeFrameMarkerType> for NativeFrameTypeConsts {
    fn from(value: &NativeFrameMarkerType) -> Self {
        match value {
            [0, 0, 0, 0] => NativeFrameTypeConsts::EndOfStream,
            [1, 0, 0, 0] => NativeFrameTypeConsts::VideoFrame,
            _ => NativeFrameTypeConsts::Unknown,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Frame {
    frame: NativeFrame,
}

#[pymethods]
impl Frame {
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
            frame: NativeFrame::Unknown,
        }
    }

    #[staticmethod]
    pub fn video_frame(frame: VideoFrame) -> Self {
        Self {
            frame: NativeFrame::VideoFrame(Box::new(frame)),
        }
    }

    #[staticmethod]
    pub fn end_of_stream(eos: EndOfStream) -> Self {
        Self {
            frame: NativeFrame::EndOfStream(eos),
        }
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self.frame, NativeFrame::Unknown)
    }

    pub fn is_end_of_stream(&self) -> bool {
        matches!(self.frame, NativeFrame::EndOfStream(_))
    }

    pub fn is_video_frame(&self) -> bool {
        matches!(self.frame, NativeFrame::VideoFrame(_))
    }

    pub fn as_end_of_stream(&self) -> Option<EndOfStream> {
        match &self.frame {
            NativeFrame::EndOfStream(eos) => Some(eos.clone()),
            _ => None,
        }
    }

    pub fn as_video_frame(&self) -> Option<VideoFrame> {
        match &self.frame {
            NativeFrame::VideoFrame(frame) => Some(*frame.clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::db::loader::Loader;
    use crate::primitives::db::{
        NativeFrameMarkerType, NativeFrameTypeConsts, NATIVE_FRAME_MARKER_LEN,
    };
    use crate::primitives::{EndOfStream, Frame, Saver};
    use crate::test::utils::gen_frame;

    #[test]
    fn test_save_load_eos() {
        pyo3::prepare_freethreaded_python();
        let saver = Saver::new(1);
        let loader = Loader::new(1);
        let eos = EndOfStream::new("test".to_string());
        let frame = Frame::end_of_stream(eos);
        let res = saver.save(frame).recv().unwrap();
        assert_eq!(
            res[(res.len() - NATIVE_FRAME_MARKER_LEN)..].as_ref(),
            NativeFrameMarkerType::from(NativeFrameTypeConsts::EndOfStream).as_ref()
        );
        let res = loader.load(res);
        let frame = res.recv().unwrap();
        assert!(frame.is_end_of_stream());
    }

    #[test]
    fn test_save_video_frame() {
        pyo3::prepare_freethreaded_python();
        let saver = Saver::new(1);
        let loader = Loader::new(1);

        let frame = Frame::video_frame(gen_frame());
        let res = saver.save(frame).recv().unwrap();
        assert_eq!(
            res[(res.len() - NATIVE_FRAME_MARKER_LEN)..].as_ref(),
            NativeFrameMarkerType::from(NativeFrameTypeConsts::VideoFrame).as_ref()
        );
        let res = loader.load(res);
        let frame = res.recv().unwrap();
        assert!(frame.is_video_frame());
    }

    #[test]
    fn test_save_unknown() {
        pyo3::prepare_freethreaded_python();
        let saver = Saver::new(1);
        let loader = Loader::new(1);

        let frame = Frame::unknown();
        let res = saver.save(frame).recv().unwrap();
        assert_eq!(
            res[(res.len() - NATIVE_FRAME_MARKER_LEN)..].as_ref(),
            NativeFrameMarkerType::from(NativeFrameTypeConsts::Unknown).as_ref()
        );
        let res = loader.load(res);
        let frame = res.recv().unwrap();
        assert!(frame.is_unknown());
    }
}
