pub mod loader;
pub mod saver;

use crate::primitives::{EndOfStream, VideoFrame};
use pyo3::{pyclass, pymethods, Py, PyAny};

#[derive(Debug, Clone)]
pub enum NativeFrame {
    EndOfStream(EndOfStream),
    VideoFrame(Box<VideoFrame>),
}

#[repr(u32)]
enum NativeFrameTypeConsts {
    EndOfStream = 0,
    VideoFrame = 1,
}

impl From<&[u8]> for NativeFrameTypeConsts {
    fn from(value: &[u8]) -> Self {
        assert_eq!(value.len(), 4);
        let v = {
            (value[0] as u32)
                + ((value[1] as u32) << 8)
                + ((value[2] as u32) << 16)
                + ((value[3] as u32) << 24)
        };

        match v {
            0 => NativeFrameTypeConsts::EndOfStream,
            1 => NativeFrameTypeConsts::VideoFrame,
            _ => panic!("Invalid NativeFrameTypeConsts value"),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Frame {
    frame_type: NativeFrame,
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
    pub fn video_frame(frame: VideoFrame) -> Self {
        Self {
            frame_type: NativeFrame::VideoFrame(Box::new(frame)),
        }
    }

    #[staticmethod]
    pub fn end_of_stream(eos: EndOfStream) -> Self {
        Self {
            frame_type: NativeFrame::EndOfStream(eos),
        }
    }

    pub fn is_end_of_stream(&self) -> bool {
        matches!(self.frame_type, NativeFrame::EndOfStream(_))
    }

    pub fn is_video_frame(&self) -> bool {
        matches!(self.frame_type, NativeFrame::VideoFrame(_))
    }

    pub fn as_end_of_stream(&self) -> Option<EndOfStream> {
        match &self.frame_type {
            NativeFrame::EndOfStream(eos) => Some(eos.clone()),
            _ => None,
        }
    }

    pub fn as_video_frame(&self) -> Option<VideoFrame> {
        match &self.frame_type {
            NativeFrame::VideoFrame(frame) => Some(*frame.clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::db::loader::Loader;
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
        assert_eq!(res[0], 0);
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
        assert_eq!(res[0], 1);
        let res = loader.load(res);
        let frame = res.recv().unwrap();
        assert!(frame.is_video_frame());
    }
}
