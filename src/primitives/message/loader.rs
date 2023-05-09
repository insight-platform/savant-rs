use crate::primitives::message::video::frame::proxy::ProxyVideoFrame;
use crate::primitives::message::video::frame::VideoFrame;
use crate::primitives::message::{
    NativeMessageMarkerType, NativeMessageTypeConsts, NATIVE_MESSAGE_MARKER_LEN,
};
use crate::primitives::{EndOfStream, Message};
use pyo3::{pyfunction, Python};

#[pyfunction]
pub fn load_message(mut bytes: Vec<u8>) -> Message {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            if bytes.len() < NATIVE_MESSAGE_MARKER_LEN {
                return Message::unknown();
            }
            let final_length = bytes.len().saturating_sub(NATIVE_MESSAGE_MARKER_LEN);
            let t = NativeMessageTypeConsts::from(
                <&NativeMessageMarkerType>::try_from(&bytes[final_length..]).unwrap(),
            );
            bytes.truncate(final_length);
            match t {
                NativeMessageTypeConsts::EndOfStream => {
                    let eos: Result<EndOfStream, _> = rkyv::from_bytes(&bytes[..]);
                    match eos {
                        Ok(eos) => Message::end_of_stream(eos),
                        Err(_) => Message::unknown(),
                    }
                }
                NativeMessageTypeConsts::VideoFrame => {
                    let f: Result<VideoFrame, _> = rkyv::from_bytes(&bytes[..]);
                    match f {
                        Ok(mut f) => {
                            f.prepare_after_load();
                            Message::video_frame(ProxyVideoFrame::new(f))
                        }
                        Err(_) => Message::unknown(),
                    }
                }
                NativeMessageTypeConsts::Unknown => Message::unknown(),
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use crate::primitives::message::loader::load_message;

    #[test]
    fn test_load_short() {
        pyo3::prepare_freethreaded_python();
        let f = load_message(vec![0; 1]);
        assert!(f.is_unknown());
    }

    #[test]
    fn test_load_invalid() {
        pyo3::prepare_freethreaded_python();
        let f = load_message(vec![0; 5]);
        assert!(f.is_unknown());
    }

    #[test]
    fn test_load_invalid_marker() {
        pyo3::prepare_freethreaded_python();
        let f = load_message(vec![0, 0, 0, 0, 1, 2, 3, 4]);
        assert!(f.is_unknown());
    }
}
