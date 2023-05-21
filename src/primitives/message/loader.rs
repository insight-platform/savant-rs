use crate::primitives::message::video::frame::InnerVideoFrame;
use crate::primitives::message::{
    NativeMessageMarkerType, NativeMessageTypeConsts, NATIVE_MESSAGE_MARKER_LEN,
};
use crate::primitives::{EndOfStream, Message, VideoFrame, VideoFrameBatch};
use crate::utils::python::no_gil;
use pyo3::pyfunction;

#[pyfunction]
#[pyo3(name = "load_message")]
pub fn load_message_py(bytes: Vec<u8>) -> Message {
    no_gil(|| load_message(bytes))
}

pub fn load_message(mut bytes: Vec<u8>) -> Message {
    if bytes.len() < NATIVE_MESSAGE_MARKER_LEN {
        return Message::unknown(format!(
            "Message is too short: {} < {}",
            bytes.len(),
            NATIVE_MESSAGE_MARKER_LEN
        ));
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
                Err(e) => Message::unknown(format!("{:?}", e)),
            }
        }
        NativeMessageTypeConsts::VideoFrame => {
            let f: Result<InnerVideoFrame, _> = rkyv::from_bytes(&bytes[..]);
            match f {
                Ok(mut f) => {
                    f.prepare_after_load();
                    Message::video_frame(VideoFrame::from_inner(f))
                }
                Err(e) => Message::unknown(format!("{:?}", e)),
            }
        }
        NativeMessageTypeConsts::VideoFrameBatch => {
            let b: Result<VideoFrameBatch, _> = rkyv::from_bytes(&bytes[..]);
            match b {
                Ok(mut b) => {
                    b.prepare_after_load();
                    Message::video_frame_batch(b)
                }
                Err(e) => Message::unknown(format!("{:?}", e)),
            }
        }
        NativeMessageTypeConsts::Unknown => {
            Message::unknown(format!("Unknown message type: {:?}", t))
        }
    }
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
