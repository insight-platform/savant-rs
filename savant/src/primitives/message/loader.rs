use crate::bytes_le_to_version;
use crate::primitives::message::video::frame::VideoFrame;
use crate::primitives::message::{
    NativeMessageMarkerType, NativeMessageTypeConsts, NATIVE_MESSAGE_MARKER_LEN, VERSION_LEN,
};
use crate::primitives::{
    EndOfStream, Message, Telemetry, VideoFrameBatch, VideoFrameProxy, VideoFrameUpdate,
};
use crate::utils::byte_buffer::ByteBuffer;
use crate::utils::python::release_gil;
use pyo3::pyfunction;
use pyo3::types::PyBytes;

/// Loads a message from a byte array. The function is GIL-free.
///
/// Parameters
/// ----------
/// bytes : bytes
///   The byte array to load the message from.
///
/// Returns
/// -------
/// savant_rs.primitives.Message
///   The loaded message.
///
#[pyfunction]
#[pyo3(name = "load_message")]
pub fn load_message_gil(bytes: Vec<u8>) -> Message {
    release_gil(|| load_message(&bytes))
}

pub fn load_message(bytes: &[u8]) -> Message {
    if bytes.len() < NATIVE_MESSAGE_MARKER_LEN + VERSION_LEN {
        return Message::unknown(format!(
            "Message is too short: {} < {}",
            bytes.len(),
            NATIVE_MESSAGE_MARKER_LEN
        ));
    }
    let final_length = bytes
        .len()
        .saturating_sub(NATIVE_MESSAGE_MARKER_LEN + VERSION_LEN);

    let version_begin_offset = final_length + NATIVE_MESSAGE_MARKER_LEN;
    let version_bytes: [u8; 4] = bytes[version_begin_offset..version_begin_offset + VERSION_LEN]
        .try_into()
        .unwrap();

    let received_version = bytes_le_to_version(version_bytes);

    if received_version != crate::version_crc32() {
        return Message::unknown(format!(
            "Message CRC32 version mismatch: {:?} != {:?}. Expected version: {}",
            received_version,
            crate::version_crc32(),
            crate::version()
        ));
    }

    let typ = NativeMessageTypeConsts::from(
        <&NativeMessageMarkerType>::try_from(
            &bytes[final_length..final_length + NATIVE_MESSAGE_MARKER_LEN],
        )
        .unwrap(),
    );
    let bytes = &bytes[..final_length];
    match typ {
        NativeMessageTypeConsts::EndOfStream => {
            let eos: Result<EndOfStream, _> = rkyv::from_bytes(bytes);
            match eos {
                Ok(eos) => Message::end_of_stream(eos),
                Err(e) => Message::unknown(format!("{:?}", e)),
            }
        }

        NativeMessageTypeConsts::Telemetry => {
            let eos: Result<Telemetry, _> = rkyv::from_bytes(bytes);
            match eos {
                Ok(t) => Message::telemetry(t),
                Err(e) => Message::unknown(format!("{:?}", e)),
            }
        }

        NativeMessageTypeConsts::VideFrameUpdate => {
            let update: Result<VideoFrameUpdate, _> = rkyv::from_bytes(bytes);
            match update {
                Ok(upd) => Message::video_frame_update(upd),
                Err(e) => Message::unknown(format!("{:?}", e)),
            }
        }

        NativeMessageTypeConsts::VideoFrame => {
            let f: Result<VideoFrame, _> = rkyv::from_bytes(bytes);
            match f {
                Ok(f) => {
                    let f = VideoFrameProxy::from_inner(f);
                    f.restore_from_snapshot();
                    Message::video_frame(f)
                }
                Err(e) => Message::unknown(format!("{:?}", e)),
            }
        }
        NativeMessageTypeConsts::VideoFrameBatch => {
            let b: Result<VideoFrameBatch, _> = rkyv::from_bytes(bytes);
            match b {
                Ok(mut b) => {
                    b.prepare_after_load();
                    Message::video_frame_batch(b)
                }
                Err(e) => Message::unknown(format!("{:?}", e)),
            }
        }
        NativeMessageTypeConsts::Unknown => {
            Message::unknown(format!("Unknown message type: {:?}", typ))
        }
    }
}

/// Loads a message from a :class:`savant_rs.utils.ByteBuffer`. The function is GIL-free.
///
/// Parameters
/// ----------
/// bytes : :class:`savant_rs.utils.ByteBuffer`
///   The byte array to load the message from.
///
/// Returns
/// -------
/// savant_rs.primitives.Message
///   The loaded message.
///
#[pyfunction]
#[pyo3(name = "load_message_from_bytebuffer")]
pub fn load_message_from_bytebuffer_gil(buffer: &ByteBuffer) -> Message {
    release_gil(|| load_message(buffer.bytes()))
}

/// Loads a message from a python bytes. The function is GIL-free.
///
/// Parameters
/// ----------
/// bytes : bytes
///   The byte buffer to load the message from.
///
/// Returns
/// -------
/// savant_rs.primitives.Message
///   The loaded message.
///
#[pyfunction]
#[pyo3(name = "load_message_from_bytes")]
pub fn load_message_from_bytes_gil(buffer: &PyBytes) -> Message {
    let bytes = buffer.as_bytes();
    release_gil(|| load_message(bytes))
}

#[cfg(test)]
mod tests {
    use crate::primitives::message::loader::load_message;

    #[test]
    fn test_load_short() {
        pyo3::prepare_freethreaded_python();
        let f = load_message(&vec![0; 1]);
        assert!(f.is_unknown());
    }

    #[test]
    fn test_load_invalid() {
        pyo3::prepare_freethreaded_python();
        let f = load_message(&vec![0; 5]);
        assert!(f.is_unknown());
    }

    #[test]
    fn test_load_invalid_marker() {
        pyo3::prepare_freethreaded_python();
        let f = load_message(&vec![0, 0, 0, 0, 1, 2, 3, 4]);
        assert!(f.is_unknown());
    }
}
