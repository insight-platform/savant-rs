use crate::primitives::message::video::frame::VideoFrame;
use crate::primitives::message::{MessageHeader, NativeMessageTypeConsts};
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
    let header_size = std::mem::size_of::<MessageHeader>();
    if bytes.len() < header_size {
        return Message::unknown(format!(
            "Message is too short: {} < {}",
            bytes.len(),
            header_size
        ));
    }
    let final_length = bytes.len().saturating_sub(header_size);

    let header: MessageHeader = *bytemuck::from_bytes::<MessageHeader>(&bytes[final_length..]);

    if header.lib_version != crate::version_to_bytes_le() {
        return Message::unknown(format!(
            "Message CRC32 version mismatch: {:?} != {:?}. Expected version: {}",
            header.lib_version,
            crate::version_crc32(),
            crate::version()
        ));
    }

    let typ = NativeMessageTypeConsts::from(&header.native_message_type);

    let bytes = &bytes[..final_length];
    let mut m = match typ {
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
    };

    m.header.labels = header.labels;
    m.header.trace_id = header.trace_id;

    m
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
