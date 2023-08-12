use crate::primitives::message::MessageEnvelope;
use crate::primitives::Message;
use crate::release_gil;
use crate::utils::byte_buffer::ByteBuffer;
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
    load_message(&bytes)
}

pub fn load_message(bytes: &[u8]) -> Message {
    let m: Result<Message, _> = rkyv::from_bytes(bytes);

    if m.is_err() {
        return Message::unknown(format!("{:?}", m.err().unwrap()));
    }

    let mut m = m.unwrap();

    if m.meta.lib_version != crate::version_to_bytes_le() {
        return Message::unknown(format!(
            "Message CRC32 version mismatch: {:?} != {:?}. Expected version: {}",
            m.meta.lib_version,
            crate::version_crc32(),
            crate::version()
        ));
    }

    match &mut m.payload {
        MessageEnvelope::VideoFrame(f) => {
            f.restore();
        }
        MessageEnvelope::VideoFrameBatch(b) => {
            b.prepare_after_load();
        }
        _ => {}
    }

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
#[pyo3(signature = (buffer, no_gil = true))]
pub fn load_message_from_bytebuffer_gil(buffer: &ByteBuffer, no_gil: bool) -> Message {
    release_gil!(no_gil, || load_message(buffer.bytes()))
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
#[pyo3(signature = (buffer, no_gil = true))]
pub fn load_message_from_bytes_gil(buffer: &PyBytes, no_gil: bool) -> Message {
    let bytes = buffer.as_bytes();
    release_gil!(no_gil, || load_message(bytes))
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
