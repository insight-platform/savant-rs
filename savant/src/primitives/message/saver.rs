use pyo3::types::PyBytes;
use pyo3::{pyfunction, PyObject};

use crate::primitives::Message;
use crate::utils::byte_buffer::ByteBuffer;
use crate::utils::python::{release_gil, with_gil};

/// Save a message to a byte array
///
/// Parameters
/// ----------
/// message: savant_rs.primitives.Message
///   The message to save
///
/// Returns
/// -------
/// bytes
///   The byte array containing the message
///
#[pyfunction]
#[pyo3(name = "save_message")]
pub fn save_message_gil(message: &Message) -> Vec<u8> {
    release_gil(|| save_message(message))
}

pub fn save_message(m: &Message) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1024);
    buf.extend_from_slice(
        rkyv::to_bytes::<_, 1024>(m)
            .expect("Failed to serialize Message")
            .as_ref(),
    );
    buf
}

/// Save a message to a byte array
///
/// Parameters
/// ----------
/// message: savant_rs.primitives.Message
///   The message to save
/// with_hash: bool
///   Whether to include a hash of the message in the returned byte buffer
///
/// Returns
/// -------
/// ByteBuffer
///   The byte buffer containing the message
///
#[pyfunction]
#[pyo3(name = "save_message_to_bytebuffer")]
#[pyo3(signature = (message, with_hash=true))]
pub fn save_message_to_bytebuffer_gil(message: &Message, with_hash: bool) -> ByteBuffer {
    release_gil(|| {
        let m = save_message(message);
        let hash_opt = if with_hash {
            Some(crc32fast::hash(&m))
        } else {
            None
        };
        ByteBuffer::new(m, hash_opt)
    })
}

/// Save a message to python bytes
///
/// Parameters
/// ----------
/// message: savant_rs.primitives.Message
///   The message to save
///
/// Returns
/// -------
/// bytes
///   The byte buffer containing the message
///
#[pyfunction]
#[pyo3(name = "save_message_to_bytes")]
pub fn save_message_to_bytes_gil(message: &Message) -> PyObject {
    let bytes = release_gil(|| save_message(message));
    with_gil(|py| {
        let bytes = PyBytes::new(py, &bytes);
        PyObject::from(bytes)
    })
}
