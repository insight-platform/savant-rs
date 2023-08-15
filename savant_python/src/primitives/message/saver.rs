use pyo3::types::PyBytes;
use pyo3::{pyfunction, PyObject};
use savant_core::fast_hash;

use crate::primitives::Message;
use crate::release_gil;
use crate::utils::byte_buffer::ByteBuffer;
use crate::with_gil;

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
pub fn save_message(m: &Message) -> Vec<u8> {
    savant_core::message::save_message(&m.0)
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
#[pyo3(signature = (message, with_hash=true, no_gil=true))]
pub fn save_message_to_bytebuffer_gil(
    message: &Message,
    with_hash: bool,
    no_gil: bool,
) -> ByteBuffer {
    release_gil!(no_gil, || {
        let m = save_message(message);
        let hash_opt = if with_hash { Some(fast_hash(&m)) } else { None };
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
#[pyo3(signature = (message, no_gil=true))]
pub fn save_message_to_bytes_gil(message: &Message, no_gil: bool) -> PyObject {
    let bytes = release_gil!(no_gil, || save_message(message));
    with_gil!(|py| {
        let bytes = PyBytes::new(py, &bytes);
        PyObject::from(bytes)
    })
}
