use crate::primitives::message::Message;
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
pub fn load_message(bytes: Vec<u8>) -> Message {
    let m = savant_core::message::load_message(&bytes);
    Message(m)
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
    release_gil!(no_gil, || Message(savant_core::message::load_message(
        buffer.bytes()
    )))
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
    release_gil!(no_gil, || Message(savant_core::message::load_message(
        bytes
    )))
}
