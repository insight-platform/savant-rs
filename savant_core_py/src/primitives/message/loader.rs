use crate::detach;
use crate::primitives::message::Message;
use crate::utils::byte_buffer::ByteBuffer;
use pyo3::types::{PyBytes, PyBytesMethods};
use pyo3::{pyfunction, Bound};

/// Loads a message from a byte array. The function is optionally GIL-free.
///
/// Parameters
/// ----------
/// bytes : bytes
///   The byte array to load the message from.
/// no_gil : bool
///   Whether to release the GIL while loading the message.
///
/// Returns
/// -------
/// savant_rs.primitives.Message
///   The loaded message.
///
#[pyfunction]
#[pyo3(name = "load_message")]
#[pyo3(signature = (bytes, no_gil = true))]
pub fn load_message_gil(bytes: Vec<u8>, no_gil: bool) -> Message {
    detach!(no_gil, || {
        let m = savant_core::message::load_message(&bytes);
        Message(m)
    })
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
    detach!(no_gil, || Message(savant_core::message::load_message(
        buffer.bytes()
    )))
}

/// Loads a message from a python bytes. The function is optionally GIL-free.
///
/// Parameters
/// ----------
/// bytes : bytes
///   The byte buffer to load the message from.
/// no_gil : bool
///   Whether to release the GIL while loading the message.
///
/// Returns
/// -------
/// savant_rs.primitives.Message
///   The loaded message.
///
#[pyfunction]
#[pyo3(name = "load_message_from_bytes")]
#[pyo3(signature = (buffer, no_gil = true))]
pub fn load_message_from_bytes_gil(buffer: &Bound<'_, PyBytes>, no_gil: bool) -> Message {
    let bytes = buffer.as_bytes();
    detach!(no_gil, || Message(savant_core::message::load_message(
        bytes
    )))
}
