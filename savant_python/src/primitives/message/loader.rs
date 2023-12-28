use crate::primitives::message::Message;
use crate::release_gil;
use crate::utils::byte_buffer::ByteBuffer;
use pyo3::types::PyBytes;
use pyo3::{pyfunction, PyResult};

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
    release_gil!(no_gil, || {
        let m = savant_core::message::load_message(&bytes);
        Message(m)
    })
}

/// Loads a protobuf message from a byte array. The function is optionally GIL-free.
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
/// Raises
/// ------
/// :class:`pyo3.exceptions.PyException`
///   If the message cannot be deserialized.
///
#[pyfunction]
#[pyo3(name = "load_pb_message")]
#[pyo3(signature = (bytes, no_gil = true))]
pub fn load_pb_message_gil(bytes: Vec<u8>, no_gil: bool) -> PyResult<Message> {
    release_gil!(no_gil, || {
        let m = savant_core::protobuf::deserialize(&bytes)
            .map_err(|e| pyo3::exceptions::PyException::new_err(format!("{:?}", e)))?;
        Ok(Message(m))
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
    release_gil!(no_gil, || Message(savant_core::message::load_message(
        buffer.bytes()
    )))
}

/// Loads a protobuf message from a :class:`savant_rs.utils.ByteBuffer`. The function is GIL-free.
///
/// Parameters
/// ----------
/// bytes : :class:`savant_rs.utils.ByteBuffer`
///   The protobuf byte array to load the message from.
///
/// Returns
/// -------
/// savant_rs.primitives.Message
///   The loaded message.
///
/// Raises
/// ------
/// :class:`pyo3.exceptions.PyException`
///   If the message cannot be deserialized.
///
#[pyfunction]
#[pyo3(name = "load_pb_message_from_bytebuffer")]
#[pyo3(signature = (buffer, no_gil = true))]
pub fn load_pb_message_from_bytebuffer_gil(buffer: &ByteBuffer, no_gil: bool) -> PyResult<Message> {
    release_gil!(no_gil, || Ok(Message(
        savant_core::protobuf::deserialize(buffer.bytes())
            .map_err(|e| pyo3::exceptions::PyException::new_err(format!("{:?}", e)))?
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
pub fn load_message_from_bytes_gil(buffer: &PyBytes, no_gil: bool) -> Message {
    let bytes = buffer.as_bytes();
    release_gil!(no_gil, || Message(savant_core::message::load_message(
        bytes
    )))
}

/// Loads a protobuf message from a python bytes. The function is optionally GIL-free.
///
/// Parameters
/// ----------
/// bytes : bytes
///   The protobuf byte buffer to load the message from.
/// no_gil : bool
///   Whether to release the GIL while loading the message.
///
/// Returns
/// -------
/// savant_rs.primitives.Message
///   The loaded message.
///
/// Raises
/// ------
/// :class:`pyo3.exceptions.PyException`
///   If the message cannot be deserialized.
///
#[pyfunction]
#[pyo3(name = "load_pb_message_from_bytes")]
#[pyo3(signature = (buffer, no_gil = true))]
pub fn load_pb_message_from_bytes_gil(buffer: &PyBytes, no_gil: bool) -> PyResult<Message> {
    let bytes = buffer.as_bytes();
    release_gil!(no_gil, || Ok(Message(
        savant_core::protobuf::deserialize(bytes)
            .map_err(|e| pyo3::exceptions::PyException::new_err(format!("{:?}", e)))?
    )))
}
