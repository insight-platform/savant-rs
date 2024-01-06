use pyo3::types::PyBytes;
use pyo3::{pyfunction, PyObject, PyResult};
use savant_core::fast_hash;

use crate::primitives::message::Message;
use crate::release_gil;
use crate::utils::byte_buffer::ByteBuffer;
use crate::with_gil;

/// Save a message to a byte array. The function is optionally GIL-free.
///
/// Parameters
/// ----------
/// message: savant_rs.primitives.Message
///   The message to save
/// no_gil: bool
///   Whether to release the GIL while saving the message
///
/// Returns
/// -------
/// bytes
///   The byte array containing the message
///
#[pyfunction]
#[pyo3(name = "save_message")]
#[pyo3(signature = (message, no_gil=true))]
pub fn save_message_gil(message: &Message, no_gil: bool) -> PyResult<Vec<u8>> {
    release_gil!(no_gil, || {
        savant_core::message::save_message(&message.0)
            .map_err(|e| pyo3::exceptions::PyException::new_err(format!("{:?}", e)))
    })
}

/// Save a message to a byte array. The function is optionally GIL-free.
///
/// Parameters
/// ----------
/// message: savant_rs.primitives.Message
///   The message to save
/// with_hash: bool
///   Whether to include a hash of the message in the returned byte buffer
/// no_gil: bool
///   Whether to release the GIL while saving the message
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
) -> PyResult<ByteBuffer> {
    release_gil!(no_gil, || {
        let m = savant_core::message::save_message(&message.0)
            .map_err(|e| pyo3::exceptions::PyException::new_err(format!("{:?}", e)))?;
        let hash_opt = if with_hash { Some(fast_hash(&m)) } else { None };
        Ok(ByteBuffer::new(m, hash_opt))
    })
}

/// Save a message to python bytes. The function is optionally GIL-free.
///
/// Parameters
/// ----------
/// message: savant_rs.primitives.Message
///   The message to save
/// no_gil: bool
///   Whether to release the GIL while saving the message
///
/// Returns
/// -------
/// bytes
///   The byte buffer containing the message
///
#[pyfunction]
#[pyo3(name = "save_message_to_bytes")]
#[pyo3(signature = (message, no_gil=true))]
pub fn save_message_to_bytes_gil(message: &Message, no_gil: bool) -> PyResult<PyObject> {
    let bytes = release_gil!(no_gil, || savant_core::message::save_message(&message.0))
        .map_err(|e| pyo3::exceptions::PyException::new_err(format!("{:?}", e)))?;
    with_gil!(|py| {
        let bytes = PyBytes::new(py, &bytes);
        Ok(PyObject::from(bytes))
    })
}
