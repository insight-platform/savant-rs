use crate::primitives::attribute::{AttributeMethods, Attributive};
use pyo3::types::PyBytes;
use pyo3::{pyfunction, PyObject, Python};
use std::collections::HashMap;

use crate::primitives::message::video::query::match_query::MatchQuery;
use crate::primitives::message::{NativeMessage, TRACE_ID_LEN};
use crate::primitives::Message;
use crate::utils::byte_buffer::ByteBuffer;
use crate::utils::python::release_gil;

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
    release_gil(|| save_message(message.clone()))
}

pub fn save_message(m: Message) -> Vec<u8> {
    match m.payload {
        NativeMessage::EndOfStream(s) => {
            let mut buf = Vec::with_capacity(32);
            buf.extend_from_slice(
                rkyv::to_bytes::<_, 32>(&s)
                    .expect("Failed to serialize EndOfStream")
                    .as_ref(),
            );
            buf.extend_from_slice(bytemuck::bytes_of(&m.header));
            buf
        }

        NativeMessage::Telemetry(mut t) => {
            t.exclude_temporary_attributes();
            let mut buf = Vec::with_capacity(32);
            buf.extend_from_slice(
                rkyv::to_bytes::<_, 32>(&t)
                    .expect("Failed to serialize Telemetry")
                    .as_ref(),
            );
            buf.extend_from_slice(bytemuck::bytes_of(&m.header));
            buf
        }

        NativeMessage::VideoFrameUpdate(update) => {
            let mut buf = Vec::with_capacity(256);
            buf.extend_from_slice(
                rkyv::to_bytes::<_, 256>(&update)
                    .expect("Failed to serialize VideoFrameUpdate")
                    .as_ref(),
            );
            buf.extend_from_slice(bytemuck::bytes_of(&m.header));
            buf
        }

        NativeMessage::VideoFrame(frame) => {
            let mut buf = Vec::with_capacity(760);
            frame.make_snapshot();
            let frame_excluded_temp_attrs = frame.exclude_temporary_attributes();

            let objects_excluded_temp_attrs = frame
                .access_objects(&MatchQuery::Idle)
                .iter()
                .map(|o| (o.get_id(), o.exclude_temporary_attributes()))
                .collect::<HashMap<_, _>>();

            let inner = frame.inner.read_recursive();
            buf.extend_from_slice(
                rkyv::to_bytes::<_, 756>(inner.as_ref())
                    .expect("Failed to serialize VideoFrame")
                    .as_ref(),
            );
            if m.header.trace_id == [0; TRACE_ID_LEN] {
                let mut h = m.header.clone();
                h.trace_id = frame.get_trace_id();
                buf.extend_from_slice(bytemuck::bytes_of(&h));
            } else {
                buf.extend_from_slice(bytemuck::bytes_of(&m.header));
            }
            drop(inner);

            frame.restore_attributes(frame_excluded_temp_attrs);

            objects_excluded_temp_attrs
                .into_iter()
                .for_each(|(id, attrs)| {
                    if let Some(o) = frame.get_object(id) {
                        o.restore_attributes(attrs)
                    }
                });

            buf
        }

        NativeMessage::VideoFrameBatch(mut b) => {
            let mut buf = Vec::with_capacity(760 * b.frames.len());
            b.prepare_before_save();
            buf.extend_from_slice(
                rkyv::to_bytes::<_, 756>(&b)
                    .expect("Failed to serialize VideoFrame")
                    .as_ref(),
            );
            buf.extend_from_slice(bytemuck::bytes_of(&m.header));
            buf
        }
        _ => {
            let mut buf = Vec::new();
            buf.extend_from_slice(bytemuck::bytes_of(&m.header));
            buf
        }
    }
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
pub fn save_message_to_bytebuffer_gil(message: Message, with_hash: bool) -> ByteBuffer {
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
pub fn save_message_to_bytes_gil(message: Message) -> PyObject {
    let bytes = release_gil(|| save_message(message));
    Python::with_gil(|py| {
        let bytes = PyBytes::new(py, &bytes);
        PyObject::from(bytes)
    })
}
