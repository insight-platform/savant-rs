use crate::primitives::attribute::AttributeMethods;
use pyo3::pyfunction;
use std::collections::HashMap;

use crate::primitives::message::video::query::Query;
use crate::primitives::message::{NativeMessage, NativeMessageMarkerType, NativeMessageTypeConsts};
use crate::primitives::Message;
use crate::utils::python::no_gil;

#[pyfunction]
#[pyo3(name = "save_message")]
pub fn save_message_gil(frame: Message) -> Vec<u8> {
    no_gil(|| save_message(frame))
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
            let t: NativeMessageMarkerType = NativeMessageTypeConsts::EndOfStream.into();
            buf.extend_from_slice(t.as_ref());
            buf
        }
        NativeMessage::VideoFrame(frame) => {
            let mut buf = Vec::with_capacity(760);
            frame.make_snapshot();
            let frame_excluded_temp_attrs = frame.exclude_temporary_attributes();

            let objects_excluded_temp_attrs = frame
                .access_objects(&Query::Idle)
                .iter()
                .map(|o| (o.get_id(), o.exclude_temporary_attributes()))
                .collect::<HashMap<_, _>>();

            let inner = frame.inner.read_recursive();
            buf.extend_from_slice(
                rkyv::to_bytes::<_, 756>(inner.as_ref())
                    .expect("Failed to serialize VideoFrame")
                    .as_ref(),
            );
            let t: NativeMessageMarkerType = NativeMessageTypeConsts::VideoFrame.into();
            buf.extend_from_slice(t.as_ref());
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
            let t: NativeMessageMarkerType = NativeMessageTypeConsts::VideoFrameBatch.into();
            buf.extend_from_slice(t.as_ref());
            buf
        }
        _ => {
            let mut buf = Vec::with_capacity(4);
            let t: NativeMessageMarkerType = NativeMessageTypeConsts::Unknown.into();
            buf.extend_from_slice(t.as_ref());
            buf
        }
    }
}
