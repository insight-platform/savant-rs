use pyo3::{pyfunction, Python};

use crate::primitives::message::{NativeMessage, NativeMessageMarkerType, NativeMessageTypeConsts};
use crate::primitives::Message;

#[pyfunction]
pub fn save_message(frame: Message) -> Vec<u8> {
    Python::with_gil(|py| {
        py.allow_threads(|| match frame.frame {
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
            NativeMessage::VideoFrame(s) => {
                let mut buf = Vec::with_capacity(760);
                let mut inner = s.inner.lock().unwrap();
                inner.prepare_before_save();
                buf.extend_from_slice(
                    rkyv::to_bytes::<_, 756>(inner.as_ref())
                        .expect("Failed to serialize VideoFrame")
                        .as_ref(),
                );
                let t: NativeMessageMarkerType = NativeMessageTypeConsts::VideoFrame.into();
                buf.extend_from_slice(t.as_ref());
                buf
            }
            _ => {
                let mut buf = Vec::with_capacity(4);
                let t: NativeMessageMarkerType = NativeMessageTypeConsts::Unknown.into();
                buf.extend_from_slice(t.as_ref());
                buf
            }
        })
    })
}
