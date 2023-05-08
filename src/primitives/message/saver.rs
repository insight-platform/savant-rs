use pyo3::{pyclass, pymethods, PyResult, Python};
use std::sync::{Arc, Mutex};

use crate::primitives::message::{NativeFrame, NativeFrameMarkerType, NativeFrameTypeConsts};
use crate::primitives::Frame;
use pyo3::exceptions::PyTypeError;
use std::sync::mpsc::Receiver;

#[pyclass]
#[derive(Debug, Clone)]
pub struct SaveResult {
    res: Arc<Mutex<Receiver<Vec<u8>>>>,
}

#[pymethods]
impl SaveResult {
    pub fn recv(&self) -> PyResult<Vec<u8>> {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                let res = self.res.lock().unwrap().recv();
                match res {
                    Ok(v) => Ok(v),
                    Err(e) => Err(PyTypeError::new_err(format!(
                        "Error receiving result: {:?}",
                        e
                    ))),
                }
            })
        })
    }
}

impl SaveResult {
    fn new(res: Receiver<Vec<u8>>) -> Self {
        Self {
            res: Arc::new(Mutex::new(res)),
        }
    }
}

#[pyclass]
pub struct Saver {
    pool: rayon::ThreadPool,
}

#[pymethods]
impl Saver {
    #[new]
    pub fn new(num: usize) -> Self {
        Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(num)
                .build()
                .unwrap(),
        }
    }

    #[staticmethod]
    pub fn default() -> Self {
        Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(num_cpus::get())
                .build()
                .unwrap(),
        }
    }

    pub fn save(&self, frame: Frame) -> SaveResult {
        let (tx, rx) = std::sync::mpsc::channel();
        self.pool.spawn(move || {
            let buf = match frame.frame {
                NativeFrame::EndOfStream(s) => {
                    let mut buf = Vec::with_capacity(32);
                    buf.extend_from_slice(
                        rkyv::to_bytes::<_, 32>(&s)
                            .expect("Failed to serialize EndOfStream")
                            .as_ref(),
                    );
                    let t: NativeFrameMarkerType = NativeFrameTypeConsts::EndOfStream.into();
                    buf.extend_from_slice(t.as_ref());
                    buf
                }
                NativeFrame::VideoFrame(mut s) => {
                    let mut buf = Vec::with_capacity(760);
                    s.prepare_before_save();
                    buf.extend_from_slice(
                        rkyv::to_bytes::<_, 756>(&*s)
                            .expect("Failed to serialize VideoFrame")
                            .as_ref(),
                    );
                    let t: NativeFrameMarkerType = NativeFrameTypeConsts::VideoFrame.into();
                    buf.extend_from_slice(t.as_ref());
                    buf
                }
                _ => {
                    let mut buf = Vec::with_capacity(4);
                    let t: NativeFrameMarkerType = NativeFrameTypeConsts::Unknown.into();
                    buf.extend_from_slice(t.as_ref());
                    buf
                }
            };
            let _ = tx.send(buf);
        });

        SaveResult::new(rx)
    }
}
