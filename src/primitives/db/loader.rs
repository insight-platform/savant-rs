use crate::primitives::db::NativeFrameTypeConsts;
use crate::primitives::Frame;
use pyo3::exceptions::PyTypeError;
use pyo3::{pyclass, pymethods, PyResult, Python};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

#[pyclass]
#[derive(Debug, Clone)]
pub struct LoadResult {
    res: Arc<Mutex<Receiver<Frame>>>,
}

#[pymethods]
impl LoadResult {
    pub fn recv(&self) -> PyResult<Frame> {
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

#[pyclass]
pub struct Loader {
    pool: rayon::ThreadPool,
}

#[pymethods]
impl Loader {
    #[new]
    pub fn new(num: usize) -> Self {
        Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(num)
                .build()
                .unwrap(),
        }
    }

    pub fn load(&self, mut bytes: Vec<u8>) -> LoadResult {
        let (tx, rx) = std::sync::mpsc::channel();
        let t = NativeFrameTypeConsts::from(&bytes[0..4]);
        bytes.drain(0..4);
        self.pool.spawn(move || {
            let f = match t {
                NativeFrameTypeConsts::EndOfStream => {
                    Frame::end_of_stream(rkyv::from_bytes(&bytes[..]).unwrap())
                }
                NativeFrameTypeConsts::VideoFrame => {
                    Frame::video_frame(rkyv::from_bytes(&bytes[..]).unwrap())
                }
            };
            tx.send(f).unwrap();
        });

        LoadResult {
            res: Arc::new(Mutex::new(rx)),
        }
    }
}
