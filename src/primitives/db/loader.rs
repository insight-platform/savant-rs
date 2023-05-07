use crate::primitives::db::{
    NativeFrameMarkerType, NativeFrameTypeConsts, NATIVE_FRAME_MARKER_LEN,
};
use crate::primitives::{Frame, VideoFrame};
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

    #[staticmethod]
    pub fn default() -> Self {
        Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(num_cpus::get())
                .build()
                .unwrap(),
        }
    }

    pub fn load(&self, mut bytes: Vec<u8>) -> LoadResult {
        let (tx, rx) = std::sync::mpsc::channel();
        let final_length = bytes.len().saturating_sub(NATIVE_FRAME_MARKER_LEN);
        let t = NativeFrameTypeConsts::from(
            <&NativeFrameMarkerType>::try_from(&bytes[final_length..]).unwrap(),
        );
        bytes.truncate(final_length);
        self.pool.spawn(move || {
            let f = match t {
                NativeFrameTypeConsts::EndOfStream => {
                    Frame::end_of_stream(rkyv::from_bytes(&bytes[..]).unwrap())
                }
                NativeFrameTypeConsts::VideoFrame => {
                    let mut f: VideoFrame = rkyv::from_bytes(&bytes[..]).unwrap();
                    f.prepare_after_load();
                    Frame::video_frame(f)
                }
                NativeFrameTypeConsts::Unknown => Frame::unknown(),
            };
            tx.send(f).unwrap();
        });

        LoadResult {
            res: Arc::new(Mutex::new(rx)),
        }
    }
}
