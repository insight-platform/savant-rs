use crate::primitives::message::{
    NativeFrameMarkerType, NativeFrameTypeConsts, NATIVE_FRAME_MARKER_LEN,
};
use crate::primitives::{EndOfStream, Frame, VideoFrame};
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
        if bytes.len() < NATIVE_FRAME_MARKER_LEN {
            tx.send(Frame::unknown()).unwrap();
        } else {
            let final_length = bytes.len().saturating_sub(NATIVE_FRAME_MARKER_LEN);
            let t = NativeFrameTypeConsts::from(
                <&NativeFrameMarkerType>::try_from(&bytes[final_length..]).unwrap(),
            );
            bytes.truncate(final_length);
            self.pool.spawn(move || {
                let f = match t {
                    NativeFrameTypeConsts::EndOfStream => {
                        let eos: Result<EndOfStream, _> = rkyv::from_bytes(&bytes[..]);
                        match eos {
                            Ok(eos) => Frame::end_of_stream(eos),
                            Err(_) => Frame::unknown(),
                        }
                    }
                    NativeFrameTypeConsts::VideoFrame => {
                        let f: Result<VideoFrame, _> = rkyv::from_bytes(&bytes[..]);
                        match f {
                            Ok(mut f) => {
                                f.prepare_after_load();
                                Frame::video_frame(f)
                            }
                            Err(_) => Frame::unknown(),
                        }
                    }
                    NativeFrameTypeConsts::Unknown => Frame::unknown(),
                };
                let _ = tx.send(f);
            });
        }
        LoadResult {
            res: Arc::new(Mutex::new(rx)),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_load_short() {
        pyo3::prepare_freethreaded_python();
        let loader = super::Loader::default();
        let res = loader.load(vec![0; 1]);
        let f = res.recv().unwrap();
        assert!(f.is_unknown());
    }

    #[test]
    fn test_load_invalid() {
        pyo3::prepare_freethreaded_python();
        let loader = super::Loader::default();
        let res = loader.load(vec![0; 5]);
        let f = res.recv().unwrap();
        assert!(f.is_unknown());
    }

    #[test]
    fn test_load_invalid_marker() {
        pyo3::prepare_freethreaded_python();
        let loader = super::Loader::default();
        let res = loader.load(vec![0, 0, 0, 0, 1, 2, 3, 4]);
        let f = res.recv().unwrap();
        assert!(f.is_unknown());
    }
}
