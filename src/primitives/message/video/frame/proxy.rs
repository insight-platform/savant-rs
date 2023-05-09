use crate::primitives::VideoFrame;
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult, Python};
use std::sync::{Arc, Mutex};

#[pyclass]
#[derive(Debug, Clone)]
pub struct ProxyVideoFrame {
    pub(crate) frame: Arc<Mutex<Box<VideoFrame>>>,
}

impl ProxyVideoFrame {
    pub fn new(object: VideoFrame) -> Self {
        ProxyVideoFrame {
            frame: Arc::new(Mutex::new(Box::new(object))),
        }
    }
}
