use crate::primitives::Object;
use pyo3::{pyclass, pymethods, Py, PyAny};
use std::sync::{Arc, Mutex};

#[pyclass]
#[derive(Debug, Clone)]
pub struct ProxyObject {
    object: Arc<Mutex<Object>>,
}

#[pymethods]
impl ProxyObject {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl ProxyObject {
    pub fn new(object: Arc<Mutex<Object>>) -> Self {
        Self { object }
    }
}
