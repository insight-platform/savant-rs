use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
#[pyclass]
pub struct Internal {
    pub v: i64,
    pub x: [u8; 128],
}

#[pymethods]
impl Internal {
    pub fn inc(&mut self) {
        self.v += 1;
    }

    pub fn val(&self) -> i64 {
        self.v
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct InternalMtx {
    pub v: Arc<Mutex<(i64, [u8; 128])>>,
}

#[pymethods]
impl InternalMtx {
    pub fn inc(&mut self) {
        let mut l = self.v.lock().unwrap();
        l.0 += 1;
    }

    pub fn val(&self) -> i64 {
        let l = self.v.lock().unwrap();
        l.0
    }
}

#[derive(Debug)]
#[pyclass]
pub struct InternalNoClone {
    pub v: i64,
    pub x: [u8; 128],
}

#[pymethods]
impl InternalNoClone {
    pub fn inc(&mut self) {
        self.v += 1;
    }

    pub fn val(&self) -> i64 {
        self.v
    }
}

#[derive(Debug)]
#[pyclass]
pub struct Wrapper {
    pub v: Py<InternalNoClone>,
}

#[pymethods]
impl Wrapper {
    #[new]
    pub fn new(v: i64) -> Self {
        Python::with_gil(|py| Self {
            v: Py::new(py, InternalNoClone { v, x: [0; 128] }).unwrap(),
        })
    }

    pub fn get(&self) -> Py<InternalNoClone> {
        Python::with_gil(|py| self.v.clone_ref(py))
    }

    pub fn inc(&self) {
        Python::with_gil(|py| {
            let mut v = self.v.as_ref(py).borrow_mut();
            v.v += 1;
        });
    }
}

#[derive(Debug)]
#[pyclass]
pub struct CopyWrapper {
    pub v: InternalMtx,
}

#[pymethods]
impl CopyWrapper {
    #[new]
    pub fn new(v: i64) -> Self {
        Self {
            v: InternalMtx {
                v: Arc::new(Mutex::new((v, [0; 128]))),
            },
        }
    }

    pub fn get(&self) -> InternalMtx {
        self.v.clone()
    }
}

#[derive(Debug)]
#[pyclass]
pub struct TakeWrapper {
    pub v: Option<Internal>,
}

#[pymethods]
impl TakeWrapper {
    #[new]
    pub fn new(v: i64) -> Self {
        Self {
            v: Some(Internal { v, x: [0; 128] }),
        }
    }

    pub fn get(&mut self) -> Internal {
        self.v.take().unwrap()
    }

    pub fn set(&mut self, v: Internal) {
        self.v = Some(v);
    }
}

#[derive(Debug)]
#[pyclass]
pub struct ProxyWrapper {
    pub v: Internal,
}

#[pymethods]
impl ProxyWrapper {
    #[new]
    pub fn new(v: i64) -> Self {
        Self {
            v: Internal { v, x: [0; 128] },
        }
    }

    pub fn get(&mut self) -> Internal {
        self.v.clone()
    }

    pub fn inc(&mut self) {
        self.v.v += 1;
    }
}
