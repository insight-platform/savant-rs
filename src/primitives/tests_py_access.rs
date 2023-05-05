use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct Internal {
    pub v: i64,
    pub x: [u8; 128]
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

#[derive(Debug)]
#[pyclass]
pub struct Wrapper {
    pub v: Py<Internal>,
}

#[pymethods]
impl Wrapper {
    #[new]
    pub fn new(v: i64) -> Self {
        Python::with_gil(|py| Self {
            v: Py::new(py, Internal { v, x: [0;128] }).unwrap(),
        })
    }

    pub fn get(&self) -> Py<Internal> {
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
    pub v: Internal,
}

#[pymethods]
impl CopyWrapper {
    #[new]
    pub fn new(v: i64) -> Self {
        Self {
            v: Internal { v, x: [0; 128] }
        }
    }

    pub fn get(&self) -> Internal {
        self.v.clone()
    }

    pub fn set(&mut self, v: Internal) {
        self.v = v;
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
            v: Some(Internal { v, x: [0; 128] })
        }
    }

    pub fn get(&mut self) -> Internal {
        self.v.take().unwrap()
    }

    pub fn set(&mut self, v: Internal) {
        self.v = Some(v);
    }
}
