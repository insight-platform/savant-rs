use pyo3::prelude::*;
use std::sync::atomic::AtomicU64;

#[pyclass]
pub struct AtomicCounter(AtomicU64);

#[pymethods]
impl AtomicCounter {
    #[new]
    fn new(initial: u64) -> Self {
        AtomicCounter(AtomicU64::new(initial))
    }

    fn set(&self, value: u64) {
        self.0.store(value, std::sync::atomic::Ordering::SeqCst);
    }

    #[getter]
    fn next(&self) -> u64 {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    #[getter]
    fn get(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}
