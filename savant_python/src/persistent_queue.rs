use crate::{release_gil, with_gil};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass]
pub struct PersistentQueue {
    queue: savant_pq::PersistentQueue,
}

#[pymethods]
impl PersistentQueue {
    #[new]
    #[pyo3(signature=(path, max_elements = 1000000000))]
    fn new(path: String, max_elements: u128) -> PyResult<Self> {
        let queue = savant_pq::PersistentQueue::new(path, max_elements).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create persistent queue: {}", e))
        })?;
        Ok(Self { queue })
    }

    #[pyo3(signature = (item, no_gil = true))]
    fn push(&mut self, item: &PyBytes, no_gil: bool) -> PyResult<()> {
        let bytes = item.as_bytes();
        release_gil!(no_gil, || self.queue.push(bytes).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to push item: {}", e))
        }))
    }

    #[pyo3(signature = (no_gil = true))]
    fn pop(&mut self, no_gil: bool) -> PyResult<Option<PyObject>> {
        release_gil!(no_gil, || self
            .queue
            .pop()
            .map(|e| with_gil!(|py| e.map(|e| PyObject::from(PyBytes::new(py, e.as_slice())))))
            .map_err(|_| PyRuntimeError::new_err("Failed to pop item")))
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn size(&self) -> PyResult<u64> {
        self.queue
            .size()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get queue size: {}", e)))
    }

    fn len(&self) -> u128 {
        self.queue.len()
    }

    #[staticmethod]
    fn remove_db(path: String) -> PyResult<()> {
        savant_pq::PersistentQueue::remove_db(path).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to remove persistent queue: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn nothing() {
        assert!(true);
    }
}
