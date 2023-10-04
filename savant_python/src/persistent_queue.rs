use crate::release_gil;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use yaque::{ReceiverBuilder, SenderBuilder};

#[pyclass]
pub struct PersistentQueue {
    sender: yaque::Sender,
    receiver: yaque::Receiver,
}

#[pymethods]
impl PersistentQueue {
    #[new]
    #[pyo3(signature=(path, segment_size = 128*1024*1024, max_segments = None, checkpoint_period_ms = 1000))]
    fn new(
        path: &str,
        segment_size: u64,
        max_segments: Option<u64>,
        checkpoint_period_ms: u64,
    ) -> PyResult<Self> {
        let sender = SenderBuilder::default()
            .segment_size(segment_size)
            .max_queue_size(max_segments.map(|n| n * segment_size))
            .open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create sender: {}", e)))?;

        let receiver = ReceiverBuilder::default()
            .save_every(Some(std::time::Duration::from_millis(checkpoint_period_ms)))
            .open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create receiver: {}", e)))?;

        Ok(Self { sender, receiver })
    }

    fn try_push(&mut self, item: &PyBytes) -> PyResult<()> {
        self.sender
            .try_send(item.as_bytes())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to push item: {}", e)))
    }

    #[pyo3(signature = (item, no_gil = true))]
    fn push(&mut self, item: &PyBytes, no_gil: bool) -> PyResult<()> {
        let bytes = item.as_bytes();
        release_gil!(no_gil, || futures::executor::block_on(async {
            self.sender
                .send(bytes)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to push item: {}", e)))
        }))
    }

    fn try_pop(&mut self) -> PyResult<Vec<u8>> {
        self.receiver
            .try_recv()
            .map(|e| e.to_vec())
            .map_err(|_| PyRuntimeError::new_err(format!("Failed to pop item")))
    }

    #[pyo3(signature = (no_gil = true))]
    fn pop(&mut self, no_gil: bool) -> PyResult<Vec<u8>> {
        release_gil!(no_gil, || futures::executor::block_on(async {
            self.receiver
                .recv()
                .await
                .map(|e| e.to_vec())
                .map_err(|_| PyRuntimeError::new_err("Failed to pop item"))
        }))
    }
}
