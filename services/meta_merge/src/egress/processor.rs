use crate::{
    configuration::CallbacksConfiguration,
    egress::{
        egress::{Egress, EgressError},
        merge_queue::{HeadReadyReason, MergeQueueError},
        payload::{EgressItem, EgressItemPy},
    },
};
use anyhow::Result;
use log::{debug, info, warn};
use pyo3::{
    types::{PyAnyMethods, PyBool, PyListMethods, PyNone},
    Py, Python,
};
use savant_core::{message::Message, transport::zeromq::NonBlockingWriter};
use savant_core_py::primitives::frame::VideoFrame;
use savant_core_py::primitives::message::Message as PyMessage;
use savant_core_py::REGISTERED_HANDLERS;

pub struct EgressProcessor {
    buffer: Egress,
    handlers: CallbacksConfiguration,
    writer: NonBlockingWriter,
}

impl EgressProcessor {
    pub fn new(
        buffer: Egress,
        handlers: CallbacksConfiguration,
        writer: NonBlockingWriter,
    ) -> Self {
        Self {
            buffer,
            handlers,
            writer,
        }
    }

    /// Process an incoming video frame from an ingress stream.
    ///
    /// If a frame with the same UUID already exists in the buffer, the merge
    /// handler is called. Otherwise the frame is pushed as a new entry.
    pub fn process_frame(
        &mut self,
        ingress_name: &str,
        topic: &str,
        frame: VideoFrame,
        data: Vec<Vec<u8>>,
        labels: Vec<String>,
    ) -> Result<()> {
        let source_id = frame.0.get_source_id();
        let uuid = frame.0.get_uuid();

        // Try to take an existing frame with the same UUID from the buffer
        let take_result = self.buffer.take_frame(source_id.clone(), uuid);
        match take_result {
            Ok(mut current_item) => {
                // Frame exists -- call the merge handler
                let ready = self.call_merge_handler(
                    ingress_name,
                    topic,
                    &mut current_item,
                    Some(frame),
                    data,
                    labels,
                )?;

                // Put the (possibly modified) frame back into the buffer
                self.buffer
                    .put_frame(source_id.clone(), current_item)
                    .map_err(|e| anyhow::anyhow!("Failed to put frame back: {}", e))?;

                if ready {
                    self.buffer.set_frame_ready(source_id, uuid)?;
                }
            }
            Err(EgressError::TakeFrameError(MergeQueueError::ItemNotFound(_))) => {
                // Frame does not exist yet -- push it
                match self
                    .buffer
                    .push_frame(frame.clone(), data.clone(), labels.clone())
                {
                    Ok(()) => {
                        // First arrival: call merge with None as incoming to let
                        // the handler initialise the frame state.
                        let mut first_item = self
                            .buffer
                            .take_frame(source_id.clone(), uuid)
                            .map_err(|e| {
                                anyhow::anyhow!("Failed to take just-pushed frame: {}", e)
                            })?;

                        let ready = self.call_merge_handler(
                            ingress_name,
                            topic,
                            &mut first_item,
                            None,
                            vec![],
                            vec![],
                        )?;

                        self.buffer
                            .put_frame(source_id.clone(), first_item)
                            .map_err(|e| anyhow::anyhow!("Failed to put frame back: {}", e))?;

                        if ready {
                            self.buffer.set_frame_ready(source_id, uuid)?;
                        }
                    }
                    Err(EgressError::PushFrameError(MergeQueueError::LateFrame(late_uuid))) => {
                        debug!(
                            target: "meta_merge::processor",
                            "Late frame detected: {:?}",
                            late_uuid
                        );
                        self.call_late_arrival_handler(frame, data, labels)?;
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            Err(e) => return Err(e.into()),
        }
        Ok(())
    }

    /// Process an incoming EOS message.
    pub fn process_eos(
        &mut self,
        source_id: String,
        data: Vec<Vec<u8>>,
        labels: Vec<String>,
    ) -> Result<()> {
        match self.buffer.push_eos(source_id, data, labels) {
            Ok(()) => Ok(()),
            Err(EgressError::PushEosError(MergeQueueError::QueueIsEmpty)) => {
                debug!(
                    target: "meta_merge::processor",
                    "EOS received but queue is empty, ignoring"
                );
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Poll for ready heads and send them to the egress socket.
    pub fn send_ready(&mut self) -> Result<()> {
        let ready_items = self
            .buffer
            .fetch_ready()
            .map_err(|e| anyhow::anyhow!("Failed to fetch ready items: {}", e))?;

        for (source_id, item, eos, reason) in ready_items {
            match reason {
                HeadReadyReason::Expired => {
                    info!(
                        target: "meta_merge::processor",
                        "Head expired for source {}", source_id
                    );
                    let message_opt = self.call_head_expire_handler(&item)?;
                    if let Some(mut message) = message_opt {
                        self.send_message(&source_id, &item, &mut message)?;
                    }
                }
                HeadReadyReason::MarkedReady => {
                    debug!(
                        target: "meta_merge::processor",
                        "Head ready for source {}", source_id
                    );
                    let message_opt = self.call_head_ready_handler(&item)?;
                    if let Some(mut message) = message_opt {
                        self.send_message(&source_id, &item, &mut message)?;
                    }
                }
            }

            // If there is an EOS attached, send it
            if let Some(eos_item) = eos {
                self.send_eos(&source_id, &eos_item)?;
            }
        }

        Ok(())
    }

    // ── Python callback helpers ──────────────────────────────────────────

    /// Call the `on_merge` Python handler.
    /// Returns `true` if the handler indicates the merge is complete (ready to send).
    fn call_merge_handler(
        &self,
        ingress_name: &str,
        topic: &str,
        current_item: &mut EgressItem,
        incoming_frame: Option<VideoFrame>,
        incoming_data: Vec<Vec<u8>>,
        incoming_labels: Vec<String>,
    ) -> Result<bool> {
        let handler_name = self.handlers.on_merge.clone();
        let current_py = current_item.to_py()?;

        let incoming_py: Option<Py<EgressItemPy>> = match incoming_frame {
            Some(frame) => {
                let item = Python::attach(|py| {
                    let data = pyo3::types::PyList::new(py, incoming_data)?.unbind();
                    let labels = pyo3::types::PyList::new(py, incoming_labels)?.unbind();
                    Py::new(
                        py,
                        EgressItemPy {
                            video_frame: frame,
                            state: pyo3::types::PyDict::new(py).unbind(),
                            data,
                            labels,
                        },
                    )
                })?;
                Some(item)
            }
            None => None,
        };

        let ready: bool = Python::attach(|py| {
            let handlers_bind = REGISTERED_HANDLERS.read();
            let handler = handlers_bind
                .get(handler_name.as_str())
                .unwrap_or_else(|| panic!("Python handler '{}' not found", handler_name));

            let result = handler.call1(
                py,
                (
                    ingress_name,
                    topic,
                    current_py.clone_ref(py),
                    incoming_py.as_ref().map(|p| p.clone_ref(py)),
                ),
            )?;

            let is_ready = result.bind(py).downcast::<PyBool>()?.extract::<bool>()?;

            // Update the current_item from the (possibly modified) Python object
            let current_bound = current_py.bind(py);
            let current_ref = current_bound.borrow();
            current_item.update_from_py(&current_ref);

            Ok::<bool, pyo3::PyErr>(is_ready)
        })?;

        Ok(ready)
    }

    /// Call the `on_late_arrival` Python handler.
    fn call_late_arrival_handler(
        &self,
        frame: VideoFrame,
        data: Vec<Vec<u8>>,
        labels: Vec<String>,
    ) -> Result<()> {
        let handler_name = self.handlers.on_late_arrival.clone();

        let item_py = Python::attach(|py| {
            let data = pyo3::types::PyList::new(py, data)?.unbind();
            let labels = pyo3::types::PyList::new(py, labels)?.unbind();
            Py::new(
                py,
                EgressItemPy {
                    video_frame: frame,
                    state: pyo3::types::PyDict::new(py).unbind(),
                    data,
                    labels,
                },
            )
        })?;

        Python::attach(|py| {
            let handlers_bind = REGISTERED_HANDLERS.read();
            let handler = handlers_bind
                .get(handler_name.as_str())
                .unwrap_or_else(|| panic!("Python handler '{}' not found", handler_name));
            handler.call1(py, (item_py.clone_ref(py),))
        })?;

        Ok(())
    }

    /// Call the `on_head_expire` Python handler.
    /// Returns `Some(Message)` if the frame should be sent, `None` to drop.
    fn call_head_expire_handler(&self, item: &EgressItem) -> Result<Option<Message>> {
        let handler_name = self.handlers.on_head_expire.clone();
        let item_py = item.to_py()?;

        self.call_message_returning_handler(&handler_name, item_py)
    }

    /// Call the `on_head_ready` Python handler.
    /// Returns `Some(Message)` if the frame should be sent, `None` to drop.
    fn call_head_ready_handler(&self, item: &EgressItem) -> Result<Option<Message>> {
        let handler_name = self.handlers.on_head_ready.clone();
        let item_py = item.to_py()?;

        self.call_message_returning_handler(&handler_name, item_py)
    }

    /// Shared helper for handlers that return `Optional[Message]`.
    fn call_message_returning_handler(
        &self,
        handler_name: &str,
        item_py: Py<EgressItemPy>,
    ) -> Result<Option<Message>> {
        let result: Option<Message> = Python::attach(|py| -> pyo3::PyResult<Option<Message>> {
            let handlers_bind = REGISTERED_HANDLERS.read();
            let handler = handlers_bind
                .get(handler_name)
                .unwrap_or_else(|| panic!("Python handler '{}' not found", handler_name));

            let result = handler.call1(py, (item_py.clone_ref(py),))?;
            let result_bound = result.bind(py);

            if result_bound.is_none() || result_bound.is_instance_of::<PyNone>() {
                return Ok(None);
            }

            let py_message: PyMessage = result_bound.extract()?;
            Ok(Some(py_message.extract()))
        })?;

        Ok(result)
    }

    // ── Egress sending helpers ───────────────────────────────────────────

    /// Send a video frame message to the egress socket.
    fn send_message(
        &mut self,
        source_id: &str,
        item: &EgressItem,
        message: &mut Message,
    ) -> Result<()> {
        // Resolve the topic (default = source_id)
        let topic = self.call_send_handler(message, item)?;
        let topic = topic.as_deref().unwrap_or(source_id);

        // Extract data bytes for the payload
        let data_bytes: Vec<Vec<u8>> = Python::attach(|py| {
            let data_list = item.data.bind(py);
            let mut bytes_vec = Vec::new();
            for elem in data_list.iter() {
                if let Ok(bytes) = elem.extract::<Vec<u8>>() {
                    bytes_vec.push(bytes);
                }
            }
            bytes_vec
        });

        let payload_slices: Vec<&[u8]> = data_bytes.iter().map(|v| v.as_slice()).collect();

        // Set routing labels from the item
        let labels: Vec<String> = Python::attach(|py| {
            let labels_list = item.labels.bind(py);
            let mut label_vec = Vec::new();
            for elem in labels_list.iter() {
                if let Ok(s) = elem.extract::<String>() {
                    label_vec.push(s);
                }
            }
            label_vec
        });
        message.set_labels(labels);

        match self.writer.send_message(topic, message, &payload_slices) {
            Ok(op_result) => {
                // Check for send result (non-blocking, best-effort)
                // Non-blocking: result may not be available immediately.
                // We do a best-effort check but don't block.
                if let Ok(Some(result)) = op_result.try_get() {
                    match result {
                        Ok(res) => {
                            debug!(
                                target: "meta_merge::processor",
                                "Message sent to egress for source {}: {:?}", source_id, res
                            );
                        }
                        Err(e) => {
                            warn!(
                                target: "meta_merge::processor",
                                "Send error for source {}: {}", source_id, e
                            );
                        }
                    }
                }
            }
            Err(e) => {
                warn!(
                    target: "meta_merge::processor",
                    "Failed to send message for source {}: {}", source_id, e
                );
            }
        }

        Ok(())
    }

    /// Send an EOS message to the egress socket.
    fn send_eos(&mut self, source_id: &str, _eos_item: &EgressItem) -> Result<()> {
        match self.writer.send_eos(source_id) {
            Ok(_op_result) => {
                info!(
                    target: "meta_merge::processor",
                    "EOS sent for source {}", source_id
                );
            }
            Err(e) => {
                warn!(
                    target: "meta_merge::processor",
                    "Failed to send EOS for source {}: {}", source_id, e
                );
            }
        }

        Ok(())
    }

    /// Call the optional `on_send` Python handler.
    /// Returns an optional topic override.
    fn call_send_handler(&self, message: &Message, item: &EgressItem) -> Result<Option<String>> {
        let handler_name = match &self.handlers.on_send {
            Some(name) => name.clone(),
            None => return Ok(None),
        };

        let py_message = PyMessage::new(message.clone());

        let result: Option<String> = Python::attach(|py| -> pyo3::PyResult<Option<String>> {
            let handlers_bind = REGISTERED_HANDLERS.read();
            let handler = handlers_bind
                .get(handler_name.as_str())
                .unwrap_or_else(|| panic!("Python handler '{}' not found", handler_name));

            // Extract state dict
            let state = item.state.clone_ref(py);
            let data = item.data.clone_ref(py);
            let labels = item.labels.clone_ref(py);

            let result = handler.call1(py, (py_message, state, data, labels))?;
            let result_bound = result.bind(py);

            if result_bound.is_none() || result_bound.is_instance_of::<PyNone>() {
                return Ok(None);
            }

            let topic: String = result_bound.extract()?;
            Ok(Some(topic))
        })?;

        Ok(result)
    }
}
