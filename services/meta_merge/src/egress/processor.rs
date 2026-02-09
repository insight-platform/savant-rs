use crate::{
    configuration::CallbacksConfiguration,
    egress::{
        egress::{Egress, EgressError},
        merge_queue::MergeQueueError,
    },
};
use anyhow::Result;
use savant_core_py::primitives::frame::VideoFrame;

pub struct EgressProcessor {
    buffer: Egress,
    handlers: CallbacksConfiguration,
}

impl EgressProcessor {
    pub fn new(buffer: Egress, handlers: CallbacksConfiguration) -> Self {
        Self { buffer, handlers }
    }
}

impl EgressProcessor {
    pub fn process(
        &mut self,
        frame: VideoFrame,
        data: Vec<Vec<u8>>,
        labels: Vec<String>,
    ) -> Result<()> {
        // get source id and uuid from frame
        let source_id = frame.0.get_source_id();
        let uuid = frame.0.get_uuid();
        // try to take frame from buffer
        let item = self.buffer.take_frame(source_id, uuid);
        match item {
            Ok(item) => {
                // call on_merge handler
                let py_item = item.to_py()?;
                // merge
                let merged = todo!();
                self.buffer.put_frame(merged)?;
            }
            Err(EgressError::TakeFrameError(MergeQueueError::ItemNotFound(uuid))) => {
                match self.buffer.push_frame(frame, data, labels) {
                    Err(EgressError::PushFrameError(MergeQueueError::LateFrame(uuid))) => {
                        todo!("check late frame")
                    }
                    _ => todo!(),
                }
            }
            x => return x.into(),
        }
        Ok(())
    }
}
