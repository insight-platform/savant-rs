pub mod pipeline_py;

use crate::primitives::message::TRACE_ID_LEN;
use crate::primitives::{VideoFrameBatch, VideoFrameProxy, VideoFrameUpdate};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::SystemTime;

#[pyclass]
#[derive(Clone, Debug, Copy)]
pub enum VideoPipelineTelemetryMessageType {
    Add,
    Move,
    Delete,
}

#[pymethods]
impl VideoPipelineTelemetryMessageType {
    fn __repr__(&self) -> String {
        match self {
            Add => "Add".to_owned(),
            Move => "Move".to_owned(),
            Delete => "Delete".to_owned(),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;
}

use VideoPipelineTelemetryMessageType::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct VideoPipelineTelemetryMessage {
    pub trace_id: [u8; TRACE_ID_LEN],
    pub timestamp_micro: u128,
    pub stage: String,
    pub message_type: VideoPipelineTelemetryMessageType,
}

#[pymethods]
impl VideoPipelineTelemetryMessage {
    fn __repr__(&self) -> String {
        format!(
            "VideoPipelineTelemetryMessage(trace_id={}, timestamp_micro={}, stage={}, message_type={})",
            self.trace_id.to_vec().iter().map(|b| format!("{:02x}", b)).collect::<Vec<String>>().join(""),
            self.timestamp_micro,
            self.stage,
            self.message_type.__repr__()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    #[new]
    pub fn new(
        trace_id: [u8; TRACE_ID_LEN],
        stage: &str,
        message_type: VideoPipelineTelemetryMessageType,
    ) -> Self {
        Self {
            trace_id,
            message_type,
            stage: stage.to_owned(),
            timestamp_micro: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_micros(),
        }
    }

    #[getter]
    pub fn get_trace_id(&self) -> Vec<u8> {
        self.trace_id.to_vec()
    }

    #[getter]
    pub fn get_timestamp_micro(&self) -> u128 {
        self.timestamp_micro
    }

    #[getter]
    pub fn get_stage(&self) -> String {
        self.stage.clone()
    }
}

#[derive(Clone, Debug, Default)]
pub struct VideoPipelineStage {
    payload: HashMap<i64, VideoPipelinePayload>,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub enum VideoPipelineStagePayloadType {
    Frame,
    Batch,
}

#[derive(Clone, Debug)]
pub enum VideoPipelinePayload {
    Frame(VideoFrameProxy, Vec<VideoFrameUpdate>),
    Batch(VideoFrameBatch, Vec<(i64, VideoFrameUpdate)>),
}

#[derive(Clone, Debug, Default)]
pub struct VideoPipeline {
    pub id_counter: i64,
    pub stages: HashMap<String, VideoPipelineStage>,
    pub stage_types: HashMap<String, VideoPipelineStagePayloadType>,
    pub telemetry: Vec<VideoPipelineTelemetryMessage>,
}

impl VideoPipeline {
    pub fn add_stage(
        &mut self,
        name: &str,
        stage_type: VideoPipelineStagePayloadType,
    ) -> anyhow::Result<()> {
        if self.stages.contains_key(name) {
            anyhow::bail!("Stage already exists")
        }
        self.stages
            .insert(name.to_owned(), VideoPipelineStage::default());
        self.stage_types.insert(name.to_owned(), stage_type);
        Ok(())
    }

    fn add_telemetry(
        &mut self,
        trace_id: [u8; TRACE_ID_LEN],
        stage: &str,
        message_type: VideoPipelineTelemetryMessageType,
    ) {
        self.telemetry.push(VideoPipelineTelemetryMessage::new(
            trace_id,
            stage,
            message_type,
        ));
    }

    fn build_telemetry(
        &mut self,
        stage_name: &str,
        telemetry_type: VideoPipelineTelemetryMessageType,
        payload: &VideoPipelinePayload,
    ) {
        match payload {
            VideoPipelinePayload::Frame(frame, _) => {
                self.add_telemetry(frame.get_trace_id(), stage_name, telemetry_type);
            }
            VideoPipelinePayload::Batch(batch, _) => {
                for frame in batch.frames.values() {
                    self.add_telemetry(frame.get_trace_id(), stage_name, telemetry_type);
                }
            }
        }
    }

    pub fn retrieve_telemetry(&mut self) -> Vec<VideoPipelineTelemetryMessage> {
        self.telemetry.drain(..).collect()
    }

    pub fn get_stage(&self, name: &str) -> Option<&VideoPipelineStage> {
        self.stages.get(name)
    }

    pub fn get_stage_mut(&mut self, name: &str) -> Option<&mut VideoPipelineStage> {
        self.stages.get_mut(name)
    }

    pub fn get_stage_type(&self, name: &str) -> Option<&VideoPipelineStagePayloadType> {
        self.stage_types.get(name)
    }

    pub fn add_frame_update(
        &mut self,
        stage: &str,
        frame_id: i64,
        update: VideoFrameUpdate,
    ) -> anyhow::Result<()> {
        if let Some(stage) = self.get_stage_mut(stage) {
            if let Some(payload) = stage.payload.get_mut(&frame_id) {
                match payload {
                    VideoPipelinePayload::Frame(_, updates) => {
                        updates.push(update);
                    }
                    _ => anyhow::bail!("Frame update can only be added to a frame payload"),
                }
            } else {
                anyhow::bail!("Frame not found in stage")
            }
        } else {
            anyhow::bail!("Stage not found")
        }
        Ok(())
    }

    pub fn add_batched_frame_update(
        &mut self,
        stage: &str,
        batch_id: i64,
        frame_id: i64,
        update: VideoFrameUpdate,
    ) -> anyhow::Result<()> {
        if let Some(stage) = self.get_stage_mut(stage) {
            if let Some(payload) = stage.payload.get_mut(&batch_id) {
                match payload {
                    VideoPipelinePayload::Batch(_, updates) => {
                        updates.push((frame_id, update));
                    }
                    _ => anyhow::bail!("Batch update can only be added to a batch payload"),
                }
            } else {
                anyhow::bail!("Batch not found in stage")
            }
        } else {
            anyhow::bail!("Stage not found")
        }
        Ok(())
    }

    pub fn add_frame(&mut self, stage_name: &str, frame: VideoFrameProxy) -> anyhow::Result<i64> {
        if matches!(
            self.get_stage_type(stage_name),
            Some(VideoPipelineStagePayloadType::Batch)
        ) {
            anyhow::bail!("Stage does not accept batched frames")
        }

        let id_counter = self.id_counter + 1;
        self.add_telemetry(frame.get_trace_id(), stage_name, Add);
        let frame_payload = VideoPipelinePayload::Frame(frame, Vec::new());
        if let Some(stage) = self.get_stage_mut(stage_name) {
            stage.payload.insert(id_counter, frame_payload);
        } else {
            anyhow::bail!("Stage not found")
        }
        self.id_counter = id_counter;
        Ok(self.id_counter)
    }

    pub fn add_batch(&mut self, stage_name: &str, batch: VideoFrameBatch) -> anyhow::Result<i64> {
        if matches!(
            self.get_stage_type(stage_name),
            Some(VideoPipelineStagePayloadType::Frame)
        ) {
            anyhow::bail!("Stage does not accept independent frames")
        }
        let id_counter = self.id_counter + 1;
        let batch_payload = VideoPipelinePayload::Batch(batch, Vec::new());
        self.build_telemetry(stage_name, Add, &batch_payload);
        if let Some(stage) = self.get_stage_mut(stage_name) {
            stage.payload.insert(id_counter, batch_payload);
        } else {
            anyhow::bail!("Stage not found")
        }
        self.id_counter = id_counter;
        Ok(self.id_counter)
    }

    pub fn delete(&mut self, stage_name: &str, id: i64) -> anyhow::Result<()> {
        if let Some(stage) = self.get_stage_mut(stage_name) {
            let removed = stage.payload.remove(&id);
            if let Some(removed) = removed {
                self.build_telemetry(stage_name, Delete, &removed);
            } else {
                anyhow::bail!("Object not found in stage")
            }
        } else {
            anyhow::bail!("Stage not found")
        }
        Ok(())
    }

    pub fn get_stage_queue_len(&self, stage: &str) -> anyhow::Result<usize> {
        if let Some(stage) = self.get_stage(stage) {
            Ok(stage.payload.len())
        } else {
            anyhow::bail!("Stage not found")
        }
    }

    pub fn get_independent_frame(
        &self,
        stage: &str,
        frame_id: i64,
    ) -> anyhow::Result<VideoFrameProxy> {
        if let Some(stage) = self.get_stage(stage) {
            if let Some(payload) = stage.payload.get(&frame_id) {
                match payload {
                    VideoPipelinePayload::Frame(frame, _) => Ok(frame.clone()),
                    _ => anyhow::bail!("Payload must be a frame"),
                }
            } else {
                anyhow::bail!("Frame not found in stage")
            }
        } else {
            anyhow::bail!("Stage not found")
        }
    }

    pub fn get_batched_frame(
        &self,
        stage: &str,
        batch_id: i64,
        frame_id: i64,
    ) -> anyhow::Result<VideoFrameProxy> {
        if let Some(stage) = self.get_stage(stage) {
            if let Some(payload) = stage.payload.get(&batch_id) {
                match payload {
                    VideoPipelinePayload::Batch(batch, _) => {
                        if let Some(frame) = batch.get(frame_id) {
                            Ok(frame)
                        } else {
                            anyhow::bail!("Frame not found in batch")
                        }
                    }
                    _ => anyhow::bail!("Payload must be a batch"),
                }
            } else {
                anyhow::bail!("Batch not found in stage")
            }
        } else {
            anyhow::bail!("Stage not found")
        }
    }

    pub fn get_batch(&self, stage: &str, batch_id: i64) -> anyhow::Result<VideoFrameBatch> {
        if let Some(stage) = self.get_stage(stage) {
            if let Some(payload) = stage.payload.get(&batch_id) {
                match payload {
                    VideoPipelinePayload::Batch(batch, _) => Ok(batch.clone()),
                    _ => anyhow::bail!("Payload must be a batch"),
                }
            } else {
                anyhow::bail!("Batch not found in stage")
            }
        } else {
            anyhow::bail!("Stage not found")
        }
    }

    pub fn apply_updates(&mut self, stage: &str, id: i64) -> anyhow::Result<()> {
        if let Some(stage) = self.get_stage_mut(stage) {
            if let Some(payload) = stage.payload.get_mut(&id) {
                match payload {
                    VideoPipelinePayload::Frame(frame, updates) => {
                        for update in updates.drain(..) {
                            frame.update(&update)?;
                        }
                    }
                    VideoPipelinePayload::Batch(batch, updates) => {
                        for (frame_id, update) in updates.drain(..) {
                            if let Some(frame) = batch.get(frame_id) {
                                frame.update(&update)?;
                            }
                        }
                    }
                }
            } else {
                anyhow::bail!("Payload not found in stage")
            }
        } else {
            anyhow::bail!("Stage not found")
        }
        Ok(())
    }

    pub fn move_as_is(
        &mut self,
        source_stage_name: &str,
        dest_stage_name: &str,
        object_ids: Vec<i64>,
    ) -> anyhow::Result<()> {
        if self.get_stage_type(source_stage_name) != self.get_stage_type(dest_stage_name) {
            anyhow::bail!("The source stage type must be the same as the destination stage type")
        }

        let source_stage_opt = self.get_stage_mut(source_stage_name);
        if source_stage_opt.is_none() {
            anyhow::bail!("Source stage not found")
        }

        let dest_stage_opt = self.get_stage_mut(dest_stage_name);
        if dest_stage_opt.is_none() {
            anyhow::bail!("Destination stage not found")
        }

        let source_stage = self.get_stage_mut(source_stage_name).unwrap();
        let mut removed_objects = Vec::new();
        for id in object_ids {
            if let Some(payload) = source_stage.payload.remove(&id) {
                removed_objects.push((id, payload));
            } else {
                anyhow::bail!("Object not found in source stage")
            }
        }

        for o in removed_objects.iter() {
            self.build_telemetry(dest_stage_name, Move, &o.1);
        }

        let dest_stage = self.get_stage_mut(dest_stage_name).unwrap();
        for o in removed_objects {
            dest_stage.payload.insert(o.0, o.1);
        }

        Ok(())
    }

    pub fn move_and_pack_frames(
        &mut self,
        source_stage_name: &str,
        dest_stage_name: &str,
        frame_ids: Vec<i64>,
    ) -> anyhow::Result<i64> {
        if matches!(
            self.get_stage_type(source_stage_name),
            Some(VideoPipelineStagePayloadType::Batch)
        ) || matches!(
            self.get_stage_type(dest_stage_name),
            Some(VideoPipelineStagePayloadType::Frame)
        ) {
            anyhow::bail!("Source stage must contain independent frames and destination stage must contain batched frames")
        }

        let batch_id = self.id_counter + 1;
        let source_stage_opt = self.get_stage_mut(source_stage_name);
        if source_stage_opt.is_none() {
            anyhow::bail!("Source stage not found")
        }

        let dest_stage_opt = self.get_stage_mut(dest_stage_name);
        if dest_stage_opt.is_none() {
            anyhow::bail!("Destination stage not found")
        }

        let source_stage = self.get_stage_mut(source_stage_name).unwrap();

        let mut batch = VideoFrameBatch::new();
        let mut batch_updates = Vec::new();
        for id in frame_ids {
            if let Some(payload) = source_stage.payload.remove(&id) {
                match payload {
                    VideoPipelinePayload::Frame(frame, updates) => {
                        batch.add(id, frame);
                        for update in updates {
                            batch_updates.push((id, update));
                        }
                    }
                    _ => anyhow::bail!("Source stage must contain independent frames"),
                }
            }
        }

        let payload = VideoPipelinePayload::Batch(batch, batch_updates);
        self.build_telemetry(dest_stage_name, Move, &payload);
        let dest_stage = self.get_stage_mut(dest_stage_name).unwrap();
        dest_stage.payload.insert(batch_id, payload);
        self.id_counter = batch_id;
        Ok(self.id_counter)
    }

    pub fn move_and_unpack_batch(
        &mut self,
        source_stage: &str,
        dest_stage: &str,
        batch_id: i64,
    ) -> anyhow::Result<HashMap<String, i64>> {
        if matches!(
            self.get_stage_type(source_stage),
            Some(VideoPipelineStagePayloadType::Frame)
        ) || matches!(
            self.get_stage_type(dest_stage),
            Some(VideoPipelineStagePayloadType::Batch)
        ) {
            anyhow::bail!("Source stage must contain batched frames and destination stage must contain independent frames")
        }

        let source_stage_opt = self.get_stage_mut(source_stage);
        if source_stage_opt.is_none() {
            anyhow::bail!("Source stage not found")
        }

        let dest_stage_opt = self.get_stage_mut(dest_stage);
        if dest_stage_opt.is_none() {
            anyhow::bail!("Destination stage not found")
        }

        let source_stage = self.get_stage_mut(source_stage).unwrap();
        let (batch, updates) = if let Some(payload) = source_stage.payload.remove(&batch_id) {
            match payload {
                VideoPipelinePayload::Batch(batch, updates) => (batch, updates),
                _ => anyhow::bail!("Source stage must contain batch"),
            }
        } else {
            anyhow::bail!("Batch not found in source stage")
        };

        let payload = VideoPipelinePayload::Batch(batch, Vec::new());
        self.build_telemetry(dest_stage, Move, &payload);

        let batch = match payload {
            VideoPipelinePayload::Batch(batch, _) => batch,
            _ => anyhow::bail!("Payload must be a batch"),
        };

        let dest_stage = self.get_stage_mut(dest_stage).unwrap();
        let mut frame_mapping = HashMap::new();
        for (frame_id, frame) in batch.frames {
            frame_mapping.insert(frame.get_source_id(), frame_id);
            dest_stage
                .payload
                .insert(frame_id, VideoPipelinePayload::Frame(frame, Vec::new()));
        }

        for (frame_id, update) in updates {
            if let Some(frame) = dest_stage.payload.get_mut(&frame_id) {
                match frame {
                    VideoPipelinePayload::Frame(_, updates) => {
                        updates.push(update);
                    }
                    _ => anyhow::bail!("Destination stage must contain independent frames"),
                }
            } else {
                anyhow::bail!("Frame not found in destination stage")
            }
        }

        Ok(frame_mapping)
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::message::video::pipeline::{
        VideoPipeline, VideoPipelineStagePayloadType,
    };
    use crate::primitives::{AttributeBuilder, AttributeValue, VideoFrameBatch, VideoFrameUpdate};
    use crate::test::utils::gen_frame;

    fn create_pipeline() -> anyhow::Result<VideoPipeline> {
        let mut pipeline = VideoPipeline::default();
        pipeline.add_stage("input", VideoPipelineStagePayloadType::Frame)?;
        pipeline.add_stage("proc1", VideoPipelineStagePayloadType::Batch)?;
        pipeline.add_stage("proc2", VideoPipelineStagePayloadType::Batch)?;
        pipeline.add_stage("output", VideoPipelineStagePayloadType::Frame)?;
        Ok(pipeline)
    }

    #[test]
    fn test_new_pipeline() -> anyhow::Result<()> {
        let pipeline = create_pipeline()?;
        assert_eq!(pipeline.id_counter, 0);
        assert_eq!(pipeline.stages.len(), 4);
        assert_eq!(pipeline.stage_types.len(), 4);
        Ok(())
    }

    #[test]
    fn test_add_duplicate_stage() -> anyhow::Result<()> {
        let mut pipeline = create_pipeline()?;
        assert!(pipeline
            .add_stage("input", VideoPipelineStagePayloadType::Frame)
            .is_err());
        Ok(())
    }

    #[test]
    fn test_get_stage() -> anyhow::Result<()> {
        let pipeline = create_pipeline()?;
        assert!(pipeline.get_stage("input").is_some());
        assert!(pipeline.get_stage("proc1").is_some());
        assert!(pipeline.get_stage("proc2").is_some());
        assert!(pipeline.get_stage("output").is_some());
        assert!(pipeline.get_stage("unknown").is_none());
        Ok(())
    }

    #[test]
    fn test_get_stage_mut() -> anyhow::Result<()> {
        let mut pipeline = create_pipeline()?;
        assert!(pipeline.get_stage_mut("input").is_some());
        assert!(pipeline.get_stage_mut("proc1").is_some());
        assert!(pipeline.get_stage_mut("proc2").is_some());
        assert!(pipeline.get_stage_mut("output").is_some());
        assert!(pipeline.get_stage_mut("unknown").is_none());
        Ok(())
    }

    #[test]
    fn test_get_stage_type() -> anyhow::Result<()> {
        let pipeline = create_pipeline()?;
        assert!(matches!(
            pipeline.get_stage_type("input"),
            Some(VideoPipelineStagePayloadType::Frame)
        ));
        assert!(matches!(
            pipeline.get_stage_type("proc1"),
            Some(VideoPipelineStagePayloadType::Batch)
        ));
        Ok(())
    }

    #[test]
    fn test_add_del_frame() -> anyhow::Result<()> {
        let mut pipeline = create_pipeline()?;
        let id = pipeline.add_frame("input", gen_frame())?;
        assert_eq!(pipeline.get_stage_queue_len("input")?, 1);
        assert!(pipeline.add_frame("proc1", gen_frame()).is_err());
        assert_eq!(pipeline.get_stage_queue_len("proc1")?, 0);

        assert!(pipeline.delete("proc1", id).is_err());
        pipeline.delete("input", id)?;
        assert_eq!(pipeline.get_stage_queue_len("input")?, 0);

        Ok(())
    }

    #[test]
    fn test_add_del_batch() -> anyhow::Result<()> {
        let mut pipeline = create_pipeline()?;
        let id = pipeline.add_batch("proc1", VideoFrameBatch::new())?;
        assert!(pipeline.add_batch("input", VideoFrameBatch::new()).is_err());

        assert!(pipeline.delete("input", id).is_err());
        pipeline.delete("proc1", id)?;

        Ok(())
    }

    #[test]
    fn test_frame_to_batch() -> anyhow::Result<()> {
        let mut pipeline = create_pipeline()?;
        let id = pipeline.add_frame("input", gen_frame())?;
        let batch_id = pipeline.move_and_pack_frames("input", "proc1", vec![id])?;

        assert!(pipeline.get_independent_frame("input", id).is_err());

        assert!(pipeline.get_batch("proc1", batch_id).is_ok());
        assert!(pipeline.get_batched_frame("proc1", batch_id, id).is_ok());

        Ok(())
    }

    #[test]
    fn test_batch_to_frame() -> anyhow::Result<()> {
        let mut pipeline = create_pipeline()?;
        let id = pipeline.add_frame("input", gen_frame())?;
        let batch_id = pipeline.move_and_pack_frames("input", "proc2", vec![id])?;
        pipeline.move_and_unpack_batch("proc2", "output", batch_id)?;
        let _frame = pipeline.get_independent_frame("output", id)?;
        Ok(())
    }

    #[test]
    fn test_batch_to_batch() -> anyhow::Result<()> {
        let mut pipeline = create_pipeline()?;
        let id = pipeline.add_frame("input", gen_frame())?;
        let batch_id = pipeline.move_and_pack_frames("input", "proc1", vec![id])?;
        pipeline.move_as_is("proc1", "proc2", vec![batch_id])?;
        let _batch = pipeline.get_batch("proc2", batch_id)?;
        let _frame = pipeline.get_batched_frame("proc2", batch_id, id)?;
        Ok(())
    }

    #[test]
    fn test_frame_to_frame() -> anyhow::Result<()> {
        let mut pipeline = create_pipeline()?;
        let id = pipeline.add_frame("input", gen_frame())?;
        pipeline.move_as_is("input", "output", vec![id])?;
        let _frame = pipeline.get_independent_frame("output", id)?;
        Ok(())
    }

    fn get_update() -> VideoFrameUpdate {
        let mut update = VideoFrameUpdate::new();
        update.add_attribute(
            &AttributeBuilder::default()
                .namespace("update".into())
                .name("attribute".into())
                .hint(None)
                .hint(Some("test".into()))
                .values(vec![AttributeValue::string("1".into(), None)])
                .build()
                .unwrap(),
        );
        update
    }

    #[test]
    fn test_frame_update() -> anyhow::Result<()> {
        let mut pipeline = create_pipeline()?;
        let id = pipeline.add_frame("input", gen_frame())?;
        let update = get_update();
        pipeline.add_frame_update("input", id, update)?;
        pipeline.apply_updates("input", id)?;
        let frame = pipeline.get_independent_frame("input", id)?;
        frame
            .get_attribute("update".to_string(), "attribute".to_string())
            .unwrap();
        Ok(())
    }

    #[test]
    fn test_batch_update() -> anyhow::Result<()> {
        let mut pipeline = create_pipeline()?;
        let id = pipeline.add_frame("input", gen_frame())?;
        let batch_id = pipeline.move_and_pack_frames("input", "proc1", vec![id])?;
        let update = get_update();
        pipeline.add_batched_frame_update("proc1", batch_id, id, update)?;
        pipeline.apply_updates("proc1", batch_id)?;
        let frame = pipeline.get_batched_frame("proc1", batch_id, id)?;
        frame
            .get_attribute("update".to_string(), "attribute".to_string())
            .unwrap();
        Ok(())
    }
}
