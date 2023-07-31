use crate::primitives::{VideoFrameBatch, VideoFrameProxy, VideoFrameUpdate};
use hashbrown::HashMap;
use pyo3::prelude::*;

#[derive(Clone, Debug)]
pub struct PipelineStage {
    payload: HashMap<i64, PipelinePayload>,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub enum PipelineStagePayloadType {
    Frame,
    Batch,
}

#[derive(Clone, Debug)]
pub enum PipelinePayload {
    Frame(VideoFrameProxy, Vec<VideoFrameUpdate>),
    Batch(VideoFrameBatch, Vec<(i64, VideoFrameUpdate)>),
}

#[derive(Clone, Debug, Default)]
pub struct VideoPipeline {
    pub id_counter: i64,
    pub stages: HashMap<String, PipelineStage>,
    pub stage_types: HashMap<String, PipelineStagePayloadType>,
}

impl VideoPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_stage(
        &mut self,
        name: &str,
        stage: PipelineStage,
        stage_type: PipelineStagePayloadType,
    ) {
        self.stages.insert(name.to_owned(), stage);
        self.stage_types.insert(name.to_owned(), stage_type);
    }

    pub fn get_stage(&self, name: &str) -> Option<&PipelineStage> {
        self.stages.get(name)
    }

    pub fn get_stage_mut(&mut self, name: &str) -> Option<&mut PipelineStage> {
        self.stages.get_mut(name)
    }

    pub fn get_stage_type(&self, name: &str) -> Option<&PipelineStagePayloadType> {
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
                    PipelinePayload::Frame(_, updates) => {
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
                    PipelinePayload::Batch(_, updates) => {
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

    pub fn add_frame(&mut self, stage: &str, frame: VideoFrameProxy) -> anyhow::Result<i64> {
        if matches!(
            self.get_stage_type(stage),
            Some(PipelineStagePayloadType::Batch)
        ) {
            anyhow::bail!("Stage does not accept batched frames")
        }

        let id_counter = self.id_counter + 1;
        if let Some(stage) = self.get_stage_mut(stage) {
            stage
                .payload
                .insert(id_counter, PipelinePayload::Frame(frame, Vec::new()));
        } else {
            anyhow::bail!("Stage not found")
        }
        self.id_counter = id_counter;
        Ok(self.id_counter)
    }

    pub fn add_batch(&mut self, stage: &str, batch: VideoFrameBatch) -> anyhow::Result<i64> {
        if matches!(
            self.get_stage_type(stage),
            Some(PipelineStagePayloadType::Frame)
        ) {
            anyhow::bail!("Stage does not accept independent frames")
        }
        let id_counter = self.id_counter + 1;
        if let Some(stage) = self.get_stage_mut(stage) {
            stage
                .payload
                .insert(id_counter, PipelinePayload::Batch(batch, Vec::new()));
        } else {
            anyhow::bail!("Stage not found")
        }
        self.id_counter = id_counter;
        Ok(self.id_counter)
    }

    pub fn del(&mut self, stage: &str, id: i64) -> anyhow::Result<()> {
        if let Some(stage) = self.get_stage_mut(stage) {
            stage.payload.remove(&id);
        } else {
            anyhow::bail!("Stage not found")
        }
        Ok(())
    }

    pub fn get_independent_frame(
        &self,
        stage: &str,
        frame_id: i64,
    ) -> anyhow::Result<VideoFrameProxy> {
        if let Some(stage) = self.get_stage(stage) {
            if let Some(payload) = stage.payload.get(&frame_id) {
                match payload {
                    PipelinePayload::Frame(frame, _) => Ok(frame.clone()),
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
                    PipelinePayload::Batch(batch, _) => {
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
                    PipelinePayload::Batch(batch, _) => Ok(batch.clone()),
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
                    PipelinePayload::Frame(frame, updates) => {
                        for update in updates.drain(..) {
                            frame.update(&update)?;
                        }
                    }
                    PipelinePayload::Batch(batch, updates) => {
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
        source_stage: &str,
        dest_stage: &str,
        object_ids: Vec<i64>,
    ) -> anyhow::Result<()> {
        if self.get_stage_type(source_stage) != self.get_stage_type(dest_stage) {
            anyhow::bail!("The source stage type must be the same as the destination stage type")
        }

        let source_stage_opt = self.get_stage_mut(source_stage);
        if !source_stage_opt.is_some() {
            anyhow::bail!("Source stage not found")
        }
        drop(source_stage_opt);

        let dest_stage_opt = self.get_stage_mut(dest_stage);
        if !dest_stage_opt.is_some() {
            anyhow::bail!("Destination stage not found")
        }
        drop(dest_stage_opt);

        let source_stage = self.get_stage_mut(source_stage).unwrap();
        let mut removed_objects = Vec::new();
        for id in object_ids {
            if let Some(payload) = source_stage.payload.remove(&id) {
                removed_objects.push((id, payload));
            } else {
                anyhow::bail!("Object not found in source stage")
            }
        }

        let dest_stage = self.get_stage_mut(dest_stage).unwrap();
        for o in removed_objects {
            dest_stage.payload.insert(o.0, o.1);
        }

        Ok(())
    }

    pub fn move_and_pack_frames(
        &mut self,
        source_stage: &str,
        dest_stage: &str,
        frame_ids: Vec<i64>,
    ) -> anyhow::Result<i64> {
        if matches!(
            self.get_stage_type(source_stage),
            Some(PipelineStagePayloadType::Batch)
        ) || matches!(
            self.get_stage_type(dest_stage),
            Some(PipelineStagePayloadType::Frame)
        ) {
            anyhow::bail!("Source stage must contain independent frames and destination stage must contain batched frames")
        }

        let batch_id = self.id_counter + 1;
        let source_stage_opt = self.get_stage_mut(source_stage);
        if !source_stage_opt.is_some() {
            anyhow::bail!("Source stage not found")
        }
        drop(source_stage_opt);

        let dest_stage_opt = self.get_stage_mut(dest_stage);
        if !dest_stage_opt.is_some() {
            anyhow::bail!("Destination stage not found")
        }
        drop(dest_stage_opt);

        let source_stage = self.get_stage_mut(source_stage).unwrap();

        let mut batch = VideoFrameBatch::new();
        let mut batch_updates = Vec::new();
        for id in frame_ids {
            if let Some(payload) = source_stage.payload.remove(&id) {
                match payload {
                    PipelinePayload::Frame(frame, updates) => {
                        batch.add(id, frame);
                        for update in updates {
                            batch_updates.push((id, update));
                        }
                    }
                    _ => anyhow::bail!("Source stage must contain independent frames"),
                }
            }
        }

        let dest_stage = self.get_stage_mut(dest_stage).unwrap();
        dest_stage
            .payload
            .insert(batch_id, PipelinePayload::Batch(batch, batch_updates));

        self.id_counter = batch_id;
        Ok(self.id_counter)
    }

    pub fn move_and_unpack_batch(
        &mut self,
        source_stage: &str,
        dest_stage: &str,
        batch_id: i64,
    ) -> anyhow::Result<()> {
        if matches!(
            self.get_stage_type(source_stage),
            Some(PipelineStagePayloadType::Frame)
        ) || matches!(
            self.get_stage_type(dest_stage),
            Some(PipelineStagePayloadType::Batch)
        ) {
            anyhow::bail!("Source stage must contain batched frames and destination stage must contain independent frames")
        }

        let source_stage_opt = self.get_stage_mut(source_stage);
        if !source_stage_opt.is_some() {
            anyhow::bail!("Source stage not found")
        }
        drop(source_stage_opt);

        let dest_stage_opt = self.get_stage_mut(dest_stage);
        if !dest_stage_opt.is_some() {
            anyhow::bail!("Destination stage not found")
        }
        drop(dest_stage_opt);

        let source_stage = self.get_stage_mut(source_stage).unwrap();
        let (batch, updates) = if let Some(payload) = source_stage.payload.remove(&batch_id) {
            match payload {
                PipelinePayload::Batch(batch, updates) => (batch, updates),
                _ => anyhow::bail!("Source stage must contain batch"),
            }
        } else {
            anyhow::bail!("Batch not found in source stage")
        };

        let dest_stage = self.get_stage_mut(dest_stage).unwrap();
        for (frame_id, frame) in batch.frames {
            dest_stage
                .payload
                .insert(frame_id, PipelinePayload::Frame(frame, Vec::new()));
        }

        for (frame_id, update) in updates {
            if let Some(frame) = dest_stage.payload.get_mut(&frame_id) {
                match frame {
                    PipelinePayload::Frame(_, updates) => {
                        updates.push(update);
                    }
                    _ => anyhow::bail!("Destination stage must contain independent frames"),
                }
            } else {
                anyhow::bail!("Frame not found in destination stage")
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {}
