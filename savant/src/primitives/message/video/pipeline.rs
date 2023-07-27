use crate::primitives::{VideoFrameBatch, VideoFrameProxy};
use hashbrown::HashMap;

#[derive(Clone, Debug)]
pub enum PipelineStage {
    IndependentFrames(HashMap<i64, VideoFrameProxy>),
    Batch(HashMap<i64, VideoFrameBatch>),
}

#[derive(Clone, Debug)]
pub struct VideoPipeline {
    pub stages: HashMap<String, PipelineStage>,
    pub stage_order: Vec<String>,
}

impl VideoPipeline {
    pub fn new() -> Self {
        Self {
            stages: HashMap::new(),
            stage_order: Vec::new(),
        }
    }

    pub fn add_stage(&mut self, name: &str, stage: PipelineStage) {
        self.stages.insert(name.to_owned(), stage);
        self.stage_order.push(name.to_owned());
    }

    pub fn get_stage(&self, name: &str) -> Option<&PipelineStage> {
        self.stages.get(name)
    }

    pub fn get_stage_mut(&mut self, name: &str) -> Option<&mut PipelineStage> {
        self.stages.get_mut(name)
    }

    pub fn get_stage_order(&self) -> &Vec<String> {
        &self.stage_order
    }

    pub fn get_stage_order_mut(&mut self) -> &mut Vec<String> {
        &mut self.stage_order
    }

    // pub fn forward_frames(
    //     &mut self,
    //     source_stage: String,
    //     dest_stage: String,
    //     frame_ids: Vec<i64>,
    //     batch_id: i64,
    // ) -> Result<()> {
    //     let source_stage = self
    //         .get_stage_mut(&source_stage)
    //         .ok_or_else(|| anyhow::anyhow!("Source stage not found"))?;
    //     let dest_stage = self
    //         .get_stage_mut(&dest_stage)
    //         .ok_or_else(|| anyhow::anyhow!("Destination stage not found"))?;
    //
    //     match (source_stage, dest_stage) {
    //         (PipelineStage::IndependentFrames(source), PipelineStage::IndependentFrames(dest)) => {
    //             for id in frame_ids {
    //                 if let Some(frame) = source.remove(&id) {
    //                     dest.insert(id, frame);
    //                 }
    //             }
    //         }
    //         (PipelineStage::IndependentFrames(source), PipelineStage::Batch(dest)) => {
    //             let mut batch = VideoFrameBatch::new();
    //             for id in frame_ids {
    //                 if let Some(frame) = source.remove(&id) {
    //                     batch.add(id, frame);
    //                 }
    //             }
    //             dest.insert(batch_id, batch);
    //         }
    //         _ => {
    //             return Err(anyhow::anyhow!(
    //                 "Source stage must contain independent frames"
    //             ))
    //         }
    //     }
    //
    //     Ok(())
    // }
}
