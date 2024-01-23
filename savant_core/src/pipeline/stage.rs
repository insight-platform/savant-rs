use crate::match_query::MatchQuery;
use crate::pipeline::implementation::Pipeline;
use crate::pipeline::{PipelinePayload, PipelineStagePayloadType};
use crate::primitives::frame::VideoFrameProxy;
use crate::primitives::frame_batch::VideoFrameBatch;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::object::VideoObjectProxy;
use crate::rust::StageStat;
use crate::rwlock::SavantRwLock;
use anyhow::bail;
use hashbrown::{HashMap, HashSet};
use opentelemetry::trace::TraceContextExt;
use opentelemetry::Context;
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct PipelineStage {
    pub name: String,
    pub stage_type: PipelineStagePayloadType,
    pub payload: SavantRwLock<HashMap<i64, PipelinePayload>>,
    pub stat: Arc<SavantRwLock<StageStat>>,
}

impl PipelineStage {
    pub fn new(name: String, stage_type: PipelineStagePayloadType) -> Self {
        Self {
            name: name.clone(),
            stage_type,
            payload: Default::default(),
            stat: Arc::new(SavantRwLock::new(StageStat::new(name))),
        }
    }

    pub fn get_stat(&self) -> Arc<SavantRwLock<StageStat>> {
        self.stat.clone()
    }

    fn with_payload_item_mut<F, T>(&self, id: i64, f: F) -> anyhow::Result<T>
    where
        F: FnOnce(&mut PipelinePayload) -> T,
    {
        let mut bind = self.payload.write();
        let payload = bind
            .get_mut(&id)
            .ok_or(anyhow::anyhow!("Payload {} not found in stage", id))?;
        Ok(f(payload))
    }

    fn with_payload_item<F, T>(&self, id: i64, f: F) -> anyhow::Result<T>
    where
        F: FnOnce(&PipelinePayload) -> T,
    {
        let bind = self.payload.read();
        let payload = bind
            .get(&id)
            .ok_or(anyhow::anyhow!("Payload {} not found in stage", id))?;
        Ok(f(payload))
    }

    fn with_payload_mut<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut HashMap<i64, PipelinePayload>) -> T,
    {
        let mut bind = self.payload.write();
        f(&mut bind)
    }

    fn with_payload<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&HashMap<i64, PipelinePayload>) -> T,
    {
        let bind = self.payload.read();
        f(&bind)
    }

    pub fn add_frame_update(&self, frame_id: i64, update: VideoFrameUpdate) -> anyhow::Result<()> {
        self.with_payload_item_mut(frame_id, move |payload| match payload {
            PipelinePayload::Frame(_, updates, _) => {
                updates.push(update);
                Ok(())
            }
            _ => bail!("Frame update can only be added to a frame payload"),
        })?
    }

    pub fn add_batched_frame_update(
        &self,
        batch_id: i64,
        frame_id: i64,
        update: VideoFrameUpdate,
    ) -> anyhow::Result<()> {
        self.with_payload_item_mut(batch_id, |payload| match payload {
            PipelinePayload::Batch(_, updates, _) => {
                updates.push((frame_id, update));
                Ok(())
            }
            _ => bail!("Batch update can only be added to a batch payload"),
        })?
    }

    fn update_stats_for_frame(&self, f: &VideoFrameProxy) {
        let mut stat_bind = self.stat.write();
        stat_bind.frame_counter += 1;
        stat_bind.queue_length += 1;
        stat_bind.object_counter += f.get_object_count();
    }

    fn update_stats_for_batch(&self, b: &VideoFrameBatch) {
        let mut stat_bind = self.stat.write();
        stat_bind.batch_counter += 1;
        stat_bind.frame_counter += b.frames.len();
        stat_bind.queue_length += 1;
        stat_bind.object_counter += b
            .frames
            .values()
            .map(|f| f.get_object_count())
            .sum::<usize>();
    }

    pub fn add_payloads<I>(&self, payloads: I) -> anyhow::Result<()>
    where
        I: IntoIterator<Item = (i64, PipelinePayload)>,
    {
        self.with_payload_mut(|bind| {
            for (id, payload) in payloads {
                if bind.contains_key(&id) {
                    bail!("Payload {} already exists", id)
                }
                match &payload {
                    PipelinePayload::Frame(f, _, _) => {
                        if self.stage_type == PipelineStagePayloadType::Batch {
                            bail!("Payload must be a batch")
                        } else {
                            self.update_stats_for_frame(f);
                        }
                    }
                    PipelinePayload::Batch(b, _, _) => {
                        if self.stage_type == PipelineStagePayloadType::Frame {
                            bail!("Payload must be a frame")
                        } else {
                            self.update_stats_for_batch(b);
                        }
                    }
                }
                bind.insert(id, payload);
            }
            Ok(())
        })
    }

    pub fn add_frame_payload(&self, frame_id: i64, payload: PipelinePayload) -> anyhow::Result<()> {
        self.with_payload_mut(|bind| {
            if bind.contains_key(&frame_id) {
                bail!("Frame {} already exists", frame_id)
            }
            match payload {
                PipelinePayload::Batch(_, _, _) => {
                    bail!("Payload must be a frame")
                }
                PipelinePayload::Frame(f, u, c) => {
                    self.update_stats_for_frame(&f);
                    bind.insert(frame_id, PipelinePayload::Frame(f, u, c));
                }
            }
            Ok(())
        })
    }

    pub fn add_batch_payload(&self, batch_id: i64, payload: PipelinePayload) -> anyhow::Result<()> {
        self.with_payload_mut(|bind| {
            if bind.contains_key(&batch_id) {
                bail!("Batch {} already exists", batch_id)
            }
            match payload {
                PipelinePayload::Frame(_, _, _) => {
                    bail!("Payload must be a batch")
                }
                PipelinePayload::Batch(b, u, c) => {
                    self.update_stats_for_batch(&b);
                    bind.insert(batch_id, PipelinePayload::Batch(b, u, c));
                }
            }
            Ok(())
        })
    }

    pub fn delete(&self, id: i64) -> Option<PipelinePayload> {
        self.with_payload_mut(|bind| {
            let res = bind.remove(&id);
            if res.is_some() {
                let mut stats_bind = self.stat.write();
                stats_bind.queue_length = bind.len();
            }
            res
        })
    }

    pub fn delete_many(&self, ids: &[i64]) -> Vec<(i64, PipelinePayload)> {
        self.with_payload_mut(|bind| {
            let mut removed = Vec::with_capacity(ids.len());
            for id in ids {
                let v = bind.remove(id);
                if let Some(p) = v {
                    removed.push((*id, p));
                }
            }
            let mut stats_bind = self.stat.write();
            stats_bind.queue_length = bind.len();
            removed
        })
    }

    pub fn len(&self) -> usize {
        self.with_payload(|bind| bind.len())
    }

    pub fn get_independent_frame(
        &self,
        frame_id: i64,
    ) -> anyhow::Result<(VideoFrameProxy, Context)> {
        self.with_payload_item(frame_id, |payload| match payload {
            PipelinePayload::Frame(frame, _, ctx) => Ok((frame.clone(), ctx.clone())),
            _ => bail!("Payload must be a frame"),
        })?
    }

    pub fn get_batched_frame(
        &self,
        batch_id: i64,
        frame_id: i64,
    ) -> anyhow::Result<(VideoFrameProxy, Context)> {
        self.with_payload_item(batch_id, |payload| match payload {
            PipelinePayload::Batch(batch, _, contexts) => {
                if let Some(frame) = batch.get(frame_id) {
                    let ctx = contexts
                        .get(&frame_id)
                        .expect("Frame context must be in the same batch");
                    Ok((frame.clone(), ctx.clone()))
                } else {
                    bail!("Frame not found in batch")
                }
            }
            _ => bail!("Payload must be a batch"),
        })?
    }

    pub fn get_batch(
        &self,
        batch_id: i64,
    ) -> anyhow::Result<(VideoFrameBatch, HashMap<i64, Context>)> {
        self.with_payload_item(batch_id, |payload| match payload {
            PipelinePayload::Batch(batch, _, contexts) => Ok((batch.clone(), contexts.clone())),
            _ => bail!("Payload must be a batch"),
        })?
    }

    pub fn apply_updates(&self, id: i64) -> anyhow::Result<()> {
        self.with_payload_item_mut(id, |payload| {
            match payload {
                PipelinePayload::Frame(frame, updates, ctx) => {
                    let _span =
                        Pipeline::get_nested_span(format!("{}/apply-updates", self.name), ctx)
                            .attach();
                    for update in updates {
                        frame.update(update)?;
                    }
                }
                PipelinePayload::Batch(batch, updates, contexts) => {
                    for (frame_id, update) in updates {
                        if let Some(frame) = batch.get(*frame_id) {
                            let _context_guard = Pipeline::get_nested_span(
                                format!("{}/apply-updates", self.name),
                                contexts.get(frame_id).unwrap(),
                            )
                            .attach();
                            frame.update(update)?;
                        }
                    }
                }
            }
            Ok(())
        })?
    }

    pub fn clear_updates(&self, id: i64) -> anyhow::Result<()> {
        self.with_payload_item_mut(id, |payload| {
            match payload {
                PipelinePayload::Frame(_, updates, ctx) => {
                    let _guard =
                        Pipeline::get_nested_span(format!("{}/clear-updates", self.name), ctx)
                            .attach();
                    updates.clear();
                }
                PipelinePayload::Batch(_, updates, ctxts) => {
                    let ids = updates.iter().map(|(id, _)| *id).collect::<HashSet<_>>();
                    let contexts = ids
                        .iter()
                        .map(|id| {
                            Pipeline::get_nested_span(
                                format!("{}/clear-updates", self.name),
                                ctxts.get(id).unwrap(),
                            )
                        })
                        .collect::<Vec<_>>();
                    updates.clear();
                    contexts.iter().for_each(|cx| cx.span().end());
                }
            }
            Ok(())
        })?
    }

    pub fn access_objects(
        &self,
        id: i64,
        query: &MatchQuery,
    ) -> anyhow::Result<HashMap<i64, Vec<VideoObjectProxy>>> {
        self.with_payload_item(id, |payload| match payload {
            PipelinePayload::Frame(frame, _, ctx) => {
                let _span = Pipeline::get_nested_span(format!("{}/access-objects", self.name), ctx)
                    .attach();
                Ok(HashMap::from([(id, frame.access_objects(query))]))
            }
            PipelinePayload::Batch(batch, _, contexts) => {
                let contexts = contexts
                    .iter()
                    .map(|(_, ctx)| {
                        Pipeline::get_nested_span(format!("{}/access-objects", self.name), ctx)
                    })
                    .collect::<Vec<_>>();
                let res = Ok(batch.access_objects(query));
                contexts.into_iter().for_each(|ctx| ctx.span().end());
                res
            }
        })?
    }
}

#[cfg(test)]
mod tests {
    use crate::match_query::MatchQuery;
    use crate::pipeline::stage::PipelineStage;
    use crate::pipeline::{PipelinePayload, PipelineStagePayloadType};
    use crate::primitives::frame_batch::VideoFrameBatch;
    use crate::primitives::frame_update::VideoFrameUpdate;
    use crate::primitives::{Attribute, WithAttributes};
    use crate::test::gen_frame;
    use anyhow::Result;
    use hashbrown::HashMap;
    use opentelemetry::Context;
    use std::default::Default;

    fn get_frame_stage() -> PipelineStage {
        let name = "stage".to_string();
        PipelineStage::new(name, PipelineStagePayloadType::Frame)
    }

    fn get_batch_stage() -> PipelineStage {
        let name = "stage".to_string();
        PipelineStage::new(name, PipelineStagePayloadType::Batch)
    }

    #[test]
    fn test_add_correct_payload() -> Result<()> {
        let stage = get_frame_stage();
        stage.add_frame_payload(
            1,
            PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
        )?;
        let stage = get_batch_stage();
        stage.add_batch_payload(
            1,
            PipelinePayload::Batch(VideoFrameBatch::new(), Vec::default(), HashMap::default()),
        )
    }

    #[test]
    fn test_add_incorrect_payload() -> Result<()> {
        let stage = get_frame_stage();
        assert!(stage
            .add_frame_payload(
                1,
                PipelinePayload::Batch(VideoFrameBatch::new(), Vec::default(), HashMap::default()),
            )
            .is_err());

        let stage = get_batch_stage();
        assert!(stage
            .add_batch_payload(
                1,
                PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
            )
            .is_err());
        Ok(())
    }

    #[test]
    fn add_correct_payload_multi() -> Result<()> {
        let stage = get_frame_stage();
        stage.add_payloads(vec![
            (
                1,
                PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
            ),
            (
                2,
                PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
            ),
        ])?;

        let stage = get_batch_stage();
        stage.add_payloads(vec![
            (
                1,
                PipelinePayload::Batch(VideoFrameBatch::new(), Vec::default(), HashMap::default()),
            ),
            (
                2,
                PipelinePayload::Batch(VideoFrameBatch::new(), Vec::default(), HashMap::default()),
            ),
        ])
    }

    #[test]
    fn add_incorrect_payload_multi() -> Result<()> {
        let stage = get_frame_stage();
        assert!(stage
            .add_payloads(vec![
                (
                    1,
                    PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
                ),
                (
                    2,
                    PipelinePayload::Batch(
                        VideoFrameBatch::new(),
                        Vec::default(),
                        HashMap::default()
                    ),
                ),
            ])
            .is_err());
        Ok(())
    }

    #[test]
    fn add_duplicate_frame_payload() -> Result<()> {
        let stage = get_frame_stage();
        stage.add_frame_payload(
            1,
            PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
        )?;
        assert!(stage
            .add_frame_payload(
                1,
                PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
            )
            .is_err());
        Ok(())
    }

    #[test]
    fn add_duplicate_batch_payload() -> Result<()> {
        let stage = get_batch_stage();
        stage.add_batch_payload(
            1,
            PipelinePayload::Batch(VideoFrameBatch::new(), Vec::default(), HashMap::default()),
        )?;
        assert!(stage
            .add_batch_payload(
                1,
                PipelinePayload::Batch(VideoFrameBatch::new(), Vec::default(), HashMap::default()),
            )
            .is_err());
        Ok(())
    }

    #[test]
    fn test_delete() {
        let stage = get_frame_stage();
        stage
            .add_frame_payload(
                1,
                PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
            )
            .unwrap();
        assert!(stage.delete(1).is_some());
        assert!(stage.delete(1).is_none());
    }

    #[test]
    fn test_delete_many() -> Result<()> {
        let stage = get_frame_stage();
        stage.add_frame_payload(
            1,
            PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
        )?;
        stage.add_frame_payload(
            2,
            PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
        )?;
        stage.add_frame_payload(
            3,
            PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
        )?;
        let removed = stage.delete_many(&[1, 2, 3]);
        assert_eq!(removed.len(), 3);
        let removed = stage.delete_many(&[1, 2, 3]);
        assert!(removed.is_empty());
        Ok(())
    }

    #[test]
    fn test_len() -> Result<()> {
        let stage = get_frame_stage();
        stage.add_frame_payload(
            1,
            PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
        )?;
        stage.add_frame_payload(
            2,
            PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
        )?;
        stage.add_frame_payload(
            3,
            PipelinePayload::Frame(gen_frame(), Vec::default(), Context::default()),
        )?;
        assert_eq!(stage.len(), 3);
        Ok(())
    }

    #[test]
    fn test_independent_frame() -> Result<()> {
        let stage = get_frame_stage();
        let frame = gen_frame();
        stage.add_frame_payload(
            1,
            PipelinePayload::Frame(frame.clone(), Vec::default(), Context::default()),
        )?;
        let (_, _) = stage.get_independent_frame(1)?;
        assert!(stage.get_independent_frame(2).is_err());
        Ok(())
    }

    #[test]
    fn test_get_batched_frame() -> Result<()> {
        let stage = get_batch_stage();
        let frame = gen_frame();
        let mut batch = VideoFrameBatch::new();
        batch.add(2, frame);
        stage.add_batch_payload(
            1,
            PipelinePayload::Batch(
                batch,
                Vec::default(),
                HashMap::from([(2, Context::default())]),
            ),
        )?;
        let (_, _) = stage.get_batched_frame(1, 2)?;
        assert!(stage.get_batched_frame(1, 3).is_err());
        Ok(())
    }

    #[test]
    fn test_get_batch() -> Result<()> {
        let stage = get_batch_stage();
        stage.add_batch_payload(
            1,
            PipelinePayload::Batch(VideoFrameBatch::new(), Vec::default(), HashMap::from([])),
        )?;
        let (_, _) = stage.get_batch(1)?;
        assert!(stage.get_batch(2).is_err());
        Ok(())
    }

    #[test]
    fn test_apply_updates_for_frame() -> Result<()> {
        let stage = get_frame_stage();
        let frame = gen_frame();
        stage.add_frame_payload(
            1,
            PipelinePayload::Frame(frame, Vec::default(), Context::default()),
        )?;
        let mut update = VideoFrameUpdate::default();
        update.add_frame_attribute(Attribute::persistent("new", "attr", vec![], &None, false));
        stage.add_frame_update(1, update)?;
        stage.apply_updates(1)?;
        let mut frame = stage.get_independent_frame(1)?.0;
        frame.get_attribute("new", "attr").unwrap();
        assert!(frame.delete_attribute("new", "attr").is_some());

        stage.clear_updates(1)?;
        stage.apply_updates(1)?;

        assert!(frame.get_attribute("new", "attr").is_none());

        Ok(())
    }

    #[test]
    fn test_apply_attributes_for_batch() -> Result<()> {
        let stage = get_batch_stage();
        let frame = gen_frame();
        let mut batch = VideoFrameBatch::new();
        batch.add(2, frame);
        stage.add_batch_payload(
            1,
            PipelinePayload::Batch(
                batch,
                Vec::default(),
                HashMap::from([(2, Context::default())]),
            ),
        )?;
        let mut update = VideoFrameUpdate::default();
        update.add_frame_attribute(Attribute::persistent("new", "attr", vec![], &None, false));
        stage.add_batched_frame_update(1, 2, update)?;
        stage.apply_updates(1)?;
        let mut frame = stage.get_batched_frame(1, 2)?.0;
        frame.get_attribute("new", "attr").unwrap();
        assert!(frame.delete_attribute("new", "attr").is_some());

        stage.clear_updates(1)?;
        stage.apply_updates(1)?;

        assert!(frame.get_attribute("new", "attr").is_none());

        Ok(())
    }

    #[test]
    fn test_access_objects_for_frame() -> Result<()> {
        let stage = get_frame_stage();
        let frame = gen_frame();
        stage.add_frame_payload(
            1,
            PipelinePayload::Frame(frame, Vec::default(), Context::default()),
        )?;
        let objects = stage.access_objects(1, &MatchQuery::Idle)?;
        assert_eq!(objects.get(&1).unwrap().len(), 3);
        Ok(())
    }

    #[test]
    fn test_access_objects_for_batch() -> Result<()> {
        let stage = get_batch_stage();
        let mut batch = VideoFrameBatch::new();
        batch.add(2, gen_frame());
        batch.add(3, gen_frame());
        stage.add_batch_payload(
            1,
            PipelinePayload::Batch(
                batch,
                Vec::default(),
                HashMap::from([(2, Context::default()), (3, Context::default())]),
            ),
        )?;
        let objects = stage.access_objects(1, &MatchQuery::Idle)?;
        assert_eq!(objects.get(&2).unwrap().len(), 3);
        assert_eq!(objects.get(&3).unwrap().len(), 3);
        Ok(())
    }
}
