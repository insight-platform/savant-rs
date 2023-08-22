use crate::match_query::MatchQuery;
use crate::pipeline::{PipelinePayload, PipelineStagePayloadType};
use crate::pipeline2::implementation::Pipeline;
use crate::primitives::frame::VideoFrameProxy;
use crate::primitives::frame_batch::VideoFrameBatch;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::object::VideoObjectProxy;
use anyhow::bail;
use hashbrown::{HashMap, HashSet};
use opentelemetry::trace::TraceContextExt;
use opentelemetry::Context;
use parking_lot::RwLock;

#[derive(Debug)]
pub(super) struct PipelineStage {
    pub stage_name: String,
    pub stage_type: PipelineStagePayloadType,
    pub payload: RwLock<HashMap<i64, PipelinePayload>>,
}

impl PipelineStage {
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
                    PipelinePayload::Frame(_, _, _) => {
                        if self.stage_type == PipelineStagePayloadType::Batch {
                            bail!("Payload must be a batch")
                        }
                    }
                    PipelinePayload::Batch(_, _, _) => {
                        if self.stage_type == PipelineStagePayloadType::Frame {
                            bail!("Payload must be a frame")
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
            if matches!(payload, PipelinePayload::Batch(_, _, _)) {
                bail!("Payload must be a frame")
            }
            bind.insert(frame_id, payload);
            Ok(())
        })
    }

    pub fn add_batch_payload(&self, batch_id: i64, payload: PipelinePayload) -> anyhow::Result<()> {
        self.with_payload_mut(|bind| {
            if bind.contains_key(&batch_id) {
                bail!("Batch {} already exists", batch_id)
            }
            if matches!(payload, PipelinePayload::Frame(_, _, _)) {
                bail!("Payload must be a batch")
            }
            bind.insert(batch_id, payload);
            Ok(())
        })
    }

    pub fn delete(&self, id: i64) -> Option<PipelinePayload> {
        self.with_payload_mut(|bind| bind.remove(&id))
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
                    let _span = Pipeline::get_nested_span(
                        format!("{}/apply-updates", self.stage_name),
                        ctx,
                    )
                    .attach();
                    for update in updates {
                        frame.update(update)?;
                    }
                }
                PipelinePayload::Batch(batch, updates, contexts) => {
                    for (frame_id, update) in updates {
                        if let Some(frame) = batch.get(*frame_id) {
                            let _context_guard = Pipeline::get_nested_span(
                                format!("{}/apply-updates", self.stage_name),
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
                    let _guard = Pipeline::get_nested_span(
                        format!("{}/clear-updates", self.stage_name),
                        ctx,
                    )
                    .attach();
                    updates.clear();
                }
                PipelinePayload::Batch(_, updates, ctxts) => {
                    let ids = updates.iter().map(|(id, _)| *id).collect::<HashSet<_>>();
                    let contexts = ids
                        .iter()
                        .map(|id| {
                            Pipeline::get_nested_span(
                                format!("{}/clear-updates", self.stage_name),
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
                let _span =
                    Pipeline::get_nested_span(format!("{}/access-objects", self.stage_name), ctx)
                        .attach();
                Ok(HashMap::from([(id, frame.access_objects(query))]))
            }
            PipelinePayload::Batch(batch, _, contexts) => {
                let contexts = contexts
                    .iter()
                    .map(|(_, ctx)| {
                        Pipeline::get_nested_span(
                            format!("{}/access-objects", self.stage_name),
                            ctx,
                        )
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
    use crate::pipeline::{PipelinePayload, PipelineStagePayloadType};
    use crate::pipeline2::stage::PipelineStage;
    use crate::primitives::frame_batch::VideoFrameBatch;
    use crate::primitives::frame_update::VideoFrameUpdate;
    use crate::primitives::{Attribute, AttributeMethods};
    use crate::test::gen_frame;
    use anyhow::Result;
    use hashbrown::HashMap;
    use opentelemetry::Context;
    use parking_lot::lock_api::RwLock;
    use std::default::Default;

    fn get_frame_stage() -> PipelineStage {
        PipelineStage {
            stage_name: "stage".to_string(),
            stage_type: PipelineStagePayloadType::Frame,
            payload: RwLock::new(HashMap::default()),
        }
    }

    fn get_batch_stage() -> PipelineStage {
        PipelineStage {
            stage_name: "stage".to_string(),
            stage_type: PipelineStagePayloadType::Batch,
            payload: RwLock::new(HashMap::default()),
        }
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
        update.add_frame_attribute(Attribute::new(
            "new".to_string(),
            "attr".to_string(),
            vec![],
            None,
            true,
        ));
        stage.add_frame_update(1, update)?;
        stage.apply_updates(1)?;
        let frame = stage.get_independent_frame(1)?.0;
        frame
            .get_attribute("new".to_string(), "attr".to_string())
            .unwrap();
        assert!(frame
            .delete_attribute("new".to_string(), "attr".to_string())
            .is_some());

        stage.clear_updates(1)?;
        stage.apply_updates(1)?;

        assert!(frame
            .get_attribute("new".to_string(), "attr".to_string())
            .is_none());

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
        update.add_frame_attribute(Attribute::new(
            "new".to_string(),
            "attr".to_string(),
            vec![],
            None,
            true,
        ));
        stage.add_batched_frame_update(1, 2, update)?;
        stage.apply_updates(1)?;
        let frame = stage.get_batched_frame(1, 2)?.0;
        frame
            .get_attribute("new".to_string(), "attr".to_string())
            .unwrap();
        assert!(frame
            .delete_attribute("new".to_string(), "attr".to_string())
            .is_some());

        stage.clear_updates(1)?;
        stage.apply_updates(1)?;

        assert!(frame
            .get_attribute("new".to_string(), "attr".to_string())
            .is_none());

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
