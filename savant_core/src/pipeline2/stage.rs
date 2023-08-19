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
                    let ctx = contexts.get(&frame_id).unwrap();
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
        frame_id: i64,
        query: &MatchQuery,
    ) -> anyhow::Result<HashMap<i64, Vec<VideoObjectProxy>>> {
        self.with_payload_item(frame_id, |payload| match payload {
            PipelinePayload::Frame(frame, _, ctx) => {
                let _span =
                    Pipeline::get_nested_span(format!("{}/access-objects", self.stage_name), ctx)
                        .attach();
                Ok(HashMap::from([(frame_id, frame.access_objects(query))]))
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
