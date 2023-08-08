use crate::primitives::message::video::query::MatchQuery;
use crate::primitives::{VideoFrameBatch, VideoFrameProxy, VideoFrameUpdate, VideoObjectProxy};
use crate::utils::get_tracer;
use opentelemetry::trace::{SpanBuilder, TraceContextExt, TraceId, Tracer};
use opentelemetry::Context;
use pyo3::prelude::*;
use std::collections::HashMap;

const DEFAULT_ROOT_SPAN_NAME: &str = "video_pipeline";

#[derive(Debug, Default)]
pub struct VideoPipelineStage {
    payload: HashMap<i64, VideoPipelinePayload>,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub enum VideoPipelineStagePayloadType {
    Frame,
    Batch,
}

#[derive(Debug)]
pub enum VideoPipelinePayload {
    Frame(VideoFrameProxy, Vec<VideoFrameUpdate>, Context),
    Batch(
        VideoFrameBatch,
        Vec<(i64, VideoFrameUpdate)>,
        HashMap<i64, Context>,
    ),
}

#[derive(Debug, Default)]
pub struct VideoPipeline {
    id_counter: i64,
    sampling_period: i64,
    frame_counter: i64,
    root_spans: HashMap<i64, Context>,
    stages: HashMap<String, VideoPipelineStage>,
    stage_types: HashMap<String, VideoPipelineStagePayloadType>,
    root_span_name: Option<String>,
}

impl VideoPipeline {
    pub fn set_root_span_name(&mut self, name: String) {
        self.root_span_name = Some(name);
    }

    pub fn set_sampling_period(&mut self, period: i64) {
        self.sampling_period = period;
    }

    pub fn get_sampling_period(&self) -> i64 {
        self.sampling_period
    }

    pub fn get_root_span_name(&self) -> String {
        self.root_span_name
            .clone()
            .unwrap_or_else(|| DEFAULT_ROOT_SPAN_NAME.to_owned())
    }

    fn get_stage_span(&self, id: i64, span_name: String) -> Context {
        let ctx = self.root_spans.get(&id).unwrap();

        if ctx.span().span_context().trace_id() == TraceId::INVALID {
            return Context::default();
        }

        let span = get_tracer().build_with_context(SpanBuilder::from_name(span_name), ctx);
        Context::current_with_span(span)
    }

    fn get_nested_span(span_name: String, parent_ctx: &Context) -> Context {
        if parent_ctx.span().span_context().trace_id() == TraceId::INVALID {
            return Context::default();
        }

        let span = get_tracer().build_with_context(SpanBuilder::from_name(span_name), parent_ctx);
        Context::current_with_span(span)
    }

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
                    VideoPipelinePayload::Frame(_, updates, _) => {
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
                    VideoPipelinePayload::Batch(_, updates, _) => {
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
        let ctx =
            if self.sampling_period <= 0 || (self.frame_counter + 1) % self.sampling_period != 0 {
                Context::default()
            } else {
                get_tracer().in_span(self.get_root_span_name(), |cx| cx)
            };
        self.add_frame_with_telemetry(stage_name, frame, ctx)
    }

    pub fn add_frame_with_telemetry(
        &mut self,
        stage_name: &str,
        frame: VideoFrameProxy,
        parent_ctx: Context,
    ) -> anyhow::Result<i64> {
        if matches!(
            self.get_stage_type(stage_name),
            Some(VideoPipelineStagePayloadType::Batch)
        ) {
            anyhow::bail!("Stage does not accept batched frames")
        }

        self.frame_counter += 1;
        let id_counter = self.id_counter + 1;

        if parent_ctx.span().span_context().trace_id() == TraceId::INVALID {
            self.root_spans.insert(id_counter, Context::default());
        } else {
            let span = get_tracer()
                .build_with_context(SpanBuilder::from_name("video-pipeline"), &parent_ctx);

            self.root_spans
                .insert(id_counter, Context::current_with_span(span));
        }

        let ctx = self.get_stage_span(id_counter, format!("add-{}", stage_name));
        let frame_payload = VideoPipelinePayload::Frame(frame, Vec::new(), ctx);
        if let Some(stage) = self.get_stage_mut(stage_name) {
            stage.payload.insert(id_counter, frame_payload);
        } else {
            anyhow::bail!("Stage not found")
        }
        self.id_counter = id_counter;
        Ok(self.id_counter)
    }

    pub fn delete(&mut self, stage_name: &str, id: i64) -> anyhow::Result<HashMap<i64, Context>> {
        if let Some(stage) = self.get_stage_mut(stage_name) {
            let removed = stage.payload.remove(&id);
            if removed.is_none() {
                anyhow::bail!("Object not found in stage")
            }
            match removed.unwrap() {
                VideoPipelinePayload::Frame(_, _, ctx) => {
                    ctx.span().end();
                    let root_ctx = self.root_spans.remove(&id).unwrap();
                    Ok(HashMap::from([(id, root_ctx)]))
                }
                VideoPipelinePayload::Batch(_, _, contexts) => Ok(contexts
                    .into_iter()
                    .map(|(id, ctx)| {
                        ctx.span().end();
                        let root_ctx = self.root_spans.remove(&id).unwrap();
                        (id, root_ctx)
                    })
                    .collect()),
            }
        } else {
            anyhow::bail!("Stage not found")
        }
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
    ) -> anyhow::Result<(VideoFrameProxy, Context)> {
        if let Some(stage) = self.get_stage(stage) {
            if let Some(payload) = stage.payload.get(&frame_id) {
                match payload {
                    VideoPipelinePayload::Frame(frame, _, ctx) => Ok((frame.clone(), ctx.clone())),
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
    ) -> anyhow::Result<(VideoFrameProxy, Context)> {
        if let Some(stage) = self.get_stage(stage) {
            if let Some(payload) = stage.payload.get(&batch_id) {
                match payload {
                    VideoPipelinePayload::Batch(batch, _, contexts) => {
                        if let Some(frame) = batch.get(frame_id) {
                            let ctx = contexts.get(&frame_id).unwrap();
                            Ok((frame, ctx.clone()))
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

    pub fn get_batch(
        &self,
        stage: &str,
        batch_id: i64,
    ) -> anyhow::Result<(VideoFrameBatch, HashMap<i64, Context>)> {
        if let Some(stage) = self.get_stage(stage) {
            if let Some(payload) = stage.payload.get(&batch_id) {
                match payload {
                    VideoPipelinePayload::Batch(batch, _, contexts) => {
                        Ok((batch.clone(), contexts.clone()))
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

    pub fn apply_updates(&mut self, stage: &str, id: i64) -> anyhow::Result<()> {
        if let Some(stage) = self.get_stage_mut(stage) {
            if let Some(payload) = stage.payload.get_mut(&id) {
                match payload {
                    VideoPipelinePayload::Frame(frame, updates, ctx) => {
                        let _span = Self::get_nested_span("apply-updates".into(), ctx).attach();
                        for update in updates.drain(..) {
                            frame.update(&update)?;
                        }
                    }
                    VideoPipelinePayload::Batch(batch, updates, contexts) => {
                        for (frame_id, update) in updates.drain(..) {
                            if let Some(frame) = batch.get(frame_id) {
                                let _context_guard = Self::get_nested_span(
                                    "apply-updates".into(),
                                    contexts.get(&frame_id).unwrap(),
                                )
                                .attach();
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

        for (id, payload) in removed_objects {
            let payload = match payload {
                VideoPipelinePayload::Frame(frame, updates, ctx) => {
                    ctx.span().end();
                    let ctx = self.get_stage_span(
                        id,
                        format!("move/{}-{}", source_stage_name, dest_stage_name),
                    );
                    VideoPipelinePayload::Frame(frame, updates, ctx)
                }
                VideoPipelinePayload::Batch(batch, updates, contexts) => {
                    let mut new_contexts = HashMap::new();
                    for (id, ctx) in contexts.iter() {
                        ctx.span().end();
                        let ctx = self.get_stage_span(
                            *id,
                            format!("move/{}-{}", source_stage_name, dest_stage_name),
                        );
                        new_contexts.insert(*id, ctx);
                    }
                    VideoPipelinePayload::Batch(batch, updates, new_contexts)
                }
            };
            let dest_stage = self.get_stage_mut(dest_stage_name).unwrap();
            dest_stage.payload.insert(id, payload);
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
        let mut contexts = HashMap::new();
        for id in frame_ids {
            if let Some(payload) = source_stage.payload.remove(&id) {
                match payload {
                    VideoPipelinePayload::Frame(frame, updates, ctx) => {
                        batch.add(id, frame);
                        contexts.insert(id, ctx);
                        for update in updates {
                            batch_updates.push((id, update));
                        }
                    }
                    _ => anyhow::bail!("Source stage must contain independent frames"),
                }
            }
        }

        let contexts = contexts
            .into_iter()
            .map(|(id, ctx)| {
                ctx.span().end();
                let ctx = self.get_stage_span(
                    id,
                    format!("move/pack/{}-{}", source_stage_name, dest_stage_name),
                );
                (id, ctx)
            })
            .collect();

        let payload = VideoPipelinePayload::Batch(batch, batch_updates, contexts);
        let dest_stage = self.get_stage_mut(dest_stage_name).unwrap();
        dest_stage.payload.insert(batch_id, payload);
        self.id_counter = batch_id;
        Ok(self.id_counter)
    }

    pub fn move_and_unpack_batch(
        &mut self,
        source_stage_name: &str,
        dest_stage_name: &str,
        batch_id: i64,
    ) -> anyhow::Result<HashMap<String, i64>> {
        if matches!(
            self.get_stage_type(source_stage_name),
            Some(VideoPipelineStagePayloadType::Frame)
        ) || matches!(
            self.get_stage_type(dest_stage_name),
            Some(VideoPipelineStagePayloadType::Batch)
        ) {
            anyhow::bail!("Source stage must contain batched frames and destination stage must contain independent frames")
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
        let (batch, updates, mut contexts) = if let Some(payload) =
            source_stage.payload.remove(&batch_id)
        {
            match payload {
                VideoPipelinePayload::Batch(batch, updates, contexts) => (batch, updates, contexts),
                _ => anyhow::bail!("Source stage must contain batch"),
            }
        } else {
            anyhow::bail!("Batch not found in source stage")
        };

        let mut frame_mapping = HashMap::new();
        for (frame_id, frame) in batch.frames {
            let ctx = contexts.remove(&frame_id).unwrap();
            ctx.span().end();
            let ctx = self.get_stage_span(
                frame_id,
                format!("move/unpack/{}-{}", source_stage_name, dest_stage_name),
            );
            frame_mapping.insert(frame.get_source_id(), frame_id);

            let dest_stage = self.get_stage_mut(dest_stage_name).unwrap();
            dest_stage.payload.insert(
                frame_id,
                VideoPipelinePayload::Frame(frame, Vec::new(), ctx),
            );
        }

        let dest_stage = self.get_stage_mut(dest_stage_name).unwrap();
        for (frame_id, update) in updates {
            if let Some(frame) = dest_stage.payload.get_mut(&frame_id) {
                match frame {
                    VideoPipelinePayload::Frame(_, updates, _) => {
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

    pub fn access_objects(
        &self,
        stage_name: &str,
        frame_id: i64,
        query: &MatchQuery,
    ) -> anyhow::Result<HashMap<i64, Vec<VideoObjectProxy>>> {
        if let Some(stage) = self.get_stage(stage_name) {
            if let Some(payload) = stage.payload.get(&frame_id) {
                match payload {
                    VideoPipelinePayload::Frame(frame, _, ctx) => {
                        let _span =
                            Self::get_nested_span("access-objects".to_string(), ctx).attach();
                        Ok(HashMap::from([(frame_id, frame.access_objects(query))]))
                    }
                    VideoPipelinePayload::Batch(batch, _, contexts) => {
                        let contexts = contexts
                            .iter()
                            .map(|(_, ctx)| {
                                Self::get_nested_span("access-objects".to_string(), ctx)
                            })
                            .collect::<Vec<_>>();
                        let res = Ok(batch.access_objects(query));
                        contexts.into_iter().for_each(|ctx| ctx.span().end());
                        res
                    }
                }
            } else {
                anyhow::bail!("Frame not found in stage")
            }
        } else {
            anyhow::bail!("Stage not found")
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::message::video::pipeline::{
        VideoPipeline, VideoPipelineStagePayloadType,
    };
    use crate::primitives::{AttributeBuilder, AttributeValue, VideoFrameUpdate};
    use crate::test::utils::gen_frame;
    use opentelemetry::global;
    use opentelemetry::sdk::export::trace::stdout;
    use opentelemetry::sdk::propagation::TraceContextPropagator;
    use opentelemetry::trace::{TraceContextExt, TraceId};
    use std::io::sink;

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
        let (frame, _) = pipeline.get_independent_frame("input", id)?;
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
        let (frame, _) = pipeline.get_batched_frame("proc1", batch_id, id)?;
        frame
            .get_attribute("update".to_string(), "attribute".to_string())
            .unwrap();
        Ok(())
    }

    #[test]
    fn test_sampling() -> anyhow::Result<()> {
        stdout::new_pipeline().with_writer(sink()).install_simple();
        global::set_text_map_propagator(TraceContextPropagator::new());

        let mut pipeline = create_pipeline()?;
        pipeline.set_sampling_period(2);

        let id = pipeline.add_frame("input", gen_frame())?;
        let (_frame, ctx) = pipeline.get_independent_frame("input", id)?;
        assert_eq!(ctx.span().span_context().trace_id(), TraceId::INVALID);

        let id = pipeline.add_frame("input", gen_frame())?;
        let (_frame, ctx) = pipeline.get_independent_frame("input", id)?;
        assert_ne!(ctx.span().span_context().trace_id(), TraceId::INVALID);

        let id = pipeline.add_frame("input", gen_frame())?;
        let (_frame, ctx) = pipeline.get_independent_frame("input", id)?;
        assert_eq!(ctx.span().span_context().trace_id(), TraceId::INVALID);

        let id = pipeline.add_frame("input", gen_frame())?;
        let (_frame, ctx) = pipeline.get_independent_frame("input", id)?;
        assert_ne!(ctx.span().span_context().trace_id(), TraceId::INVALID);

        Ok(())
    }

    #[test]
    fn test_no_tracing() -> anyhow::Result<()> {
        stdout::new_pipeline().with_writer(sink()).install_simple();
        global::set_text_map_propagator(TraceContextPropagator::new());

        let mut pipeline = create_pipeline()?;
        pipeline.set_sampling_period(0);

        let id = pipeline.add_frame("input", gen_frame())?;
        let (_frame, ctx) = pipeline.get_independent_frame("input", id)?;
        assert_eq!(ctx.span().span_context().trace_id(), TraceId::INVALID);

        let id = pipeline.add_frame("input", gen_frame())?;
        let (_frame, ctx) = pipeline.get_independent_frame("input", id)?;
        assert_eq!(ctx.span().span_context().trace_id(), TraceId::INVALID);

        Ok(())
    }

    #[test]
    fn test_tracing_every() -> anyhow::Result<()> {
        stdout::new_pipeline().with_writer(sink()).install_simple();
        global::set_text_map_propagator(TraceContextPropagator::new());

        let mut pipeline = create_pipeline()?;
        pipeline.set_sampling_period(1);

        let id = pipeline.add_frame("input", gen_frame())?;
        let (_frame, ctx) = pipeline.get_independent_frame("input", id)?;
        assert_ne!(ctx.span().span_context().trace_id(), TraceId::INVALID);

        let id = pipeline.add_frame("input", gen_frame())?;
        let (_frame, ctx) = pipeline.get_independent_frame("input", id)?;
        assert_ne!(ctx.span().span_context().trace_id(), TraceId::INVALID);

        Ok(())
    }
}
