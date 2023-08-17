use crate::match_query::MatchQuery;
use crate::primitives::frame::VideoFrameProxy;
use crate::primitives::frame_batch::VideoFrameBatch;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::object::VideoObjectProxy;
pub use implementation::PipelineStagePayloadType;
use opentelemetry::Context;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Default, Debug)]
pub struct Pipeline(Arc<RwLock<implementation::Pipeline>>);

impl Pipeline {
    pub fn memory_handle(&self) -> usize {
        self as *const Self as usize
    }

    pub fn set_root_span_name(&self, name: String) {
        self.0.write().set_root_span_name(name)
    }

    pub fn set_sampling_period(&self, period: i64) {
        self.0.write().set_sampling_period(period)
    }

    pub fn get_sampling_period(&self) -> i64 {
        self.0.read().get_sampling_period()
    }

    pub fn get_root_span_name(&self) -> String {
        self.0.read().get_root_span_name()
    }

    pub fn add_stage(
        &self,
        name: &str,
        stage_type: PipelineStagePayloadType,
    ) -> anyhow::Result<()> {
        self.0.write().add_stage(name, stage_type)
    }

    pub fn get_stage_type(&self, name: &str) -> Option<PipelineStagePayloadType> {
        self.0.read().get_stage_type(name).cloned()
    }

    pub fn add_frame_update(&self, frame_id: i64, update: VideoFrameUpdate) -> anyhow::Result<()> {
        self.0.write().add_frame_update(frame_id, update)
    }

    pub fn add_batched_frame_update(
        &self,
        batch_id: i64,
        frame_id: i64,
        update: VideoFrameUpdate,
    ) -> anyhow::Result<()> {
        self.0
            .write()
            .add_batched_frame_update(batch_id, frame_id, update)
    }

    pub fn add_frame(&self, stage_name: &str, frame: VideoFrameProxy) -> anyhow::Result<i64> {
        self.0.write().add_frame(stage_name, frame)
    }

    pub fn add_frame_with_telemetry(
        &self,
        stage_name: &str,
        frame: VideoFrameProxy,
        parent_ctx: Context,
    ) -> anyhow::Result<i64> {
        self.0
            .write()
            .add_frame_with_telemetry(stage_name, frame, parent_ctx)
    }

    pub fn delete(&self, id: i64) -> anyhow::Result<HashMap<i64, Context>> {
        self.0.write().delete(id)
    }

    pub fn get_stage_queue_len(&self, stage: &str) -> anyhow::Result<usize> {
        self.0.read().get_stage_queue_len(stage)
    }

    pub fn get_id_locations(&self) -> HashMap<i64, String> {
        self.0.read().get_id_locations().clone()
    }

    pub fn get_independent_frame(
        &self,
        frame_id: i64,
    ) -> anyhow::Result<(VideoFrameProxy, Context)> {
        self.0.read().get_independent_frame(frame_id)
    }

    pub fn get_batched_frame(
        &self,
        batch_id: i64,
        frame_id: i64,
    ) -> anyhow::Result<(VideoFrameProxy, Context)> {
        self.0.read().get_batched_frame(batch_id, frame_id)
    }

    pub fn get_batch(
        &self,
        batch_id: i64,
    ) -> anyhow::Result<(VideoFrameBatch, HashMap<i64, Context>)> {
        self.0.read().get_batch(batch_id)
    }

    pub fn apply_updates(&self, id: i64) -> anyhow::Result<()> {
        self.0.read().apply_updates(id)
    }

    pub fn clear_updates(&self, id: i64) -> anyhow::Result<()> {
        self.0.write().clear_updates(id)
    }

    pub fn move_as_is(&self, dest_stage_name: &str, object_ids: Vec<i64>) -> anyhow::Result<()> {
        self.0.write().move_as_is(dest_stage_name, object_ids)
    }

    pub fn move_and_pack_frames(
        &self,
        dest_stage_name: &str,
        frame_ids: Vec<i64>,
    ) -> anyhow::Result<i64> {
        self.0
            .write()
            .move_and_pack_frames(dest_stage_name, frame_ids)
    }

    pub fn move_and_unpack_batch(
        &self,
        dest_stage_name: &str,
        batch_id: i64,
    ) -> anyhow::Result<HashMap<String, i64>> {
        self.0
            .write()
            .move_and_unpack_batch(dest_stage_name, batch_id)
    }

    pub fn access_objects(
        &self,
        frame_id: i64,
        query: &MatchQuery,
    ) -> anyhow::Result<HashMap<i64, Vec<VideoObjectProxy>>> {
        self.0.read().access_objects(frame_id, query)
    }
}

mod implementation {
    use crate::get_tracer;
    use crate::match_query::MatchQuery;
    use crate::primitives::frame::VideoFrameProxy;
    use crate::primitives::frame_batch::VideoFrameBatch;
    use crate::primitives::frame_update::VideoFrameUpdate;
    use crate::primitives::object::VideoObjectProxy;
    use anyhow::bail;
    use opentelemetry::trace::{SpanBuilder, TraceContextExt, TraceId, Tracer};
    use opentelemetry::Context;
    use std::collections::{HashMap, HashSet};
    const DEFAULT_ROOT_SPAN_NAME: &str = "video_pipeline";

    #[derive(Debug, Default)]
    struct PipelineStage {
        payload: HashMap<i64, VideoPipelinePayload>,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub enum PipelineStagePayloadType {
        Frame,
        Batch,
    }

    #[derive(Debug)]
    enum VideoPipelinePayload {
        Frame(VideoFrameProxy, Vec<VideoFrameUpdate>, Context),
        Batch(
            VideoFrameBatch,
            Vec<(i64, VideoFrameUpdate)>,
            HashMap<i64, Context>,
        ),
    }

    #[derive(Debug, Default)]
    pub struct Pipeline {
        id_counter: i64,
        sampling_period: i64,
        frame_counter: i64,
        root_spans: HashMap<i64, Context>,
        stages: HashMap<String, PipelineStage>,
        stage_types: HashMap<String, PipelineStagePayloadType>,
        id_locations: HashMap<i64, String>,
        root_span_name: Option<String>,
    }

    impl Pipeline {
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

            let span =
                get_tracer().build_with_context(SpanBuilder::from_name(span_name), parent_ctx);
            Context::current_with_span(span)
        }

        pub fn add_stage(
            &mut self,
            name: &str,
            stage_type: PipelineStagePayloadType,
        ) -> anyhow::Result<()> {
            if self.stages.contains_key(name) {
                bail!("Stage already exists")
            }
            self.stages
                .insert(name.to_owned(), PipelineStage::default());
            self.stage_types.insert(name.to_owned(), stage_type);
            Ok(())
        }

        fn get_stage(&self, name: &str) -> Option<&PipelineStage> {
            self.stages.get(name)
        }

        fn get_stage_mut(&mut self, name: &str) -> Option<&mut PipelineStage> {
            self.stages.get_mut(name)
        }

        pub fn get_stage_type(&self, name: &str) -> Option<&PipelineStagePayloadType> {
            self.stage_types.get(name)
        }

        pub fn add_frame_update(
            &mut self,
            frame_id: i64,
            update: VideoFrameUpdate,
        ) -> anyhow::Result<()> {
            let stage_name = &self.get_stage_for_id(frame_id)?;
            if let Some(stage) = self.get_stage_mut(stage_name) {
                if let Some(payload) = stage.payload.get_mut(&frame_id) {
                    match payload {
                        VideoPipelinePayload::Frame(_, updates, _) => {
                            updates.push(update);
                        }
                        _ => bail!("Frame update can only be added to a frame payload"),
                    }
                } else {
                    bail!("Frame not found in stage")
                }
            } else {
                bail!("Stage not found")
            }
            Ok(())
        }

        pub fn add_batched_frame_update(
            &mut self,
            batch_id: i64,
            frame_id: i64,
            update: VideoFrameUpdate,
        ) -> anyhow::Result<()> {
            let stage_name = &self.get_stage_for_id(batch_id)?;
            if let Some(stage) = self.get_stage_mut(stage_name) {
                if let Some(payload) = stage.payload.get_mut(&batch_id) {
                    match payload {
                        VideoPipelinePayload::Batch(_, updates, _) => {
                            updates.push((frame_id, update));
                        }
                        _ => bail!("Batch update can only be added to a batch payload"),
                    }
                } else {
                    bail!("Batch not found in stage")
                }
            } else {
                bail!("Stage not found")
            }
            Ok(())
        }

        pub fn add_frame(
            &mut self,
            stage_name: &str,
            frame: VideoFrameProxy,
        ) -> anyhow::Result<i64> {
            let ctx = if self.sampling_period <= 0
                || (self.frame_counter + 1) % self.sampling_period != 0
            {
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
                Some(PipelineStagePayloadType::Batch)
            ) {
                bail!("Stage does not accept batched frames")
            }

            self.frame_counter += 1;
            let id_counter = self.id_counter + 1;

            if parent_ctx.span().span_context().trace_id() == TraceId::INVALID {
                self.root_spans.insert(id_counter, Context::default());
            } else {
                let span = get_tracer().build_with_context(
                    SpanBuilder::from_name(self.get_root_span_name()),
                    &parent_ctx,
                );

                self.root_spans
                    .insert(id_counter, Context::current_with_span(span));
            }

            let ctx = self.get_stage_span(id_counter, format!("add/{}", stage_name));
            let frame_payload = VideoPipelinePayload::Frame(frame, Vec::new(), ctx);
            if let Some(stage) = self.get_stage_mut(stage_name) {
                stage.payload.insert(id_counter, frame_payload);
            } else {
                bail!("Stage not found")
            }
            self.id_counter = id_counter;
            self.id_locations.insert(id_counter, stage_name.to_owned());
            Ok(self.id_counter)
        }

        pub fn delete(&mut self, id: i64) -> anyhow::Result<HashMap<i64, Context>> {
            let stage_name = &self
                .id_locations
                .remove(&id)
                .ok_or(anyhow::anyhow!("Object location not found"))?;

            if let Some(stage) = self.get_stage_mut(stage_name) {
                let removed = stage.payload.remove(&id);
                if removed.is_none() {
                    bail!("Object not found in stage")
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
                bail!("Stage not found")
            }
        }

        pub fn get_stage_queue_len(&self, stage: &str) -> anyhow::Result<usize> {
            if let Some(stage) = self.get_stage(stage) {
                Ok(stage.payload.len())
            } else {
                bail!("Stage not found")
            }
        }

        pub fn get_id_locations(&self) -> &HashMap<i64, String> {
            &self.id_locations
        }

        fn get_stage_for_id(&self, id: i64) -> anyhow::Result<String> {
            if let Some(stage) = self.id_locations.get(&id) {
                Ok(stage.to_string())
            } else {
                bail!("Object location not found")
            }
        }

        pub fn get_independent_frame(
            &self,
            frame_id: i64,
        ) -> anyhow::Result<(VideoFrameProxy, Context)> {
            let stage_name = self.get_stage_for_id(frame_id)?;
            if let Some(stage) = self.get_stage(&stage_name) {
                if let Some(payload) = stage.payload.get(&frame_id) {
                    match payload {
                        VideoPipelinePayload::Frame(frame, _, ctx) => {
                            Ok((frame.clone(), ctx.clone()))
                        }
                        _ => bail!("Payload must be a frame"),
                    }
                } else {
                    bail!("Frame not found in stage")
                }
            } else {
                bail!("Stage not found")
            }
        }

        pub fn get_batched_frame(
            &self,
            batch_id: i64,
            frame_id: i64,
        ) -> anyhow::Result<(VideoFrameProxy, Context)> {
            let stage_name = self.get_stage_for_id(batch_id)?;
            if let Some(stage) = self.get_stage(&stage_name) {
                if let Some(payload) = stage.payload.get(&batch_id) {
                    match payload {
                        VideoPipelinePayload::Batch(batch, _, contexts) => {
                            if let Some(frame) = batch.get(frame_id) {
                                let ctx = contexts.get(&frame_id).unwrap();
                                Ok((frame, ctx.clone()))
                            } else {
                                bail!("Frame not found in batch")
                            }
                        }
                        _ => bail!("Payload must be a batch"),
                    }
                } else {
                    bail!("Batch not found in stage")
                }
            } else {
                bail!("Stage not found")
            }
        }

        pub fn get_batch(
            &self,
            batch_id: i64,
        ) -> anyhow::Result<(VideoFrameBatch, HashMap<i64, Context>)> {
            let stage_name = self.get_stage_for_id(batch_id)?;
            if let Some(stage) = self.get_stage(&stage_name) {
                if let Some(payload) = stage.payload.get(&batch_id) {
                    match payload {
                        VideoPipelinePayload::Batch(batch, _, contexts) => {
                            Ok((batch.clone(), contexts.clone()))
                        }
                        _ => bail!("Payload must be a batch"),
                    }
                } else {
                    bail!("Batch not found in stage")
                }
            } else {
                bail!("Stage not found")
            }
        }

        pub fn apply_updates(&self, id: i64) -> anyhow::Result<()> {
            let stage_name = self.get_stage_for_id(id)?;
            if let Some(stage) = self.get_stage(&stage_name) {
                if let Some(payload) = stage.payload.get(&id) {
                    match payload {
                        VideoPipelinePayload::Frame(frame, updates, ctx) => {
                            let _span =
                                Self::get_nested_span(format!("{}/apply-updates", stage_name), ctx)
                                    .attach();
                            for update in updates {
                                frame.update(update)?;
                            }
                        }
                        VideoPipelinePayload::Batch(batch, updates, contexts) => {
                            for (frame_id, update) in updates {
                                if let Some(frame) = batch.get(*frame_id) {
                                    let _context_guard = Self::get_nested_span(
                                        format!("{}/apply-updates", stage_name),
                                        contexts.get(frame_id).unwrap(),
                                    )
                                    .attach();
                                    frame.update(update)?;
                                }
                            }
                        }
                    }
                } else {
                    bail!("Payload not found in stage")
                }
            } else {
                bail!("Stage not found")
            }
            Ok(())
        }

        pub fn clear_updates(&mut self, id: i64) -> anyhow::Result<()> {
            let stage_name = self.get_stage_for_id(id)?;
            if let Some(stage) = self.get_stage_mut(&stage_name) {
                if let Some(payload) = stage.payload.get_mut(&id) {
                    match payload {
                        VideoPipelinePayload::Frame(_, updates, ctx) => {
                            let _guard =
                                Self::get_nested_span(format!("{}/clear-updates", stage_name), ctx)
                                    .attach();
                            updates.clear();
                        }
                        VideoPipelinePayload::Batch(_, updates, ctxts) => {
                            let ids = updates.iter().map(|(id, _)| *id).collect::<HashSet<_>>();
                            let contexts = ids
                                .iter()
                                .map(|id| {
                                    Self::get_nested_span(
                                        format!("{}/clear-updates", stage_name),
                                        ctxts.get(id).unwrap(),
                                    )
                                })
                                .collect::<Vec<_>>();
                            updates.clear();
                            contexts.iter().for_each(|cx| cx.span().end());
                        }
                    }
                } else {
                    bail!("Payload not found in stage")
                }
            } else {
                bail!("Stage not found")
            }
            Ok(())
        }

        pub fn move_as_is(
            &mut self,
            dest_stage_name: &str,
            object_ids: Vec<i64>,
        ) -> anyhow::Result<()> {
            if object_ids.is_empty() {
                bail!("Object IDs cannot be empty")
            }
            let source_stage_name = &self.get_stage_for_id(object_ids[0])?;
            if self.get_stage_type(source_stage_name) != self.get_stage_type(dest_stage_name) {
                bail!("The source stage type must be the same as the destination stage type")
            }

            let source_stage_opt = self.get_stage_mut(source_stage_name);
            if source_stage_opt.is_none() {
                bail!("Source stage not found")
            }

            let dest_stage_opt = self.get_stage_mut(dest_stage_name);
            if dest_stage_opt.is_none() {
                bail!("Destination stage not found")
            }

            for id in &object_ids {
                let object_source_stage = &self.get_stage_for_id(*id)?;
                if object_source_stage != source_stage_name {
                    bail!("All objects must be in the same source stage")
                }
            }

            let source_stage = self.get_stage_mut(source_stage_name).unwrap();
            let mut removed_objects = Vec::new();
            for id in object_ids {
                if let Some(payload) = source_stage.payload.remove(&id) {
                    removed_objects.push((id, payload));
                } else {
                    bail!("Object not found in source stage")
                }
            }

            for (id, payload) in removed_objects {
                self.id_locations.insert(id, dest_stage_name.to_owned());
                let payload = match payload {
                    VideoPipelinePayload::Frame(frame, updates, ctx) => {
                        ctx.span().end();
                        let ctx = self.get_stage_span(id, format!("stage/{}", dest_stage_name));
                        VideoPipelinePayload::Frame(frame, updates, ctx)
                    }
                    VideoPipelinePayload::Batch(batch, updates, contexts) => {
                        let mut new_contexts = HashMap::new();
                        for (id, ctx) in contexts.iter() {
                            ctx.span().end();
                            let ctx =
                                self.get_stage_span(*id, format!("stage/{}", dest_stage_name));
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
            dest_stage_name: &str,
            frame_ids: Vec<i64>,
        ) -> anyhow::Result<i64> {
            if frame_ids.is_empty() {
                bail!("Frame IDs cannot be empty")
            }
            let source_stage_name = &self.get_stage_for_id(frame_ids[0])?;

            if matches!(
                self.get_stage_type(source_stage_name),
                Some(PipelineStagePayloadType::Batch)
            ) || matches!(
                self.get_stage_type(dest_stage_name),
                Some(PipelineStagePayloadType::Frame)
            ) {
                bail!("Source stage must contain independent frames and destination stage must contain batched frames")
            }

            let batch_id = self.id_counter + 1;
            let source_stage_opt = self.get_stage_mut(source_stage_name);
            if source_stage_opt.is_none() {
                bail!("Source stage not found")
            }

            let dest_stage_opt = self.get_stage_mut(dest_stage_name);
            if dest_stage_opt.is_none() {
                bail!("Destination stage not found")
            }

            for id in &frame_ids {
                self.id_locations.insert(*id, dest_stage_name.to_owned());
            }

            let mut batch = VideoFrameBatch::new();
            let mut batch_updates = Vec::new();
            let mut contexts = HashMap::new();

            let source_stage = self.get_stage_mut(source_stage_name).unwrap();
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
                        _ => bail!("Source stage must contain independent frames"),
                    }
                }
            }

            let contexts = contexts
                .into_iter()
                .map(|(id, ctx)| {
                    ctx.span().end();
                    let ctx = self.get_stage_span(id, format!("stage/{}", dest_stage_name));
                    (id, ctx)
                })
                .collect();

            let payload = VideoPipelinePayload::Batch(batch, batch_updates, contexts);
            let dest_stage = self.get_stage_mut(dest_stage_name).unwrap();
            dest_stage.payload.insert(batch_id, payload);
            self.id_counter = batch_id;
            self.id_locations
                .insert(batch_id, dest_stage_name.to_owned());
            Ok(self.id_counter)
        }

        pub fn move_and_unpack_batch(
            &mut self,
            dest_stage_name: &str,
            batch_id: i64,
        ) -> anyhow::Result<HashMap<String, i64>> {
            let source_stage_name = &self.get_stage_for_id(batch_id)?;
            if matches!(
                self.get_stage_type(source_stage_name),
                Some(PipelineStagePayloadType::Frame)
            ) || matches!(
                self.get_stage_type(dest_stage_name),
                Some(PipelineStagePayloadType::Batch)
            ) {
                bail!("Source stage must contain batched frames and destination stage must contain independent frames")
            }

            let source_stage_opt = self.get_stage_mut(source_stage_name);
            if source_stage_opt.is_none() {
                bail!("Source stage not found")
            }

            let dest_stage_opt = self.get_stage_mut(dest_stage_name);
            if dest_stage_opt.is_none() {
                bail!("Destination stage not found")
            }

            let source_stage = self.get_stage_mut(source_stage_name).unwrap();
            let (batch, updates, mut contexts) =
                if let Some(payload) = source_stage.payload.remove(&batch_id) {
                    match payload {
                        VideoPipelinePayload::Batch(batch, updates, contexts) => {
                            (batch, updates, contexts)
                        }
                        _ => bail!("Source stage must contain batch"),
                    }
                } else {
                    bail!("Batch not found in source stage")
                };

            self.id_locations.remove(&batch_id);
            let mut frame_mapping = HashMap::new();
            for (frame_id, frame) in batch.frames {
                self.id_locations
                    .insert(frame_id, dest_stage_name.to_owned());
                let ctx = contexts.remove(&frame_id).unwrap();
                ctx.span().end();
                let ctx = self.get_stage_span(frame_id, format!("stage/{}", dest_stage_name));
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
                        _ => bail!("Destination stage must contain independent frames"),
                    }
                } else {
                    bail!("Frame not found in destination stage")
                }
            }

            Ok(frame_mapping)
        }

        pub fn access_objects(
            &self,
            frame_id: i64,
            query: &MatchQuery,
        ) -> anyhow::Result<HashMap<i64, Vec<VideoObjectProxy>>> {
            let stage_name = &self.get_stage_for_id(frame_id)?;
            if let Some(stage) = self.get_stage(stage_name) {
                if let Some(payload) = stage.payload.get(&frame_id) {
                    match payload {
                        VideoPipelinePayload::Frame(frame, _, ctx) => {
                            let _span = Self::get_nested_span(
                                format!("{}/access-objects", stage_name),
                                ctx,
                            )
                            .attach();
                            Ok(HashMap::from([(frame_id, frame.access_objects(query))]))
                        }
                        VideoPipelinePayload::Batch(batch, _, contexts) => {
                            let contexts = contexts
                                .iter()
                                .map(|(_, ctx)| {
                                    Self::get_nested_span(
                                        format!("{}/access-objects", stage_name),
                                        ctx,
                                    )
                                })
                                .collect::<Vec<_>>();
                            let res = Ok(batch.access_objects(query));
                            contexts.into_iter().for_each(|ctx| ctx.span().end());
                            res
                        }
                    }
                } else {
                    bail!("Frame not found in stage")
                }
            } else {
                bail!("Stage not found")
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::pipeline::implementation::{Pipeline, PipelineStagePayloadType};
        use crate::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
        use crate::primitives::frame_update::VideoFrameUpdate;
        use crate::primitives::{Attribute, AttributeMethods};
        use crate::test::gen_frame;
        use opentelemetry::global;
        use opentelemetry::sdk::export::trace::stdout;
        use opentelemetry::sdk::propagation::TraceContextPropagator;
        use opentelemetry::trace::{TraceContextExt, TraceId};
        use std::io::sink;

        fn create_pipeline() -> anyhow::Result<Pipeline> {
            let mut pipeline = Pipeline::default();
            pipeline.add_stage("input", PipelineStagePayloadType::Frame)?;
            pipeline.add_stage("proc1", PipelineStagePayloadType::Batch)?;
            pipeline.add_stage("proc2", PipelineStagePayloadType::Batch)?;
            pipeline.add_stage("output", PipelineStagePayloadType::Frame)?;
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
                .add_stage("input", PipelineStagePayloadType::Frame)
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
                Some(PipelineStagePayloadType::Frame)
            ));
            assert!(matches!(
                pipeline.get_stage_type("proc1"),
                Some(PipelineStagePayloadType::Batch)
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

            pipeline.delete(id)?;
            assert_eq!(pipeline.get_stage_queue_len("input")?, 0);

            Ok(())
        }

        #[test]
        fn test_frame_to_batch() -> anyhow::Result<()> {
            let mut pipeline = create_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let batch_id = pipeline.move_and_pack_frames("proc1", vec![id])?;

            assert!(pipeline.get_independent_frame(id).is_err());

            assert!(pipeline.get_batch(batch_id).is_ok());
            assert!(pipeline.get_batched_frame(batch_id, id).is_ok());

            Ok(())
        }

        #[test]
        fn test_batch_to_frame() -> anyhow::Result<()> {
            let mut pipeline = create_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let batch_id = pipeline.move_and_pack_frames("proc2", vec![id])?;
            pipeline.move_and_unpack_batch("output", batch_id)?;
            let _frame = pipeline.get_independent_frame(id)?;
            Ok(())
        }

        #[test]
        fn test_batch_to_batch() -> anyhow::Result<()> {
            let mut pipeline = create_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let batch_id = pipeline.move_and_pack_frames("proc1", vec![id])?;
            pipeline.move_as_is("proc2", vec![batch_id])?;
            let _batch = pipeline.get_batch(batch_id)?;
            let _frame = pipeline.get_batched_frame(batch_id, id)?;
            Ok(())
        }

        #[test]
        fn test_frame_to_frame() -> anyhow::Result<()> {
            let mut pipeline = create_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            pipeline.move_as_is("output", vec![id])?;
            let _frame = pipeline.get_independent_frame(id)?;
            Ok(())
        }

        fn get_update() -> VideoFrameUpdate {
            let mut update = VideoFrameUpdate::default();
            update.add_attribute(Attribute::persistent(
                "update".into(),
                "attribute".into(),
                vec![AttributeValue::new(
                    AttributeValueVariant::String("1".into()),
                    None,
                )],
                Some("test".into()),
            ));
            update
        }

        #[test]
        fn test_frame_update() -> anyhow::Result<()> {
            let mut pipeline = create_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let update = get_update();
            pipeline.add_frame_update(id, update)?;
            pipeline.apply_updates(id)?;
            let (frame, _) = pipeline.get_independent_frame(id)?;
            frame
                .get_attribute("update".to_string(), "attribute".to_string())
                .unwrap();
            Ok(())
        }

        #[test]
        fn test_batch_update() -> anyhow::Result<()> {
            let mut pipeline = create_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let batch_id = pipeline.move_and_pack_frames("proc1", vec![id])?;
            let update = get_update();
            pipeline.add_batched_frame_update(batch_id, id, update)?;
            pipeline.apply_updates(batch_id)?;
            pipeline.clear_updates(batch_id)?;
            let (frame, _) = pipeline.get_batched_frame(batch_id, id)?;
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
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_eq!(ctx.span().span_context().trace_id(), TraceId::INVALID);

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_ne!(ctx.span().span_context().trace_id(), TraceId::INVALID);

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_eq!(ctx.span().span_context().trace_id(), TraceId::INVALID);

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
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
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_eq!(ctx.span().span_context().trace_id(), TraceId::INVALID);

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
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
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_ne!(ctx.span().span_context().trace_id(), TraceId::INVALID);

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_ne!(ctx.span().span_context().trace_id(), TraceId::INVALID);

            Ok(())
        }
    }
}
