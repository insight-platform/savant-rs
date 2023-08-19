use crate::match_query::MatchQuery;
use crate::pipeline::PipelineStagePayloadType;
use crate::primitives::frame::VideoFrameProxy;
use crate::primitives::frame_batch::VideoFrameBatch;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::object::VideoObjectProxy;
use anyhow::Result;
use hashbrown::HashMap;
use opentelemetry::Context;
use std::sync::Arc;

#[derive(Clone, Default, Debug)]
pub struct Pipeline(Arc<implementation::Pipeline>);

impl Pipeline {
    pub fn new(stages: Vec<(String, PipelineStagePayloadType)>) -> Result<Self> {
        Ok(Self(Arc::new(implementation::Pipeline::new(stages)?)))
    }

    pub fn memory_handle(&self) -> usize {
        self as *const Self as usize
    }

    pub fn set_root_span_name(&self, name: String) -> Result<()> {
        self.0.set_root_span_name(name)
    }

    pub fn set_sampling_period(&self, period: i64) -> Result<()> {
        self.0.set_sampling_period(period)
    }

    pub fn get_sampling_period(&self) -> i64 {
        *self.0.get_sampling_period()
    }

    pub fn get_root_span_name(&self) -> String {
        self.0.get_root_span_name().clone()
    }

    pub fn get_stage_type(&self, name: &str) -> Option<PipelineStagePayloadType> {
        self.0.get_stage_type(name)
    }

    pub fn add_frame_update(&self, frame_id: i64, update: VideoFrameUpdate) -> Result<()> {
        self.0.add_frame_update(frame_id, update)
    }

    pub fn add_batched_frame_update(
        &self,
        batch_id: i64,
        frame_id: i64,
        update: VideoFrameUpdate,
    ) -> Result<()> {
        self.0.add_batched_frame_update(batch_id, frame_id, update)
    }

    pub fn add_frame(&self, stage_name: &str, frame: VideoFrameProxy) -> Result<i64> {
        self.0.add_frame(stage_name, frame)
    }

    pub fn add_frame_with_telemetry(
        &self,
        stage_name: &str,
        frame: VideoFrameProxy,
        parent_ctx: Context,
    ) -> Result<i64> {
        self.0
            .add_frame_with_telemetry(stage_name, frame, parent_ctx)
    }

    pub fn delete(&self, id: i64) -> Result<HashMap<i64, Context>> {
        self.0.delete(id)
    }

    pub fn get_stage_queue_len(&self, stage: &str) -> Result<usize> {
        self.0.get_stage_queue_len(stage)
    }

    pub fn get_independent_frame(&self, frame_id: i64) -> Result<(VideoFrameProxy, Context)> {
        self.0.get_independent_frame(frame_id)
    }

    pub fn get_batched_frame(
        &self,
        batch_id: i64,
        frame_id: i64,
    ) -> Result<(VideoFrameProxy, Context)> {
        self.0.get_batched_frame(batch_id, frame_id)
    }

    pub fn get_batch(&self, batch_id: i64) -> Result<(VideoFrameBatch, HashMap<i64, Context>)> {
        self.0.get_batch(batch_id)
    }

    pub fn apply_updates(&self, id: i64) -> Result<()> {
        self.0.apply_updates(id)
    }

    pub fn clear_updates(&self, id: i64) -> Result<()> {
        self.0.clear_updates(id)
    }

    pub fn move_as_is(&self, dest_stage_name: &str, object_ids: Vec<i64>) -> Result<()> {
        self.0.move_as_is(dest_stage_name, object_ids)
    }

    pub fn move_and_pack_frames(&self, dest_stage_name: &str, frame_ids: Vec<i64>) -> Result<i64> {
        self.0.move_and_pack_frames(dest_stage_name, frame_ids)
    }

    pub fn move_and_unpack_batch(&self, dest_stage_name: &str, batch_id: i64) -> Result<Vec<i64>> {
        self.0.move_and_unpack_batch(dest_stage_name, batch_id)
    }

    pub fn access_objects(
        &self,
        frame_id: i64,
        query: &MatchQuery,
    ) -> Result<HashMap<i64, Vec<VideoObjectProxy>>> {
        self.0.access_objects(frame_id, query)
    }

    pub fn get_id_locations_len(&self) -> usize {
        self.0.get_id_locations_len()
    }
}

mod implementation {
    use crate::get_tracer;
    use crate::match_query::MatchQuery;
    use crate::pipeline::{PipelinePayload, PipelineStagePayloadType};
    use crate::primitives::frame::VideoFrameProxy;
    use crate::primitives::frame_batch::VideoFrameBatch;
    use crate::primitives::frame_update::VideoFrameUpdate;
    use crate::primitives::object::VideoObjectProxy;
    use anyhow::{bail, Result};
    use hashbrown::{HashMap, HashSet};
    use opentelemetry::trace::{SpanBuilder, TraceContextExt, TraceId, Tracer};
    use opentelemetry::Context;
    use parking_lot::RwLock;
    use std::sync::atomic::Ordering;
    use std::sync::OnceLock;

    const DEFAULT_ROOT_SPAN_NAME: &str = "video_pipeline";

    #[derive(Debug)]
    struct PipelineStage {
        pub stage_name: String,
        pub stage_type: PipelineStagePayloadType,
        pub payload: RwLock<HashMap<i64, PipelinePayload>>,
    }

    impl PipelineStage {
        fn with_payload_item_mut<F, T>(&self, id: i64, f: F) -> Result<T>
        where
            F: FnOnce(&mut PipelinePayload) -> T,
        {
            let mut bind = self.payload.write();
            let payload = bind
                .get_mut(&id)
                .ok_or(anyhow::anyhow!("Payload {} not found in stage", id))?;
            Ok(f(payload))
        }

        fn with_payload_item<F, T>(&self, id: i64, f: F) -> Result<T>
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

        pub fn add_frame_update(&self, frame_id: i64, update: VideoFrameUpdate) -> Result<()> {
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
        ) -> Result<()> {
            self.with_payload_item_mut(batch_id, |payload| match payload {
                PipelinePayload::Batch(_, updates, _) => {
                    updates.push((frame_id, update));
                    Ok(())
                }
                _ => bail!("Batch update can only be added to a batch payload"),
            })?
        }

        pub fn add_payloads<I>(&self, payloads: I) -> Result<()>
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

        pub fn add_frame_payload(&self, frame_id: i64, payload: PipelinePayload) -> Result<()> {
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

        pub fn add_batch_payload(&self, batch_id: i64, payload: PipelinePayload) -> Result<()> {
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

        pub fn get_independent_frame(&self, frame_id: i64) -> Result<(VideoFrameProxy, Context)> {
            self.with_payload_item(frame_id, |payload| match payload {
                PipelinePayload::Frame(frame, _, ctx) => Ok((frame.clone(), ctx.clone())),
                _ => bail!("Payload must be a frame"),
            })?
        }

        pub fn get_batched_frame(
            &self,
            batch_id: i64,
            frame_id: i64,
        ) -> Result<(VideoFrameProxy, Context)> {
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

        pub fn get_batch(&self, batch_id: i64) -> Result<(VideoFrameBatch, HashMap<i64, Context>)> {
            self.with_payload_item(batch_id, |payload| match payload {
                PipelinePayload::Batch(batch, _, contexts) => Ok((batch.clone(), contexts.clone())),
                _ => bail!("Payload must be a batch"),
            })?
        }

        pub fn apply_updates(&self, id: i64) -> Result<()> {
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

        pub fn clear_updates(&self, id: i64) -> Result<()> {
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
        ) -> Result<HashMap<i64, Vec<VideoObjectProxy>>> {
            self.with_payload_item(frame_id, |payload| match payload {
                PipelinePayload::Frame(frame, _, ctx) => {
                    let _span = Pipeline::get_nested_span(
                        format!("{}/access-objects", self.stage_name),
                        ctx,
                    )
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

    #[derive(Debug, Default)]
    pub struct Pipeline {
        id_counter: std::sync::atomic::AtomicI64,
        frame_counter: std::sync::atomic::AtomicI64,
        root_spans: RwLock<HashMap<i64, Context>>,
        stages: HashMap<String, PipelineStage>,
        frame_locations: RwLock<HashMap<i64, String>>,
        sampling_period: OnceLock<i64>,
        root_span_name: OnceLock<String>,
    }

    impl Pipeline {
        pub fn new(stages: Vec<(String, PipelineStagePayloadType)>) -> Result<Self> {
            let mut pipeline = Self::default();
            for (name, stage_type) in stages {
                pipeline.add_stage(name, stage_type)?;
            }
            Ok(pipeline)
        }

        pub fn get_id_locations_len(&self) -> usize {
            self.frame_locations.read().len()
        }

        pub fn set_root_span_name(&self, name: String) -> Result<()> {
            self.root_span_name.set(name).map_err(|last| {
                anyhow::anyhow!(
                    "Root span name can only be set once. Current value: {}",
                    last
                )
            })
        }

        pub fn set_sampling_period(&self, period: i64) -> Result<()> {
            self.sampling_period.set(period).map_err(|last| {
                anyhow::anyhow!(
                    "Sampling period can only be set once. Current value: {}",
                    last
                )
            })
        }

        pub fn get_sampling_period(&self) -> &i64 {
            self.sampling_period.get_or_init(|| 0)
        }

        pub fn get_root_span_name(&self) -> &String {
            self.root_span_name
                .get_or_init(|| DEFAULT_ROOT_SPAN_NAME.to_owned())
        }

        fn get_stage_span(&self, id: i64, span_name: String) -> Context {
            let bind = self.root_spans.read();
            let ctx = bind.get(&id).unwrap();

            if ctx.span().span_context().trace_id() == TraceId::INVALID {
                return Context::default();
            }

            let span = get_tracer().build_with_context(SpanBuilder::from_name(span_name), ctx);
            Context::current_with_span(span)
        }

        pub(self) fn get_nested_span(span_name: String, parent_ctx: &Context) -> Context {
            if parent_ctx.span().span_context().trace_id() == TraceId::INVALID {
                return Context::default();
            }

            let span =
                get_tracer().build_with_context(SpanBuilder::from_name(span_name), parent_ctx);
            Context::current_with_span(span)
        }

        fn add_stage(&mut self, name: String, stage_type: PipelineStagePayloadType) -> Result<()> {
            if self.stages.contains_key(&name) {
                bail!("Stage already exists")
            }
            self.stages.insert(
                name.to_owned(),
                PipelineStage {
                    stage_name: name,
                    stage_type,
                    payload: Default::default(),
                },
            );
            Ok(())
        }

        pub fn get_stage_type(&self, name: &str) -> Option<PipelineStagePayloadType> {
            self.stages.get(name).map(|stage| stage.stage_type.clone())
        }

        pub fn add_frame_update(&self, frame_id: i64, update: VideoFrameUpdate) -> Result<()> {
            let stage_name = &self.get_stage_for_id(frame_id)?;
            if let Some(stage) = self.stages.get(stage_name) {
                stage.add_frame_update(frame_id, update)
            } else {
                bail!("Stage not found")
            }
        }

        pub fn add_batched_frame_update(
            &self,
            batch_id: i64,
            frame_id: i64,
            update: VideoFrameUpdate,
        ) -> Result<()> {
            let stage_name = &self.get_stage_for_id(batch_id)?;
            if let Some(stage) = self.stages.get(stage_name) {
                stage.add_batched_frame_update(batch_id, frame_id, update)
            } else {
                bail!("Stage not found")
            }
        }

        pub fn add_frame(&self, stage_name: &str, frame: VideoFrameProxy) -> Result<i64> {
            let sampling_period = self.get_sampling_period();
            let next_frame = self.frame_counter.load(Ordering::SeqCst) + 1;
            let ctx = if *sampling_period <= 0 || next_frame % *sampling_period != 0 {
                Context::default()
            } else {
                get_tracer().in_span(self.get_root_span_name().clone(), |cx| cx)
            };
            self.add_frame_with_telemetry(stage_name, frame, ctx)
        }

        pub fn add_frame_with_telemetry(
            &self,
            stage_name: &str,
            frame: VideoFrameProxy,
            parent_ctx: Context,
        ) -> Result<i64> {
            if matches!(
                self.get_stage_type(stage_name),
                Some(PipelineStagePayloadType::Batch)
            ) {
                bail!("Stage does not accept batched frames")
            }

            self.frame_counter.fetch_add(1, Ordering::SeqCst);
            let id_counter = self.id_counter.fetch_add(1, Ordering::SeqCst) + 1;

            if parent_ctx.span().span_context().trace_id() == TraceId::INVALID {
                self.root_spans
                    .write()
                    .insert(id_counter, Context::default());
            } else {
                let span = get_tracer().build_with_context(
                    SpanBuilder::from_name(self.get_root_span_name().clone()),
                    &parent_ctx,
                );

                self.root_spans
                    .write()
                    .insert(id_counter, Context::current_with_span(span));
            }

            let ctx = self.get_stage_span(id_counter, format!("add/{}", stage_name));
            let frame_payload = PipelinePayload::Frame(frame, Vec::new(), ctx);

            if let Some(stage) = self.stages.get(stage_name) {
                stage.add_frame_payload(id_counter, frame_payload)?;
            } else {
                bail!("Stage not found")
            }

            self.frame_locations
                .write()
                .insert(id_counter, stage_name.to_owned());

            Ok(id_counter)
        }

        pub fn delete(&self, id: i64) -> Result<HashMap<i64, Context>> {
            let stage_name = self
                .frame_locations
                .write()
                .remove(&id)
                .ok_or(anyhow::anyhow!("Object location not found"))?;

            if let Some(stage) = self.stages.get(&stage_name) {
                let removed = stage.delete(id);
                if removed.is_none() {
                    bail!("Object not found in stage")
                }

                let mut bind = self.root_spans.write();
                match removed.unwrap() {
                    PipelinePayload::Frame(_, _, ctx) => {
                        ctx.span().end();
                        let root_ctx = bind.remove(&id).unwrap();
                        Ok(HashMap::from([(id, root_ctx)]))
                    }
                    PipelinePayload::Batch(_, _, contexts) => Ok({
                        let mut bind = self.root_spans.write();
                        contexts
                            .into_iter()
                            .map(|(id, ctx)| {
                                ctx.span().end();
                                let root_ctx = bind.remove(&id).unwrap();
                                (id, root_ctx)
                            })
                            .collect()
                    }),
                }
            } else {
                bail!("Stage not found")
            }
        }

        pub fn get_stage_queue_len(&self, stage: &str) -> Result<usize> {
            if let Some(stage) = self.stages.get(stage) {
                Ok(stage.len())
            } else {
                bail!("Stage not found")
            }
        }

        fn get_stage_for_id(&self, id: i64) -> Result<String> {
            let bind = self.frame_locations.read();
            if let Some(stage) = bind.get(&id) {
                Ok(stage.clone())
            } else {
                bail!("Object location not found")
            }
        }

        fn get_stages_for_ids(&self, ids: &[i64]) -> Result<Vec<(i64, String)>> {
            let bind = self.frame_locations.read();
            let mut results = Vec::with_capacity(ids.len());
            for id in ids {
                let val = bind.get(id);
                if val.is_none() {
                    bail!("Object location not found for {}", id)
                }
                results.push((*id, val.cloned().unwrap()));
            }
            Ok(results)
        }

        pub fn get_independent_frame(&self, frame_id: i64) -> Result<(VideoFrameProxy, Context)> {
            let stage_name = self.get_stage_for_id(frame_id)?;
            if let Some(stage) = self.stages.get(&stage_name) {
                stage.get_independent_frame(frame_id)
            } else {
                bail!("Stage not found")
            }
        }

        pub fn get_batched_frame(
            &self,
            batch_id: i64,
            frame_id: i64,
        ) -> Result<(VideoFrameProxy, Context)> {
            let stage_name = self.get_stage_for_id(batch_id)?;
            if let Some(stage) = self.stages.get(&stage_name) {
                stage.get_batched_frame(batch_id, frame_id)
            } else {
                bail!("Stage not found")
            }
        }

        pub fn get_batch(&self, batch_id: i64) -> Result<(VideoFrameBatch, HashMap<i64, Context>)> {
            let stage_name = self.get_stage_for_id(batch_id)?;
            if let Some(stage) = self.stages.get(&stage_name) {
                stage.get_batch(batch_id)
            } else {
                bail!("Stage not found")
            }
        }

        pub fn apply_updates(&self, id: i64) -> Result<()> {
            let stage_name = self.get_stage_for_id(id)?;
            if let Some(stage) = self.stages.get(&stage_name) {
                stage.apply_updates(id)
            } else {
                bail!("Stage not found")
            }
        }

        pub fn clear_updates(&self, id: i64) -> Result<()> {
            let stage_name = self.get_stage_for_id(id)?;
            if let Some(stage) = self.stages.get(&stage_name) {
                stage.clear_updates(id)
            } else {
                bail!("Stage not found")
            }
        }

        fn update_frame_locations(&self, ids: &[i64], stage_name: &str) {
            self.frame_locations
                .write()
                .extend(ids.iter().map(|id| (*id, stage_name.to_owned())));
        }

        fn check_ids_in_the_same_stage(&self, ids: &[i64]) -> Result<String> {
            if ids.is_empty() {
                bail!("Object IDs cannot be empty")
            }

            let mut stage_names = self
                .get_stages_for_ids(ids)?
                .into_iter()
                .map(|(_, name)| name);

            let stage_name = stage_names.next().unwrap();

            for current_name in stage_names {
                if current_name != stage_name {
                    bail!("All objects must be in the same stage")
                }
            }
            Ok(stage_name)
        }

        pub fn move_as_is(&self, dest_stage_name: &str, object_ids: Vec<i64>) -> Result<()> {
            let source_stage_name = self.check_ids_in_the_same_stage(&object_ids)?;

            if self.get_stage_type(&source_stage_name) != self.get_stage_type(dest_stage_name) {
                bail!("The source stage type must be the same as the destination stage type")
            }

            let source_stage_opt = self.stages.get(&source_stage_name);
            if source_stage_opt.is_none() {
                bail!("Source stage not found")
            }

            let dest_stage_opt = self.stages.get(dest_stage_name);
            if dest_stage_opt.is_none() {
                bail!("Destination stage not found")
            }

            let removed_objects = source_stage_opt
                .map(|stage| stage.delete_many(&object_ids))
                .unwrap();

            self.update_frame_locations(&object_ids, dest_stage_name);

            let mut payloads = Vec::with_capacity(removed_objects.len());
            for (id, payload) in removed_objects {
                let payload = match payload {
                    PipelinePayload::Frame(frame, updates, ctx) => {
                        ctx.span().end();
                        let ctx = self.get_stage_span(id, format!("stage/{}", dest_stage_name));
                        PipelinePayload::Frame(frame, updates, ctx)
                    }
                    PipelinePayload::Batch(batch, updates, contexts) => {
                        let mut new_contexts = HashMap::new();
                        for (id, ctx) in contexts.iter() {
                            ctx.span().end();
                            let ctx =
                                self.get_stage_span(*id, format!("stage/{}", dest_stage_name));
                            new_contexts.insert(*id, ctx);
                        }
                        PipelinePayload::Batch(batch, updates, new_contexts)
                    }
                };
                payloads.push((id, payload));
            }

            dest_stage_opt
                .map(|dest_stage| dest_stage.add_payloads(payloads))
                .unwrap()?;

            Ok(())
        }

        pub fn move_and_pack_frames(
            &self,
            dest_stage_name: &str,
            frame_ids: Vec<i64>,
        ) -> Result<i64> {
            let source_stage_name = self.check_ids_in_the_same_stage(&frame_ids)?;

            if matches!(
                self.get_stage_type(&source_stage_name),
                Some(PipelineStagePayloadType::Batch)
            ) || matches!(
                self.get_stage_type(dest_stage_name),
                Some(PipelineStagePayloadType::Frame)
            ) {
                bail!("Source stage must contain independent frames and destination stage must contain batched frames")
            }

            let batch_id = self.id_counter.fetch_add(1, Ordering::SeqCst) + 1;
            let source_stage_opt = self.stages.get(&source_stage_name);
            if source_stage_opt.is_none() {
                bail!("Source stage not found")
            }

            let dest_stage_opt = self.stages.get(dest_stage_name);
            if dest_stage_opt.is_none() {
                bail!("Destination stage not found")
            }

            self.update_frame_locations(&frame_ids, dest_stage_name);

            let mut batch = VideoFrameBatch::new();
            let mut batch_updates = Vec::new();
            let mut contexts = HashMap::new();

            for id in frame_ids {
                if let Some(payload) = source_stage_opt
                    .map(|source_stage| source_stage.delete(id))
                    .unwrap()
                {
                    match payload {
                        PipelinePayload::Frame(frame, updates, ctx) => {
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

            let payload = PipelinePayload::Batch(batch, batch_updates, contexts);

            dest_stage_opt
                .map(|dest_stage| dest_stage.add_batch_payload(batch_id, payload))
                .unwrap()?;

            self.frame_locations
                .write()
                .insert(batch_id, dest_stage_name.to_owned());

            Ok(batch_id)
        }

        pub fn move_and_unpack_batch(
            &self,
            dest_stage_name: &str,
            batch_id: i64,
        ) -> Result<Vec<i64>> {
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

            let source_stage_opt = self.stages.get(source_stage_name);
            if source_stage_opt.is_none() {
                bail!("Source stage not found")
            }

            let dest_stage_opt = self.stages.get(dest_stage_name);
            if dest_stage_opt.is_none() {
                bail!("Destination stage not found")
            }

            let (batch, updates, mut contexts) = if let Some(payload) = source_stage_opt
                .map(|stage| stage.delete(batch_id))
                .unwrap()
            {
                match payload {
                    PipelinePayload::Batch(batch, updates, contexts) => (batch, updates, contexts),
                    _ => bail!("Source stage must contain batch"),
                }
            } else {
                bail!("Batch not found in source stage")
            };

            self.frame_locations.write().remove(&batch_id);

            let frame_ids = batch.frames.keys().cloned().collect::<Vec<_>>();
            self.update_frame_locations(&frame_ids, dest_stage_name);

            let mut payloads = HashMap::with_capacity(batch.frames.len());
            for (frame_id, frame) in batch.frames {
                let ctx = contexts.remove(&frame_id).unwrap();
                ctx.span().end();
                let ctx = self.get_stage_span(frame_id, format!("stage/{}", dest_stage_name));

                payloads.insert(frame_id, PipelinePayload::Frame(frame, Vec::new(), ctx));
            }

            for (frame_id, update) in updates {
                if let Some(frame) = payloads.get_mut(&frame_id) {
                    match frame {
                        PipelinePayload::Frame(_, updates, _) => {
                            updates.push(update);
                        }
                        _ => bail!("Destination stage must contain independent frames"),
                    }
                } else {
                    bail!("Frame not found in destination stage")
                }
            }
            dest_stage_opt
                .map(|dest_stage| dest_stage.add_payloads(payloads))
                .unwrap()?;

            Ok(frame_ids)
        }

        pub fn access_objects(
            &self,
            frame_id: i64,
            query: &MatchQuery,
        ) -> Result<HashMap<i64, Vec<VideoObjectProxy>>> {
            let stage_name = &self.get_stage_for_id(frame_id)?;
            let stage_opt = self.stages.get(stage_name);
            if stage_opt.is_none() {
                bail!("Stage {} not found", stage_name);
            }

            stage_opt
                .map(|stage| stage.access_objects(frame_id, query))
                .unwrap()
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::pipeline2::implementation::{Pipeline, PipelineStagePayloadType};
        use crate::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
        use crate::primitives::frame_update::VideoFrameUpdate;
        use crate::primitives::{Attribute, AttributeMethods};
        use crate::test::gen_frame;
        use opentelemetry::global;
        use opentelemetry::sdk::export::trace::stdout;
        use opentelemetry::sdk::propagation::TraceContextPropagator;
        use opentelemetry::trace::{TraceContextExt, TraceId};
        use std::io::sink;
        use std::sync::atomic::Ordering;

        fn create_pipeline() -> anyhow::Result<Pipeline> {
            let pipeline = Pipeline::new(vec![
                ("input".to_string(), PipelineStagePayloadType::Frame),
                ("proc1".to_string(), PipelineStagePayloadType::Batch),
                ("proc2".to_string(), PipelineStagePayloadType::Batch),
                ("output".to_string(), PipelineStagePayloadType::Frame),
            ])?;
            Ok(pipeline)
        }

        #[test]
        fn test_new_pipeline() -> anyhow::Result<()> {
            let pipeline = create_pipeline()?;
            assert_eq!(pipeline.id_counter.load(Ordering::SeqCst), 0);
            assert_eq!(pipeline.stages.len(), 4);
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
            let pipeline = create_pipeline()?;
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
            let pipeline = create_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let batch_id = pipeline.move_and_pack_frames("proc1", vec![id])?;

            assert!(pipeline.get_independent_frame(id).is_err());

            assert!(pipeline.get_batch(batch_id).is_ok());
            assert!(pipeline.get_batched_frame(batch_id, id).is_ok());

            Ok(())
        }

        #[test]
        fn test_batch_to_frame() -> anyhow::Result<()> {
            let pipeline = create_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let batch_id = pipeline.move_and_pack_frames("proc2", vec![id])?;
            pipeline.move_and_unpack_batch("output", batch_id)?;
            let _frame = pipeline.get_independent_frame(id)?;
            Ok(())
        }

        #[test]
        fn test_batch_to_batch() -> anyhow::Result<()> {
            let pipeline = create_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let batch_id = pipeline.move_and_pack_frames("proc1", vec![id])?;
            pipeline.move_as_is("proc2", vec![batch_id])?;
            let _batch = pipeline.get_batch(batch_id)?;
            let _frame = pipeline.get_batched_frame(batch_id, id)?;
            Ok(())
        }

        #[test]
        fn test_frame_to_frame() -> anyhow::Result<()> {
            let pipeline = create_pipeline()?;
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
            let pipeline = create_pipeline()?;
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
            let pipeline = create_pipeline()?;
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

            let pipeline = create_pipeline()?;
            pipeline.set_sampling_period(2)?;

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

            let pipeline = create_pipeline()?;
            pipeline.set_sampling_period(0)?;

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

            let pipeline = create_pipeline()?;
            pipeline.set_sampling_period(1)?;

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
