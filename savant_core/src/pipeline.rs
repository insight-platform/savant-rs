use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Result;
use hashbrown::HashMap;
use opentelemetry::Context;

pub use implementation::PipelineConfiguration;
pub use implementation::PipelineConfigurationBuilder;

use crate::match_query::MatchQuery;
use crate::pipeline::stage::PipelineStage;
use crate::primitives::attribute_value::AttributeValue;
use crate::primitives::frame::VideoFrameProxy;
use crate::primitives::frame_batch::VideoFrameBatch;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::object::BorrowedVideoObject;
use crate::webserver::{register_pipeline, unregister_pipeline};

const MAX_TRACKED_STREAMS: usize = 8192; // defines how many streams are tracked for the frame ordering

pub mod stage;
pub mod stage_function_loader;
pub mod stage_plugin_sample;
pub mod stats;

pub trait PipelineStageFunction: Send {
    fn set_pipeline(&mut self, pipeline: Pipeline);
    fn get_pipeline(&self) -> &Option<Pipeline>;
    fn call(
        &self,
        id: i64,
        stage: &PipelineStage,
        order: PipelineStageFunctionOrder,
        payload: &mut PipelinePayload,
    ) -> Result<()>;
}

#[derive(Debug)]
pub enum PipelineStageFunctionOrder {
    Ingress,
    Egress,
}

#[repr(C)]
#[derive(Default, Debug, Clone)]
pub struct PluginParams {
    pub params: HashMap<String, AttributeValue>,
}

pub type PipelineStageFunctionFactory =
    fn(name: &str, parameters: PluginParams) -> *mut (dyn PipelineStageFunction);

#[derive(Clone, Debug, PartialEq)]
pub enum PipelineStagePayloadType {
    Frame,
    Batch,
}

#[derive(Debug)]
pub enum PipelinePayload {
    Frame(
        VideoFrameProxy,
        Vec<VideoFrameUpdate>,
        Context,
        Option<String>,
        SystemTime,
    ),
    Batch(
        VideoFrameBatch,
        Vec<(i64, VideoFrameUpdate)>,
        HashMap<i64, Context>,
        Option<String>,
        Vec<SystemTime>,
    ),
}

#[derive(Clone, Default, Debug)]
pub struct Pipeline(pub(crate) Arc<implementation::Pipeline>);

impl Pipeline {
    #[allow(clippy::type_complexity)]
    pub fn new(
        name: &str,
        stages: Vec<(
            String,
            PipelineStagePayloadType,
            Option<Box<dyn PipelineStageFunction>>,
            Option<Box<dyn PipelineStageFunction>>,
        )>,
        configuration: PipelineConfiguration,
    ) -> Result<Self> {
        let pipeline = Arc::new(implementation::Pipeline::new(name, stages, configuration)?);
        let p = Self(pipeline);
        register_pipeline(p.0.clone());
        Ok(p)
    }

    pub fn get_name(&self) -> String {
        self.0.get_name()
    }

    pub fn get_stage_name(&self, stage_id: usize) -> Option<String> {
        self.0.get_stage_name(stage_id)
    }

    pub fn get_stat_records(&self, max_n: usize) -> Vec<stats::FrameProcessingStatRecord> {
        self.0.get_stat_records(max_n)
    }

    pub fn get_stat_records_newer_than(&self, id: i64) -> Vec<stats::FrameProcessingStatRecord> {
        self.0.get_stat_records_newer_than(id)
    }

    pub fn log_final_fps(&self) {
        self.0.log_final_fps()
    }

    pub fn memory_handle(&self) -> usize {
        self as *const Self as usize
    }

    pub fn clear_source_ordering(&self, source_id: &str) -> Result<()> {
        self.0.clear_source_ordering(source_id)
    }

    pub fn set_root_span_name(&self, name: &str) -> Result<()> {
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

    pub fn get_stage_type(&self, name: &str) -> Result<PipelineStagePayloadType> {
        self.0.find_stage_type(name, 0)
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
    ) -> Result<HashMap<i64, Vec<BorrowedVideoObject>>> {
        self.0.access_objects(frame_id, query)
    }

    pub fn get_id_locations_len(&self) -> usize {
        self.0.get_id_locations_len()
    }

    pub fn get_keyframe_history(&self, frame: &VideoFrameProxy) -> Option<Vec<(u128, i64)>> {
        self.0.get_keyframe_history(frame)
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unregister_pipeline(self.0.clone());
    }
}

pub(super) mod implementation {
    use std::collections::VecDeque;
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::sync::OnceLock;
    use std::time::SystemTime;

    use anyhow::{anyhow, bail, Result};
    use derive_builder::Builder;
    use hashbrown::HashMap;
    use lru::LruCache;
    use opentelemetry::trace::{SpanBuilder, TraceContextExt, Tracer};
    use opentelemetry::{Context, KeyValue};

    use crate::get_tracer;
    use crate::match_query::MatchQuery;
    use crate::pipeline::stage::PipelineStage;
    use crate::pipeline::stats::{FrameProcessingStatRecord, Stats};
    use crate::pipeline::{
        PipelinePayload, PipelineStageFunction, PipelineStagePayloadType, MAX_TRACKED_STREAMS,
    };
    use crate::primitives::frame::VideoFrameProxy;
    use crate::primitives::frame_batch::VideoFrameBatch;
    use crate::primitives::frame_update::VideoFrameUpdate;
    use crate::primitives::object::BorrowedVideoObject;
    use crate::rwlock::SavantRwLock;

    const DEFAULT_ROOT_SPAN_NAME: &str = "video_pipeline";

    #[derive(Builder, Default, Debug, Clone)]
    pub struct PipelineConfiguration {
        #[builder(default = "false")]
        pub append_frame_meta_to_otlp_span: bool,
        #[builder(default = "Some(1000)")]
        pub timestamp_period: Option<i64>,
        #[builder(default = "Some(1000)")]
        pub frame_period: Option<i64>,
        #[builder(default = "10")]
        pub collection_history: usize,
        #[builder(default = "60")]
        pub keyframe_history: usize,
    }

    #[derive(Debug)]
    pub struct Pipeline {
        name: OnceLock<String>,
        id_counter: AtomicI64,
        frame_counter: AtomicI64,
        root_spans: SavantRwLock<HashMap<i64, Context>>,
        stages: Vec<PipelineStage>,
        frame_locations: SavantRwLock<HashMap<i64, usize>>,
        frame_ordering: SavantRwLock<LruCache<String, i64>>,
        keyframe_tracking: SavantRwLock<LruCache<u64, u128>>,
        keyframe_history: SavantRwLock<LruCache<u64, VecDeque<(u128, i64)>>>,
        sampling_period: OnceLock<i64>,
        root_span_name: OnceLock<String>,
        configuration: PipelineConfiguration,
        stats: Stats,
    }

    impl Default for Pipeline {
        fn default() -> Self {
            Self {
                name: OnceLock::new(),
                id_counter: AtomicI64::new(0),
                frame_counter: AtomicI64::new(0),
                root_spans: SavantRwLock::new(HashMap::new()),
                stages: Vec::new(),
                frame_locations: SavantRwLock::new(HashMap::new()),
                frame_ordering: SavantRwLock::new(LruCache::new(
                    NonZeroUsize::try_from(MAX_TRACKED_STREAMS).unwrap(),
                )),
                keyframe_tracking: SavantRwLock::new(LruCache::new(
                    NonZeroUsize::try_from(MAX_TRACKED_STREAMS).unwrap(),
                )),
                keyframe_history: SavantRwLock::new(LruCache::new(
                    NonZeroUsize::try_from(MAX_TRACKED_STREAMS).unwrap(),
                )),
                sampling_period: OnceLock::new(),
                root_span_name: OnceLock::new(),
                configuration: PipelineConfiguration::default(),
                stats: Stats::default(),
            }
        }
    }

    unsafe impl Send for Pipeline {}

    unsafe impl Sync for Pipeline {}

    impl Pipeline {
        pub fn get_name(&self) -> String {
            self.name.get().cloned().expect("Pipeline name is not set")
        }

        pub fn get_stage_name(&self, stage_id: usize) -> Option<String> {
            self.stages.get(stage_id).map(|s| s.name.clone())
        }

        fn add_stage(
            &mut self,
            name: String,
            stage_type: PipelineStagePayloadType,
            ingress_function: Option<Box<dyn PipelineStageFunction>>,
            egress_function: Option<Box<dyn PipelineStageFunction>>,
        ) -> Result<()> {
            if self.find_stage(&name, 0).is_ok() {
                bail!("Stage with name {} already exists", name)
            }

            let stage = PipelineStage::new(
                self.stages.len(),
                name,
                stage_type,
                ingress_function,
                egress_function,
            );
            let stat = stage.get_stat();
            self.stats.add_stage_stats(stat);
            self.stages.push(stage);
            Ok(())
        }

        #[allow(clippy::type_complexity)]
        pub fn new(
            name: &str,
            stages: Vec<(
                String,
                PipelineStagePayloadType,
                Option<Box<dyn PipelineStageFunction>>,
                Option<Box<dyn PipelineStageFunction>>,
            )>,
            configuration: PipelineConfiguration,
        ) -> Result<Self> {
            let stats = Stats::new(
                configuration.collection_history,
                configuration.frame_period,
                configuration.timestamp_period,
            );
            let mut pipeline = Self {
                name: OnceLock::from(name.to_string()),
                configuration,
                stats,
                ..Default::default()
            };

            for (name, stage_type, ingress_function, egress_function) in stages {
                pipeline.add_stage(name, stage_type, ingress_function, egress_function)?;
            }
            Ok(pipeline)
        }

        pub fn get_stat_records(&self, max_n: usize) -> Vec<FrameProcessingStatRecord> {
            self.stats.get_records(max_n)
        }

        pub fn get_stat_records_newer_than(&self, id: i64) -> Vec<FrameProcessingStatRecord> {
            self.stats.get_records_newer_than(id)
        }

        pub fn log_final_fps(&self) {
            self.stats.log_final_fps()
        }

        pub fn get_id_locations_len(&self) -> usize {
            self.frame_locations.read().len()
        }

        pub fn set_root_span_name(&self, name: &str) -> Result<()> {
            self.root_span_name.set(name.to_string()).map_err(|last| {
                anyhow::anyhow!(
                    "The root span name can only be set once. The current value: {}",
                    last
                )
            })
        }

        pub fn set_sampling_period(&self, period: i64) -> Result<()> {
            self.sampling_period.set(period).map_err(|last| {
                anyhow::anyhow!(
                    "The sampling period can only be set once. The current value: {}",
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

            if !ctx.span().span_context().is_valid() {
                return Context::default();
            }

            let span = get_tracer().build_with_context(SpanBuilder::from_name(span_name), ctx);
            Context::current_with_span(span)
        }

        pub(crate) fn get_nested_span(span_name: String, parent_ctx: &Context) -> Context {
            if !parent_ctx.span().span_context().is_valid() {
                return Context::default();
            }

            let span =
                get_tracer().build_with_context(SpanBuilder::from_name(span_name), parent_ctx);
            Context::current_with_span(span)
        }

        pub fn find_stage_type(
            &self,
            name: &str,
            start_from: usize,
        ) -> Result<PipelineStagePayloadType> {
            let (_, s) = self.find_stage(name, start_from)?;
            Ok(s.stage_type.clone())
        }

        pub fn add_frame_update(&self, frame_id: i64, update: VideoFrameUpdate) -> Result<()> {
            let cur_stage = self.get_stage_for_id(frame_id)?;
            if let Some(stage) = self.stages.get(cur_stage) {
                stage.add_frame_update(frame_id, update)
            } else {
                bail!(
                    "Stage ID={} not found when adding update to frame {}",
                    cur_stage,
                    frame_id
                )
            }
        }

        pub fn add_batched_frame_update(
            &self,
            batch_id: i64,
            frame_id: i64,
            update: VideoFrameUpdate,
        ) -> Result<()> {
            let cur_stage = self.get_stage_for_id(batch_id)?;
            if let Some(stage) = self.stages.get(cur_stage) {
                stage.add_batched_frame_update(batch_id, frame_id, update)
            } else {
                bail!(
                    "Stage ID={} not found when adding update to frame {} in batch {}",
                    cur_stage,
                    frame_id,
                    batch_id
                )
            }
        }

        fn find_stage(
            &self,
            stage_name: &str,
            start_from: usize,
        ) -> Result<(usize, &PipelineStage)> {
            if self.stages.is_empty() {
                bail!("Pipeline is empty. Looked for stage {}", stage_name)
            }

            let res = self.stages[start_from..]
                .iter()
                .enumerate()
                .map(|(i, s)| (i + start_from, s))
                .find(|(_, s)| s.name == stage_name);

            if let Some((index, stage)) = res {
                Ok((index, stage))
            } else {
                let source_stage = self.stages[start_from].name.as_str();
                // try to start from the beginning to find the out of order situation
                let res = self
                    .stages
                    .iter()
                    .enumerate()
                    .find(|(_, s)| s.name == stage_name);

                if let Some((index, _)) = res {
                    bail!(
                        "Stage {} is out of order. Stage index {} < {}, must be greater. Source stage is {}",
                        stage_name,
                        index,
                        start_from,
                        source_stage
                    )
                } else {
                    bail!(
                        "Stage {} not found. Source stage is {}",
                        stage_name,
                        source_stage
                    )
                }
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
            mut frame: VideoFrameProxy,
            parent_ctx: Context,
        ) -> Result<i64> {
            if matches!(
                self.find_stage_type(stage_name, 0)?,
                PipelineStagePayloadType::Batch
            ) {
                bail!("Stage does not accept batched frames")
            }

            self.frame_counter.fetch_add(1, Ordering::SeqCst);
            let id_counter = self.id_counter.fetch_add(1, Ordering::SeqCst) + 1;
            let source_id = frame.get_source_id();

            if !parent_ctx.span().span_context().is_valid() {
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
            let source_id_compatibility_hash = frame.stream_compatibility_hash();
            let mut ordering = self.frame_ordering.write();
            let prev_ordering_seq = ordering.get(&source_id);
            if let Some(prev) = prev_ordering_seq {
                frame.set_previous_frame_seq_id(Some(*prev));
            } else {
                frame.set_previous_frame_seq_id(None);
            }
            ordering.put(source_id, id_counter);

            let mut keyframe_tracking = self.keyframe_tracking.write();
            let mut keyframe_history = self.keyframe_history.write();
            let last_keyframe = keyframe_tracking.get(&source_id_compatibility_hash);
            if let Some(last) = last_keyframe {
                frame.set_previous_keyframe(Some(*last));
            } else {
                frame.set_previous_keyframe(None);
            }
            if let Some(true) = frame.get_keyframe() {
                keyframe_tracking.put(source_id_compatibility_hash, frame.get_uuid_u128());
            }

            let history_entry = keyframe_history.get_mut(&source_id_compatibility_hash);
            if let Some(h) = history_entry {
                h.push_back((frame.get_uuid_u128(), frame.get_pts()));
                if h.len() > self.configuration.keyframe_history {
                    h.pop_front();
                }
            } else {
                keyframe_history.put(
                    source_id_compatibility_hash,
                    VecDeque::from(vec![(frame.get_uuid_u128(), frame.get_pts())]),
                );
            }

            let ctx = self.get_stage_span(id_counter, format!("add/{}", stage_name));
            let frame_payload =
                PipelinePayload::Frame(frame, Vec::new(), ctx, None, SystemTime::now());

            let (index, stage) = self.find_stage(stage_name, 0)?;
            stage.add_frame_payload(id_counter, frame_payload)?;
            self.frame_locations.write().insert(id_counter, index);

            log::trace!(target: "savant_rs::pipeline", "Added frame {} to stage {}", id_counter, stage_name);
            Ok(id_counter)
        }

        pub fn get_keyframe_history(&self, frame: &VideoFrameProxy) -> Option<Vec<(u128, i64)>> {
            let mut keyframe_history = self.keyframe_history.write();
            keyframe_history
                .get(&frame.stream_compatibility_hash())
                .map(|h| h.iter().cloned().collect())
        }

        pub fn clear_source_ordering(&self, source_id: &str) -> Result<()> {
            let mut ordering = self.frame_ordering.write();
            ordering.pop(source_id).ok_or_else(|| {
                anyhow!(
                    "Unable to remove ordering info for source id = {}",
                    source_id
                )
            })?;
            Ok(())
        }

        fn add_frame_json(&self, frame: &VideoFrameProxy, ctx: &Context) {
            if self.configuration.append_frame_meta_to_otlp_span {
                let json = frame.get_json();
                ctx.span().set_attribute(KeyValue::new("frame_json", json));
            }
        }

        pub fn delete(&self, id: i64) -> Result<HashMap<i64, Context>> {
            let stage = self
                .frame_locations
                .write()
                .remove(&id)
                .ok_or(anyhow::anyhow!("Object {} location not found", id))?;

            self.stats.kick_off();

            if let Some(stage) = self.stages.get(stage) {
                log::trace!(target: "savant_rs::pipeline", "Delete object {} from the stage {}", id, stage.name);
                let removed = stage.delete(id)?;
                if removed.is_none() {
                    bail!("Object {} is not found in the stage {}", id, stage.name)
                }

                let mut bind = self.root_spans.write();
                match removed.unwrap() {
                    PipelinePayload::Frame(frame, _, ctx, _, _) => {
                        self.stats.register_frame(frame.get_object_count());
                        self.add_frame_json(&frame, &ctx);
                        ctx.span().end();
                        let root_ctx = bind.remove(&id).unwrap();
                        Ok(HashMap::from([(id, root_ctx)]))
                    }
                    PipelinePayload::Batch(batch, _, contexts, _, _) => Ok({
                        let mut bind = self.root_spans.write();
                        contexts
                            .into_iter()
                            .map(|(frame_id, ctx)| {
                                let frame_opt = batch.get(frame_id);
                                if let Some(frame) = frame_opt {
                                    self.stats.register_frame(frame.get_object_count());
                                    self.add_frame_json(&frame, &ctx);
                                } else {
                                    bail!(
                                        "Frame {} not found in batch {} in the stage {}",
                                        frame_id,
                                        id,
                                        stage.name
                                    )
                                }
                                ctx.span().end();
                                let root_ctx = bind.remove(&id).unwrap();
                                Ok((id, root_ctx))
                            })
                            .collect::<Result<HashMap<_, _>, _>>()?
                    }),
                }
            } else {
                bail!("Stage ID={} not found (when removing object {})", stage, id)
            }
        }

        pub fn get_stage_queue_len(&self, stage: &str) -> Result<usize> {
            let (_, stage) = self.find_stage(stage, 0)?;
            Ok(stage.len())
        }

        fn get_stage_for_id(&self, id: i64) -> Result<usize> {
            let bind = self.frame_locations.read();
            if let Some(stage) = bind.get(&id) {
                Ok(*stage)
            } else {
                bail!("Object {} location not found.", id)
            }
        }

        fn get_stages_for_ids(&self, ids: &[i64]) -> Result<Vec<(i64, usize)>> {
            let bind = self.frame_locations.read();
            let mut results = Vec::with_capacity(ids.len());
            for id in ids {
                let val = bind.get(id);
                if val.is_none() {
                    bail!("Object {} location not found.", id)
                }
                results.push((*id, *val.unwrap()));
            }
            Ok(results)
        }

        pub fn get_independent_frame(&self, frame_id: i64) -> Result<(VideoFrameProxy, Context)> {
            let stage = self.get_stage_for_id(frame_id)?;
            if let Some(stage) = self.stages.get(stage) {
                stage.get_independent_frame(frame_id)
            } else {
                bail!("Stage not found (when getting frame {})", frame_id)
            }
        }

        pub fn get_batched_frame(
            &self,
            batch_id: i64,
            frame_id: i64,
        ) -> Result<(VideoFrameProxy, Context)> {
            let stage = self.get_stage_for_id(batch_id)?;
            if let Some(stage) = self.stages.get(stage) {
                stage.get_batched_frame(batch_id, frame_id)
            } else {
                bail!(
                    "Stage not found (when getting frame {} from batch {})",
                    frame_id,
                    batch_id
                )
            }
        }

        pub fn get_batch(&self, batch_id: i64) -> Result<(VideoFrameBatch, HashMap<i64, Context>)> {
            let stage = self.get_stage_for_id(batch_id)?;
            if let Some(stage) = self.stages.get(stage) {
                stage.get_batch(batch_id)
            } else {
                bail!(
                    "Stage ID={} not found (when getting batch {})",
                    stage,
                    batch_id
                )
            }
        }

        pub fn apply_updates(&self, id: i64) -> Result<()> {
            let stage = self.get_stage_for_id(id)?;
            if let Some(stage) = self.stages.get(stage) {
                stage.apply_updates(id)
            } else {
                bail!(
                    "Stage ID={} not found (when applying updates to object {})",
                    stage,
                    id
                )
            }
        }

        pub fn clear_updates(&self, id: i64) -> Result<()> {
            let stage = self.get_stage_for_id(id)?;
            if let Some(stage) = self.stages.get(stage) {
                stage.clear_updates(id)
            } else {
                bail!(
                    "Stage ID={} not found (when clearing updates to object {})",
                    stage,
                    id
                )
            }
        }

        fn update_frame_locations(&self, ids: &[i64], index: usize) {
            self.frame_locations
                .write()
                .extend(ids.iter().map(|id| (*id, index)));
        }

        fn check_ids_in_the_same_stage(&self, ids: &[i64]) -> Result<usize> {
            if ids.is_empty() {
                bail!("Object IDs cannot be empty")
            }

            let mut stages = self
                .get_stages_for_ids(ids)?
                .into_iter()
                .map(|(_, name)| name);

            let stage = stages.next().unwrap();

            for current_stage in stages {
                if current_stage != stage {
                    bail!(
                        "All objects {:?} must be in the same stage with ID={}",
                        ids,
                        stage
                    )
                }
            }
            Ok(stage)
        }

        pub fn move_as_is(&self, dest_stage_name: &str, object_ids: Vec<i64>) -> Result<()> {
            let source_index = self.check_ids_in_the_same_stage(&object_ids)?;
            let source_stage_opt = self.stages.get(source_index);
            if source_stage_opt.is_none() {
                bail!(
                    "Source stage ID={} not found for object IDs {:?}",
                    source_index,
                    object_ids
                )
            }
            let source_stage = source_stage_opt.unwrap();
            log::trace!(
                target: "savant_rs::pipeline", "Moving objects {:?} of type {:?} as is from stage {} to stage {}", 
                object_ids, source_stage.stage_type, source_stage.name, dest_stage_name);
            let (dest_index, dest_stage) = self.find_stage(dest_stage_name, source_index)?;

            if source_stage.stage_type != dest_stage.stage_type {
                bail!("The source stage type for {} ({:?}) must be the same as the destination stage type for {} ({:?})", 
                    source_stage.name, source_stage.stage_type, dest_stage.name, dest_stage.stage_type)
            }

            let removed_objects = source_stage_opt
                .as_ref()
                .expect("Stage must be defined according to the previous check")
                .delete_many(&object_ids)?;

            self.update_frame_locations(&object_ids, dest_index);

            let mut payloads = Vec::with_capacity(removed_objects.len());
            for (id, payload) in removed_objects {
                let payload = match payload {
                    PipelinePayload::Frame(frame, updates, ctx, source_index, time) => {
                        self.add_frame_json(&frame, &ctx);
                        ctx.span().end();
                        let ctx = self.get_stage_span(id, format!("stage/{}", dest_stage_name));
                        PipelinePayload::Frame(frame, updates, ctx, source_index, time)
                    }
                    PipelinePayload::Batch(batch, updates, contexts, source_index, times) => {
                        let mut new_contexts = HashMap::with_capacity(contexts.len());
                        for (frame_id, ctx) in contexts.iter() {
                            let frame_opt = batch.get(*frame_id);
                            if let Some(frame) = frame_opt {
                                self.add_frame_json(&frame, ctx);
                            } else {
                                bail!("Frame {} not found in batch {}", frame_id, id)
                            }
                            ctx.span().end();
                            let ctx = self
                                .get_stage_span(*frame_id, format!("stage/{}", dest_stage_name));
                            new_contexts.insert(*frame_id, ctx);
                        }
                        PipelinePayload::Batch(batch, updates, new_contexts, source_index, times)
                    }
                };
                payloads.push((id, payload));
            }

            dest_stage.add_payloads(payloads)?;

            Ok(())
        }

        pub fn move_and_pack_frames(
            &self,
            dest_stage_name: &str,
            frame_ids: Vec<i64>,
        ) -> Result<i64> {
            let source_index = self.check_ids_in_the_same_stage(&frame_ids)?;
            let source_stage_opt = self.stages.get(source_index);
            if source_stage_opt.is_none() {
                bail!(
                    "Source stage ID={} not found for frame IDs {:?}",
                    source_index,
                    frame_ids
                )
            }
            let source_stage = source_stage_opt.unwrap();
            log::trace!(target: "savant_rs::pipeline", "Moving and packing frames {:?} from stage {} to stage {}", frame_ids, source_stage.name, dest_stage_name);
            let (dest_index, dest_stage) = self.find_stage(dest_stage_name, source_index)?;

            if matches!(source_stage.stage_type, PipelineStagePayloadType::Batch)
                || matches!(dest_stage.stage_type, PipelineStagePayloadType::Frame)
            {
                bail!("Source stage {} must contain independent frames and destination stage must contain batched frames", source_stage.name)
            }

            let batch_id = self.id_counter.fetch_add(1, Ordering::SeqCst) + 1;

            self.update_frame_locations(&frame_ids, dest_index);

            let default_size = frame_ids.len();

            let mut batch = VideoFrameBatch::with_capacity(default_size);
            let mut batch_updates = Vec::with_capacity(default_size);
            let mut contexts = HashMap::with_capacity(default_size);

            let mut last_stage: Option<String> = None;
            let mut last_times: Vec<SystemTime> = Vec::with_capacity(batch.frames.len());
            for id in frame_ids {
                if let Some(payload) = source_stage_opt
                    .as_ref()
                    .expect("Stage must be defined according to the previous check")
                    .delete(id)?
                {
                    match payload {
                        PipelinePayload::Frame(frame, updates, ctx, ls, lt) => {
                            last_stage = ls;
                            last_times.push(lt);
                            batch.add(id, frame);
                            contexts.insert(id, ctx);
                            for update in updates {
                                batch_updates.push((id, update));
                            }
                        }
                        _ => bail!(
                            "Source stage {} must contain independent frames",
                            source_stage.name
                        ),
                    }
                }
            }

            let contexts = contexts
                .into_iter()
                .map(|(frame_id, ctx)| {
                    let frame_opt = batch.get(frame_id);
                    if let Some(frame) = frame_opt {
                        self.add_frame_json(&frame, &ctx);
                    } else {
                        bail!("Frame {} not found in batch {}", frame_id, batch_id)
                    }
                    ctx.span().end();
                    let ctx = self.get_stage_span(frame_id, format!("stage/{}", dest_stage_name));
                    Ok((frame_id, ctx))
                })
                .collect::<Result<HashMap<_, _>, _>>()?;

            let payload =
                PipelinePayload::Batch(batch, batch_updates, contexts, last_stage, last_times);
            dest_stage.add_batch_payload(batch_id, payload)?;
            self.frame_locations.write().insert(batch_id, dest_index);
            log::trace!(target: "savant_rs::pipeline", "Created batch {} to stage {}", batch_id, dest_stage_name);
            Ok(batch_id)
        }

        pub fn move_and_unpack_batch(
            &self,
            dest_stage_name: &str,
            batch_id: i64,
        ) -> Result<Vec<i64>> {
            let source_index = self.get_stage_for_id(batch_id)?;
            let source_stage_opt = self.stages.get(source_index);
            if source_stage_opt.is_none() {
                bail!(
                    "Source stage ID={} not found for batch {}",
                    source_index,
                    batch_id
                )
            }
            let source_stage = source_stage_opt.unwrap();
            log::trace!(target: "savant_rs::pipeline", "Moving and unpacking batch {} from stage {} to stage {}", batch_id, source_stage.name, dest_stage_name);
            let (dest_index, dest_stage) = self.find_stage(dest_stage_name, source_index)?;

            if matches!(source_stage.stage_type, PipelineStagePayloadType::Frame)
                || matches!(dest_stage.stage_type, PipelineStagePayloadType::Batch)
            {
                bail!("Source stage {} must contain batched frames and destination stage must contain independent frames", source_stage.name)
            }

            let (batch, updates, mut contexts, last_stage, last_times) = if let Some(payload) =
                source_stage_opt
                    .as_ref()
                    .expect("The stage must be defined according to the previous check")
                    .delete(batch_id)?
            {
                match payload {
                    PipelinePayload::Batch(batch, updates, contexts, last_stage, last_times) => {
                        (batch, updates, contexts, last_stage, last_times)
                    }
                    _ => bail!("Source stage {} must contain batch", source_stage.name),
                }
            } else {
                bail!("Batch not found in source stage {}", source_stage.name)
            };

            self.frame_locations.write().remove(&batch_id);

            let frame_ids = batch.frames.keys().cloned().collect::<Vec<_>>();
            self.update_frame_locations(&frame_ids, dest_index);

            let mut payloads = HashMap::with_capacity(batch.frames.len());
            for (frame_id, frame) in batch.frames {
                let ctx = contexts.remove(&frame_id).unwrap();
                self.add_frame_json(&frame, &ctx);
                ctx.span().end();
                let ctx = self.get_stage_span(frame_id, format!("stage/{}", dest_stage_name));

                payloads.insert(
                    frame_id,
                    PipelinePayload::Frame(
                        frame,
                        Vec::new(),
                        ctx,
                        last_stage.clone(),
                        last_times.first().unwrap_or(&SystemTime::now()).to_owned(),
                    ),
                );
            }

            for (frame_id, update) in updates {
                if let Some(frame) = payloads.get_mut(&frame_id) {
                    match frame {
                        PipelinePayload::Frame(_, updates, _, _, _) => {
                            updates.push(update);
                        }
                        _ => bail!(
                            "Destination stage {} must contain independent frames",
                            dest_stage.name
                        ),
                    }
                } else {
                    bail!("Frame not found in destination stage {}", dest_stage.name)
                }
            }

            dest_stage.add_payloads(payloads)?;

            Ok(frame_ids)
        }

        pub fn access_objects(
            &self,
            frame_id: i64,
            query: &MatchQuery,
        ) -> Result<HashMap<i64, Vec<BorrowedVideoObject>>> {
            let stage = self.get_stage_for_id(frame_id)?;
            let stage_opt = self.stages.get(stage);
            if stage_opt.is_none() {
                bail!("Stage ID={} not found", stage);
            }

            stage_opt
                .map(|stage| stage.access_objects(frame_id, query))
                .unwrap()
        }
    }

    #[cfg(test)]
    pub(crate) fn create_test_pipeline() -> anyhow::Result<Pipeline> {
        let pipeline = Pipeline::new(
            "test_pipeline",
            vec![
                (
                    "input".to_string(),
                    PipelineStagePayloadType::Frame,
                    None,
                    None,
                ),
                (
                    "proc1".to_string(),
                    PipelineStagePayloadType::Batch,
                    None,
                    None,
                ),
                (
                    "proc2".to_string(),
                    PipelineStagePayloadType::Batch,
                    None,
                    None,
                ),
                (
                    "output".to_string(),
                    PipelineStagePayloadType::Frame,
                    None,
                    None,
                ),
            ],
            PipelineConfigurationBuilder::default()
                .append_frame_meta_to_otlp_span(true)
                .frame_period(Some(2))
                .build()
                .unwrap(),
        )?;
        Ok(pipeline)
    }

    #[cfg(test)]
    mod tests {
        use std::sync::atomic::Ordering;
        use std::sync::Once;
        use std::thread::sleep;
        use std::time::Duration;

        use opentelemetry::trace::TraceContextExt;

        use crate::pipeline::implementation::{create_test_pipeline, PipelineStagePayloadType};
        use crate::primitives::attribute_value::AttributeValue;
        use crate::primitives::frame_update::VideoFrameUpdate;
        use crate::primitives::{Attribute, WithAttributes};
        use crate::telemetry::{init, TelemetryConfiguration};
        use crate::test::gen_frame;

        static INIT: Once = Once::new();

        fn init_telemetry() {
            INIT.call_once(|| init(&TelemetryConfiguration::no_op()))
        }

        #[test]
        fn test_new_pipeline() -> anyhow::Result<()> {
            let pipeline = create_test_pipeline()?;
            assert_eq!(pipeline.id_counter.load(Ordering::SeqCst), 0);
            assert_eq!(pipeline.stages.len(), 4);
            Ok(())
        }

        #[test]
        fn test_get_stage_type() -> anyhow::Result<()> {
            let pipeline = create_test_pipeline()?;
            assert!(matches!(
                pipeline.find_stage_type("input", 0)?,
                PipelineStagePayloadType::Frame
            ));
            assert!(matches!(
                pipeline.find_stage_type("proc1", 0)?,
                PipelineStagePayloadType::Batch
            ));
            Ok(())
        }

        #[test]
        fn test_find_stages() -> anyhow::Result<()> {
            let pipeline = create_test_pipeline()?;
            assert!(pipeline.find_stage("input", 0).is_ok());
            assert!(pipeline.find_stage("input", 1).is_err());
            Ok(())
        }

        #[test]
        fn test_add_del_frame() -> anyhow::Result<()> {
            let pipeline = create_test_pipeline()?;
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
            let pipeline = create_test_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let batch_id = pipeline.move_and_pack_frames("proc1", vec![id])?;

            assert!(pipeline.get_independent_frame(id).is_err());
            assert_eq!(pipeline.get_stage_queue_len("input")?, 0);
            assert_eq!(pipeline.get_stage_queue_len("proc1")?, 1);
            pipeline.get_batch(batch_id)?;
            pipeline.get_batched_frame(batch_id, id)?;
            Ok(())
        }

        #[test]
        fn test_ordering() -> anyhow::Result<()> {
            let pipeline = create_test_pipeline()?;
            let frame = gen_frame();
            let id = pipeline.add_frame("input", frame.clone())?;
            let res = frame.get_previous_frame_seq_id();
            assert_eq!(res, None);
            let frame = gen_frame();
            let _ = pipeline.add_frame("input", frame.clone())?;
            let res = frame.get_previous_frame_seq_id();
            assert_eq!(res, Some(id));
            Ok(())
        }

        #[test]
        fn test_batch_to_frame() -> anyhow::Result<()> {
            let pipeline = create_test_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let batch_id = pipeline.move_and_pack_frames("proc2", vec![id])?;
            assert_eq!(pipeline.get_stage_queue_len("input")?, 0);
            assert_eq!(pipeline.get_stage_queue_len("proc2")?, 1);
            assert_eq!(pipeline.get_stage_queue_len("output")?, 0);
            pipeline.move_and_unpack_batch("output", batch_id)?;
            assert_eq!(pipeline.get_stage_queue_len("input")?, 0);
            assert_eq!(pipeline.get_stage_queue_len("proc2")?, 0);
            assert_eq!(pipeline.get_stage_queue_len("output")?, 1);
            let _frame = pipeline.get_independent_frame(id)?;
            Ok(())
        }

        #[test]
        fn test_batch_to_batch() -> anyhow::Result<()> {
            let pipeline = create_test_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let batch_id = pipeline.move_and_pack_frames("proc1", vec![id])?;
            pipeline.move_as_is("proc2", vec![batch_id])?;
            assert_eq!(pipeline.get_stage_queue_len("input")?, 0);
            assert_eq!(pipeline.get_stage_queue_len("proc1")?, 0);
            assert_eq!(pipeline.get_stage_queue_len("proc2")?, 1);
            let _batch = pipeline.get_batch(batch_id)?;
            let _frame = pipeline.get_batched_frame(batch_id, id)?;
            Ok(())
        }

        #[test]
        fn test_frame_to_frame() -> anyhow::Result<()> {
            let pipeline = create_test_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            pipeline.move_as_is("output", vec![id])?;
            assert_eq!(pipeline.get_stage_queue_len("input")?, 0);
            assert_eq!(pipeline.get_stage_queue_len("output")?, 1);
            let _frame = pipeline.get_independent_frame(id)?;
            Ok(())
        }

        fn get_update() -> VideoFrameUpdate {
            let mut update = VideoFrameUpdate::default();
            update.add_frame_attribute(Attribute::persistent(
                "update",
                "attribute",
                vec![AttributeValue::string("1".into(), None)],
                &Some("test"),
                false,
            ));
            update
        }

        #[test]
        fn test_frame_update() -> anyhow::Result<()> {
            let pipeline = create_test_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let update = get_update();
            pipeline.add_frame_update(id, update)?;
            pipeline.apply_updates(id)?;
            let (frame, _) = pipeline.get_independent_frame(id)?;
            frame.get_attribute("update", "attribute").unwrap();
            Ok(())
        }

        #[test]
        fn test_batch_update() -> anyhow::Result<()> {
            let pipeline = create_test_pipeline()?;
            let id = pipeline.add_frame("input", gen_frame())?;
            let batch_id = pipeline.move_and_pack_frames("proc1", vec![id])?;
            let update = get_update();
            pipeline.add_batched_frame_update(batch_id, id, update)?;
            pipeline.apply_updates(batch_id)?;
            pipeline.clear_updates(batch_id)?;
            let (frame, _) = pipeline.get_batched_frame(batch_id, id)?;
            frame.get_attribute("update", "attribute").unwrap();
            Ok(())
        }

        #[test]
        fn test_sampling() -> anyhow::Result<()> {
            init_telemetry();

            let pipeline = create_test_pipeline()?;
            pipeline.set_sampling_period(2)?;

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_eq!(ctx.span().span_context().is_valid(), false);

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_eq!(ctx.span().span_context().is_valid(), true);

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_eq!(ctx.span().span_context().is_valid(), false);

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_eq!(ctx.span().span_context().is_valid(), true);

            Ok(())
        }

        #[test]
        fn test_no_tracing() -> anyhow::Result<()> {
            init_telemetry();

            let pipeline = create_test_pipeline()?;
            pipeline.set_sampling_period(0)?;

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_eq!(ctx.span().span_context().is_valid(), false);

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_eq!(ctx.span().span_context().is_valid(), false);

            Ok(())
        }

        #[test]
        fn test_tracing_every() -> anyhow::Result<()> {
            init_telemetry();

            let pipeline = create_test_pipeline()?;
            pipeline.set_sampling_period(1)?;

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_eq!(ctx.span().span_context().is_valid(), true);

            let id = pipeline.add_frame("input", gen_frame())?;
            let (_frame, ctx) = pipeline.get_independent_frame(id)?;
            assert_eq!(ctx.span().span_context().is_valid(), true);

            Ok(())
        }

        #[test]
        fn test_stats() -> anyhow::Result<()> {
            let pipeline = create_test_pipeline()?;
            let id1 = pipeline.add_frame("input", gen_frame())?;
            // sleep for 5 ms
            sleep(Duration::from_millis(5));
            let id2 = pipeline.add_frame("input", gen_frame())?;
            // sleep for 3 ms
            sleep(Duration::from_millis(3));
            let batch_id = pipeline.move_and_pack_frames("proc1", vec![id1, id2])?;
            // sleep for 2 ms
            sleep(Duration::from_millis(2));
            pipeline.move_as_is("proc2", vec![batch_id])?;
            // sleep for 1 ms
            sleep(Duration::from_millis(1));
            let ids = pipeline.move_and_unpack_batch("output", batch_id)?;
            ids.iter().for_each(|id| {
                pipeline.delete(*id).unwrap();
            });
            pipeline.log_final_fps();
            let _ = pipeline.get_stat_records(10);
            Ok(())
        }
    }
}
