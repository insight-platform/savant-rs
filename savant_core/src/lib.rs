pub mod consts;
pub mod cplugin;
pub mod deadlock_detection;
pub mod draw;
pub mod eval_cache;
pub mod eval_context;
pub mod eval_resolvers;
/// A trait to serialize various objects to json.
pub mod json_api;
pub mod macros;
pub mod match_query;
pub mod message;
pub mod otlp;
pub mod pipeline;
pub mod pluggable_udf_api;
pub mod primitives;
pub mod protobuf;
mod savant_rwlock;
pub mod symbol_mapper;
pub mod telemetry;
pub mod test;
pub mod transmutes;
pub mod transport;

use opentelemetry::global;
use opentelemetry::global::BoxedTracer;

#[inline]
pub fn round_2_digits(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

#[inline]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

#[inline]
pub fn fast_hash(bytes: &[u8]) -> u32 {
    crc32fast::hash(bytes)
}

#[inline]
pub fn get_tracer() -> BoxedTracer {
    global::tracer("video_pipeline")
}

pub mod rust {
    pub use super::otlp::PropagatedContext;
    pub use super::pipeline::stats::FrameProcessingStatRecord;
    pub use super::pipeline::stats::FrameProcessingStatRecordType;
    pub use super::pipeline::stats::StageStat;
    pub use super::pipeline::Pipeline;
    pub use super::pipeline::PipelineConfiguration;
    pub use super::pipeline::PipelineConfigurationBuilder;
    pub use super::pipeline::PipelineStagePayloadType;
    pub use super::symbol_mapper::RegistrationPolicy;
    pub use super::symbol_mapper::SymbolMapper;
}
