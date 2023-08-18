pub mod consts;
pub mod cplugin;
pub mod deadlock_detection;
pub mod draw;
pub mod eval_cache;
pub mod eval_context;
pub mod eval_resolvers;
pub mod macros;
pub mod match_query;
pub mod message;
pub mod otlp;
pub mod pipeline;
pub mod pluggable_udf_api;
pub mod primitives;
pub mod symbol_mapper;
pub mod telemetry;
pub mod test;
/// A trait to serialize various objects to json.
pub mod to_json_value;
pub mod transmutes;

use lazy_static::lazy_static;
use opentelemetry::global;
use opentelemetry::global::BoxedTracer;

lazy_static! {
    static ref VERSION_CRC32: u32 = crc32fast::hash(env!("CARGO_PKG_VERSION").as_bytes());
}

#[inline]
pub fn round_2_digits(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

#[inline]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

/// Returns version in CRC32 format
///
#[inline]
pub fn version_crc32() -> u32 {
    *VERSION_CRC32
}

#[inline]
pub fn version_to_bytes_le() -> [u8; 4] {
    VERSION_CRC32.to_le_bytes()
}

#[inline]
pub fn bytes_le_to_version(bytes: [u8; 4]) -> u32 {
    u32::from_le_bytes(bytes)
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
    pub use super::pipeline::*;
    pub use super::symbol_mapper::RegistrationPolicy;
    pub use super::symbol_mapper::SymbolMapper;
}
