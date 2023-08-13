pub mod eval_cache;
pub mod eval_resolvers;
pub mod match_query;
pub mod primitives;
pub mod symbol_mapper;
pub mod telemetry;
/// A trait to serialize various objects to json.
pub mod to_json_value;
pub mod udf_api;

use lazy_static::lazy_static;

lazy_static! {
    static ref VERSION_CRC32: u32 = crc32fast::hash(env!("CARGO_PKG_VERSION").as_bytes());
}

pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

/// Returns version in CRC32 format
///
pub fn version_crc32() -> u32 {
    *VERSION_CRC32
}

pub fn version_to_bytes_le() -> [u8; 4] {
    VERSION_CRC32.to_le_bytes()
}

pub fn bytes_le_to_version(bytes: [u8; 4]) -> u32 {
    u32::from_le_bytes(bytes)
}

pub fn fast_hash(bytes: &[u8]) -> u32 {
    crc32fast::hash(bytes)
}
