pub mod match_query;
pub mod py;

use std::collections::HashMap;

use crate::primitives::VideoObjectProxy;

pub use match_query::*;

pub type VideoObjectsProxyBatch = HashMap<i64, Vec<VideoObjectProxy>>;
