pub mod functions;
pub mod macros;
pub mod match_query;
pub mod py;

use evalexpr::*;
use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use crate::primitives::VideoObjectProxy;

pub use crate::query_and as and;
pub use crate::query_not as not;
pub use crate::query_or as or;
pub use functions::*;
pub use match_query::*;

pub type VideoObjectsProxyBatch = HashMap<i64, Vec<VideoObjectProxy>>;

const MAX_JMES_CACHE_SIZE: usize = 1024;

lazy_static! {
    static ref COMPILED_EVAL_EXPR: Mutex<lru::LruCache<String, Arc<Node>>> = Mutex::new(
        lru::LruCache::new(std::num::NonZeroUsize::new(MAX_JMES_CACHE_SIZE).unwrap())
    );
    static ref COMPILED_JMP_FILTER: Mutex<lru::LruCache<String, Arc<jmespath::Expression<'static>>>> =
        Mutex::new(lru::LruCache::new(
            std::num::NonZeroUsize::new(MAX_JMES_CACHE_SIZE).unwrap()
        ));
}

fn get_compiled_jmp_filter(query: &str) -> anyhow::Result<Arc<jmespath::Expression>> {
    let mut compiled_jmp_filter = COMPILED_JMP_FILTER.lock();
    if let Some(c) = compiled_jmp_filter.get(query) {
        return Ok(c.clone());
    }
    let c = Arc::new(jmespath::compile(query)?);
    compiled_jmp_filter.put(query.to_string(), c.clone());
    Ok(c)
}

fn get_compiled_eval_expr(query: &str) -> anyhow::Result<Arc<Node>> {
    let mut compiled_eval_expr = COMPILED_EVAL_EXPR.lock();
    if let Some(c) = compiled_eval_expr.get(query) {
        return Ok(c.clone());
    }
    let c = Arc::new(build_operator_tree(query)?);
    compiled_eval_expr.put(query.to_string(), c.clone());
    Ok(c)
}
