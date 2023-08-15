use evalexpr::{build_operator_tree, Node};
use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::sync::Arc;

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

pub fn get_compiled_jmp_filter(query: &str) -> anyhow::Result<Arc<jmespath::Expression>> {
    let mut compiled_jmp_filter = COMPILED_JMP_FILTER.lock();
    if let Some(c) = compiled_jmp_filter.get(query) {
        return Ok(c.clone());
    }
    let c = Arc::new(jmespath::compile(query)?);
    compiled_jmp_filter.put(query.to_string(), c.clone());
    Ok(c)
}

pub fn get_compiled_eval_expr(query: &str) -> anyhow::Result<Arc<Node>> {
    let mut compiled_eval_expr = COMPILED_EVAL_EXPR.lock();
    if let Some(c) = compiled_eval_expr.get(query) {
        return Ok(c.clone());
    }
    let c = Arc::new(build_operator_tree(query)?);
    compiled_eval_expr.put(query.to_string(), c.clone());
    Ok(c)
}
