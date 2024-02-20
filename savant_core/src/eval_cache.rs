use crate::eval_context::GlobalContext;
use crate::eval_resolvers::{
    config_resolver_name, env_resolver_name, etcd_resolver_name, utility_resolver_name,
};
use evalexpr::{build_operator_tree, Node, Value};
use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::sync::Arc;

const MAX_JMES_CACHE_SIZE: usize = 1024;
const MAX_EVAL_EXPR_CACHE_SIZE: usize = 1024;
const MAX_EVAL_RESULTS_CACHE_SIZE: usize = 1024;

lazy_static! {
    static ref COMPILED_EVAL_EXPR: Mutex<lru::LruCache<String, Arc<Node>>> = Mutex::new(
        lru::LruCache::new(std::num::NonZeroUsize::new(MAX_EVAL_EXPR_CACHE_SIZE).unwrap())
    );
    static ref COMPILED_JMP_FILTER: Mutex<lru::LruCache<String, Arc<jmespath::Expression<'static>>>> =
        Mutex::new(lru::LruCache::new(
            std::num::NonZeroUsize::new(MAX_JMES_CACHE_SIZE).unwrap()
        ));
    static ref EVAL_RESULTS: Mutex<lru::LruCache<String, (u128, evalexpr::Value)>> = Mutex::new(
        lru::LruCache::new(std::num::NonZeroUsize::new(MAX_EVAL_RESULTS_CACHE_SIZE).unwrap())
    );
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

pub fn eval_expr(query: &str, ttl: u64) -> anyhow::Result<(Value, bool)> {
    let expr = get_compiled_eval_expr(query)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();

    if ttl > 0 {
        let mut eval_results = EVAL_RESULTS.lock();
        let res_opt = eval_results.get(query).map(|(exp, v)| (*exp, v.clone()));
        if let Some((expires, res)) = res_opt {
            if expires > now {
                return Ok((res, true));
            }
        }
    }

    let mut context = GlobalContext::new(&[
        utility_resolver_name(),
        etcd_resolver_name(),
        config_resolver_name(),
        env_resolver_name(),
    ]);

    let res = expr.eval_with_context_mut(&mut context)?;

    if ttl > 0 {
        let mut eval_results = EVAL_RESULTS.lock();
        eval_results.put(query.to_string(), (now + ttl as u128, res.clone()));
    }

    Ok((res, false))
}

#[cfg(test)]
mod tests {
    use crate::eval_resolvers::register_env_resolver;

    #[test]
    fn test_eval_expr() {
        use super::*;
        register_env_resolver();

        let (res, cached) = eval_expr("1 + 1", 0).unwrap();
        assert_eq!(res, Value::from(2));
        assert!(!cached);

        let (res, cached) = eval_expr("env(\"PATH\", \"\")", 0).unwrap();
        assert_eq!(res, Value::from(std::env::var("PATH").unwrap()));
        assert!(!cached);

        let (res, cached) = eval_expr("x = 1; x", 1000).unwrap();
        assert_eq!(res, Value::from(1));
        assert!(!cached);

        let (res, cached) = eval_expr("x = 1; x", 1000).unwrap();
        assert_eq!(res, Value::from(1));
        assert!(cached);
    }
}
