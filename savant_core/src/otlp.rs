use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::{global, Context};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static CURRENT_CONTEXTS: RefCell<Vec<Context>> = RefCell::new(vec![Context::default()]);
}

pub fn push_context(ctx: Context) {
    CURRENT_CONTEXTS.with(|contexts| {
        contexts.borrow_mut().push(ctx);
    });
}

pub fn pop_context() {
    CURRENT_CONTEXTS.with(|contexts| {
        contexts.borrow_mut().pop();
    });
}

pub fn current_context() -> Context {
    CURRENT_CONTEXTS.with(|contexts| {
        let contexts = contexts.borrow();
        contexts.last().unwrap().clone()
    })
}

pub fn current_context_depth() -> usize {
    CURRENT_CONTEXTS.with(|contexts| contexts.borrow().len())
}

pub fn with_current_context<F, R>(f: F) -> R
where
    F: FnOnce(&Context) -> R,
{
    CURRENT_CONTEXTS.with(|contexts| {
        let contexts = contexts.borrow();
        f(contexts.last().as_ref().unwrap())
    })
}

#[derive(Debug, Clone, Default)]
pub struct PropagatedContext(pub HashMap<String, String>);

impl Injector for PropagatedContext {
    fn set(&mut self, key: &str, value: String) {
        self.0.insert(key.to_string(), value);
    }
}

impl Extractor for PropagatedContext {
    fn get(&self, key: &str) -> Option<&str> {
        let key = key.to_owned();
        self.0.get(&key).map(|v| v.as_ref())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_ref()).collect()
    }
}

impl PropagatedContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inject(context: &Context) -> Self {
        global::get_text_map_propagator(|propagator| {
            let mut propagation_context = PropagatedContext::new();
            propagator.inject_context(context, &mut propagation_context);
            propagation_context
        })
    }

    pub fn extract(&self) -> Context {
        global::get_text_map_propagator(|propagator| propagator.extract(self))
    }
}
