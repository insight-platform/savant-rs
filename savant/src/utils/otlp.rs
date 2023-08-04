use crate::utils::get_tracer;
use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::trace::{SpanBuilder, TraceContextExt, Tracer};
use opentelemetry::{global, Context, KeyValue};
use pyo3::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

#[pyclass]
#[derive(Debug, Clone)]
pub struct OTLPSpan(pub(crate) Context);

#[pymethods]
impl OTLPSpan {
    #[pyo3(signature=(name, attrs=None))]
    fn nested_span(&self, name: String, attrs: Option<HashMap<String, String>>) -> OTLPSpan {
        let span = get_tracer().build_with_context(SpanBuilder::from_name(name), &self.0);
        let ctx = Context::current_with_span(span);
        if let Some(attrs) = attrs {
            for (k, v) in attrs {
                ctx.span().set_attribute(KeyValue::new(k, v));
            }
        }
        OTLPSpan(ctx)
    }

    fn propagate(&self) -> PropagatedContext {
        PropagatedContext::inject(&self.0)
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!(
            "{self:?}, span_id={}",
            self.0.span().span_context().span_id()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Clone, Default)]
#[archive(check_bytes)]
pub struct PropagatedContext(HashMap<String, String>);

#[pymethods]
impl PropagatedContext {
    #[pyo3(signature=(name, attrs=None))]
    fn nested_span(&self, name: String, attrs: Option<HashMap<String, String>>) -> OTLPSpan {
        let context = self.extract();
        let span = get_tracer().build_with_context(SpanBuilder::from_name(name), &context);
        let ctx = Context::current_with_span(span);
        if let Some(attrs) = attrs {
            for (k, v) in attrs {
                ctx.span().set_attribute(KeyValue::new(k, v));
            }
        }
        OTLPSpan(ctx)
    }

    fn as_dict(&self) -> HashMap<String, String> {
        self.0.clone()
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl PropagatedContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inject(context: &opentelemetry::Context) -> Self {
        global::get_text_map_propagator(|propagator| {
            let mut propagation_context = PropagatedContext::new();
            propagator.inject_context(context, &mut propagation_context);
            propagation_context
        })
    }

    pub fn extract(&self) -> opentelemetry::Context {
        global::get_text_map_propagator(|propagator| propagator.extract(self))
    }
}

impl Injector for PropagatedContext {
    fn set(&mut self, key: &str, value: String) {
        self.0.insert(key.to_owned(), value);
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
