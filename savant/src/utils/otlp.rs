use crate::utils::get_tracer;
use crate::utils::python::release_gil;
use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::trace::{SpanBuilder, Status, TraceContextExt, Tracer};
use opentelemetry::{global, Array, Context, KeyValue, StringValue, Value};
use pyo3::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

#[pyclass]
#[derive(Debug, Clone)]
pub struct OTLPSpan(pub(crate) Context);

impl OTLPSpan {
    fn build_attributes(attributes: HashMap<String, String>) -> Vec<KeyValue> {
        attributes
            .into_iter()
            .map(|(k, v)| KeyValue::new(k, v))
            .collect()
    }
}

#[pymethods]
impl OTLPSpan {
    #[new]
    fn new(name: String) -> OTLPSpan {
        let span = get_tracer().build(SpanBuilder::from_name(name));
        let ctx = Context::current_with_span(span);
        OTLPSpan(ctx)
    }

    fn nested_span(&self, name: String) -> OTLPSpan {
        release_gil(|| {
            let span = get_tracer().build_with_context(SpanBuilder::from_name(name), &self.0);
            let ctx = Context::current_with_span(span);
            OTLPSpan(ctx)
        })
    }

    fn propagate(&self) -> PropagatedContext {
        release_gil(|| PropagatedContext::inject(&self.0))
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

    fn __enter__<'p>(slf: PyRef<'p, Self>, _py: Python<'p>) -> PyResult<PyRef<'p, Self>> {
        Ok(slf)
    }

    fn trace_id(&self) -> String {
        format!("{:?}", self.0.span().span_context().trace_id())
    }

    fn span_id(&self) -> String {
        format!("{:?}", self.0.span().span_context().span_id())
    }

    fn set_string_attribute(&self, key: String, value: String) {
        self.0.span().set_attribute(KeyValue::new(key, value));
    }

    fn set_string_vec_attribute(&self, key: String, value: Vec<String>) {
        self.0.span().set_attribute(KeyValue::new(
            key,
            Value::Array(Array::String(
                value.into_iter().map(StringValue::from).collect(),
            )),
        ));
    }

    fn set_bool_attribute(&self, key: String, value: bool) {
        self.0.span().set_attribute(KeyValue::new(key, value));
    }

    fn set_bool_vec_attribute(&self, key: String, value: Vec<bool>) {
        self.0
            .span()
            .set_attribute(KeyValue::new(key, Value::Array(Array::Bool(value))));
    }

    fn set_int_attribute(&self, key: String, value: i64) {
        self.0.span().set_attribute(KeyValue::new(key, value));
    }

    fn set_int_vec_attribute(&self, key: String, value: Vec<i64>) {
        self.0
            .span()
            .set_attribute(KeyValue::new(key, Value::Array(Array::I64(value))));
    }

    fn set_float_attribute(&self, key: String, value: f64) {
        self.0.span().set_attribute(KeyValue::new(key, value));
    }

    fn set_float_vec_attribute(&self, key: String, value: Vec<f64>) {
        self.0
            .span()
            .set_attribute(KeyValue::new(key, Value::Array(Array::F64(value))));
    }

    #[pyo3(signature = (name, attributes = HashMap::default()))]
    fn add_event(&self, name: String, attributes: HashMap<String, String>) {
        release_gil(|| {
            self.0
                .span()
                .add_event(name, OTLPSpan::build_attributes(attributes))
        });
    }

    fn set_status_error(&self, message: String) {
        self.0.span().set_status(Status::Error {
            description: message.into(),
        });
    }

    fn set_status_ok(&self) {
        self.0.span().set_status(Status::Ok);
    }

    fn set_status_unset(&self) {
        self.0.span().set_status(Status::Unset);
    }

    fn __exit__(
        &self,
        _exc_type: Option<&PyAny>,
        _exc_value: Option<&PyAny>,
        _traceback: Option<&PyAny>,
    ) -> PyResult<()> {
        release_gil(|| self.0.span().end());
        Ok(())
    }
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Clone, Default)]
#[archive(check_bytes)]
pub struct PropagatedContext(HashMap<String, String>);

#[pymethods]
impl PropagatedContext {
    fn nested_span(&self, name: String) -> OTLPSpan {
        release_gil(|| {
            let context = self.extract();
            let span = get_tracer().build_with_context(SpanBuilder::from_name(name), &context);
            let ctx = Context::current_with_span(span);
            OTLPSpan(ctx)
        })
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
