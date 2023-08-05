use crate::utils::get_tracer;
use crate::utils::python::release_gil;
use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::trace::{SpanBuilder, Status, TraceContextExt, Tracer};
use opentelemetry::{global, Array, Context, KeyValue, StringValue, Value};
use pyo3::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

/// A Span to be used locally. Works as a guard (use with `with` statement).
///
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
    /// Create a root span with the given name for a new trace. Can be used as `__init__` method.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the span.
    ///
    /// Returns
    /// -------
    /// OTLPSpan
    ///   The created span.
    ///
    #[staticmethod]
    fn constructor(name: String) -> OTLPSpan {
        OTLPSpan::new(name)
    }

    #[new]
    fn new(name: String) -> OTLPSpan {
        let span = get_tracer().build(SpanBuilder::from_name(name));
        let ctx = Context::current_with_span(span);
        OTLPSpan(ctx)
    }

    /// Create a child span with the given name.
    ///
    /// GIL Management: This method releases the GIL.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the span.
    ///
    fn nested_span(&self, name: String) -> OTLPSpan {
        release_gil(|| {
            let span = get_tracer().build_with_context(SpanBuilder::from_name(name), &self.0);
            let ctx = Context::current_with_span(span);
            OTLPSpan(ctx)
        })
    }

    /// Creates a propagation context from the current span.
    ///
    /// GIL Management: This method releases the GIL.
    ///
    /// Returns
    /// -------
    /// :py:class:`PropagatedContext`
    ///   The created context.
    ///
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

    /// Returns the trace ID of the span.
    ///
    /// Returns
    /// -------
    /// str
    ///   The trace ID.
    ///
    fn trace_id(&self) -> String {
        format!("{:?}", self.0.span().span_context().trace_id())
    }

    /// Returns the span ID of the span.
    ///
    /// Returns
    /// -------
    /// str
    ///   The span ID.
    ///
    fn span_id(&self) -> String {
        format!("{:?}", self.0.span().span_context().span_id())
    }

    /// Install the attribute with `str` value.
    ///
    fn set_string_attribute(&self, key: String, value: String) {
        self.0.span().set_attribute(KeyValue::new(key, value));
    }

    /// Install the attribute with string array value (`list[str]`).
    ///
    fn set_string_vec_attribute(&self, key: String, value: Vec<String>) {
        self.0.span().set_attribute(KeyValue::new(
            key,
            Value::Array(Array::String(
                value.into_iter().map(StringValue::from).collect(),
            )),
        ));
    }

    /// Install the attribute with `bool` value.
    ///
    fn set_bool_attribute(&self, key: String, value: bool) {
        self.0.span().set_attribute(KeyValue::new(key, value));
    }

    /// Install the attribute with bool array value (`list[bool]`).
    ///
    fn set_bool_vec_attribute(&self, key: String, value: Vec<bool>) {
        self.0
            .span()
            .set_attribute(KeyValue::new(key, Value::Array(Array::Bool(value))));
    }

    /// Install the attribute with `int` value. Must be a valid int64, large numbers are not supported (use string instead).
    ///
    fn set_int_attribute(&self, key: String, value: i64) {
        self.0.span().set_attribute(KeyValue::new(key, value));
    }

    /// Install the attribute with int array value (`list[int]`). Must be a valid int64, large numbers are not supported (use string instead).
    ///
    fn set_int_vec_attribute(&self, key: String, value: Vec<i64>) {
        self.0
            .span()
            .set_attribute(KeyValue::new(key, Value::Array(Array::I64(value))));
    }

    /// Install the attribute with `float` value.
    ///
    fn set_float_attribute(&self, key: String, value: f64) {
        self.0.span().set_attribute(KeyValue::new(key, value));
    }

    /// Install the attribute with float array value (`list[float]`).
    ///
    fn set_float_vec_attribute(&self, key: String, value: Vec<f64>) {
        self.0
            .span()
            .set_attribute(KeyValue::new(key, Value::Array(Array::F64(value))));
    }

    /// Adds an event to the span.
    ///
    /// GIL Management: This method releases the GIL.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the event.
    /// attributes : dict[str, str]
    ///   The attributes of the event.
    ///
    #[pyo3(signature = (name, attributes = HashMap::default()))]
    fn add_event(&self, name: String, attributes: HashMap<String, String>) {
        release_gil(|| {
            self.0
                .span()
                .add_event(name, OTLPSpan::build_attributes(attributes))
        });
    }

    /// Configures the span status as Error with the given message.
    ///
    fn set_status_error(&self, message: String) {
        self.0.span().set_status(Status::Error {
            description: message.into(),
        });
    }

    /// Configures the span status as OK.
    ///
    fn set_status_ok(&self) {
        self.0.span().set_status(Status::Ok);
    }

    /// Configures the span status as Unset.
    ///
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

/// Represents a context that can be propagated to remote system like Python code
///
///
#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Clone, Default)]
#[archive(check_bytes)]
pub struct PropagatedContext(HashMap<String, String>);

#[pymethods]
impl PropagatedContext {
    /// Create a new child span
    ///
    /// GIL Management: GIL is released during the call
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///   Name of the span
    ///
    /// Returns
    /// -------
    /// OTLPSpan
    ///   A new span
    ///
    fn nested_span(&self, name: String) -> OTLPSpan {
        release_gil(|| {
            let context = self.extract();
            let span = get_tracer().build_with_context(SpanBuilder::from_name(name), &context);
            let ctx = Context::current_with_span(span);
            OTLPSpan(ctx)
        })
    }

    /// Returns the context in the form of a dictionary
    ///
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
