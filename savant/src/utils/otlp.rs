use crate::logging::{log_level_enabled, log_message, LogLevel};
use crate::utils::get_tracer;
use crate::utils::python::{release_gil, with_gil};
use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::trace::{SpanBuilder, Status, TraceContextExt, TraceId, Tracer};
use opentelemetry::{global, Array, Context, KeyValue, StringValue, Value};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyTraceback;
use rkyv::{Archive, Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::thread::ThreadId;

thread_local! {
    static CURRENT_CONTEXTS: RefCell<Vec<Context>> = RefCell::new(vec![Context::default()]);
}

fn push_context(ctx: Context) {
    CURRENT_CONTEXTS.with(|contexts| {
        contexts.borrow_mut().push(ctx);
    });
}

fn pop_context() {
    CURRENT_CONTEXTS.with(|contexts| {
        contexts.borrow_mut().pop();
    });
}

pub(crate) fn current_context() -> Context {
    CURRENT_CONTEXTS.with(|contexts| {
        let contexts = contexts.borrow();
        contexts.last().unwrap().clone()
    })
}

pub(crate) fn with_current_context<F, R>(f: F) -> R
where
    F: FnOnce(&Context) -> R,
{
    CURRENT_CONTEXTS.with(|contexts| {
        let contexts = contexts.borrow();
        f(contexts.last().as_ref().unwrap())
    })
}

/// A Span to be used locally. Works as a guard (use with `with` statement).
///
#[pyclass]
#[derive(Debug, Clone)]
pub struct TelemetrySpan(pub(crate) Context, ThreadId);

impl TelemetrySpan {
    fn build_attributes(attributes: HashMap<String, String>) -> Vec<KeyValue> {
        attributes
            .into_iter()
            .map(|(k, v)| KeyValue::new(k, v))
            .collect()
    }

    pub(crate) fn from_context(ctx: Context) -> TelemetrySpan {
        TelemetrySpan(ctx, TelemetrySpan::thread_id())
    }

    fn thread_id() -> ThreadId {
        std::thread::current().id()
    }

    fn ensure_same_thread(&self) {
        if self.1 != TelemetrySpan::thread_id() {
            panic!("Span used in a different thread than it was created in");
        }
    }
}

#[pymethods]
impl TelemetrySpan {
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
    fn constructor(name: &str) -> TelemetrySpan {
        TelemetrySpan::new(name)
    }

    #[staticmethod]
    fn current() -> TelemetrySpan {
        TelemetrySpan::from_context(current_context())
    }

    #[staticmethod]
    fn context_depth() -> usize {
        CURRENT_CONTEXTS.with(|contexts| contexts.borrow().len())
    }

    #[new]
    fn new(name: &str) -> TelemetrySpan {
        TelemetrySpan(
            get_tracer().in_span(name.to_string(), |ctx| ctx),
            TelemetrySpan::thread_id(),
        )
    }

    #[staticmethod]
    fn default() -> TelemetrySpan {
        TelemetrySpan::from_context(Context::default())
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
    /// Returns
    /// --------
    /// :py:class:`TelemetrySpan`
    ///   new span
    ///
    fn nested_span(&self, name: &str) -> TelemetrySpan {
        release_gil(|| {
            let parent_ctx = &self.0;

            if parent_ctx.span().span_context().trace_id() == TraceId::INVALID {
                return TelemetrySpan::default();
            }

            let span = get_tracer()
                .build_with_context(SpanBuilder::from_name(name.to_string()), parent_ctx);
            let ctx = Context::current_with_span(span);
            TelemetrySpan(ctx, TelemetrySpan::thread_id())
        })
    }

    /// Creates a nested span only when log level is enabled
    ///
    /// GIL Management: This method releases the GIL.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the span.
    ///
    /// Returns
    /// -------
    /// :py:class:`MaybeTelemetrySpan`
    ///   a span that maybe a nested if the log level is enabled
    ///
    fn nested_span_when_loglevel_active(
        &self,
        name: &str,
        log_level: LogLevel,
    ) -> MaybeTelemetrySpan {
        MaybeTelemetrySpan::new(if log_level_enabled(log_level) {
            Some(self.nested_span(name))
        } else {
            None
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
        self.ensure_same_thread();
        release_gil(|| PropagatedContext::inject(&self.0))
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        self.ensure_same_thread();
        format!(
            "{self:?}, span_id={}",
            self.0.span().span_context().span_id()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __enter__<'p>(slf: PyRef<'p, Self>, _py: Python<'p>) -> PyResult<PyRef<'p, Self>> {
        let o = slf.deref();
        o.enter();
        Ok(slf)
    }

    pub(crate) fn enter(&self) {
        self.ensure_same_thread();
        push_context(self.0.clone());
    }

    fn __exit__(
        &self,
        exc_type: Option<&PyAny>,
        exc_value: Option<&PyAny>,
        traceback: Option<&PyAny>,
    ) -> PyResult<()> {
        with_gil(|py| {
            if let Some(e) = exc_type {
                let mut attrs = HashMap::new();

                self.0.span().set_status(Status::Error {
                    description: "python.exception".into(),
                });

                attrs.insert("python.exception.type".to_string(), format!("{:?}", e));

                if let Some(v) = exc_value {
                    if let Ok(e) = PyAny::downcast::<PyException>(v) {
                        attrs.insert("python.exception.value".to_string(), e.to_string());
                    }
                }

                if let Some(t) = traceback {
                    let traceback = PyAny::downcast::<PyTraceback>(t).unwrap();
                    if let Ok(formatted) = traceback.format() {
                        attrs.insert("python.exception.traceback".to_string(), formatted);
                    }
                }

                attrs.insert("python.version".into(), py.version().to_string());
                release_gil(|| {
                    log_message(
                        LogLevel::Error,
                        "python::exception".to_string(),
                        "Exception occurred".to_string(),
                        Some(
                            attrs
                                .iter()
                                .map(|(k, v)| KeyValue::new(k.clone(), v.clone()))
                                .collect(),
                        ),
                    );

                    self.add_event("python.exception".to_string(), attrs);
                });
            } else {
                self.0.span().set_status(Status::Ok);
            }
        });

        release_gil(|| {
            self.0.span().end();
            pop_context();
        });

        Ok(())
    }

    /// Returns the trace ID of the span.
    ///
    /// Returns
    /// -------
    /// str
    ///   The trace ID.
    ///
    fn trace_id(&self) -> String {
        self.ensure_same_thread();
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
        self.ensure_same_thread();
        format!("{:?}", self.0.span().span_context().span_id())
    }

    /// Install the attribute with `str` value.
    ///
    fn set_string_attribute(&self, key: String, value: String) {
        self.ensure_same_thread();
        self.0.span().set_attribute(KeyValue::new(key, value));
    }

    /// Install the attribute with string array value (`list[str]`).
    ///
    fn set_string_vec_attribute(&self, key: String, value: Vec<String>) {
        self.ensure_same_thread();
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
        self.ensure_same_thread();
        self.0.span().set_attribute(KeyValue::new(key, value));
    }

    /// Install the attribute with bool array value (`list[bool]`).
    ///
    fn set_bool_vec_attribute(&self, key: String, value: Vec<bool>) {
        self.ensure_same_thread();
        self.0
            .span()
            .set_attribute(KeyValue::new(key, Value::Array(Array::Bool(value))));
    }

    /// Install the attribute with `int` value. Must be a valid int64, large numbers are not supported (use string instead).
    ///
    fn set_int_attribute(&self, key: String, value: i64) {
        self.ensure_same_thread();
        self.0.span().set_attribute(KeyValue::new(key, value));
    }

    /// Install the attribute with int array value (`list[int]`). Must be a valid int64, large numbers are not supported (use string instead).
    ///
    fn set_int_vec_attribute(&self, key: String, value: Vec<i64>) {
        self.ensure_same_thread();
        self.0
            .span()
            .set_attribute(KeyValue::new(key, Value::Array(Array::I64(value))));
    }

    /// Install the attribute with `float` value.
    ///
    fn set_float_attribute(&self, key: String, value: f64) {
        self.ensure_same_thread();
        self.0.span().set_attribute(KeyValue::new(key, value));
    }

    /// Install the attribute with float array value (`list[float]`).
    ///
    fn set_float_vec_attribute(&self, key: String, value: Vec<f64>) {
        self.ensure_same_thread();
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
        self.ensure_same_thread();
        release_gil(|| {
            self.0
                .span()
                .add_event(name, TelemetrySpan::build_attributes(attributes))
        });
    }

    /// Configures the span status as Error with the given message.
    ///
    fn set_status_error(&self, message: String) {
        self.ensure_same_thread();
        self.0.span().set_status(Status::Error {
            description: message.into(),
        });
    }

    /// Configures the span status as OK.
    ///
    fn set_status_ok(&self) {
        self.ensure_same_thread();
        self.0.span().set_status(Status::Ok);
    }

    /// Configures the span status as Unset.
    ///
    fn set_status_unset(&self) {
        self.ensure_same_thread();
        self.0.span().set_status(Status::Unset);
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct MaybeTelemetrySpan {
    span: Option<TelemetrySpan>,
}

#[pymethods]
impl MaybeTelemetrySpan {
    #[new]
    fn new(span: Option<TelemetrySpan>) -> MaybeTelemetrySpan {
        MaybeTelemetrySpan { span }
    }

    fn nested_span(&self, name: &str) -> MaybeTelemetrySpan {
        if let Some(span) = &self.span {
            MaybeTelemetrySpan {
                span: Some(span.nested_span(name)),
            }
        } else {
            MaybeTelemetrySpan { span: None }
        }
    }

    fn nested_span_when_loglevel_active(
        &self,
        name: &str,
        log_level: LogLevel,
    ) -> MaybeTelemetrySpan {
        match &self.span {
            None => MaybeTelemetrySpan::new(None),
            Some(span) => MaybeTelemetrySpan::new(if log_level_enabled(log_level) {
                Some(span.nested_span(name))
            } else {
                None
            }),
        }
    }

    fn __enter__<'p>(slf: PyRef<'p, Self>, _py: Python<'p>) {
        if let Some(span) = &slf.span {
            span.enter();
        }
    }

    fn __exit__(
        &self,
        exc_type: Option<&PyAny>,
        exc_value: Option<&PyAny>,
        traceback: Option<&PyAny>,
    ) -> PyResult<()> {
        if let Some(span) = &self.span {
            span.__exit__(exc_type, exc_value, traceback)?;
        }
        Ok(())
    }

    #[staticmethod]
    fn current() -> TelemetrySpan {
        TelemetrySpan::from_context(current_context())
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
    fn nested_span(&self, name: &str) -> TelemetrySpan {
        release_gil(|| {
            let parent_ctx = self.extract();
            if parent_ctx.span().span_context().trace_id() == TraceId::INVALID {
                return TelemetrySpan::default();
            }
            let span = get_tracer()
                .build_with_context(SpanBuilder::from_name(name.to_string()), &parent_ctx);
            let ctx = Context::current_with_span(span);
            TelemetrySpan(ctx, TelemetrySpan::thread_id())
        })
    }

    /// Creates a nested span only when log level is enabled
    ///
    /// GIL Management: This method releases the GIL.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the span.
    ///
    /// Returns
    /// -------
    /// :py:class:`MaybeTelemetrySpan`
    ///   a span that maybe a nested if the log level is enabled
    ///
    fn nested_span_when_loglevel_active(
        &self,
        name: &str,
        log_level: LogLevel,
    ) -> MaybeTelemetrySpan {
        MaybeTelemetrySpan::new(if log_level_enabled(log_level) {
            Some(self.nested_span(name))
        } else {
            None
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
