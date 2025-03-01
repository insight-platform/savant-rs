use hashbrown::HashMap;
use lazy_static::lazy_static;
use parking_lot::RwLock;
use pyo3::prelude::*;

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq, Debug)]
pub enum FlowResult {
    CustomSuccess2,
    CustomSuccess1,
    CustomSuccess,
    Ok,
    NotLinked,
    Flushing,
    Eos,
    NotNegotiated,
    Error,
    NotSupported,
    CustomError,
    CustomError1,
    CustomError2,
}

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq, Debug)]
pub enum InvocationReason {
    Buffer,
    SinkEvent,
    SourceEvent,
    StateChange,
}

lazy_static! {
    pub static ref REGISTERED_HANDLERS: RwLock<HashMap<String, Py<PyAny>>> =
        RwLock::new(HashMap::new());
}

#[pyfunction]
pub fn register_handler(name: &str, handler: Bound<'_, PyAny>) -> PyResult<()> {
    let mut handlers = REGISTERED_HANDLERS.write();
    let unbound = handler.unbind();
    handlers.insert(name.to_string(), unbound);
    Ok(())
}
