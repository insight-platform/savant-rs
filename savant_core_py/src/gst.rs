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
    IngressMessageTransformer,
}
