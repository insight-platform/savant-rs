use crate::zmq::basic_types::{ReaderSocketType, TopicPrefixSpec, WriterSocketType};
use crate::zmq::configs::{ReaderConfig, ReaderConfigBuilder, WriterConfig, WriterConfigBuilder};
use crate::zmq::results::{
    ReaderResultEndOfStream, ReaderResultMessage, ReaderResultPrefixMismatch, ReaderResultTimeout,
    WriterResultAck, WriterResultAckTimeout, WriterResultSendTimeout, WriterResultSuccess,
};
use pyo3::prelude::*;

pub mod basic_types;
pub mod configs;
pub mod nonblocking;
pub mod results;
pub mod sync;

#[pymodule]
pub fn zmq(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WriterSocketType>()?;
    m.add_class::<WriterConfigBuilder>()?;
    m.add_class::<WriterConfig>()?;
    m.add_class::<WriterResultSendTimeout>()?;
    m.add_class::<WriterResultAckTimeout>()?;
    m.add_class::<WriterResultAck>()?;
    m.add_class::<WriterResultSuccess>()?;

    m.add_class::<sync::Writer>()?;
    m.add_class::<nonblocking::NonBlockingWriter>()?;
    m.add_class::<nonblocking::WriteOperationResult>()?;

    m.add_class::<ReaderSocketType>()?;
    m.add_class::<TopicPrefixSpec>()?;
    m.add_class::<ReaderConfigBuilder>()?;
    m.add_class::<ReaderConfig>()?;
    m.add_class::<ReaderResultMessage>()?;
    m.add_class::<ReaderResultEndOfStream>()?;
    m.add_class::<ReaderResultTimeout>()?;
    m.add_class::<ReaderResultPrefixMismatch>()?;

    m.add_class::<sync::Reader>()?;
    m.add_class::<nonblocking::NonBlockingReader>()?;

    Ok(())
}
