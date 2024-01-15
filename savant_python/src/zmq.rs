use crate::zmq::basic_types::{ReaderSocketType, TopicPrefixSpec, WriterSocketType};
use crate::zmq::configs::{ReaderConfig, ReaderConfigBuilder, WriterConfig, WriterConfigBuilder};
use crate::zmq::results::{
    ReaderResultEndOfStream, ReaderResultMessage, ReaderResultPrefixMismatch, ReaderResultTimeout,
    WriterResultAck, WriterResultAckTimeout, WriterResultSendTimeout, WriterResultSuccess,
};
use pyo3::prelude::*;

pub mod basic_types;
pub mod blocking;
pub mod configs;
pub mod nonblocking;
pub mod results;

#[pymodule]
pub fn zmq(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WriterSocketType>()?; // PYI
    m.add_class::<WriterConfigBuilder>()?; // PYI
    m.add_class::<WriterConfig>()?; // PYI
    m.add_class::<WriterResultSendTimeout>()?; // PYI
    m.add_class::<WriterResultAckTimeout>()?; // PYI
    m.add_class::<WriterResultAck>()?; // PYI
    m.add_class::<WriterResultSuccess>()?; // PYI

    m.add_class::<blocking::BlockingWriter>()?; // PYI
    m.add_class::<nonblocking::NonBlockingWriter>()?; // PYI
    m.add_class::<nonblocking::WriteOperationResult>()?; // PYI

    m.add_class::<ReaderSocketType>()?; // PYI
    m.add_class::<TopicPrefixSpec>()?; // PYI
    m.add_class::<ReaderConfigBuilder>()?; // PYI
    m.add_class::<ReaderConfig>()?; // PYI
    m.add_class::<ReaderResultMessage>()?; // PYI
    m.add_class::<ReaderResultEndOfStream>()?; // PYI
    m.add_class::<ReaderResultTimeout>()?; // PYI
    m.add_class::<ReaderResultPrefixMismatch>()?; // PYI

    m.add_class::<blocking::BlockingReader>()?; // PYI
    m.add_class::<nonblocking::NonBlockingReader>()?;

    Ok(())
}
