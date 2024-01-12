use pyo3::{pyclass, pymethods, Py, PyAny};
use savant_core::transport::zeromq;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Represents a socket type for a writer socket.
///
#[pyclass]
#[derive(Debug, Clone, Hash)]
pub enum WriterSocketType {
    Pub,
    Dealer,
    Req,
}

#[pymethods]
impl WriterSocketType {
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl From<zeromq::WriterSocketType> for WriterSocketType {
    fn from(socket_type: zeromq::WriterSocketType) -> Self {
        match socket_type {
            zeromq::WriterSocketType::Pub => Self::Pub,
            zeromq::WriterSocketType::Dealer => Self::Dealer,
            zeromq::WriterSocketType::Req => Self::Req,
        }
    }
}

impl From<WriterSocketType> for zeromq::WriterSocketType {
    fn from(socket_type: WriterSocketType) -> Self {
        match socket_type {
            WriterSocketType::Pub => Self::Pub,
            WriterSocketType::Dealer => Self::Dealer,
            WriterSocketType::Req => Self::Req,
        }
    }
}

/// Represents a socket type for a reader socket.
///
#[pyclass]
#[derive(Debug, Clone, Hash)]
pub enum ReaderSocketType {
    Sub,
    Router,
    Rep,
}

#[pymethods]
impl ReaderSocketType {
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl From<zeromq::ReaderSocketType> for ReaderSocketType {
    fn from(socket_type: zeromq::ReaderSocketType) -> Self {
        match socket_type {
            zeromq::ReaderSocketType::Sub => Self::Sub,
            zeromq::ReaderSocketType::Router => Self::Router,
            zeromq::ReaderSocketType::Rep => Self::Rep,
        }
    }
}

impl From<ReaderSocketType> for zeromq::ReaderSocketType {
    fn from(socket_type: ReaderSocketType) -> Self {
        match socket_type {
            ReaderSocketType::Sub => Self::Sub,
            ReaderSocketType::Router => Self::Router,
            ReaderSocketType::Rep => Self::Rep,
        }
    }
}

/// The object is used to configure the rules to pass messages from a writer to a reader
/// based on either exact topic match or a prefix match.
///
#[pyclass]
#[derive(Debug, Clone)]
pub struct TopicPrefixSpec(pub(crate) zeromq::TopicPrefixSpec);

#[pymethods]
impl TopicPrefixSpec {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Creates a match rule for exact topic match
    ///
    /// Parameters
    /// ----------
    /// id: str
    ///   The topic to match
    ///
    #[staticmethod]
    pub fn source_id(id: &str) -> Self {
        Self(zeromq::TopicPrefixSpec::SourceId(id.to_string()))
    }

    /// Creates a match rule for prefix match
    ///
    /// Parameters
    /// ----------
    /// prefix: str
    ///   The prefix to match
    ///
    #[staticmethod]
    pub fn prefix(prefix: &str) -> Self {
        Self(zeromq::TopicPrefixSpec::Prefix(prefix.to_string()))
    }

    /// Creates a match rule for no match
    ///
    #[staticmethod]
    pub fn none() -> Self {
        Self(zeromq::TopicPrefixSpec::None)
    }
}
