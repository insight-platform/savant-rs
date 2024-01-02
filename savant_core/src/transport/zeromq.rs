use anyhow::bail;
use lazy_static::lazy_static;
use log::debug;
use lru::LruCache;
use std::num::NonZeroUsize;

pub mod reader;
mod reader_config;

pub use reader::Reader;
pub use reader_config::{ReaderConfig, ReaderConfigBuilder};
use std::mem;

const RECEIVE_TIMEOUT: i32 = 1000;
const SENDER_RECEIVE_TIMEOUT: i32 = 5000;
const RECEIVE_HWM: i32 = 50;
const SEND_HWM: i32 = 50;
const REQ_RECEIVE_RETRIES: i32 = 3;
const EOS_CONFIRMATION_RETRIES: usize = 3;

const ROUTING_ID_CACHE_SIZE: usize = 512;

const CONFIRMATION_MESSAGE: &[u8] = b"OK";
const END_OF_STREAM_MESSAGE: &[u8] = b"EOS";
const IPC_PERMISSIONS: u32 = 0o777;

const ZMQ_ACK_LINGER: i32 = 100;

#[derive(Clone, Debug, PartialEq)]
pub enum ReaderSocketType {
    Sub,
    Router,
    Rep,
}
#[derive(Clone, Debug, PartialEq)]
pub enum WriterSocketType {
    Pub,
    Dealer,
    Req,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SocketType {
    Reader(ReaderSocketType),
    Writer(WriterSocketType),
}

lazy_static! {
    static ref SOCKET_URI_PATTERN: regex::Regex =
        regex::Regex::new(r"([a-z]+\+[a-z]+:)?((?:tcp|ipc)://[^:]+)(:.+)?").unwrap();
    static ref SOCKET_OPTIONS_PATTERN: regex::Regex =
        regex::Regex::new(r"(pub|sub|req|rep|dealer|router)\+(bind|connect):").unwrap();
}

pub struct ZmqSocketUri {
    pub endpoint: String,
    pub source: Option<String>,
    pub bind: Option<bool>,
    pub socket_type: Option<SocketType>,
}

pub fn parse_zmq_socket_uri(uri: String) -> anyhow::Result<ZmqSocketUri> {
    let endpoint;
    let source;
    let mut socket_type = None;
    let mut bind = None;
    if let Some(captures) = SOCKET_URI_PATTERN.captures(&uri) {
        if let Some(options) = captures.get(1) {
            let options = options.as_str();
            if let Some(captures) = SOCKET_OPTIONS_PATTERN.captures(options) {
                let socket_type_str = captures.get(1).unwrap().as_str();
                let bind_str = captures.get(2).unwrap().as_str();

                socket_type = Some(match socket_type_str {
                    "sub" => SocketType::Reader(ReaderSocketType::Sub),
                    "router" => SocketType::Reader(ReaderSocketType::Router),
                    "rep" => SocketType::Reader(ReaderSocketType::Rep),

                    "pub" => SocketType::Writer(WriterSocketType::Pub),
                    "dealer" => SocketType::Writer(WriterSocketType::Dealer),
                    "req" => SocketType::Writer(WriterSocketType::Req),

                    _ => bail!("Unknown socket type {}", socket_type_str),
                });

                bind = Some(match bind_str {
                    "bind" => true,
                    "connect" => false,
                    _ => bail!("Unknown bind type {}", bind_str),
                });
            } else {
                bail!("Invalid socket options {}", options);
            }
        }

        endpoint = captures.get(2).unwrap().as_str().to_string();
        source = if let Some(source) = captures.get(3) {
            if let Some(SocketType::Writer(_)) = socket_type {
                Some(source.as_str()[1..].to_string())
            } else {
                bail!("Source specification is not allowed for reader sockets");
            }
        } else {
            None
        };
    } else {
        bail!("Invalid ZeroMQ socket URI {}", uri);
    }

    Ok(ZmqSocketUri {
        endpoint,
        bind,
        socket_type,
        source,
    })
}

#[derive(Debug, Clone)]
pub enum TopicPrefixSpec {
    SourceId(String),
    Prefix(String),
    None,
}

impl TopicPrefixSpec {
    pub fn source_id(source_id: &str) -> Self {
        Self::SourceId(source_id.to_string())
    }

    pub fn prefix(prefix: &str) -> Self {
        Self::Prefix(prefix.to_string())
    }

    pub fn none() -> Self {
        Self::None
    }

    pub fn get(&self) -> String {
        match self {
            Self::SourceId(source_id) => format!("{}", source_id),
            Self::Prefix(prefix) => prefix.clone(),
            Self::None => "".to_string(),
        }
    }

    pub fn matches(&self, topic: &[u8]) -> bool {
        match self {
            Self::SourceId(source_id) => topic.eq(source_id.as_bytes()),
            Self::Prefix(prefix) => topic.starts_with(prefix.as_bytes()),
            Self::None => true,
        }
    }
}

struct RoutingIdFilter {
    ids: hashbrown::HashMap<Vec<u8>, Vec<u8>>,
    expired_routing_ids: LruCache<(Vec<u8>, Vec<u8>), ()>,
}

impl RoutingIdFilter {
    pub fn new(size: usize) -> anyhow::Result<Self> {
        debug!(target: "savant_rs.zeromq.routing-filter", "Creating routing id filter with LRU cache size = {}", size);
        Ok(Self {
            ids: hashbrown::HashMap::with_capacity(size),
            expired_routing_ids: LruCache::new(NonZeroUsize::try_from(size)?),
        })
    }

    pub fn allow(&mut self, topic: &[u8], routing_id: &Option<&Vec<u8>>) -> bool {
        if routing_id.is_none() {
            debug!(target: "savant_rs.zeromq.routing-filter", "Message without routing id always allowed");
            return true;
        }
        let routing_id = routing_id.unwrap();
        let current_valid_routing_id = self.ids.entry(topic.to_vec()).or_insert(routing_id.clone());
        debug!(target: "savant_rs.zeromq.routing-filter",
            "The current registered routing id: {:?}, the received routing id: {:?}",
            current_valid_routing_id, routing_id
        );

        if current_valid_routing_id == routing_id {
            debug!(target: "savant_rs.zeromq.routing-filter", "The current routing id {:?} is the same as the received one {:?}. Message is allowed.", 
                current_valid_routing_id, routing_id);
            true
        } else if self
            .expired_routing_ids
            .contains(&(topic.to_vec(), routing_id.clone()))
        {
            debug!(target: "savant_rs.zeromq.routing-filter", "The received routing id {:?} is found among old routing ids. Message is not allowed.", 
                routing_id);
            // routing id is outdated and we do not allow it anymore
            false
        } else {
            // routing id is new (because it is not in the cache) and we allow it.
            debug!(target: "savant_rs.zeromq.routing-filter", "The received routing id {:?} is new. The previous routing id {:?} is added to the expired. Message is allowed.", 
                routing_id, current_valid_routing_id);
            self.expired_routing_ids
                .put((topic.to_vec(), current_valid_routing_id.clone()), ());
            self.ids.entry(topic.to_vec()).and_modify(|id| {
                id.clone_from_slice(&routing_id);
            });
            true
        }
    }
}

#[allow(dead_code)]
enum Socket {
    ZmqSocket(zmq::Socket),
    MockSocket(Vec<Vec<u8>>),
}

#[allow(dead_code)]
impl Socket {
    fn send_multipart(&mut self, parts: &[&[u8]], flags: i32) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.send_multipart(parts, flags).map_err(|e| e.into()),
            Socket::MockSocket(data) => {
                data.clear();
                data.extend(parts.iter().map(|p| p.to_vec()));
                Ok(())
            }
        }
    }

    fn send(&mut self, m: &[u8], flags: i32) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.send(m, flags).map_err(|e| e.into()),
            Socket::MockSocket(data) => {
                data.clear();
                data.push(m.to_vec());
                Ok(())
            }
        }
    }

    fn recv_multipart(&mut self, flags: i32) -> Result<Vec<Vec<u8>>, zmq::Error> {
        match self {
            Socket::ZmqSocket(socket) => socket.recv_multipart(flags),
            Socket::MockSocket(data) => {
                let data = mem::replace(data, vec![]);
                Ok(data)
            }
        }
    }

    fn set_rcvhwm(&self, hwm: i32) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.set_rcvhwm(hwm).map_err(|e| e.into()),
            Socket::MockSocket(_) => Ok(()),
        }
    }

    fn set_rcvtimeo(&self, timeout: i32) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.set_rcvtimeo(timeout).map_err(|e| e.into()),
            Socket::MockSocket(_) => Ok(()),
        }
    }

    fn set_linger(&self, linger: i32) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.set_linger(linger).map_err(|e| e.into()),
            Socket::MockSocket(_) => Ok(()),
        }
    }

    fn set_subscribe(&self, topic: &[u8]) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.set_subscribe(topic).map_err(|e| e.into()),
            Socket::MockSocket(_) => Ok(()),
        }
    }

    fn bind(&self, endpoint: &str) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.bind(endpoint).map_err(|e| e.into()),
            Socket::MockSocket(_) => Ok(()),
        }
    }

    fn connect(&self, endpoint: &str) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.connect(endpoint).map_err(|e| e.into()),
            Socket::MockSocket(_) => Ok(()),
        }
    }

    fn take_buffer(&mut self) -> Vec<Vec<u8>> {
        match self {
            Socket::ZmqSocket(_) => unreachable!("Cannot take buffer from ZMQ socket. The function is implemented only for testing purposes."),
            Socket::MockSocket(data) => mem::replace(data, vec![]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::zeromq::reader_config::ReaderConfig;
    #[test]
    fn test_parse_only_uri() {
        let uri = "ipc:///tmp/address".to_string();
        let res = parse_zmq_socket_uri(uri).unwrap();
        assert_eq!(res.endpoint, "ipc:///tmp/address");
        assert!(res.bind.is_none());
        assert!(res.socket_type.is_none());
    }

    #[test]
    fn test_parse_uri_with_options() {
        let uri = "pub+bind:tcp://address".to_string();
        let res = parse_zmq_socket_uri(uri).unwrap();
        assert_eq!(res.endpoint, "tcp://address");
        assert!(matches!(res.bind, Some(true)));
        assert!(matches!(
            res.socket_type,
            Some(SocketType::Writer(WriterSocketType::Pub))
        ));
    }

    #[test]
    fn test_parse_writer_with_options_and_source() {
        let uri = "pub+bind:tcp://address:1234".to_string();
        let res = parse_zmq_socket_uri(uri).unwrap();
        assert_eq!(res.endpoint, "tcp://address");
        assert!(matches!(res.bind, Some(true)));
        assert!(matches!(
            res.socket_type,
            Some(SocketType::Writer(WriterSocketType::Pub))
        ));
        assert_eq!(res.source, Some("1234".to_string()));
    }

    #[test]
    fn test_parse_reader_with_options_and_source() {
        let uri = "sub+bind:tcp://address:1234".to_string();
        let res = parse_zmq_socket_uri(uri);
        assert!(res.is_err());
    }

    #[test]
    fn test_wrong_protocol() {
        let uri = "sub+bind:udp://address".to_string();
        let res = parse_zmq_socket_uri(uri);
        assert!(res.is_err());
    }

    #[test]
    fn test_routing_id_filter() {
        let mut filter = RoutingIdFilter::new(2).unwrap();

        let topic1 = vec![1, 2, 3];
        let topic2 = vec![1, 2, 4];

        let routing_id = vec![4, 5, 6];
        let routing_id2 = vec![7, 8, 9];

        assert!(filter.allow(&topic1, &None));
        assert!(filter.allow(&topic1, &Some(&routing_id)));
        assert!(filter.allow(&topic1, &Some(&routing_id2)));
        assert!(!filter.allow(&topic1, &Some(&routing_id)));
        assert!(filter.allow(&topic1, &Some(&routing_id2)));

        assert!(filter.allow(&topic2, &None));
        assert!(filter.allow(&topic2, &Some(&routing_id2)));
        assert!(filter.allow(&topic2, &Some(&routing_id)));
        assert!(!filter.allow(&topic2, &Some(&routing_id2)));
        assert!(filter.allow(&topic2, &Some(&routing_id)));
    }

    #[test]
    fn test_reader_config_default_build_fails() -> anyhow::Result<()> {
        let config = ReaderConfig::new().build();
        assert!(config.is_err());
        Ok(())
    }

    #[test]
    fn test_topic_prefix_spec() {
        let spec = TopicPrefixSpec::source_id("source_id");
        assert!(spec.matches(b"source_id"));
        assert!(!spec.matches(b"source_id2"));
        assert!(!spec.matches(b"source_id/abc"));
        assert!(!spec.matches(b"source_id/abc/def"));

        let spec = TopicPrefixSpec::prefix("prefix");
        assert!(spec.matches(b"prefix"));
        assert!(spec.matches(b"prefix/abc"));
        assert!(spec.matches(b"prefix/abc/def"));
        assert!(!spec.matches(b"prefi"));
        assert!(!spec.matches(b"prefi/abc"));
        assert!(!spec.matches(b"prefi/abc/def"));

        let spec = TopicPrefixSpec::none();
        assert!(spec.matches(b"prefix"));
        assert!(spec.matches(b"prefix/abc"));
        assert!(spec.matches(b"prefix/abc/def"));
        assert!(spec.matches(b"source_id"));
        assert!(spec.matches(b"source_id/abc"));
        assert!(spec.matches(b"source_id/abc/def"));
    }
}
