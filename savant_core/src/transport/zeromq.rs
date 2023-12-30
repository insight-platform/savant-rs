use anyhow::bail;
use lazy_static::lazy_static;
use log::debug;
use lru::LruCache;
use std::num::NonZeroUsize;

pub mod reader;
mod reader_config;

pub use reader::Reader;
pub use reader_config::{ReaderConfig, ReaderConfigBuilder};

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
pub enum TopicPrefix {
    SourceId(String),
    Prefix(String),
    None,
}

impl TopicPrefix {
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
            Self::SourceId(source_id) => format!("{}/", source_id),
            Self::Prefix(prefix) => prefix.clone(),
            Self::None => "".to_string(),
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

    pub fn allow(&mut self, topic: &[u8], routing_id: &Option<Vec<u8>>) -> bool {
        if routing_id.is_none() {
            debug!(target: "savant_rs.zeromq.routing-filter", "Message without routing id always allowed");
            return true;
        }
        let routing_id = routing_id.clone().unwrap();
        let current_valid_routing_id = self.ids.entry(topic.to_vec()).or_insert(routing_id.clone());
        debug!(target: "savant_rs.zeromq.routing-filter",
            "The current registered routing id: {:?}, the received routing id: {:?}",
            current_valid_routing_id, routing_id
        );

        if current_valid_routing_id == &routing_id {
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
        assert!(filter.allow(&topic1, &Some(routing_id.clone())));
        assert!(filter.allow(&topic1, &Some(routing_id2.clone())));
        assert!(!filter.allow(&topic1, &Some(routing_id.clone())));
        assert!(filter.allow(&topic1, &Some(routing_id2.clone())));

        assert!(filter.allow(&topic2, &None));
        assert!(filter.allow(&topic2, &Some(routing_id2.clone())));
        assert!(filter.allow(&topic2, &Some(routing_id.clone())));
        assert!(!filter.allow(&topic2, &Some(routing_id2.clone())));
        assert!(filter.allow(&topic2, &Some(routing_id.clone())));
    }

    #[test]
    fn test_reader_config_default_build_fails() -> anyhow::Result<()> {
        let config = ReaderConfig::new().build();
        assert!(config.is_err());
        Ok(())
    }
}
