use crate::message::Message;
use crate::protobuf::serialize;
use crate::transport::zeromq::{
    create_ipc_dirs, set_ipc_permissions, MockSocketResponder, Socket, WriterConfig,
    WriterSocketType, CONFIRMATION_MESSAGE, END_OF_STREAM_MESSAGE, ZMQ_LINGER,
};
use crate::TEST_ENV;
use anyhow::bail;
use log::{debug, info, warn};

pub struct Writer {
    context: Option<zmq::Context>,
    config: WriterConfig,
    socket: Option<Socket<MockResponder>>,
}

impl From<&WriterSocketType> for zmq::SocketType {
    fn from(socket_type: &WriterSocketType) -> Self {
        match socket_type {
            WriterSocketType::Pub => zmq::SocketType::PUB,
            WriterSocketType::Dealer => zmq::SocketType::DEALER,
            WriterSocketType::Req => zmq::SocketType::REQ,
        }
    }
}

#[derive(Debug)]
pub enum WriterResult {
    SendTimeout,
    AckTimeout(u128),
    Ack {
        retries_spent: i32,
        time_spent: u128,
    },
    Success(u128),
}

#[derive(Default)]
struct MockResponder;

#[cfg(test)]
impl MockSocketResponder for MockResponder {
    fn fix(&mut self, data: &mut Vec<Vec<u8>>) {
        if data.len() == 2 && data[1] == END_OF_STREAM_MESSAGE {
            data.pop();
            data.push(CONFIRMATION_MESSAGE.to_vec());
        } else if data.len() == 1 && data[0] == END_OF_STREAM_MESSAGE {
            panic!("Wrong data format, topic is missing");
        } else {
            data.clear();
            data.push(CONFIRMATION_MESSAGE.to_vec());
        }
    }
}
#[cfg(not(test))]
impl MockSocketResponder for MockResponder {}

#[cfg(not(test))]
fn new_socket(
    config: &WriterConfig,
    context: &zmq::Context,
) -> anyhow::Result<Socket<MockResponder>> {
    Ok(Socket::ZmqSocket(
        context.socket(config.socket_type().into())?,
    ))
}
#[cfg(test)]
fn new_socket(
    _config: &WriterConfig,
    _context: &zmq::Context,
) -> anyhow::Result<Socket<MockResponder>> {
    Ok(Socket::MockSocket(vec![], MockResponder {}))
}

impl Writer {
    pub fn new(config: &WriterConfig) -> anyhow::Result<Self> {
        let context = zmq::Context::new();
        let socket = new_socket(config, &context)?;

        socket.set_sndhwm(*config.send_hwm())?;
        socket.set_sndtimeo(*config.send_timeout())?;
        socket.set_linger(ZMQ_LINGER)?;

        if !TEST_ENV && config.endpoint().starts_with("ipc://") {
            create_ipc_dirs(config.endpoint())?;
            if let Some(permissions) = config.fix_ipc_permissions() {
                set_ipc_permissions(config.endpoint(), *permissions)?;
            }
        }

        if *config.bind() {
            socket.bind(config.endpoint())?;
        } else {
            socket.connect(config.endpoint())?;
        }

        Ok(Self {
            context: Some(context),
            config: config.clone(),
            socket: Some(socket),
        })
    }

    pub fn destroy(&mut self) -> anyhow::Result<()> {
        info!(
            target: "savant_rs::zeromq::writer",
            "Destroying ZeroMQ socket for endpoint {}",
            self.config.endpoint()
        );
        self.socket.take();
        self.context.take();
        info!(
            target: "savant_rs::zeromq::writer",
            "ZeroMQ socket for endpoint {} destroyed",
            self.config.endpoint()
        );
        Ok(())
    }

    pub fn is_alive(&self) -> bool {
        self.socket.is_some()
    }

    pub fn send_eos(&mut self, topic: &[u8]) -> anyhow::Result<WriterResult> {
        self.send(topic, END_OF_STREAM_MESSAGE, &[])
    }

    pub fn send_message(
        &mut self,
        topic: &[u8],
        m: &Message,
        extra_parts: &[&[u8]],
    ) -> anyhow::Result<WriterResult> {
        let serialized = serialize(m)?;
        self.send(topic, &serialized, extra_parts)
    }

    fn send(
        &mut self,
        topic: &[u8],
        m: &[u8],
        extra_parts: &[&[u8]],
    ) -> anyhow::Result<WriterResult> {
        if self.socket.is_none() {
            bail!("ZeroMQ socket is no longer alive");
        }
        let socket = self.socket.as_mut().unwrap();
        let extra_parts_iter = extra_parts.iter().cloned();
        let parts = vec![topic, &m]
            .into_iter()
            .chain(extra_parts_iter)
            .collect::<Vec<_>>();
        debug!(
            target: "savant_rs::zeromq::writer",
            "Sending message to ZeroMQ socket: {:?}", parts);
        let mut send_retries = *self.config.send_retries();
        while send_retries >= 0 {
            let res = socket.send_multipart(&parts, 0);
            if let Err(e) = res {
                warn!(
                    target: "savant_rs::zeromq::writer",
                    "Failed to send message to ZeroMQ socket. Error is {:?}", e);
                if let zmq::Error::EAGAIN = e {
                    warn!(
                        target: "savant_rs::zeromq::writer",
                        "Retrying to send message to ZeroMQ socket, retries left: {}",
                        send_retries
                    );
                    send_retries -= 1;
                    continue;
                } else {
                    bail!("Failed to send message to ZeroMQ socket. Error is {:?}", e);
                }
            }
            break;
        }
        let start = std::time::Instant::now();
        if self.config.socket_type() == &WriterSocketType::Req
            || (m == END_OF_STREAM_MESSAGE && self.config.socket_type() != &WriterSocketType::Pub)
        {
            let mut receive_retries = *self.config.receive_retries();
            while receive_retries >= 0 {
                let res = socket.recv_multipart(0);
                debug!(
                    target: "savant_rs::zeromq::writer",
                    "Received message from ZeroMQ socket: {:?}", res);
                if let Err(e) = res {
                    warn!(
                        target: "savant_rs::zeromq::writer",
                        "Failed to receive message from ZeroMQ socket. Error is {:?}",
                        e
                    );
                    if let zmq::Error::EAGAIN = e {
                        warn!(
                            target: "savant_rs::zeromq::writer",
                            "Retrying to receive message from ZeroMQ socket, retries left: {}",
                            receive_retries
                        );
                        receive_retries -= 1;
                        continue;
                    } else {
                        bail!(
                            "Failed to receive message from ZeroMQ socket. Error is {:?}",
                            e
                        );
                    }
                }
                if m == END_OF_STREAM_MESSAGE {
                    let res = res.as_ref().unwrap();
                    if res.last().unwrap().as_slice() != CONFIRMATION_MESSAGE {
                        bail!(
                            "Failed to receive confirmation message from ZeroMQ socket. \
                            Received message is {:?}",
                            res
                        );
                    }
                }
                return Ok(WriterResult::Ack {
                    retries_spent: *self.config.receive_retries() - receive_retries,
                    time_spent: start.elapsed().as_millis(),
                });
            }
            return Ok(WriterResult::AckTimeout(start.elapsed().as_millis()));
        }
        let spent = start.elapsed().as_millis();
        debug!(
            target: "savant_rs::zeromq::writer",
            "Message sent to ZeroMQ socket. Time spent: {} ms", spent);
        Ok(WriterResult::Success(spent))
    }
}

#[cfg(test)]
mod tests {
    mod tests_with_response {
        use super::super::{Writer, WriterConfig};
        use crate::message::Message;
        use crate::test::gen_frame;
        use crate::transport::zeromq::WriterResult;
        #[test]
        fn test_dealer_eos_op() -> anyhow::Result<()> {
            let mut writer = Writer::new(
                &WriterConfig::new()
                    .url("dealer+bind:ipc:///tmp/test")?
                    .with_receive_retries(3)?
                    .build()?,
            )?;
            let res = writer.send_eos(b"test")?;
            assert!(matches!(res, WriterResult::Ack {
                retries_spent,
                time_spent: _
            } if retries_spent == 0));
            Ok(())
        }
        #[test]
        fn test_req_eos_op() -> anyhow::Result<()> {
            let mut writer = Writer::new(
                &WriterConfig::new()
                    .url("req+bind:ipc:///tmp/test")?
                    .with_receive_retries(3)?
                    .build()?,
            )?;
            let res = writer.send_eos(b"test")?;
            assert!(matches!(res, WriterResult::Ack {
                retries_spent,
                time_spent: _
            } if retries_spent == 0));
            Ok(())
        }
        #[test]
        fn test_req_message_op() -> anyhow::Result<()> {
            let mut writer = Writer::new(
                &WriterConfig::new()
                    .url("req+bind:ipc:///tmp/test")?
                    .with_receive_retries(3)?
                    .build()?,
            )?;
            let m = Message::video_frame(&gen_frame());
            let res = writer.send_message(b"test", &m, &[b"abc"])?;
            assert!(matches!(res, WriterResult::Ack {
                retries_spent,
                time_spent: _
            } if retries_spent == 0));
            Ok(())
        }
    }

    mod tests_without_response {
        use crate::message::Message;
        use crate::test::gen_frame;
        use crate::transport::zeromq::{Writer, WriterConfig, WriterResult};

        #[test]
        fn test_dealer_op() -> anyhow::Result<()> {
            let mut writer = Writer::new(
                &WriterConfig::new()
                    .url("dealer+bind:ipc:///tmp/test")?
                    .with_receive_retries(3)?
                    .build()?,
            )?;
            let m = Message::video_frame(&gen_frame());
            let res = writer.send_message(b"test", &m, &[b"abc"])?;
            assert!(matches!(res, WriterResult::Success(_)));
            Ok(())
        }
    }
}
