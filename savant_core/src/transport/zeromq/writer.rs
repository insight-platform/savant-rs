use crate::message::Message;
use crate::primitives::eos::EndOfStream;
use crate::protobuf::{deserialize, serialize};
use crate::transport::zeromq::{
    create_ipc_dirs, set_ipc_permissions, MockSocketResponder, Socket, SocketProvider,
    WriterConfig, WriterSocketType, CONFIRMATION_MESSAGE, ZMQ_LINGER,
};
use crate::utils::bytes_to_hex_string;
use anyhow::bail;
use log::{debug, info, warn};
use std::str::from_utf8;

pub struct Writer<R: MockSocketResponder, P: SocketProvider<R>> {
    context: Option<zmq::Context>,
    config: WriterConfig,
    socket: Option<Socket<R>>,
    phony: std::marker::PhantomData<P>,
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
        send_retries_spent: i32,
        receive_retries_spent: i32,
        time_spent: u128,
    },
    Success {
        retries_spent: i32,
        time_spent: u128,
    },
}

#[derive(Default)]
struct MockResponder;

impl MockSocketResponder for MockResponder {
    fn fix(&mut self, data: &mut Vec<Vec<u8>>) {
        if data.len() == 2 {
            let eos = deserialize(&data[1]);
            if eos.is_ok() {
                data.pop();
                data.push(CONFIRMATION_MESSAGE.to_vec());
            }
        } else if data.len() == 1 {
            panic!("Wrong data format, topic is missing");
        } else {
            data.clear();
            data.push(CONFIRMATION_MESSAGE.to_vec());
        }
    }
}

impl<R: MockSocketResponder, P: SocketProvider<R> + Default> Writer<R, P> {
    pub fn new(config: &WriterConfig) -> anyhow::Result<Self> {
        let context = zmq::Context::new();
        let p = P::default();
        let socket = p.new_socket(&context, config.socket_type().into())?;

        socket.set_sndhwm(*config.send_hwm())?;
        socket.set_sndtimeo(*config.send_timeout())?;
        socket.set_linger(ZMQ_LINGER)?;

        if *config.socket_type() != WriterSocketType::Pub {
            socket.set_rcvtimeo(*config.receive_timeout())?;
            socket.set_rcvhwm(*config.receive_hwm())?;
        }

        if *config.bind() {
            if matches!(&socket, Socket::ZmqSocket(_)) && config.endpoint().starts_with("ipc://") {
                create_ipc_dirs(config.endpoint())?;
            }

            socket.bind(config.endpoint())?;

            if matches!(&socket, Socket::ZmqSocket(_)) && config.endpoint().starts_with("ipc://") {
                if let Some(permissions) = config.fix_ipc_permissions() {
                    set_ipc_permissions(config.endpoint(), *permissions)?;
                }
            }
        } else {
            socket.connect(config.endpoint())?;
        }

        Ok(Self {
            context: Some(context),
            config: config.clone(),
            socket: Some(socket),
            phony: std::marker::PhantomData,
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

    pub fn is_started(&self) -> bool {
        self.socket.is_some()
    }

    pub fn send_eos(&mut self, topic: &str) -> anyhow::Result<WriterResult> {
        let m = Message::end_of_stream(EndOfStream::new(topic.to_string()));
        self.send(topic.as_bytes(), &m, &[])
    }

    pub fn send_message(
        &mut self,
        topic: &str,
        m: &Message,
        extra_parts: &[&[u8]],
    ) -> anyhow::Result<WriterResult> {
        self.send(topic.as_bytes(), m, extra_parts)
    }

    fn send(
        &mut self,
        topic: &[u8],
        m: &Message,
        extra_parts: &[&[u8]],
    ) -> anyhow::Result<WriterResult> {
        if self.socket.is_none() {
            bail!("ZeroMQ socket is no longer alive");
        }
        let socket = self.socket.as_mut().unwrap();
        let extra_parts_iter = extra_parts.iter().cloned();
        let serialized_message = serialize(m)?;
        let parts = vec![topic, &serialized_message]
            .into_iter()
            .chain(extra_parts_iter)
            .collect::<Vec<_>>();
        debug!(
            target: "savant_rs::zeromq::writer",
            "Sending message to ZeroMQ socket: {} {:?}",
            from_utf8(topic).unwrap_or(&bytes_to_hex_string(topic)),
            m);
        let mut send_retries = *self.config.send_retries();
        while send_retries >= 0 {
            let res = socket.send_multipart(&parts, 0);
            if let Err(e) = res {
                warn!(
                    target: "savant_rs::zeromq::writer",
                    "Failed to send message to ZeroMQ socket. Error is [{}] {:?}", e.to_raw(), e);
                if let zmq::Error::EAGAIN = e {
                    warn!(
                        target: "savant_rs::zeromq::writer",
                        "Retrying to send message to ZeroMQ socket, retries left: {}",
                        send_retries
                    );
                    send_retries -= 1;
                    continue;
                } else {
                    bail!(
                        "Failed to send message to ZeroMQ socket. Error is [{}] {:?}",
                        e.to_raw(),
                        e
                    );
                }
            }
            break;
        }

        if send_retries < 0 {
            warn!(
                target: "savant_rs::zeromq::writer",
                "Failed to send message to ZeroMQ socket. Send retries spent: {}",
                *self.config.send_retries()
            );
            return Ok(WriterResult::SendTimeout);
        }

        let start = std::time::Instant::now();
        if self.config.socket_type() == &WriterSocketType::Req
            || (m.is_end_of_stream() && self.config.socket_type() != &WriterSocketType::Pub)
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
                        "Failed to receive message from ZeroMQ socket. Error is [{}] {:?}",
                        e.to_raw(),
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
                            "Failed to receive message from ZeroMQ socket. Error is [{}] {:?}",
                            e.to_raw(),
                            e
                        );
                    }
                }
                if m.is_end_of_stream() {
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
                    send_retries_spent: *self.config.send_retries() - send_retries,
                    receive_retries_spent: *self.config.receive_retries() - receive_retries,
                    time_spent: start.elapsed().as_millis(),
                });
            }
            return Ok(WriterResult::AckTimeout(start.elapsed().as_millis()));
        }
        let spent = start.elapsed().as_millis();
        debug!(
            target: "savant_rs::zeromq::writer",
            "Message sent to ZeroMQ socket. Time spent: {} ms", spent);
        Ok(WriterResult::Success {
            retries_spent: *self.config.send_retries() - send_retries,
            time_spent: spent,
        })
    }
}

#[cfg(test)]
mod tests {
    mod tests_with_response {
        use super::super::{Writer, WriterConfig};
        use crate::message::Message;
        use crate::test::gen_frame;
        use crate::transport::zeromq::writer::MockResponder;
        use crate::transport::zeromq::{MockSocketProvider, WriterResult};

        #[test]
        fn test_dealer_eos_op() -> anyhow::Result<()> {
            let mut writer = Writer::<MockResponder, MockSocketProvider>::new(
                &WriterConfig::new()
                    .url("dealer+bind:ipc:///tmp/test")?
                    .with_receive_retries(3)?
                    .build()?,
            )?;
            let res = writer.send_eos("test")?;
            assert!(matches!(res, WriterResult::Ack {
                receive_retries_spent,
                send_retries_spent,
                time_spent: _
            } if receive_retries_spent == 0 && send_retries_spent == 0));
            Ok(())
        }
        #[test]
        fn test_req_eos_op() -> anyhow::Result<()> {
            let mut writer = Writer::<MockResponder, MockSocketProvider>::new(
                &WriterConfig::new()
                    .url("req+bind:ipc:///tmp/test")?
                    .with_receive_retries(3)?
                    .build()?,
            )?;
            let res = writer.send_eos("test")?;
            assert!(matches!(res, WriterResult::Ack {
                receive_retries_spent,
                send_retries_spent,
                time_spent: _
            } if receive_retries_spent == 0 && send_retries_spent == 0));
            Ok(())
        }
        #[test]
        fn test_req_message_op() -> anyhow::Result<()> {
            let mut writer = Writer::<MockResponder, MockSocketProvider>::new(
                &WriterConfig::new()
                    .url("req+bind:ipc:///tmp/test")?
                    .with_receive_retries(3)?
                    .build()?,
            )?;
            let m = Message::video_frame(&gen_frame());
            let res = writer.send_message("test", &m, &[b"abc"])?;
            assert!(matches!(res, WriterResult::Ack {
                receive_retries_spent,
                send_retries_spent,
                time_spent: _
            } if receive_retries_spent == 0 && send_retries_spent == 0));
            Ok(())
        }
    }

    mod tests_without_response {
        use crate::message::Message;
        use crate::test::gen_frame;
        use crate::transport::zeromq::writer::MockResponder;
        use crate::transport::zeromq::{MockSocketProvider, Writer, WriterConfig, WriterResult};

        #[test]
        fn test_dealer_op() -> anyhow::Result<()> {
            let mut writer = Writer::<MockResponder, MockSocketProvider>::new(
                &WriterConfig::new()
                    .url("dealer+bind:ipc:///tmp/test")?
                    .with_receive_retries(3)?
                    .build()?,
            )?;
            let m = Message::video_frame(&gen_frame());
            let res = writer.send_message("test", &m, &[b"abc"])?;
            assert!(matches!(res, WriterResult::Success {
                retries_spent,
                time_spent: _
            } if retries_spent == 0));
            Ok(())
        }
    }
}
