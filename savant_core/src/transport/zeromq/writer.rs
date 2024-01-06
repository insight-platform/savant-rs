use crate::message::Message;
use crate::protobuf::serialize;
use crate::transport::zeromq::{
    create_ipc_dirs, set_ipc_permissions, Socket, WriterConfig, WriterSocketType,
    END_OF_STREAM_MESSAGE, ZMQ_LINGER,
};
use crate::TEST_ENV;
use anyhow::bail;
use log::{debug, info, warn};

pub struct Writer {
    context: Option<zmq::Context>,
    config: WriterConfig,
    socket: Option<Socket>,
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

#[cfg(not(test))]
fn new_socket(config: &WriterConfig, context: &zmq::Context) -> anyhow::Result<Socket> {
    Ok(Socket::ZmqSocket(
        context.socket(config.socket_type().into())?,
    ))
}
#[cfg(test)]
fn new_socket(_config: &WriterConfig, _context: &zmq::Context) -> anyhow::Result<Socket> {
    Ok(Socket::MockSocket(vec![]))
}

impl Writer {
    pub fn new(config: &WriterConfig) -> anyhow::Result<Self> {
        let context = zmq::Context::new();
        let socket: Socket = new_socket(config, &context)?;

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
            "Destroying ZeroMQ socket for endpoint {}",
            self.config.endpoint()
        );
        self.socket.take();
        self.context.take();
        info!(
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
        debug!("Sending message to ZeroMQ socket: {:?}", parts);
        let res = socket.send_multipart(&parts, 0);
        if let Err(e) = res {
            warn!("Failed to send message to ZeroMQ socket. Error is {:?}", e);
            if let zmq::Error::EAGAIN = e {
                return Ok(WriterResult::SendTimeout);
            } else {
                bail!("Failed to send message to ZeroMQ socket. Error is {:?}", e);
            }
        }
        let start = std::time::Instant::now();
        if self.config.socket_type() == &WriterSocketType::Req
            || (m == END_OF_STREAM_MESSAGE && self.config.socket_type() != &WriterSocketType::Pub)
        {
            let mut receive_retries = *self.config.receive_retries();
            while receive_retries >= 0 {
                let res = socket.recv_multipart(0);
                debug!("Received message from ZeroMQ socket: {:?}", res);
                if let Err(e) = res {
                    warn!(
                        "Failed to receive message from ZeroMQ socket. Error is {:?}",
                        e
                    );
                    if let zmq::Error::EAGAIN = e {
                        warn!(
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
                return Ok(WriterResult::Ack {
                    retries_spent: *self.config.receive_retries() - receive_retries,
                    time_spent: start.elapsed().as_millis(),
                });
            }
            return Ok(WriterResult::AckTimeout(start.elapsed().as_millis()));
        }
        let spent = start.elapsed().as_millis();
        debug!("Message sent to ZeroMQ socket. Time spent: {} ms", spent);
        Ok(WriterResult::Success(spent))
    }
}
