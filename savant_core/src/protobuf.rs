use crate::message::{Message, MessageEnvelope, MessageMeta};
use crate::otlp::PropagatedContext;
use savant_protobuf::generated;

mod serialize;

pub use generated::{UserData, VideoFrame, VideoFrameBatch, VideoFrameUpdate, VideoObject};
pub use serialize::from_pb;
pub use serialize::Error;
pub use serialize::ToProtobuf;

impl From<&Message> for generated::Message {
    fn from(m: &Message) -> Self {
        generated::Message {
            protocol_version: m.meta().protocol_version.clone(),
            routing_labels: m.meta().routing_labels.clone(),
            propagated_context: m.meta().span_context.0.clone(),
            seq_id: m.meta().seq_id,
            content: Some(m.payload().into()),
        }
    }
}

impl TryFrom<&generated::Message> for Message {
    type Error = Error;

    fn try_from(m: &generated::Message) -> Result<Self, Self::Error> {
        let (protocol_version, routing_labels, propagated_context, seq_id) = (
            m.protocol_version.clone(),
            m.routing_labels.clone(),
            PropagatedContext(m.propagated_context.clone()),
            m.seq_id,
        );

        let meta = MessageMeta {
            protocol_version,
            routing_labels,
            span_context: propagated_context,
            seq_id,
        };

        let message_content = m
            .content
            .as_ref()
            .expect("Unexpected absense of message content");
        let payload = MessageEnvelope::try_from(message_content)?;

        Ok(Message { meta, payload })
    }
}

pub fn serialize(m: &Message) -> Result<Vec<u8>, Error> {
    use prost::Message as ProstMessage;
    let message = generated::Message::from(m);
    let mut buf = Vec::new();
    message.encode(&mut buf)?;
    Ok(buf)
}

pub fn deserialize(bytes: &[u8]) -> Result<Message, Error> {
    use prost::Message as ProstMessage;
    let message = generated::Message::decode(bytes)?;
    let m = Message::try_from(&message)?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use crate::primitives::eos::EndOfStream;
    use crate::protobuf::{deserialize, serialize};

    #[test]
    fn test_eos_message() {
        let source = "source_id".to_string();
        let eos = crate::message::Message::end_of_stream(EndOfStream::new(source.clone()));
        let serialized = serialize(&eos).unwrap();
        let restored = deserialize(&serialized).unwrap();
        assert_eq!(eos.meta.seq_id, restored.meta.seq_id);
        assert_eq!(eos.meta.routing_labels, restored.meta.routing_labels);
        assert_eq!(eos.meta.span_context.0, restored.meta.span_context.0);
        assert_eq!(eos.meta.protocol_version, restored.meta.protocol_version);
        assert!(
            matches!(eos.payload(), crate::message::MessageEnvelope::EndOfStream(EndOfStream {source_id: v}) if v == &source),
        );
    }
}
