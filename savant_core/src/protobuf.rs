use crate::message::Message;

pub(crate) mod generated;
mod serialize;

impl From<Message> for generated::Message {
    fn from(m: Message) -> Self {
        generated::Message {
            lib_version: m.meta().lib_version.clone(),
            routing_labels: m.meta().routing_labels.clone(),
            propagated_context: m.meta().span_context.0.clone(),
            seq_id: m.meta().seq_id,
            content: Some(m.payload().into()),
        }
    }
}

impl TryFrom<generated::Message> for Message {
    type Error = String;

    fn try_from(_: generated::Message) -> Result<Self, Self::Error> {
        todo!()
    }
}

pub fn serialize(_: Message) -> Vec<u8> {
    todo!()
}

pub fn deserialize(_: &[u8]) -> Message {
    todo!()
}
